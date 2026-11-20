"""One edge node: everything that runs beside one robot.

A node owns one site, one robot backend, one database, the tool registry, the runbooks, and the
runs in progress. The console, the chat path, the trial pipeline and the verification scripts
all go through it, so there is one place where a site's state lives and one place that stops it.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from . import access
from .config import get_settings
from .robot.needle import DeclaredNeedle
from .robot.site import build_site
from .runtime.graph import Runner, apply_restart_rule
from .runtime.trace import Tracer
from .store import Database, dumps
from .tools.registry import Principal, ToolContext, ToolRegistry


class Bus:
    """Events for the console, with increasing ids so a page that reconnects can ask for what it
    missed. Kept in memory; a restart starts a new stream."""

    def __init__(self, keep: int = 2000):
        self._events: list[dict[str, Any]] = []
        self._next = 1
        self._keep = keep
        self._cond = threading.Condition()

    def publish(self, kind: str, data: dict[str, Any]) -> int:
        with self._cond:
            ev = {"id": self._next, "kind": kind, "at": time.time(), "data": data}
            self._next += 1
            self._events.append(ev)
            del self._events[:-self._keep]
            self._cond.notify_all()
            return ev["id"]

    def since(self, last_id: int, wait_s: float = 0.0) -> list[dict[str, Any]]:
        with self._cond:
            out = [e for e in self._events if e["id"] > last_id]
            if not out and wait_s > 0:
                self._cond.wait(wait_s)
                out = [e for e in self._events if e["id"] > last_id]
            return out


@dataclass
class RunBox:
    """A copilot run in progress or finished, with what is needed to resume or stop it."""
    run_id: str
    who: Principal
    agent: str
    session: str
    ctx: ToolContext
    cfg: Any
    state: Any = None
    done: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def build_registry() -> tuple[ToolRegistry, Any, Any]:
    """Every tool of the system and the runbooks. Which of them a person may use is decided per
    call from the principal; which of them an agent is offered is in vxagent/profiles.py."""
    from .fleet.tools import register_fleet
    from .service.tools import register_service_work
    from .service.tickets import register_tickets
    from .skills.index import default_library, register_bench, register_skills
    from .tools.builtin import register_builtin
    from .tools.dataset import register_dataset
    from .tools.knowledge import register_knowledge
    from .tools.memory import register_memory
    from .tools.service import register_service
    from .tools.trialtools import register_trial_tools

    reg = register_builtin()
    # The diagnostic checks of task 04 work on one recorded run and are started on a ticket
    # (vxagent/service/diagnosis.py); they are not offered to a copilot as tools.
    for add in (register_knowledge, register_memory, register_service, register_bench,
                register_dataset, register_tickets, register_trial_tools, register_service_work, register_fleet):
        add(reg)
    library = default_library(reg, strict=True)          # a node has every tool, so every runbook must load
    catalogue = register_skills(reg, library, [n for n in reg.names()])
    return reg, library, catalogue


class Node:
    def __init__(self, site: Any, spec: dict[str, Any], db: Database, *, llm: Any = None, small_llm: Any = None,
                 knowledge: Any = None, tick: Callable[[float], None] = time.sleep, rosma_dir: Path | None = None,
                 dvrk_dir: Path | None = None, config_dir: Path | None = None, site_file: Path | None = None,
                 perception: Callable[[], dict[str, Any]] | None = None, fleet: Any = None, software: str = "",
                 demo_state: bool = True):
        self.site, self.spec, self.db = site, spec, db
        self.name = site.name
        self.llm, self.small_llm, self.knowledge, self.tick = llm, small_llm or llm, knowledge, tick
        self.rosma_dir, self.dvrk_dir, self.site_file = rosma_dir, dvrk_dir, site_file
        self.robot = site.twin()
        self.backend = "twin"
        self.bus = Bus()
        self.tracer = Tracer(db, on_event=lambda e: self.bus.publish("span", e))
        self.registry, self.library, self.catalogue = build_registry()
        self.needle = DeclaredNeedle(site).attach(self.robot)
        self.perception = perception if perception is not None else self._declared_perception
        self.boxes: dict[str, RunBox] = {}
        self.trials: dict[str, Any] = {}                  # run id -> Env of a trial
        self.policy: dict[str, Any] | None = None         # contract and demonstrations, loaded on first use
        self.fleet = fleet                                # the outbox of this node (task 13), or None
        self.software = software or spec.get("software_release", "")
        self.started = time.time()
        self.lock = threading.RLock()
        self.store = self.schemas = self.instruments = None
        if dvrk_dir is not None and site_file is not None and config_dir is not None:
            from .robot import config as cfg
            from .structured.rules import InstrumentSet
            self.store = cfg.ConfigStore(config_dir, cfg.site_sources(site_file, dvrk_dir))
            self.schemas = cfg.SchemaSet(dvrk_dir / "share" / "schemas")
            self.instruments = InstrumentSet(dvrk_dir / "share" / "tool")
        self.robot.listeners.append(lambda s: self.bus.publish("robot", {"seq": s.seq}))
        access.ready(db)
        # Start-up rule: after a restart the arm is assumed disabled, and runs that were moving
        # must be looked at by a person before they go on.
        self.needs_reverification = apply_restart_rule(db)
        self.adopt_stored_site()
        if demo_state:
            self.declare_demo_state()

    # ------------------------------------------------------------------ construction

    @classmethod
    def open(cls, site_name: str = "site_a", db: Database | None = None, runtime: Path | None = None, **kw: Any) -> "Node":
        from .robot.kinematics import load_chain
        s = get_settings()
        path = s.sites_dir / f"{site_name}.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec["_file"] = path.name
        chain = load_chain(s.dvrk_dir / spec["kinematic_file"], s.dvrk_dir / spec["instrument_file"])
        site = build_site(spec, chain)
        runtime = (runtime or s.runtime_dir) / site.name
        runtime.mkdir(parents=True, exist_ok=True)
        if "rosma_dir" not in kw:
            # Recordings belong to a site. This demonstration gives the ROSMA recordings to site A.
            own = s.rosma_dir if spec.get("recordings") == "rosma" else runtime / "episodes"
            own.mkdir(parents=True, exist_ok=True)
            kw["rosma_dir"] = own
        return cls(site, spec, db or Database(runtime / "node.sqlite"), dvrk_dir=s.dvrk_dir, site_file=path,
                   config_dir=runtime / "config", **kw)

    def declare_demo_state(self) -> None:
        """The state a demonstration starts from: a calibration record and scene records of one
        session. Both are DECLARED; nothing was calibrated."""
        from .diagnostics.episode import calibration
        from .structured import proposals
        from .tools.service import load_calibration_record, store_calibration_record
        if load_calibration_record(self.db, self.name) is None:
            cal = calibration(None, self.site.scene.workspace_center, np.random.default_rng(17))
            store_calibration_record(self.db, self.name, {
                "id": cal["id"], "T_est": cal["T_est"].tolist(), "seen_m": cal["seen_m"].tolist(),
                "touched_m": cal["touched_m"].tolist(), "method": "point pairs (declared for the demonstration)",
                "session": "s1"})
            proposals.set_calibration(self.db, self.name, True, "calibrated (declared for the demonstration)")
            self.db.execute("INSERT OR REPLACE INTO kv(k,v) VALUES (?,?)", (f"scene_session:{self.name}", json.dumps("s1")))

    def _declared_perception(self) -> dict[str, Any]:
        seen = self.needle.observe()
        return {"visible": bool(seen.get("visible", True)), "latency_s": 0.04, "source": "declared"}

    def adopt_stored_site(self) -> bool:
        """At start-up: if changes were applied to the site file before this process started, the
        site in force is the stored version, not the file as published. (Found in the browser run:
        after a restart an applied limit was shown, and enforced, as it was before the change.)"""
        if self.store is None or not self.store.versions(self.site_target):
            return False
        self.reload_site()
        return True

    def reload_site(self) -> None:
        """Take the site description from the configuration store after a change was applied."""
        if self.store is None:
            return
        doc, _, _ = self.store.current(self.site_target)
        doc = dict(doc)
        doc["_file"] = self.spec.get("_file", "")
        with self.lock:
            new = build_site(doc, self.site.chain)
            self.site, self.spec = new, doc
            self.needle.site = new
        if self.fleet is not None:                          # the fleet sees that this site's configuration moved on
            self.fleet.meta["config_version"] = str(self.store.current(self.site_target)[2])
        self.bus.publish("site", {"profile_version": str(new.profile.version)})

    @property
    def site_target(self) -> str:
        return f"site/{self.site_file.name}" if self.site_file is not None else "site"

    # ------------------------------------------------------------------ contexts and runs

    def ctx(self, who: Principal, run_id: str) -> ToolContext:
        from .tools.memory import fact_sink
        ctx = ToolContext(principal=who, db=self.db, tracer=self.tracer, run_id=run_id, robot=self.robot, site=self.site,
                          tick=self.tick)
        ctx.extras.update({"rosma_dir": self.rosma_dir, "library": self.library, "needle": self.needle, "node": self})
        if self.knowledge is not None:
            ctx.extras["knowledge"] = self.knowledge
        ctx.extras["on_result"] = fact_sink(ctx)
        return ctx

    def new_run_id(self, prefix: str = "run") -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def stop_all(self, by: str) -> dict[str, Any]:
        """The operator's stop. It never goes through a model: it sets the stop flag of every run
        of this node, and the code that is moving the arm sees it within one control tick."""
        stopped = []
        for box in list(self.boxes.values()):
            if not box.done:
                box.ctx.stop_event.set()
                stopped.append(box.run_id)
        for run_id, env in list(self.trials.items()):
            env.stop_event.set()
            stopped.append(run_id)
        self.db.audit(by, "-", "stop", self.name, True, f"{len(stopped)} runs")
        self.bus.publish("stop", {"by": by, "runs": stopped})
        return {"stopped_runs": stopped, "robot_seq": self.robot.state().seq}

    # ------------------------------------------------------------------ the policy trial

    def load_policy(self) -> dict[str, Any]:
        if self.policy is None:
            from .data import quality, surgicai
            from .policy.contract import contract_from_demonstrations
            demos = quality.from_surgicai("Approach")
            version = json.loads((surgicai.folder() / "MANIFEST.json").read_text(encoding="utf-8"))["tree"][:10]
            contract, _ = contract_from_demonstrations(demos, None, version)
            self.policy = {"contract": contract, "source": "SurgicAI expert demonstrations (Approach)",
                           "demos": [{"id": e.episode_id, "poses": np.array(e.notes["poses"], dtype=float), "actions": e.actions,
                                      "goal": e.notes["goal"]} for e in demos],
                           "seen": np.concatenate([e.position for e in demos[:5]])}
        return self.policy

    def trial_extras(self, demo: int = 0, needle: dict[str, bool] | None = None) -> dict[str, Any]:
        from .policy import compat
        from .policy.adapter import Adapter
        pol = self.load_policy()
        d = pol["demos"][demo % len(pol["demos"])]
        ad = Adapter.anchored(pol["contract"], d["goal"], self.site.poses["pregrasp"].T)
        if needle is not None:
            self.needle = DeclaredNeedle(self.site, **needle).attach(self.robot)
        cp = compat.check(pol["contract"], self.site, "start", "pregrasp", 10.0, policy_start=d["poses"][0],
                          policy_goal=d["goal"], seen_positions=pol["seen"])
        tool_dir = self.dvrk_dir / "share" / "tool" if self.dvrk_dir is not None else None
        return {"contract": pol["contract"], "adapter": ad, "demo": d, "needle": self.needle, "compat": cp,
                "perception": self.perception, "tool_dir": tool_dir}

    def trial_sheet(self, operator: str, extras: dict[str, Any]) -> Any:
        from .pipelines import trial
        d = extras["demo"]
        return trial.make_sheet(self.site, operator, {"dataset": "SurgicAI", "episode_id": d["id"], "arm": "PSM1",
                                                      "first_sample": 0, "last_sample": len(d["actions"]), "start_s": 0.0,
                                                      "end_s": len(d["actions"]) / 10.0})

    def preflight_rows(self, operator: str, demo: int = 0) -> list[dict[str, Any]]:
        from .pipelines import policy_trial as PT
        from .policy import compat
        from .robot.preflight import run_preflight
        x = self.trial_extras(demo)
        rt = SimpleNamespace(env=SimpleNamespace(extras=x, site=self.site))
        poses = PT.trial_poses(rt)
        sheet = self.trial_sheet(operator, x).model_dump()
        return run_preflight(self.robot, self.site, sheet, self.db, start_T=poses["start"], goal_T=poses["goal"],
                             compat=x["compat"], in_range=compat.in_training_range(x["contract"], poses["start"], poses["goal"]),
                             perception=x["perception"], tool_dir=x["tool_dir"])

    def rehearse_trial(self, operator: str, demo: int = 0) -> dict[str, Any]:
        from .pipelines import policy_trial as PT
        x = self.trial_extras(demo)
        sheet = self.trial_sheet(operator, x).model_dump()
        rt = SimpleNamespace(env=SimpleNamespace(extras=x, site=self.site), state={"sheet": sheet})
        steps, rejections = PT.run_rehearsal(self.robot, self.site, PT.stage_sequence(rt))
        return {"on": "a copy of the twin; the arm did not move", "rollout": x["demo"]["id"], "control_steps": steps,
                "expected_s": round(sum(steps.values()) / self.robot.rate_hz, 1), "rejections": rejections,
                "clean": not rejections}

    def start_trial(self, who: Principal, demo: int = 0, needle: dict[str, bool] | None = None, use_llm: bool = True) -> Any:
        from .pipelines import policy_trial as PT
        from .pipelines import trial
        x = self.trial_extras(demo, needle)
        env = trial.make_env(self.site, self.robot, self.db, llm=self.llm if use_llm else None, extras=x, tick=self.tick,
                             on_event=lambda ev, st: self.bus.publish("trial", {"run_id": ev.get("run_id"), "n": ev.get("n"),
                                                                                "type": ev.get("type")}))
        run_id = self.new_run_id("trial")
        self.trials[run_id] = env
        res = PT.start(env, self.trial_sheet(who.user_id, x), who, run_id)
        self._after_trial(res)
        return res

    def resume_trial(self, run_id: str, by: Principal | None = None) -> Any:
        from .pipelines import policy_trial as PT
        env = self.trials.get(run_id)
        if env is None:
            raise KeyError(f"no trial {run_id} on this node")
        res = Runner(PT.build_graph(), env).resume(run_id, by)
        self._after_trial(res)
        return res

    def _after_trial(self, res: Any) -> None:
        if res.status == "finished":
            self.trials.pop(res.run_id, None)
            failed = (res.state["data"].get("preflight") or {}).get("failed") or []
            if self.fleet is not None:
                self.fleet.record_trial(res, failed)

    # ------------------------------------------------------------------ health, without data access

    def health(self) -> dict[str, Any]:
        import shutil
        s = get_settings()
        out: dict[str, Any] = {"site": self.name, "backend": self.backend, "software": self.software,
                               "uptime_s": round(time.time() - self.started, 1)}
        try:
            st = self.robot.state()
            out["robot"] = {"reachable": True, "ready": bool(st.ready), "fault": bool(st.fault)}
        except Exception as exc:                                             # a backend that fails is not reachable
            out["robot"] = {"reachable": False, "error": type(exc).__name__}
        ping = getattr(self.llm, "health", None)
        out["model_server"] = ping() if callable(ping) else {"reachable": self.llm is not None, "checked": False}
        kn = getattr(self.knowledge, "health", None)
        out["retrieval"] = kn() if callable(kn) else {"configured": self.knowledge is not None, "checked": False}
        t0 = time.perf_counter()
        try:
            seen = self.perception()
            out["perception"] = {"reachable": True, "latency_s": round(float(seen.get("latency_s", time.perf_counter() - t0)), 3)}
        except Exception as exc:
            out["perception"] = {"reachable": False, "error": type(exc).__name__}
        out["disk_free_gb"] = round(shutil.disk_usage(str(s.data_dir if s.data_dir.exists() else Path("."))).free / 1e9, 1)
        out["pending_approvals"] = len(self.db.query("SELECT 1 FROM approvals WHERE status='pending'"))
        out["outbox_backlog"] = self.fleet.backlog() if self.fleet is not None else None
        out["runs_needing_reverification"] = len(self.needs_reverification)
        return out


def robot_view(node: Node) -> dict[str, Any]:
    """What the console shows of the robot: state read by code, never by a model."""
    from .robot.motion import state_report
    out = state_report(node.robot.state(), node.site.scene)
    out.update({"site": node.name, "arm": node.site.arm, "backend": node.backend, "read_at": time.time(),
                "safety_profile": f"{node.site.profile.name} v{node.site.profile.version}",
                "needle": {**node.needle.observe(), "source": "DECLARED"}})
    return out

