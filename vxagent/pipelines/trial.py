"""The trial pipeline: one attempt at reach, grasp, hold check and place on a physical arm.

    preflight -+-> stop -> END
               +-> dry_run -+-> stop -> END
                            +-> operator_go -+-> stop -> END            (rejected, or nobody answered)
                                             +-> reach -> grasp -> hold_check -> place --+
                                                    |        |          |          |      |
                                                    +--------+----------+----------+-> recover
                                                                                          |
                                                                  verdict -> confirm -> END

It is a fixed workflow. The language model writes two texts (why a pre-flight failed, and the
summary at the end) and neither can change the route.

The twin has no needle. Whether the needle is held comes from an outcome that is declared for
the run before it starts and recorded as declared; on a real arm it would come from a sensor
or from the camera.
"""
from __future__ import annotations

import json
import math
import re
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ..data import episodes as ep_store
from ..llm.base import LLMError
from ..robot.kinematics import pose_distance
from ..robot.motion import MotionStopped, check_goal, execute_path
from ..runtime.events import EventLog
from ..runtime.graph import END, Env, Graph, Limits, Runner, RunResult, Step, approval_gate
from ..runtime.trace import Tracer
from ..runtime.watchdog import jaw, move
from ..store import Database, dumps
from ..structured import proposals
from ..structured.schemas import RunSheet
from ..tools.registry import Principal
from . import paths

RELEASE_HEIGHT_MM = 10.0       # the jaw is opened during recovery only this close to the pad, or with no needle
AT_POSE_MM = 1.0


# ------------------------------------------------------------------------------ what the run is given

def make_sheet(site: Any, operator: str, path_source: dict[str, Any] | None = None,
               recovery_preapproved: bool = True) -> RunSheet:
    return RunSheet(site=site.name, arm=site.arm, instrument=Path(site.chain.source.get("instrument", "")).stem,
                    stages=["reach", "grasp", "hold_check", "place"], envelope_profile=site.profile.name,
                    envelope_version=str(site.profile.version), start_pose="start", grasp_pose="pregrasp",
                    place_pose="place",
                    acceptance={"reach_tolerance_mm": 1.0, "hold_required": True, "max_recoveries": 1},
                    operator=operator, path_source=path_source, outcome_source="declared",
                    recovery_preapproved=recovery_preapproved)


def declare_outcome(db: Database, run_id: str, held: bool, declared_by: str) -> None:
    """DECLARED, not measured: whether the needle is in the jaw once it has closed."""
    db.execute("INSERT OR REPLACE INTO kv(k,v) VALUES (?,?)",
               (f"outcome:{run_id}", dumps({"held": held, "declared_by": declared_by, "at": time.time()})))


def prepare_path(db: Database, site: Any, sheet: RunSheet, run_id: str, rosma_dir: Path | None = None) -> dict[str, Any]:
    """Map the recorded stretch named on the run sheet into the site and keep the waypoints
    with the run. Without a path source the approach is a straight line."""
    start, goal = site.poses[sheet.start_pose].T, site.poses[sheet.grasp_pose].T
    if sheet.path_source is None:
        way, info = [goal.copy()], {"source": "straight line", "waypoints": 1}
    else:
        src = sheet.path_source
        ep = ep_store.get_episode(src.episode_id, rosma_dir)
        if ep is None:
            raise FileNotFoundError(f"no recording {src.episode_id!r}")
        t, p = paths.positions(ep, src.arm)
        seg = p[src.first_sample:src.last_sample + 1]
        way, info = paths.fit_path(seg, start, goal)
        info.update({"source": f"{src.dataset} {src.episode_id} {src.arm} {src.start_s}-{src.end_s} s",
                     "recorded_samples": int(len(seg)),
                     "recorded_net_mm": round(float(np.linalg.norm(seg[-1] - seg[0])) * 1000.0, 2),
                     "recorded_path_mm": round(float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()) * 1000.0, 2)})
    db.execute("INSERT OR REPLACE INTO kv(k,v) VALUES (?,?)",
               (f"path:{run_id}", dumps({"info": info, "waypoints": [w.tolist() for w in way]})))
    return info


def _waypoints(rt: Runner) -> list[np.ndarray]:
    cache = rt.env.extras.setdefault("waypoints", {})
    if rt.run_id not in cache:
        row = rt.env.db.one("SELECT v FROM kv WHERE k=?", (f"path:{rt.run_id}",))
        cache[rt.run_id] = [np.array(w, dtype=float) for w in json.loads(row["v"])["waypoints"]] if row else []
    return cache[rt.run_id]


# ------------------------------------------------------------------------------ looking at the robot

def needle_held(rt: Runner) -> bool:
    d = rt.state.get("data", {})
    if not d.get("grasp_done") or d.get("released"):
        return False
    row = rt.env.db.one("SELECT v FROM kv WHERE k=?", (f"outcome:{rt.run_id}",))
    return bool(json.loads(row["v"])["held"]) if row else False


def observe(rt: Runner) -> dict[str, Any]:
    s, site = rt.env.robot.state(), rt.env.site
    p = s.measured_cp[:3, 3]
    return {"operating": {"powered": s.powered, "enabled": s.enabled, "homed": s.homed, "fault": s.fault, "seq": s.seq},
            "pose": {"position_mm": [round(float(x) * 1000, 2) for x in p],
                     "height_over_pad_mm": round(site.scene.height(p) * 1000, 2),
                     "jaw_deg": round(math.degrees(s.jaw), 1), "seq": s.seq},
            "needle_held": {"value": needle_held(rt), "seq": s.seq, "source": rt.state["sheet"]["outcome_source"]}}


def _mm_to(rt: Runner, pose: str) -> float:
    return pose_distance(rt.env.robot.state().measured_cp, rt.env.site.poses[pose].T)[0] * 1000.0


def _ready(rt: Runner) -> tuple[bool, str]:
    s = rt.env.robot.state()
    return s.ready, f"powered {s.powered}, enabled {s.enabled}, homed {s.homed}, fault {s.fault}"


def _no_fault(rt: Runner) -> tuple[bool, str]:
    s = rt.env.robot.state()
    return s.powered and s.fault is None, f"powered {s.powered}, fault {s.fault}"


def _calibration(rt: Runner) -> tuple[bool, str]:
    c = proposals.calibration(rt.env.db, rt.env.site.name)
    return bool(c["valid"]), c["why"]


def _envelope(rt: Runner) -> tuple[bool, str]:
    sheet, prof = rt.state["sheet"], rt.env.site.profile
    same = sheet["envelope_profile"] == prof.name and sheet["envelope_version"] == str(prof.version)
    return same, (f"run sheet: {sheet['envelope_profile']} v{sheet['envelope_version']}; "
                  f"site now: {prof.name} v{prof.version}")


def _go(rt: Runner) -> tuple[bool, str]:
    return bool(rt.state["task"]["go"]), "the operator's go for this run"


def _at_grasp(rt: Runner) -> tuple[bool, str]:
    d = _mm_to(rt, rt.state["sheet"]["grasp_pose"])
    return d <= rt.state["sheet"]["acceptance"]["reach_tolerance_mm"], f"{d:.2f} mm from the grasp pose"


def _held(rt: Runner) -> tuple[bool, str]:
    return needle_held(rt), f"needle held: {needle_held(rt)} ({rt.state['sheet']['outcome_source']})"


CHECKS = {"arm_ready": _ready, "no_fault": _no_fault, "calibration_valid": _calibration,
          "envelope_unchanged": _envelope, "go_granted": _go, "at_grasp_pose": _at_grasp, "needle_held": _held}


# ------------------------------------------------------------------------------ the two texts

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def model_text(rt: Runner, purpose: str, facts: dict[str, Any], template: str) -> tuple[str, str]:
    """A short text for the operator. The model gets the facts and nothing else; if its text
    contains a number that is not among them, the template is used instead."""
    text, source, tokens = template, "template", 0
    llm = rt.env.llm
    if llm is not None:
        allowed = [float(x) for x in _NUM.findall(dumps(facts))]
        try:
            with rt.env.tracer.span(rt.run_id, "llm", purpose, input=facts) as sp:
                resp = llm.chat([
                    {"role": "system", "content":
                        "You write a short note for the operator of a surgical-robot research set-up. Use only the "
                        "facts given. Do not add numbers, causes or advice that are not in the facts. Plain "
                        "sentences, three at most, no lists."},
                    {"role": "user", "content": f"{purpose}\nFacts:\n{dumps(facts)}"}], temperature=0.0, max_tokens=220)
                sp.prompt_tokens, sp.completion_tokens = resp.usage.prompt_tokens, resp.usage.completion_tokens
                sp.output = resp.content
            tokens = resp.usage.total
            said = [float(x) for x in _NUM.findall(resp.content)]
            foreign = [x for x in said if not any(abs(x - a) <= 0.0501 for a in allowed)]
            if resp.content.strip() and not foreign:
                text, source = resp.content.strip(), "model"
            else:
                source = "template (the model's text had a number that is not in the facts)" if foreign \
                    else "template (the model returned nothing)"
        except LLMError as exc:
            source = f"template (model {exc.kind})"
    rt.emit("model_text", {"node": rt.state["task"]["stage"], "purpose": purpose, "source": source, "tokens": tokens,
                           "text": text[:700]})
    return text, source


# ------------------------------------------------------------------------------ nodes

def _check(rt: Runner, node: str, name: str, ok: bool, detail: str) -> bool:
    rt.emit("check_passed" if ok else "check_failed", {"node": node, "check": name, "detail": detail})
    return ok


def preflight(rt: Runner) -> Step:
    env, site, sheet = rt.env, rt.env.site, rt.state["sheet"]
    s = env.robot.state()
    failed: list[dict[str, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        if not _check(rt, "preflight", name, ok, detail):
            failed.append({"check": name, "detail": detail})

    check("run sheet is for this site and arm", sheet["site"] == site.name and sheet["arm"] == site.arm,
          f"run sheet {sheet['site']}/{sheet['arm']}, site {site.name}/{site.arm}")
    have = Path(site.chain.source.get("instrument", "")).stem
    check("instrument on the run sheet is the one the site is configured for", sheet["instrument"] == have,
          f"run sheet {sheet['instrument']}, site {have}")
    ok, detail = _envelope(rt)
    check("safety envelope is the one on the run sheet", ok, detail)
    ok, detail = _ready(rt)
    check("arm is powered, enabled and homed, with no fault", ok, detail)
    ok, detail = _calibration(rt)
    check("calibration is valid", ok, detail)
    for role in ("start_pose", "grasp_pose", "place_pose"):
        name = sheet[role]
        if name not in site.poses:
            check(f"{role} exists", False, f"no pose named {name!r} at this site")
            continue
        bad = check_goal(site.profile, site.scene, site.chain, site.poses[name].T, s.q)
        check(f"{role} {name} is accepted by the safety filter", not bad,
              "accepted" if not bad else "rejected by " + ", ".join(v.rule for v in bad))
    if sheet["start_pose"] in site.poses:
        d = _mm_to(rt, sheet["start_pose"])
        check("arm is at the start pose", d <= AT_POSE_MM, f"{d:.2f} mm from {sheet['start_pose']}")
    check("jaw is open", abs(s.jaw - site.jaw_angles["open"]) < 1e-6, f"jaw at {math.degrees(s.jaw):.1f} deg")
    check("approach path is available", len(_waypoints(rt)) > 0, f"{len(_waypoints(rt))} waypoints")
    if failed:
        return Step("failed", {"data": {"preflight": {"failed": failed}, "outcome": "preflight_failed"}})
    return Step("passed", {"data": {"preflight": {"failed": []}}})


def dry_run(rt: Runner) -> Step:
    """Rehearse the whole trial, and the recovery from the grasp pose, on a copy of the twin."""
    env, site, sheet = rt.env, rt.env.site, rt.state["sheet"]
    copy = env.robot.clone()
    steps: dict[str, int] = {}
    rejections: list[dict[str, Any]] = []

    def follow(twin: Any, label: str, targets: list[np.ndarray]) -> bool:
        twin.begin_motion(label)
        n = 0
        try:
            for T in targets:
                n += execute_path(twin, site.chain, site.profile, site.scene, T, label, tick=lambda dt: None)["steps"]
        except MotionStopped as exc:
            rejections.append({"stage": label, "after_control_steps": n + exc.steps_done, "kind": exc.kind,
                               "violations": [v.as_dict() for v in exc.violations]})
            return False
        finally:
            twin.end_motion(label)
            steps[label] = n
        return True

    start, place_T = site.poses[sheet["start_pose"]].T, site.poses[sheet["place_pose"]].T
    if follow(copy, "reach", _waypoints(rt)):
        at_grasp = copy.clone()
        follow(copy, "place", [place_T]) and follow(copy, "retreat", [start])
        lift = at_grasp.state().setpoint_cp.copy()
        lift[:3, 3] += site.scene.surface_normal * max(0.0, site.scene.height(start[:3, 3]) - site.scene.height(lift[:3, 3]))
        follow(at_grasp, "recover", [lift, start])
    total = sum(steps.values())
    _check(rt, "dry_run", "the trial and its recovery complete on a copy of the twin", not rejections,
           f"{total} control steps" if not rejections else f"rejected in {rejections[0]['stage']}")
    update: dict[str, Any] = {"dry_run": {"control_steps": steps, "expected_s": round(total / env.robot.rate_hz, 1),
                                          "rejections": rejections}}
    if rejections:
        update["outcome"] = "dry_run_rejected"
    return Step("rejected" if rejections else "clean", {"data": update})


def operator_go(rt: Runner) -> Step:
    verdict = approval_gate(rt, "operator_go", grants_go=True)
    if verdict == "approved":
        return Step("go")
    return Step(verdict, {"data": {"outcome": "operator_rejected" if verdict == "rejected" else "operator_timeout"}})


def _budget(rt: Runner, stage: str) -> int:
    """An action may take twice the control steps its rehearsal took, plus ten."""
    planned = rt.state["data"].get("dry_run", {}).get("control_steps", {}).get(stage, 40)
    return 2 * planned + 10


def reach(rt: Runner) -> Step:
    sheet, way = rt.state["sheet"], _waypoints(rt)
    p = rt.env.robot.state().measured_cp[:3, 3]
    # A reach that is entered again after a restart goes on from the waypoint nearest to where
    # the arm is. It does not go back to the beginning of the path.
    nearest = int(np.argmin([np.linalg.norm(w[:3, 3] - p) for w in way]))
    res = move(rt, "reach", way[nearest:], _budget(rt, "reach"))
    d, tol = _mm_to(rt, sheet["grasp_pose"]), sheet["acceptance"]["reach_tolerance_mm"]
    ok = res["status"] == "succeeded" and d <= tol
    _check(rt, "reach", "tip is within tolerance of the grasp pose", ok,
           f"{d:.3f} mm, tolerance {tol} mm, action {res['status']}" + (f" ({res['reason']})" if res["reason"] else ""))
    return Step("reached" if ok else "not_reached",
                {"data": {"reach": {"error_mm": round(d, 3), "status": res["status"], "reason": res["reason"],
                                    "control_steps": res["steps"], "from_waypoint": nearest}}})


def grasp(rt: Runner) -> Step:
    res = jaw(rt, "grasp", rt.env.site.jaw_angles["closed"])
    if res["status"] != "succeeded":
        return Step("failed", {"data": {"grasp": {"status": res["status"], "reason": res["reason"]}}})
    return Step("next", {"data": {"grasp_done": True}})


def hold_check(rt: Runner) -> Step:
    held = needle_held(rt)
    _check(rt, "hold_check", "needle is held", held, f"{rt.state['sheet']['outcome_source']}: held {held}")
    return Step("held" if held else "not_held", {"data": {"held_at_check": held}})


def place(rt: Runner) -> Step:
    site, sheet = rt.env.site, rt.state["sheet"]
    res = move(rt, "place", [site.poses[sheet["place_pose"]].T], _budget(rt, "place"))
    if res["status"] != "succeeded":
        return Step("failed", {"data": {"place": {"status": res["status"], "reason": res["reason"]}}})
    res = jaw(rt, "release", site.jaw_angles["open"])
    if res["status"] != "succeeded":
        return Step("failed", {"data": {"place": {"status": res["status"], "reason": res["reason"]}}})
    res = move(rt, "retreat", [site.poses[sheet["start_pose"]].T], _budget(rt, "retreat"))
    if res["status"] != "succeeded":
        return Step("failed", {"data": {"released": True, "place": {"status": res["status"], "reason": res["reason"]}}})
    return Step("next", {"data": {"placed": True, "released": True}})


def recover(rt: Runner) -> Step:
    """Open the jaw if that is safe, lift to the height of the start pose, return to the start
    pose. Every motion goes through the safety filter like any other."""
    env, site, state = rt.env, rt.env.site, rt.state
    sheet, robot = state["sheet"], rt.env.robot
    last = state["history"][-1] if state["history"] else {"node": "?", "route": "?"}
    why = state["task"]["why"] or f"{last['node']}:{last['route']}"
    came_from = state["task"].get("stopped_at") or last["node"]
    # A recovery agreed before the run covers a failed check or a silent watchdog. It does not
    # cover the operator's own stop: after a stop, nothing moves until the operator says so.
    preapproved = bool(sheet["recovery_preapproved"]) and not why.startswith("operator_stop")
    if not preapproved:
        verdict = approval_gate(rt, "recover")
        if verdict != "approved":
            return Step("rejected", {"data": {"recovery": {"done": False, "reason": why, "from": came_from,
                                                           "not_done_because": f"recovery {verdict}"}}})
        env.stop_event.clear()                        # approving the recovery releases the stop
    rt.emit("recovery_started", {"from": came_from, "reason": why, "preapproved": preapproved})
    done: list[str] = []
    released = bool(state["data"].get("released"))
    s = robot.state()
    if not s.ready:
        robot.enable()
        robot.home()
        done.append("enabled and homed the arm (homing does not move an arm that carries an instrument)")
        s = robot.state()
    held, h_mm = needle_held(rt), site.scene.height(s.measured_cp[:3, 3]) * 1000.0
    if abs(s.jaw - site.jaw_angles["open"]) > 1e-6:
        if not held or h_mm <= RELEASE_HEIGHT_MM:
            if jaw(rt, "recover:open_jaw", site.jaw_angles["open"])["status"] != "succeeded":
                return Step("failed", {"data": {"recovery": {"done": False, "reason": why, "from": came_from, "steps": done}}})
            released = True
            done.append("opened the jaw" + (f" {h_mm:.1f} mm over the pad, setting the needle down" if held else ""))
        else:
            done.append(f"kept the jaw closed: the needle is held {h_mm:.1f} mm over the pad")
    start = site.poses[sheet["start_pose"]].T
    lift = robot.state().setpoint_cp.copy()
    lift[:3, 3] += site.scene.surface_normal * max(0.0, site.scene.height(start[:3, 3]) - site.scene.height(lift[:3, 3]))
    ok = True
    for label, target, what in (("recover:lift", lift, "lifted to the height of the start pose"),
                                ("recover:return", start, "returned to the start pose")):
        res = move(rt, label, [target], 120)
        if res["status"] != "succeeded":
            ok = False
            done.append(f"{label} did not finish: {res['reason']}")
            break
        done.append(what)
    d = _mm_to(rt, sheet["start_pose"])
    ok = ok and d <= AT_POSE_MM
    return Step("recovered" if ok else "failed",
                {"data": {"released": released, "recovery": {"done": ok, "reason": why, "from": came_from, "steps": done,
                                                              "preapproved": preapproved,
                                                              "error_to_start_mm": round(d, 3)}}})


def verdict(rt: Runner) -> Step:
    d, s, sheet = rt.state["data"], rt.env.robot.state(), rt.state["sheet"]
    err = _mm_to(rt, sheet["start_pose"])
    rec = d.get("recovery")
    if d.get("placed") and not rec:
        outcome = "success"
    elif rec and rec.get("done"):
        outcome = "recovered"
    else:
        outcome = "needs_manual_recovery"
    return Step("next", {"data": {"outcome": outcome, "verdict": {
        "arm_at_start_pose": err <= AT_POSE_MM, "error_to_start_mm": round(err, 3), "arm_ready": s.ready,
        "arm_busy": s.busy, "jaw_deg": round(math.degrees(s.jaw), 1),
        "why": rt.state["task"]["why"] or (rec or {}).get("reason")}}})


def _facts(rt: Runner) -> dict[str, Any]:
    st, d = rt.state, rt.state["data"]
    return {"outcome": d.get("outcome"), "why": st["task"]["why"] or (d.get("recovery") or {}).get("reason"),
            "stages_run": [h["node"] for h in st["history"]],
            "reach_error_mm": (d.get("reach") or {}).get("error_mm"), "needle_held_at_check": d.get("held_at_check"),
            "needle_held_source": st["sheet"]["outcome_source"],
            "recovery_steps": (d.get("recovery") or {}).get("steps"),
            "arm_at_start_pose": (d.get("verdict") or {}).get("arm_at_start_pose"),
            "tip_travel_mm": st["usage"]["path_mm"], "motions": len(st["actions"]),
            "failed_preflight_checks": (d.get("preflight") or {}).get("failed")}


def confirm(rt: Runner) -> Step:
    f = _facts(rt)
    template = (f"Trial outcome: {f['outcome']}. Stages run: {', '.join(f['stages_run'])}. "
                f"Tip travel {f['tip_travel_mm']} mm in {f['motions']} motions. "
                + (f"Reason: {f['why']}. " if f["why"] else "")
                + ("The arm is at the start pose." if f["arm_at_start_pose"] else "The arm is NOT at the start pose."))
    text, source = model_text(rt, "Summarise this trial for the operator.", f, template)
    return Step("next", {"data": {"summary": text, "summary_source": source}})


def stop(rt: Runner) -> Step:
    """Ends a run in which nothing has moved."""
    d, why = rt.state["data"], rt.state["task"]["why"]
    outcome = d.get("outcome") or f"stopped:{why or 'unknown'}"
    update: dict[str, Any] = {"outcome": outcome, "motions_executed": len(rt.state["actions"])}
    failed = (d.get("preflight") or {}).get("failed")
    if failed:
        template = "The trial did not start. Failed pre-flight checks: " + "; ".join(
            f"{x['check']} ({x['detail']})" for x in failed) + ". Nothing was moved."
        text, source = model_text(rt, "Explain to the operator why the trial did not start. Nothing was moved.",
                                  {"failed_preflight_checks": failed, "arm_moved": False}, template)
        update.update({"explanation": text, "explanation_source": source})
    return Step("next", {"data": update})


# ------------------------------------------------------------------------------ the graph

def abort_to(state: dict[str, Any], node: str, limits: Limits) -> str:
    """Where a run goes when a node cannot run or was ended from the outside."""
    if node == "recover":
        return "verdict"                              # recovery itself failed: a person takes over
    if not state["actions"]:
        return "stop"                                 # nothing has moved, so there is nothing to recover
    if state["usage"]["recoveries"] >= limits.max_recoveries:
        return "verdict"
    return "recover"


def build_graph() -> Graph:
    g = Graph("preflight", abort_to, exempt=("recover", "verdict", "confirm", "stop"))
    g.abort_targets = {"recover", "stop", "verdict"}
    g.add_node("preflight", preflight)
    g.add_node("dry_run", dry_run)
    g.add_node("operator_go", operator_go, requires=("arm_ready", "calibration_valid", "envelope_unchanged"))
    g.add_node("reach", reach, requires=("arm_ready", "calibration_valid", "envelope_unchanged", "go_granted"))
    g.add_node("grasp", grasp, requires=("arm_ready", "go_granted", "at_grasp_pose"))
    g.add_node("hold_check", hold_check, requires=("arm_ready",))
    g.add_node("place", place, requires=("arm_ready", "go_granted", "needle_held"))
    g.add_node("recover", recover, requires=("no_fault",))
    g.add_node("verdict", verdict)
    g.add_node("confirm", confirm)
    g.add_node("stop", stop)
    g.add_conditional("preflight", {"passed": "dry_run", "failed": "stop"})
    g.add_conditional("dry_run", {"clean": "operator_go", "rejected": "stop"})
    g.add_conditional("operator_go", {"go": "reach", "rejected": "stop", "timeout": "stop"})
    g.add_conditional("reach", {"reached": "grasp", "not_reached": "recover"})
    g.add_conditional("grasp", {"next": "hold_check", "failed": "recover"})
    g.add_conditional("hold_check", {"held": "place", "not_held": "recover"})
    g.add_conditional("place", {"next": "verdict", "failed": "recover"})
    g.add_conditional("recover", {"recovered": "verdict", "failed": "verdict", "rejected": "verdict"})
    g.add_edge("verdict", "confirm")
    g.add_edge("confirm", END)
    g.add_edge("stop", END)
    return g


def make_env(site: Any, robot: Any, db: Database, llm: Any = None, **kw: Any) -> Env:
    return Env(robot=robot, site=site, db=db, log=EventLog(db), tracer=Tracer(db), checks=dict(CHECKS),
               observe=observe, llm=llm, **kw)


def start_trial(env: Env, sheet: RunSheet, who: Principal, limits: Limits | None = None, run_id: str | None = None,
                held: bool | None = None, rosma_dir: Path | None = None) -> tuple[RunResult, dict[str, Any]]:
    """Validate the run sheet, attach the approach path and the declared outcome, and run until
    the trial ends or waits for a person."""
    if "trial:run" not in who.scopes or who.site != sheet.site:
        raise PermissionError(f"{who.user_id} may not run a trial at {sheet.site}")
    run_id = run_id or f"trial_{uuid.uuid4().hex[:10]}"
    info = prepare_path(env.db, env.site, sheet, run_id, rosma_dir)
    if held is not None:
        declare_outcome(env.db, run_id, held, who.user_id)
    runner = Runner(build_graph(), env, limits)
    return runner.start(run_id, sheet.model_dump(), who), info
