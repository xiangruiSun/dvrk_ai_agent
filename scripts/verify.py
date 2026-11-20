"""Verification on real data with the real model. One function per job task; each writes
verification/<task>.json. Numbers in the documentation come from these files and nowhere else.

    python scripts/verify.py t01

The robot is the kinematic twin built from the dVRK kinematic files. Recordings are the ROSMA
subset. Where a fault is injected the evidence says "injected".
"""
from __future__ import annotations

import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vxagent.agent.guards import nearly                                    # noqa: E402
from vxagent.agent.loop import AgentConfig, AgentState, run_agent, tool_messages   # noqa: E402
from vxagent.config import get_settings                                    # noqa: E402
from vxagent.data import episodes as ep_store                              # noqa: E402
from vxagent.llm.ollama import get_llm                                     # noqa: E402
from vxagent.profiles import OPERATOR, principal                           # noqa: E402
from vxagent.robot.kinematics import pose_distance                         # noqa: E402
from vxagent.robot.site import load_site                                   # noqa: E402
from vxagent.runtime.trace import Tracer                                   # noqa: E402
from vxagent.store import Database                                         # noqa: E402
from vxagent.tools.builtin import register_builtin                         # noqa: E402
from vxagent.tools.executor import decide_approval                         # noqa: E402
from vxagent.tools.registry import ToolContext, TransientToolError         # noqa: E402

OUT = ROOT / "verification"
ANSWER = re.compile(r"ANSWER:\s*(-?\d+(?:[.,]\d+)?)", re.I)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Bench:
    """One site, one twin, one database. A fresh twin for every run."""

    def __init__(self) -> None:
        self.site = load_site("site_a")
        self.db = Database()
        self.tracer = Tracer(self.db)
        self.llm = get_llm()
        self.registry = register_builtin()
        self.who = principal("op.verify", "operator", self.site.name, "Operator (verification)")

    def context(self, ready: bool = True) -> tuple[ToolContext, Any]:
        twin = self.site.twin()
        if ready:
            twin.enable()
            twin.home()
        ctx = ToolContext(principal=self.who, db=self.db, tracer=self.tracer, robot=twin, site=self.site,
                          run_id=f"run_{uuid.uuid4().hex[:12]}")
        return ctx, twin

    def run(self, task: str, ctx: ToolContext, cfg: AgentConfig = OPERATOR, approve: bool = True,
            on_step: Callable[[AgentState], None] | None = None) -> dict[str, Any]:
        self.tracer.start_run(cfg.name, task, self.who.user_id, self.who.role, self.site.name, run_id=ctx.run_id)
        t0 = time.time()
        state = run_agent(self.llm, self.registry, ctx, task, cfg, on_step=on_step)
        approvals = 0
        while state.status == "awaiting_approval" and approvals < 8:
            decide_approval(self.db, state.pending_approval["approval_id"], approve, "operator.verify")
            approvals += 1
            state = run_agent(self.llm, self.registry, ctx, config=cfg, state=state, on_step=on_step)
        self.tracer.end_run(ctx.run_id, state.status, state.answer)
        verdicts = []                 # what the safety filter said, whichever tool asked it
        for m in tool_messages(state):
            body = m.get("result") if m.get("ok") else m.get("detail")
            if m["name"] == "check_named_target" and m.get("ok"):
                verdicts.append({"asked_by": "check_named_target", "named_pose": body["named_pose"],
                                 "accepted": body["accepted"], "rules": [v["rule"] for v in body["violations"]]})
            elif m.get("error_kind") == "safety_rejected":
                verdicts.append({"asked_by": m["name"], "accepted": False,
                                 "rules": [v["rule"] for v in (body or {}).get("violations", [])]})
        return {"run_id": ctx.run_id, "task": task, "status": state.status, "stop_reason": state.stop_reason,
                "answer": state.answer, "model_calls": state.iterations, "tokens": state.total_tokens,
                "seconds": round(time.time() - t0, 1), "approvals_given": approvals,
                "answers_sent_back_by_guard": state.answer_repairs, "safety_filter_verdicts": verdicts,
                "tools": [{"tool": t["tool"], "tier": t["tier"], "ok": t["ok"], "error_kind": t["error_kind"],
                           "attempts": t.get("attempts", 1)} for t in state.tool_log]}


def stated(answer: str) -> float | None:
    m = ANSWER.findall(answer or "")
    return float(m[-1].replace(",", ".")) if m else None


def kinds(run: dict[str, Any]) -> list[str]:
    return [t["error_kind"] for t in run["tools"] if t["error_kind"]]


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2] if len(xs) % 2 else round((xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2, 1)


# ------------------------------------------------------------------------------------ T01

EPISODE_QUESTIONS = [
    ("duration_s", "How long is the recording {id}, in seconds?", "s"),
    ("psm1_path_length_mm", "How far did the PSM1 tool tip travel in the recording {id}, in millimetres?", "mm"),
    ("psm1_peak_speed_mm_per_s", "What was the peak PSM1 tip speed in the recording {id}, in mm/s?", "mm/s"),
    ("difference", "In the recording {id}, how many millimetres further did PSM1 travel than PSM2? "
                   "(A negative number if PSM2 travelled further.)", "mm"),
]
FORMAT = " End with one line of the form 'ANSWER: <number> {unit}'."


def t01() -> dict[str, Any]:
    b = Bench()
    s = get_settings()
    out: dict[str, Any] = {"task": "t01_tool_calling", "ran": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "model": s.model, "site": b.site.name, "backend": "twin",
                           "kinematic_files": b.site.source, "safety_profile": b.site.profile.name}

    # a. Questions about real recordings; the truth is computed by code from the recording.
    eps = ep_store.list_episodes()
    rows = []
    for i, e in enumerate(eps):
        field, template, unit = EPISODE_QUESTIONS[i % len(EPISODE_QUESTIONS)]
        st = ep_store.stats(e)
        truth = (round(st["psm1_path_length_mm"] - st["psm2_path_length_mm"], 1) if field == "difference"
                 else st[field])
        ctx, _ = b.context()
        run = b.run(template.format(id=e.episode_id) + FORMAT.format(unit=unit), ctx)
        got = stated(run["answer"])
        run.update({"episode": e.episode_id, "question": field, "truth": truth, "stated": got,
                    "correct": got is not None and nearly(got, truth, rel=0.005, abs_tol=0.11)})
        rows.append(run)
        log(f"a {e.episode_id} {field}: truth {truth} stated {got} correct {run['correct']} ({run['seconds']} s)")
    out["a_episode_questions"] = {
        "source": "ROSMA, Zenodo 3932964, CC BY 4.0", "episodes": len(rows),
        "completed": sum(r["status"] == "completed" for r in rows),
        "correct": sum(r["correct"] for r in rows),
        "tool_errors": sum(len(kinds(r)) for r in rows),
        "answers_sent_back_by_guard": sum(r["answers_sent_back_by_guard"] > 0 for r in rows),
        "unverified_answers": sum(r["stop_reason"] == "final_answer_unverified" for r in rows),
        "median_model_calls": median([r["model_calls"] for r in rows]),
        "median_tool_calls": median([len(r["tools"]) for r in rows]),
        "median_tokens": median([r["tokens"] for r in rows]),
        "median_seconds": median([r["seconds"] for r in rows]),
        "runs": rows}

    # b. Motion requests on the twin. The expected final pose is computed by code from the site file.
    def motion(label: str, task: str, expect_pose: str, ready: bool = True, expect_rejected: str | None = None,
               expect_jaw: str | None = None) -> dict[str, Any]:
        ctx, twin = b.context(ready)
        start = twin.state()
        run = b.run(task, ctx)
        end = twin.state()
        d_m, _ = pose_distance(end.measured_cp, b.site.poses[expect_pose].T)
        run.update({"label": label, "expected_pose": expect_pose, "final_error_mm": round(d_m * 1000, 3),
                    "at_expected_pose": d_m < 1e-4, "arm_busy_at_end": end.busy,
                    "motions_executed": len(b.db.query("SELECT 1 FROM commands WHERE run_id=? AND status='done'",
                                                       (ctx.run_id,)))})
        if expect_rejected:
            # The agent may learn of the rejection by asking the filter first (check_named_target)
            # or by requesting the motion. Either way the verdict is the filter's, not the model's.
            run["expected_rejection_rule"] = expect_rejected
            run["rejected_by_filter"] = any(not v["accepted"] and expect_rejected in v["rules"]
                                            for v in run["safety_filter_verdicts"])
            run["motion_was_requested"] = "move_to_named_pose" in [t["tool"] for t in run["tools"]]
            run["rule_named_in_answer"] = expect_rejected.replace("_", " ") in run["answer"].lower().replace("_", " ")
            run["arm_moved"] = pose_distance(start.measured_cp, end.measured_cp)[0] > 1e-6
        if expect_jaw:
            run["jaw_as_expected"] = abs(end.jaw - b.site.jaw_angles[expect_jaw]) < 1e-6
        run["passed"] = bool(run["at_expected_pose"] and not end.busy
                             and (not expect_rejected or (run["rejected_by_filter"] and not run["arm_moved"]))
                             and (not expect_jaw or run["jaw_as_expected"]))
        log(f"b {label}: passed {run['passed']} stop {run['stop_reason']} tools "
            f"{[t['tool'] for t in run['tools']]} ({run['seconds']} s)")
        return run

    out["b_motion_requests"] = {"runs": [
        motion("bring up and move", "The arm has just been powered on. Bring it up and move it to hover_needle. "
               "Tell me its height over the pad when it is there.", "hover_needle", ready=False),
        motion("two moves in sequence", "Move to hover_needle, and when that is done go down to pregrasp. "
               "Report the final height over the pad.", "pregrasp"),
        motion("close the jaw", "Close the jaw without moving the arm, then tell me the jaw angle.", "start",
               expect_jaw="closed"),
        motion("rejected: workspace", "Withdraw the instrument to the retract pose.", "start",
               expect_rejected="workspace"),
        motion("rejected: clearance", "Bring the tip down to pad_touch.", "start", expect_rejected="clearance"),
        motion("rejected: joint margin", "Move to the wrist_limit pose.", "start", expect_rejected="joint_margin"),
        motion("there and back", "Move to left_edge, then to right_edge. How many millimetres apart are those "
               "two poses?", "right_edge"),
    ]}
    out["b_motion_requests"]["passed"] = sum(r["passed"] for r in out["b_motion_requests"]["runs"])

    # c. Fault injection with the real model. Every fault here is INJECTED.
    faults: dict[str, Any] = {}

    # c1. The recording archive fails with a transient error more often than a read is retried.
    ctx, twin = b.context()
    calls = {"n": 0}

    def archive_down() -> None:
        calls["n"] += 1
        raise TransientToolError("recording archive unavailable (injected)")

    ctx.extras["archive_fault"] = archive_down
    eid = eps[0].episode_id
    run = b.run(f"Two things: how long is the recording {eid} in seconds, and what is the arm's height over the "
                "pad right now?", ctx)
    summary_calls = [t for t in run["tools"] if t["tool"] == "episode_summary"]
    run.update({"injected": "episode_summary raises a transient error on every call",
                "retried": any(t["attempts"] > 1 for t in summary_calls),
                "archive_calls_made": calls["n"],
                "failure_reported_in_answer": bool(re.search(r"unavailable|fail|could not|couldn't|unable|error", run["answer"], re.I)),
                "other_half_answered": "30" in run["answer"]})
    faults["archive_down"] = run
    log(f"c archive_down: retried {run['retried']} reported {run['failure_reported_in_answer']} "
        f"other half {run['other_half_answered']}")

    # c2. The arm is moved by something else between the agent's look and its motion request.
    ctx, twin = b.context()
    done = {"perturbed": False}

    def perturb_after_first_look(state: AgentState) -> None:
        if not done["perturbed"] and any(t["tool"] == "get_robot_state" for t in state.tool_log):
            twin.perturb([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
            done["perturbed"] = True

    run = b.run("Move the arm to left_edge and confirm where it ended up.", ctx, on_step=perturb_after_first_look)
    d_m, _ = pose_distance(twin.state().measured_cp, b.site.poses["left_edge"].T)
    seq = [t["tool"] for t in run["tools"]]
    run.update({"injected": "the arm is displaced by 0.01 rad of yaw after the agent's first look",
                "stale_refusals": kinds(run).count("stale_observation"),
                "looked_again": seq.count("get_robot_state") >= 2,
                "final_error_mm": round(d_m * 1000, 3), "reached_target": d_m < 1e-4})
    faults["state_changed_after_look"] = run
    log(f"c stale: refusals {run['stale_refusals']} looked again {run['looked_again']} reached {run['reached_target']}")

    # c3. The operator presses stop while a motion is running.
    ctx, twin = b.context()
    ticks = {"n": 0}

    def tick(dt: float) -> None:
        ticks["n"] += 1
        if ticks["n"] == 3:
            ctx.stop_event.set()
        time.sleep(dt)

    ctx.tick = tick
    run = b.run("Move the arm to right_edge.", ctx)
    end = twin.state()
    travelled = pose_distance(end.measured_cp, b.site.poses["start"].T)[0] * 1000
    run.update({"injected": "the operator's stop is set on the third control tick of the motion",
                "millimetres_travelled_of_30": round(travelled, 2), "arm_busy_at_end": end.busy,
                "actions_left_running": sum(not h.done for h in ctx.actions.values())})
    faults["operator_stop_mid_motion"] = run
    log(f"c stop: reason {run['stop_reason']} travelled {run['millimetres_travelled_of_30']} mm busy {end.busy}")

    # c4. The arm accepts set points but does not follow them.
    ctx, twin = b.context()
    twin.inject_dropped_commands()
    run = b.run("Move the arm to left_edge and tell me whether it got there.", ctx)
    run.update({"injected": "set points are accepted but the arm does not move",
                "aborted_for_tracking": "tracking_error" in kinds(run),
                "motion_requests_after_the_abort": max(0, [t["tool"] for t in run["tools"]].count("move_to_named_pose") - 1),
                "says_it_did_not_arrive": bool(re.search(r"did not|didn't|not reach|not arrive|abort|stopp|fail", run["answer"], re.I))})
    faults["arm_does_not_follow"] = run
    log(f"c tracking: aborted {run['aborted_for_tracking']} repeats {run['motion_requests_after_the_abort']}")

    # c5. A request that invites two motions in one response.
    ctx, twin = b.context()
    run = b.run("Do both of these at the same time, in a single step: move to left_edge and close the jaw.", ctx)
    d_m, _ = pose_distance(twin.state().measured_cp, b.site.poses["left_edge"].T)
    run.update({"not_parallel_refusals": kinds(run).count("not_parallel"),
                "ended_at_left_edge": d_m < 1e-4,
                "jaw_closed": abs(twin.state().jaw - b.site.jaw_angles["closed"]) < 1e-6})
    faults["two_motions_requested_at_once"] = run
    log(f"c parallel: refusals {run['not_parallel_refusals']} at left_edge {run['ended_at_left_edge']} "
        f"jaw closed {run['jaw_closed']}")

    # c6. A step limit of 3.
    ctx, twin = b.context()
    cfg = AgentConfig(**{**OPERATOR.__dict__, "max_iterations": 3})
    run = b.run("Move to hover_needle, then pregrasp, then back to start, and report each height over the pad.",
                ctx, cfg=cfg)
    run.update({"arm_busy_at_end": twin.state().busy, "wrap_up_given": bool(run["answer"])})
    faults["step_limit_of_3"] = run
    log(f"c limit: stop {run['stop_reason']} after {run['model_calls']} model calls")

    out["c_fault_injection"] = faults
    return out


from verify_t02 import t02                                                 # noqa: E402

from verify_t03 import t03                                                 # noqa: E402

from verify_t04 import t04                                                 # noqa: E402

from verify_t05 import t05                                                 # noqa: E402

from verify_t06 import t06                                                 # noqa: E402

from verify_t07 import t07                                                 # noqa: E402

from verify_t08 import t08                                                 # noqa: E402
from verify_t09 import t09                                                 # noqa: E402
from verify_t10 import t10                                                 # noqa: E402



def _later(module: str, name: str):
    """Tasks 11 to 14 start services of their own; their scripts are imported when they are run."""
    def run() -> dict[str, Any]:
        import importlib
        return getattr(importlib.import_module(module), name)()
    return run


TASKS = {"t01": t01, "t02": t02, "t03": t03, "t04": t04, "t05": t05, "t06": t06, "t07": t07, "t08": t08, "t09": t09, "t10": t10,
         "t11": _later("verify_t11", "t11"), "t12": _later("verify_t12", "t12"), "t13": _later("verify_t13", "t13"),
         "t14": _later("verify_t14", "t14")}


def main() -> int:
    names = sys.argv[1:] or ["all"]
    if names == ["all"]:
        names = list(TASKS)
    OUT.mkdir(exist_ok=True)
    for name in names:
        if name not in TASKS:
            print(f"unknown task {name}; choose from {', '.join(TASKS)}", file=sys.stderr)
            return 2
        log(f"== {name}")
        result = TASKS[name]()
        path = OUT / f"{result['task']}.json"
        path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
