"""The trial pipeline for a simulation-trained policy (task 09), on the runtime of task 03.

    preflight -+-> stop -> END
               +-> rehearse -+-> stop -> END
                             +-> operator_go -+-> stop -> END
                                              +-> stage -> reach -> grasp -> retention -+-> place --+
                                                     |       |        |          |      |           |
                                                     +-------+--------+----------+------+-> recover-+
                                                                                                    |
                                                                  verdict -> confirm -> record -> END

Reach replays a recorded expert rollout: the policy's actions, one per control tick, are taken
through the adapter generated from the policy contract and through the safety filter. Grasp is
a geometric procedure, not a learned one. Whether the needle is held comes from a needle that
is DECLARED for the run and follows the arm (the twin has none). The machine verdict is
computed from measurements; the operator confirms it or overrides it with a reason, and an
override is kept beside the machine verdict, never in its place.
"""
from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np

from ..policy import compat as policy_compat
from ..policy.adapter import Adapter
from ..robot.kinematics import angle_between, pose_distance
from ..robot.motion import MotionStopped, execute_path
from ..robot.preflight import PERCEPTION_LIMIT_S, run_preflight
from ..robot.safety import plan_steps
from ..runtime.graph import END, Env, Graph, Interrupt, Limits, Runner, RunResult, Step, approval_gate
from ..runtime.watchdog import jaw, move, run_action
from ..store import dumps
from ..structured.schemas import RunSheet
from ..tools.registry import Principal
from . import trial

REACH_MM, REACH_DEG, MAX_ACTIONS = 5.0, 15.0, 200       # published reach criterion and action budget
DESCENT_MM, LIFT_MM, HOLD_S = 4.0, 14.0, 2.0
PLACE_HEIGHTS_MM = (15.0, 8.0, 3.5)                      # approach heights over the entry, as published


def _x(rt: Runner) -> dict[str, Any]:
    return rt.env.extras


def trial_poses(rt: Runner) -> dict[str, np.ndarray]:
    """The poses of this trial in the site's frame: where the rollout starts, its goal, the grasp."""
    x = _x(rt)
    if "poses" not in x:
        ad: Adapter = x["adapter"]
        demo = x["demo"]
        goal = ad.pose_to_site(demo["goal"])
        grasp = goal.copy()
        grasp[:3, 3] -= rt.env.site.scene.surface_normal * DESCENT_MM / 1000.0
        lifted = grasp.copy()
        lifted[:3, 3] += rt.env.site.scene.surface_normal * LIFT_MM / 1000.0
        x["poses"] = {"start": ad.pose_to_site(demo["poses"][0]), "goal": goal, "grasp": grasp, "lifted": lifted}
    return x["poses"]


def rollout_targets(rt: Runner) -> list[np.ndarray]:
    """One site-frame target per policy action: pose + action * step size, through the adapter.
    The jaw component of the actions is not used: the jaw belongs to the grasp procedure."""
    ad: Adapter = _x(rt)["adapter"]
    demo = _x(rt)["demo"]
    pose = np.array(demo["poses"][0], dtype=float)
    out = []
    for a in demo["actions"][:MAX_ACTIONS]:
        pose = ad.step(pose, a)
        out.append(ad.pose_to_site(pose))
    return out


def place_targets(rt: Runner) -> list[np.ndarray]:
    site, sheet = rt.env.site, rt.state["sheet"]
    entry = site.poses[sheet["place_pose"]].T
    base = trial_poses(rt)["lifted"]
    n = site.scene.surface_normal
    out = []
    for h in PLACE_HEIGHTS_MM:
        T = base.copy()
        p = entry[:3, 3]
        T[:3, 3] = p + n * (h / 1000.0 - site.scene.height(p))
        out.append(T)
    return out


# ------------------------------------------------------------------------------ nodes

def preflight(rt: Runner) -> Step:
    env, x, sheet = rt.env, _x(rt), rt.state["sheet"]
    poses = trial_poses(rt)
    rows = run_preflight(env.robot, env.site, sheet, env.db, start_T=poses["start"], goal_T=poses["goal"],
                         compat=x.get("compat"), in_range=policy_compat.in_training_range(x["contract"], poses["start"], poses["goal"]),
                         perception=x.get("perception"), tool_dir=x.get("tool_dir"))
    for r in rows:
        trial._check(rt, "preflight", r["check"], r["ok"], r["detail"])
    failed = [{"check": r["check"], "problem": r["problem"], "detail": r["detail"], "repair": r["repair"]} for r in rows if not r["ok"]]
    if failed:
        return Step("failed", {"data": {"preflight": {"checks": len(rows), "failed": failed}, "outcome": "preflight_failed"}})
    return Step("passed", {"data": {"preflight": {"checks": len(rows), "failed": []}}})


def run_rehearsal(robot: Any, site: Any, sequence: list[tuple[str, list[np.ndarray]]]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Follow the stages on a copy of the twin through the safety filter: (control steps per stage, rejections)."""
    copy = robot.clone()
    steps: dict[str, int] = {}
    rejections: list[dict[str, Any]] = []
    for label, targets in sequence:
        copy.begin_motion(label)
        n = 0
        try:
            for T in targets:
                n += execute_path(copy, site.chain, site.profile, site.scene, T, label, tick=lambda dt: None)["steps"]
        except MotionStopped as exc:
            rejections.append({"stage": label, "after_control_steps": n + exc.steps_done, "kind": exc.kind,
                               "violations": [v.as_dict() for v in exc.violations]})
        finally:
            copy.end_motion(label)
            steps[label] = n
        if rejections:
            break
    return steps, rejections


def stage_sequence(rt: Any) -> list[tuple[str, list[np.ndarray]]]:
    site, poses = rt.env.site, trial_poses(rt)
    return [("stage", [poses["start"]]), ("reach", rollout_targets(rt)), ("descent", [poses["grasp"]]),
            ("lift", [poses["lifted"]]), ("place", place_targets(rt)), ("retreat", [site.poses[rt.state["sheet"]["start_pose"]].T])]


def rehearse(rt: Runner) -> Step:
    """The whole stage sequence on a copy of the twin, through the same safety filter."""
    env = rt.env
    steps, rejections = run_rehearsal(env.robot, env.site, stage_sequence(rt))
    trial._check(rt, "rehearse", "the stage sequence completes on a copy of the twin", not rejections,
                 f"{sum(steps.values())} control steps" if not rejections else f"rejected in {rejections[0]['stage']}")
    update: dict[str, Any] = {"dry_run": {"control_steps": steps, "expected_s": round(sum(steps.values()) / env.robot.rate_hz, 1),
                                          "rejections": rejections}}
    if rejections:
        update["outcome"] = "rehearsal_rejected"
    return Step("rejected" if rejections else "clean", {"data": update})


def operator_go(rt: Runner) -> Step:
    verdict = approval_gate(rt, "operator_go", grants_go=True)
    if verdict == "approved":
        return Step("go")
    return Step(verdict, {"data": {"outcome": "operator_rejected" if verdict == "rejected" else "operator_timeout"}})


def _staged(rt: Runner, label: str, targets: list[np.ndarray], budget: int) -> dict[str, Any]:
    """Staged activation: the first target of a stage is sent alone and the arm's tracking is
    checked before the rest of the stage is sent."""
    site = rt.env.site
    planned = plan_steps(site.profile, site.scene, rt.env.robot.state().setpoint_cp, targets[0])
    first = move(rt, f"{label}:first_step", [planned[0] if planned else targets[0]], 4)
    s = rt.env.robot.state()
    d_m, d_rad = pose_distance(s.setpoint_cp, s.measured_cp)
    ok = first["status"] == "succeeded" and d_m <= rt.env.site.profile.track_pos_m and d_rad <= rt.env.site.profile.track_rot_rad
    trial._check(rt, label, "first step of the stage is tracked", ok,
                 f"action {first['status']}, set point to measured {d_m * 1000:.3f} mm, {math.degrees(d_rad):.3f} deg")
    if not ok:
        return {**first, "first_step_ok": False}
    rest = move(rt, label, targets, budget)
    return {**rest, "first_step_ok": True, "steps": rest["steps"] + first["steps"], "path_mm": round(rest["path_mm"] + first["path_mm"], 2)}


def stage(rt: Runner) -> Step:
    """Bring the arm to the pose the rollout starts from, with the jaw open."""
    site = rt.env.site
    if abs(rt.env.robot.state().jaw - site.jaw_angles["open"]) > 1e-6:
        jaw(rt, "stage:open_jaw", site.jaw_angles["open"])
    res = _staged(rt, "stage", [trial_poses(rt)["start"]], 80)
    d, _ = pose_distance(rt.env.robot.state().measured_cp, trial_poses(rt)["start"])
    ok = res["status"] == "succeeded" and d * 1000 <= 1.0
    return Step("ready" if ok else "failed", {"data": {"stage": {"status": res["status"], "reason": res["reason"],
                                                                "error_mm": round(d * 1000, 3)}}})


def reach(rt: Runner) -> Step:
    targets = rollout_targets(rt)
    goal = trial_poses(rt)["goal"]
    res = _staged(rt, "reach", targets, 2 * len(targets) + 20)
    s = rt.env.robot.state()
    d_mm = pose_distance(s.measured_cp, goal)[0] * 1000.0
    d_deg = math.degrees(angle_between(s.measured_cp[:3, :3], goal[:3, :3]))
    ok = res["status"] == "succeeded" and d_mm <= REACH_MM and d_deg <= REACH_DEG
    trial._check(rt, "reach", "the measured pose is within 5 mm and 15 degrees of the approach goal", ok,
                 f"{d_mm:.3f} mm, {d_deg:.3f} deg, action {res['status']}" + (f" ({res['reason']})" if res["reason"] else ""))
    return Step("reached" if ok else "not_reached",
                {"data": {"reach": {"error_mm": round(d_mm, 3), "error_deg": round(d_deg, 3), "status": res["status"],
                                    "reason": res["reason"], "policy_actions": len(targets), "control_steps": res["steps"],
                                    "violations": res.get("violations")}}})


def grasp(rt: Runner) -> Step:
    """Geometric: bounded descent, close, lift, hold for two seconds."""
    env, site, poses = rt.env, rt.env.site, trial_poses(rt)
    needle = _x(rt).get("needle")
    if needle is not None:
        needle.grasp_p = poses["grasp"][:3, 3].copy()
    log: list[str] = []
    for label, fn in (("grasp:descent", lambda: _staged(rt, "grasp:descent", [poses["grasp"]], 20)),
                      ("grasp:close", lambda: jaw(rt, "grasp:close", site.jaw_angles["closed"]))):
        res = fn()
        log.append(f"{label} {res['status']}")
        if res["status"] != "succeeded":
            return Step("failed", {"data": {"grasp": {"steps": log, "reason": res["reason"]}}})
    before = needle.observe() if needle is not None else {"held": False, "height_over_pad_mm": 0.0}
    res = _staged(rt, "grasp:lift", [poses["lifted"]], 40)
    log.append(f"grasp:lift {res['status']}")
    if res["status"] != "succeeded":
        return Step("failed", {"data": {"grasp_done": True, "grasp": {"steps": log, "reason": res["reason"]}}})
    seen = _see(rt, "grasp")
    if seen is not None and not seen["available"]:
        # Whether the needle came up can only be observed. Without perception this stage does not guess.
        return Step("failed", {"data": {"grasp_done": True, "grasp": {"steps": log, "reason": "perception_unavailable",
                                                                      "perception": seen["detail"]}}})
    after_lift = needle.observe() if needle is not None else dict(before)
    env.tick(HOLD_S)
    if needle is not None:
        needle.held_still(HOLD_S)
    after_hold = needle.observe() if needle is not None else dict(before)
    return Step("next", {"data": {"grasp_done": True, "grasp": {
        "steps": log + [f"held for {HOLD_S:g} s"], "needle_at_grip": before, "needle_after_lift": after_lift,
        "needle_after_hold": after_hold, "source": rt.state["sheet"]["outcome_source"]}}})


def retention(rt: Runner) -> Step:
    """Held only if the needle came up with the jaws and is still there after the hold."""
    g = rt.state["data"]["grasp"]
    rose = g["needle_after_lift"]["height_over_pad_mm"] - g["needle_at_grip"]["height_over_pad_mm"]
    held = bool(g["needle_after_lift"]["held"] and rose >= 5.0 and g["needle_after_hold"]["held"])
    trial._check(rt, "retention", "the needle moved with the jaws and is still held after the hold", held,
                 f"held after lift {g['needle_after_lift']['held']}, rose {rose:.1f} mm, held after hold "
                 f"{g['needle_after_hold']['held']} ({g['source']})")
    return Step("held" if held else "not_held", {"data": {"held_at_check": held, "needle_rose_mm": round(rose, 2)}})


def _see(rt: Runner, stage: str) -> dict[str, Any] | None:
    """Ask the perception source during a stage. None if the trial has none configured (pre-flight
    then failed already). A late or missing answer is "unavailable", recorded as an event."""
    source = _x(rt).get("perception")
    if source is None:
        return None
    try:
        out = source()
        took = float(out.get("latency_s", 0.0))
        ok, detail = took <= PERCEPTION_LIMIT_S, f"answered in {took:.2f} s"
    except Exception as exc:                            # a source that fails did not answer
        ok, detail = False, f"no answer: {type(exc).__name__}"
    rt.emit("check_passed" if ok else "check_failed",
            {"node": stage, "check": "perception answered within its time limit", "detail": detail})
    return {"available": ok, "detail": detail}


def place(rt: Runner) -> Step:
    site, sheet = rt.env.site, rt.state["sheet"]
    seen = _see(rt, "place")
    carried = seen is not None and not seen["available"]      # this stage may go on with the carried needle pose
    res = _staged(rt, "place", place_targets(rt), 120)
    if res["status"] != "succeeded":
        return Step("failed", {"data": {"place": {"status": res["status"], "reason": res["reason"]}}})
    needle = _x(rt).get("needle")
    still = bool(needle.held) if needle is not None else False
    res = jaw(rt, "place:release", site.jaw_angles["open"])
    released = dict(needle.released_at) if needle is not None and needle.released_at else None
    entry = site.poses[sheet["place_pose"]].T[:3, 3]
    place_data: dict[str, Any] = {"needle_held_on_arrival": still, "release": res["status"],
                                  "needle_pose_source": "carried from the last reliable needle-to-gripper relation (perception unavailable)"
                                  if carried else "the needle-to-gripper relation of the grasp"}
    if released:
        tip = np.array([released["x_m"], released["y_m"], released["z_m"]])
        off = tip - entry
        lateral = off - site.scene.surface_normal * float(np.dot(off, site.scene.surface_normal))
        place_data.update({"needle_height_over_pad_mm_at_release": released["height_over_pad_mm"],
                           "lateral_distance_from_the_entry_mm": round(float(np.linalg.norm(lateral)) * 1000, 3)})
    back = move(rt, "place:retreat", [site.poses[sheet["start_pose"]].T], 120)
    if back["status"] != "succeeded":
        return Step("failed", {"data": {"released": True, "place": {**place_data, "reason": back["reason"]}}})
    return Step("next", {"data": {"placed": True, "released": True, "place": place_data}})


def verdict(rt: Runner) -> Step:
    """The machine verdict, from measurements."""
    st, d = rt.state, rt.state["data"]
    events = rt.env.log.events(rt.run_id)
    rejections = sum(1 for a in st["actions"] if a["status"] == "aborted")
    stops = sum(1 for e in events if e["type"] in ("operator_stop", "watchdog", "robot_fault", "limit_reached", "precondition_failed"))
    reach_ok = bool(d.get("reach")) and d["reach"]["status"] == "succeeded" and d["reach"]["error_mm"] <= REACH_MM \
        and d["reach"]["error_deg"] <= REACH_DEG
    rec = d.get("recovery")
    p = d.get("place") or {}
    placed_ok = bool(d.get("placed")) and p.get("needle_held_on_arrival") and p.get("lateral_distance_from_the_entry_mm", 99) <= 2.0
    if placed_ok and reach_ok and d.get("held_at_check") and not rec:
        machine, why = "success", None
    elif rec and rec.get("done"):
        machine, why = "failed_recovered", rec.get("reason")
    elif rec is None and d.get("placed"):
        machine, why = "failed", "placed without the needle or away from the entry"
    else:
        machine, why = "needs_manual_recovery", st["task"]["why"] or (rec or {}).get("reason")
    err = pose_distance(rt.env.robot.state().measured_cp, rt.env.site.poses[st["sheet"]["start_pose"]].T)[0] * 1000
    return Step("next", {"data": {"outcome": machine, "verdict": {
        "machine": machine, "why": why, "reach_tolerance_met": reach_ok, "needle_held": bool(d.get("held_at_check")),
        "needle_height_over_pad_mm_at_release": p.get("needle_height_over_pad_mm_at_release"),
        "lateral_distance_from_the_entry_mm": p.get("lateral_distance_from_the_entry_mm"),
        "motions_stopped_or_rejected": rejections, "stops": stops, "recovery_ran": rec is not None,
        "arm_at_start_pose": err <= 1.0, "outcome_source": st["sheet"]["outcome_source"]}}})


def confirm(rt: Runner) -> Step:
    """The run pauses here. The operator confirms the machine verdict or overrides it with a reason."""
    row = rt.env.db.one("SELECT v FROM kv WHERE k=?", (f"verdict_decision:{rt.run_id}",))
    machine = rt.state["data"]["verdict"]["machine"]
    if row is None:
        raise Interrupt({"kind": "operator_confirmation", "machine_verdict": machine})
    dec = json.loads(row["v"])
    if dec["decision"] == "override" and len((dec.get("reason") or "").strip()) < 10:
        rt.env.db.execute("DELETE FROM kv WHERE k=?", (f"verdict_decision:{rt.run_id}",))
        rt.emit("check_failed", {"node": "confirm", "check": "an override has a reason", "detail": "refused: no reason given"})
        raise Interrupt({"kind": "operator_confirmation", "machine_verdict": machine, "note": "an override needs a reason"})
    operator = {"decision": dec["decision"], "by": dec["by"], "verdict": machine if dec["decision"] == "confirm" else dec["to"],
                "reason": dec.get("reason") or None}
    return Step("next", {"data": {"operator_verdict": operator}})        # kept beside the machine verdict


def decide_verdict(db: Any, run_id: str, by: str, decision: str = "confirm", to: str | None = None, reason: str = "") -> None:
    if decision not in ("confirm", "override"):
        raise ValueError("decision is confirm or override")
    db.execute("INSERT OR REPLACE INTO kv(k,v) VALUES (?,?)",
               (f"verdict_decision:{run_id}", dumps({"decision": decision, "to": to, "reason": reason, "by": by, "at": time.time()})))


def record(rt: Runner) -> Step:
    """The trial record, and one summary written from it."""
    st, d = rt.state, rt.state["data"]
    rec = {"run_id": rt.run_id, "site": st["sheet"]["site"], "arm": st["sheet"]["arm"], "instrument": st["sheet"]["instrument"],
           "operator": st["sheet"]["operator"], "rollout": _x(rt)["demo"]["id"], "envelope": f"{st['sheet']['envelope_profile']} v{st['sheet']['envelope_version']}",
           "stages_run": [h["node"] for h in st["history"]], "reach": d.get("reach"), "needle_rose_mm": d.get("needle_rose_mm"),
           "place": d.get("place"), "recovery": d.get("recovery"), "machine_verdict": d["verdict"], "operator_verdict": d.get("operator_verdict"),
           "tip_travel_mm": st["usage"]["path_mm"], "motions": len(st["actions"])}
    v, o = d["verdict"], d.get("operator_verdict") or {}
    template = (f"Trial {rt.run_id}: machine verdict {v['machine']}" + (f" ({v['why']})" if v["why"] else "")
                + f". Reach error {(d.get('reach') or {}).get('error_mm')} mm. Needle held at the check: {v['needle_held']} "
                f"({v['outcome_source']}). Operator: {o.get('decision')}" + (f" to {o.get('verdict')}, reason: {o.get('reason')}" if o.get("decision") == "override" else "") + ".")
    text, source = trial.model_text(rt, "Summarise this trial record for the lab's log. State the machine verdict and the "
                                    "operator's decision separately.", rec, template)
    return Step("next", {"data": {"record": rec, "summary": text, "summary_source": source}})


def stop(rt: Runner) -> Step:
    d, why = rt.state["data"], rt.state["task"]["why"]
    update: dict[str, Any] = {"outcome": d.get("outcome") or f"stopped:{why or 'unknown'}", "motions_executed": len(rt.state["actions"])}
    failed = (d.get("preflight") or {}).get("failed")
    if failed:
        update["repairs"] = sorted({f["repair"] for f in failed if f.get("repair")})
    return Step("next", {"data": update})


def abort_to(state: dict[str, Any], node: str, limits: Limits) -> str:
    if node == "recover":
        return "verdict"
    if not state["actions"]:
        return "stop"
    if state["usage"]["recoveries"] >= limits.max_recoveries:
        return "verdict"
    return "recover"


def build_graph() -> Graph:
    g = Graph("preflight", abort_to, exempt=("recover", "verdict", "confirm", "record", "stop"))
    g.abort_targets = {"recover", "stop", "verdict"}
    ready = ("arm_ready", "calibration_valid", "envelope_unchanged", "go_granted")
    g.add_node("preflight", preflight)
    g.add_node("rehearse", rehearse)
    g.add_node("operator_go", operator_go, requires=("arm_ready", "calibration_valid", "envelope_unchanged"))
    g.add_node("stage", stage, requires=ready)
    g.add_node("reach", reach, requires=ready)
    g.add_node("grasp", grasp, requires=ready)
    g.add_node("retention", retention, requires=("arm_ready",))
    g.add_node("place", place, requires=("arm_ready", "go_granted", "needle_held"))
    g.add_node("recover", trial.recover, requires=("no_fault",))
    g.add_node("verdict", verdict)
    g.add_node("confirm", confirm)
    g.add_node("record", record)
    g.add_node("stop", stop)
    g.add_conditional("preflight", {"passed": "rehearse", "failed": "stop"})
    g.add_conditional("rehearse", {"clean": "operator_go", "rejected": "stop"})
    g.add_conditional("operator_go", {"go": "stage", "rejected": "stop", "timeout": "stop"})
    g.add_conditional("stage", {"ready": "reach", "failed": "recover"})
    g.add_conditional("reach", {"reached": "grasp", "not_reached": "recover"})
    g.add_conditional("grasp", {"next": "retention", "failed": "recover"})
    g.add_conditional("retention", {"held": "place", "not_held": "recover"})
    g.add_conditional("place", {"next": "verdict", "failed": "recover"})
    g.add_conditional("recover", {"recovered": "verdict", "failed": "verdict", "rejected": "verdict"})
    g.add_edge("verdict", "confirm")
    g.add_edge("confirm", "record")
    g.add_edge("record", END)
    g.add_edge("stop", END)
    return g


def start(env: Env, sheet: RunSheet, who: Principal, run_id: str, limits: Limits | None = None) -> RunResult:
    """env.extras must hold: contract, adapter, demo {id, poses, actions, goal}, and for a full
    pre-flight: compat, perception, needle, tool_dir."""
    if "trial:run" not in who.scopes or who.site != sheet.site:
        raise PermissionError(f"{who.user_id} may not run a trial at {sheet.site}")
    return Runner(build_graph(), env, limits).start(run_id, sheet.model_dump(), who)


__all__ = ["build_graph", "decide_verdict", "start", "rollout_targets", "trial_poses", "run_action"]
