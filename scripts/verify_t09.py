"""Verification of task 09 (trial pipeline for a simulation-trained policy), on the twin.

    python scripts/verify.py t09     ->  verification/t09_trial_pipeline.json

Reach replays real expert rollouts (SurgicAI, MIT) through the policy contract's adapter and the
safety filter. The needle is DECLARED per run. Nothing here measures task success on a robot.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verify_t10 import measured_contract                                   # noqa: E402
from vxagent.config import get_settings                                    # noqa: E402
from vxagent.data import episodes as rosma                                 # noqa: E402
from vxagent.data import quality                                           # noqa: E402
from vxagent.diagnostics.episode import calibration                        # noqa: E402
from vxagent.llm.ollama import get_llm                                     # noqa: E402
from vxagent.memory import site as site_memory                             # noqa: E402
from vxagent.pipelines import policy_trial as PT                           # noqa: E402
from vxagent.pipelines import trial                                        # noqa: E402
from vxagent.policy import compat                                          # noqa: E402
from vxagent.policy.adapter import Adapter                                 # noqa: E402
from vxagent.profiles import principal                                     # noqa: E402
from vxagent.robot.kinematics import from_rotvec, interpolate              # noqa: E402
from vxagent.robot.needle import DeclaredNeedle                            # noqa: E402
from vxagent.robot.safety import check_pose, check_step, check_tracking    # noqa: E402
from vxagent.robot.site import build_site, load_site                       # noqa: E402
from vxagent.runtime.graph import Runner                                   # noqa: E402
from vxagent.store import Database                                         # noqa: E402
from vxagent.structured import proposals                                   # noqa: E402
from vxagent.tools.executor import decide_approval                         # noqa: E402
from vxagent.tools.service import store_calibration_record                 # noqa: E402

SMOKE = bool(os.environ.get("VX_SMOKE"))


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def boundary_table(site: Any) -> list[dict[str, Any]]:
    """A case just inside and just outside every limit of the safety filter."""
    p, sc, ch = site.profile, site.scene, site.chain
    start = site.poses["start"].T
    rows: list[dict[str, Any]] = []

    def shifted(T: np.ndarray, d_mm: Any = (0, 0, 0), rot_deg: float = 0.0) -> np.ndarray:
        out = T.copy()
        out[:3, 3] += np.array(d_mm, dtype=float) / 1000.0
        if rot_deg:
            out[:3, :3] = from_rotvec([0.0, 0.0, math.radians(rot_deg)]) @ out[:3, :3]
        return out

    def step(name: str, T_from: np.ndarray, T_to: np.ndarray, rule: str, expect_reject: bool) -> None:
        q, _ = ch.ik(T_to, site.home_q)
        got = [v.rule for v in check_step(p, sc, ch, T_from, T_to, q)]
        rows.append({"limit": name, "expected": "rejected" if expect_reject else "accepted",
                     "got": "rejected" if rule in got else "accepted", "rules_fired": got})

    def pose(name: str, T: np.ndarray, rule: str, expect_reject: bool, q: Any = None) -> None:
        if q is None:
            q, _ = ch.ik(T, site.home_q)
        got = [v.rule for v in check_pose(p, sc, ch, T, q)]
        rows.append({"limit": name, "expected": "rejected" if expect_reject else "accepted",
                     "got": "rejected" if rule in got else "accepted", "rules_fired": got})

    step("free-space step 5 mm: 4.9 mm", start, shifted(start, (4.9, 0, 0)), "step_position", False)
    step("free-space step 5 mm: 5.1 mm", start, shifted(start, (5.1, 0, 0)), "step_position", True)
    step("free-space rotation 5 deg: 4.9 deg", start, shifted(start, rot_deg=4.9), "step_rotation", False)
    step("free-space rotation 5 deg: 5.1 deg", start, shifted(start, rot_deg=5.1), "step_rotation", True)
    near = site.poses["pregrasp"].T                               # 8 mm over the pad: inside the near zone
    step("near step 2 mm: 1.9 mm", near, shifted(near, (1.9, 0, 0)), "step_position", False)
    step("near step 2 mm: 2.1 mm", near, shifted(near, (2.1, 0, 0)), "step_position", True)
    step("near rotation 3 deg: 2.9 deg", near, shifted(near, rot_deg=2.9), "step_rotation", False)
    step("near rotation 3 deg: 3.1 deg", near, shifted(near, rot_deg=3.1), "step_rotation", True)
    needle = sc.near_points[0]
    far, close = start.copy(), start.copy()
    far[:3, 3] = needle + sc.surface_normal * 0.0 + np.array([0.0, 0.0, 0.0])
    up = sc.surface_normal
    far[:3, 3] = needle + up * 0.0215                             # 21.5 mm above the needle, which lies on the pad
    close[:3, 3] = needle + up * 0.0195
    side = np.array([1.0, 0.0, 0.0])
    step("near zone starts 20 mm from the needle: a 3 mm step at 21.5 mm", far, shifted(far, side * 3.0 + up * 3.0e0 * 0), "step_position", False)
    step("near zone starts 20 mm from the needle: a 3 mm step at 19.5 mm", close, shifted(close, side * 3.0), "step_position", True)
    half = p.workspace_half_m * 1000
    pose(f"workspace +/- {half:g} mm: {half - 0.1:g} mm", shifted(start, (half - 0.1, 0, 0)), "workspace", False)
    pose(f"workspace +/- {half:g} mm: {half + 0.1:g} mm", shifted(start, (half + 0.1, 0, 0)), "workspace", True)
    clear = p.clearance_m * 1000
    pad = start.copy()
    pad[:3, 3] = sc.surface_point + np.array([0.015, 0.010, 0.0])
    pose(f"clearance {clear:g} mm over the pad: {clear + 0.1:g} mm", shifted(pad, tuple(up * (clear + 0.1))), "clearance", False)
    pose(f"clearance {clear:g} mm over the pad: {clear - 0.1:g} mm", shifted(pad, tuple(up * (clear - 0.1))), "clearance", True)
    j = 4                                                         # wrist pitch
    margin = math.degrees(p.joint_margin_rad)
    q_in, q_out = np.array(site.home_q, dtype=float), np.array(site.home_q, dtype=float)
    q_in[j] = ch.qmax[j] - math.radians(margin + 0.2)
    q_out[j] = ch.qmax[j] - math.radians(margin - 0.2)
    pose(f"joint margin {margin:g} deg: {margin + 0.2:g} deg from the wrist pitch limit", ch.fk(q_in), "joint_margin", False, q_in)
    pose(f"joint margin {margin:g} deg: {margin - 0.2:g} deg from the wrist pitch limit", ch.fk(q_out), "joint_margin", True, q_out)
    for name, d_mm, rot, rule in (("tracking stop 2 mm: 1.9 mm", 1.9, 0.0, "tracking_position"), ("tracking stop 2 mm: 2.1 mm", 2.1, 0.0, "tracking_position"),
                                  ("tracking stop 2 deg: 1.9 deg", 0.0, 1.9, "tracking_rotation"), ("tracking stop 2 deg: 2.1 deg", 0.0, 2.1, "tracking_rotation")):
        got = [v.rule for v in check_tracking(p, start, start, shifted(start, (d_mm, 0, 0), rot))]
        expect = d_mm > 2.0 or rot > 2.0
        rows.append({"limit": name, "expected": "rejected" if expect else "accepted", "got": "rejected" if rule in got else "accepted",
                     "rules_fired": got})
    got = [v.rule for v in check_tracking(p, start, shifted(start, (0.5, 0, 0)), shifted(start, (0.5, 0, 0)))]
    rows.append({"limit": "a set point that is not the requested target (clipped by inverse kinematics)", "expected": "rejected",
                 "got": "rejected" if "clipped" in got else "accepted", "rules_fired": got})
    for r in rows:
        r["match"] = r["expected"] == r["got"]
    return rows


def kabsch(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t that best take the points A onto B."""
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cb - R @ ca


def t09() -> dict[str, Any]:
    s = get_settings()
    site = load_site("site_a")
    llm = get_llm()
    db = Database()
    op = principal("op.lee", "operator", site.name)
    out: dict[str, Any] = {"task": "t09_trial_pipeline" + ("_smoke" if SMOKE else ""), "ran": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "model": s.model, "site": site.name, "backend": "twin", "control_rate_hz": 10.0,
                           "needle": "DECLARED per run: the twin has no needle",
                           "not_measured": "task success on a robot; the published success rates belong to the paper's physical trials"}

    # ---------------------------------------------------------------- a. safety filter boundary table
    table = boundary_table(site)
    out["a_safety_filter_boundaries"] = {"cases": len(table), "match": sum(r["match"] for r in table), "rows": table}
    log(f"a boundaries: {out['a_safety_filter_boundaries']['match']}/{len(table)}")

    # ---------------------------------------------------------------- b. the twin's kinematics against recordings
    rows = []
    for e in rosma.list_episodes()[:(2 if SMOKE else 12)]:
        t, cols = rosma.load(e)
        q = np.stack([cols[f"PSM1_joint_position_{i}"] for i in range(1, 7)], axis=1)[::10]
        rec = np.stack([cols[f"PSM1_position_{a}"] for a in "xyz"], axis=1)[::10]
        inside = ((q >= site.chain.qmin - 1e-6) & (q <= site.chain.qmax + 1e-6)).all(axis=1)
        fk = np.array([site.chain.fk(x)[:3, 3] for x in q])
        R, tr = kabsch(fk, rec)
        res = np.linalg.norm(fk @ R.T + tr - rec, axis=1) * 1000
        raw = np.linalg.norm(fk - rec, axis=1) * 1000
        rows.append({"episode": e.episode_id, "samples_used": int(len(q)), "samples_inside_the_joint_limits": int(inside.sum()),
                     "difference_without_alignment_mm_median": round(float(np.median(raw)), 2),
                     "residual_after_one_rigid_alignment_mm": {"median": round(float(np.median(res)), 3), "p95": round(float(np.percentile(res, 95)), 3),
                                                               "max": round(float(res.max()), 3)},
                     "rotation_of_the_alignment_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))), 2),
                     "translation_of_the_alignment_mm": [round(float(x) * 1000, 1) for x in tr]})
    out["b_twin_against_recordings"] = {
        "what": "forward kinematics of the twin (PSM.json with LARGE_NEEDLE_DRIVER_400006.json) on the PSM1 joint values recorded "
                "in ROSMA, against the PSM1 positions recorded in the same samples. The recordings do not say which instrument "
                "was mounted or which frame the positions are in, so one rigid transform per recording is fitted first.",
        "episodes": len(rows), "median_of_the_median_residuals_mm": round(float(np.median([r["residual_after_one_rigid_alignment_mm"]["median"] for r in rows])), 3),
        "largest_p95_residual_mm": max(r["residual_after_one_rigid_alignment_mm"]["p95"] for r in rows), "rows": rows}
    log(f"b twin vs ROSMA: median residual {out['b_twin_against_recordings']['median_of_the_median_residuals_mm']} mm")

    # ---------------------------------------------------------------- the pieces of a trial
    approach = quality.from_surgicai("Approach")
    contract, _ = measured_contract(approach)
    spec = json.loads((s.sites_dir / "site_a.json").read_text(encoding="utf-8"))
    tool_dir = s.dvrk_dir / "share" / "tool"

    def prepare(i: int, needle: dict[str, bool] | None = None, site_: Any = None, ready: bool = True, use_llm: bool = False,
                cal_fault: dict[str, Any] | None = None, session: str = "s1", scene_session: str | None = "s1",
                perception: Any = "ok", contract_: Any = None, demo_shift: Any = None) -> tuple[Any, Any, Any, Any]:
        st = site_ or site
        c = contract_ or contract
        e = approach[i % len(approach)]
        poses = np.array(e.notes["poses"], dtype=float)
        if demo_shift is not None:
            poses = poses.copy()
            poses[:, :3] += np.array(demo_shift, dtype=float)
            poses[-1, :3] = e.notes["poses"][-1][:3]
        demo = {"id": e.episode_id, "poses": poses, "actions": e.actions, "goal": e.notes["goal"]}
        twin = st.twin()
        if ready:
            twin.enable()
            twin.home()
        ad = Adapter.anchored(c, demo["goal"], st.poses["pregrasp"].T)
        for table_ in ("site_facts", "site_events"):
            site_memory._ready(db)
            db.execute(f"DELETE FROM {table_} WHERE site=?", (st.name,))
        db.execute("DELETE FROM kv WHERE k IN (?,?)", (f"calrec:{st.name}", f"scene_session:{st.name}"))
        proposals.set_calibration(db, st.name, True, "calibrated (verification)")
        if cal_fault != "missing":
            cal = calibration(cal_fault if isinstance(cal_fault, dict) and "cause" in cal_fault else None, st.scene.workspace_center,
                              np.random.default_rng(17))
            T = cal["T_est"].copy()
            if cal_fault == "malformed":
                T[:3, :3] *= 1.2
            store_calibration_record(db, st.name, {"id": cal["id"], "T_est": T.tolist(), "seen_m": cal["seen_m"].tolist(),
                                                   "touched_m": cal["touched_m"].tolist(), "method": "point pairs", "session": session})
        if scene_session is not None:
            db.execute("INSERT OR REPLACE INTO kv(k,v) VALUES (?,?)", (f"scene_session:{st.name}", json.dumps(scene_session)))
        nd = DeclaredNeedle(st, **(needle or {})).attach(twin)
        look = {"ok": lambda: {"visible": True, "latency_s": 0.04}, "slow": lambda: {"visible": True, "latency_s": 0.9},
                "none": None}[perception]
        cp = compat.check(c, st, "start", "pregrasp", 10.0, policy_start=demo["poses"][0], policy_goal=demo["goal"],
                          seen_positions=np.concatenate([x.position for x in approach[:5]]))
        env = trial.make_env(st, twin, db, llm=llm if use_llm else None,
                             extras={"contract": c, "adapter": ad, "demo": demo, "needle": nd, "compat": cp, "perception": look,
                                     "tool_dir": tool_dir})
        sheet = trial.make_sheet(st, op.user_id, {"dataset": "SurgicAI", "episode_id": e.episode_id, "arm": "PSM1", "first_sample": 0,
                                                  "last_sample": len(e.actions), "start_s": 0.0, "end_s": len(e.actions) / 10.0})
        return env, sheet, twin, nd

    def commands(run_id: str) -> int:
        return len(db.query("SELECT 1 FROM commands WHERE run_id=?", (run_id,)))

    # ---------------------------------------------------------------- c. pre-flight matrix
    other = copy.deepcopy(spec)
    other["safety_profile"]["near_step_mm"] = 1.0
    tight_site = build_site(other, site.chain)
    cases: list[tuple[str, dict[str, Any], str, str]] = [
        ("arm not enabled", {"ready": False}, "arm is powered, enabled and homed, with no fault", "bring_up"),
        ("another instrument on the run sheet", {"sheet_instrument": "MEGA_NEEDLE_DRIVER_400194"},
         "instrument is the one on the run sheet and has an instrument file", "escalate_to_engineer"),
        ("no calibration record", {"cal_fault": "missing"}, "a calibration record exists", "calibration_validity"),
        ("calibration record with a scaled rotation", {"cal_fault": "malformed"}, "the calibration record is well formed", "calibration_validity"),
        ("camera arm moved since the calibration", {"event": "ecm_moved"}, "the calibration has not been ended by an event", "calibration_validity"),
        ("calibration off by 6 mm", {"cal_fault": {"cause": "calibration_offset", "mm": 6.0, "direction": [1.0, 0.0, 0.0]}},
         "the calibration residual is under the site's limit", "calibration_validity"),
        ("scene records from another session", {"scene_session": "s0"},
         "pad-plane and site records are of the same session as the calibration", "rig_check"),
        ("site whose near step is 1 mm", {"site_": tight_site}, "the policy contract is compatible with the site", "escalate_to_engineer"),
        ("rollout that starts 30 mm from where the policy was trained", {"demo_shift": [3.0, 0.0, 0.0]},
         "start and target lie in the policy's training range and inside the workspace", "escalate_to_engineer"),
        ("perception answers after 0.9 s", {"perception": "slow"}, "the perception source answered within its time limit", "rig_check"),
    ]
    pre = []
    for label, kw, check_name, repair in (cases[:3] if SMOKE else cases):
        kw = dict(kw)
        sheet_instrument, event = kw.pop("sheet_instrument", None), kw.pop("event", None)
        env, sheet, twin, _ = prepare(3, **kw)
        if sheet_instrument:
            sheet = sheet.model_copy(update={"instrument": sheet_instrument})
        if event:
            site_memory.record_event(db, env.site.name, event, "eng.kim", "verification")
        seq0, run_id = twin.state().seq, f"pf_{len(pre)}_{int(time.time() * 1000) % 100000}"
        res = PT.start(env, sheet, op, run_id)
        failed = (res.state["data"].get("preflight") or {}).get("failed", [])
        names = [f["check"] for f in failed]
        pre.append({"case": label, "outcome": res.outcome, "failed_checks": names, "repairs_named": sorted({f["repair"] for f in failed}),
                    "expected_check": check_name, "expected_repair": repair, "motions_commanded": commands(run_id),
                    "robot_state_changed": twin.state().seq != seq0,
                    "as_expected": res.outcome == "preflight_failed" and check_name in names and repair in {f["repair"] for f in failed}
                    and commands(run_id) == 0})
        log(f"c {label}: {res.outcome} failed {names} repairs {pre[-1]['repairs_named']}")
    out["c_preflight_matrix"] = {"cases": len(pre), "as_expected": sum(p["as_expected"] for p in pre),
                                 "motions_commanded_in_all_cases": sum(p["motions_commanded"] for p in pre), "rows": pre}

    # ---------------------------------------------------------------- d. trials end to end
    plan = ([{}] * 10 + [{"slips_on_lift": True}] * 4 + [{"slips_while_holding": True}] * 3 + [{"grasp_succeeds": False}] * 3)
    if SMOKE:
        plan = [{}, {"slips_on_lift": True}]
    trials = []
    for i, needle in enumerate(plan):
        env, sheet, twin, nd = prepare(i, needle=needle, use_llm=True)
        env.tick = time.sleep
        run_id = f"pt_{i:02d}_{int(time.time()) % 100000}"
        t0 = time.time()
        res = PT.start(env, sheet, op, run_id)
        runner = Runner(PT.build_graph(), env)
        decide_approval(db, res.waiting_for["approval_id"], True, op.user_id)
        res = runner.resume(run_id)
        paused_for = (res.waiting_for or {}).get("kind")
        machine = res.state["data"]["verdict"]["machine"]
        override = None
        if i == 1:                                    # an override without a reason must be refused
            PT.decide_verdict(db, run_id, op.user_id, "override", to="failed", reason="")
            res = runner.resume(run_id)
            override = {"without_a_reason": "refused" if res.status == "waiting" else "accepted"}
            PT.decide_verdict(db, run_id, op.user_id, "override", to="failed", reason="The needle sat crooked in the jaw on video.")
        else:
            PT.decide_verdict(db, run_id, op.user_id, "confirm")
        res = runner.resume(run_id)
        d = res.state["data"]
        expected = "success" if not needle else "failed_recovered"
        trials.append({"rollout": env.extras["demo"]["id"], "needle_declared": nd.declared(), "needle_log": nd.log,
                       "route": [h["node"] for h in res.state["history"]], "paused_for": paused_for,
                       "machine_verdict": d["verdict"], "operator_verdict": d.get("operator_verdict"), "override_test": override,
                       "machine_verdict_kept_beside_an_override": d["verdict"]["machine"] == machine,
                       "reach": d.get("reach"), "needle_rose_mm": d.get("needle_rose_mm"), "place": d.get("place"),
                       "recovery": d.get("recovery"), "tip_travel_mm": res.state["usage"]["path_mm"],
                       "motions": [(a["label"], a["status"]) for a in res.state["actions"]],
                       "first_steps_checked": sum(1 for e in runner.env.log.events(run_id) if e["type"] == "check_passed"
                                                  and e["data"]["check"] == "first step of the stage is tracked"),
                       "summary": d.get("summary"), "summary_source": d.get("summary_source"),
                       "expected": expected, "as_expected": machine == expected, "seconds": round(time.time() - t0, 1)})
        log(f"d {trials[-1]['rollout']} {needle or 'holds'}: {machine}, reach {d['reach']['error_mm']} mm / {d['reach']['error_deg']} deg "
            f"({trials[-1]['seconds']} s, summary by {d.get('summary_source')})")
    reach_mm = [t["reach"]["error_mm"] for t in trials]
    reach_deg = [t["reach"]["error_deg"] for t in trials]
    out["d_trials"] = {
        "trials": len(trials), "as_expected": sum(t["as_expected"] for t in trials),
        "machine_verdicts": {v: sum(t["machine_verdict"]["machine"] == v for t in trials) for v in sorted({t["machine_verdict"]["machine"] for t in trials})},
        "reach_within_5_mm_and_15_deg": sum(t["machine_verdict"]["reach_tolerance_met"] for t in trials),
        "reach_error_mm_max": max(reach_mm), "reach_error_deg_max": max(reach_deg),
        "policy_actions_min_max": [min(t["reach"]["policy_actions"] for t in trials), max(t["reach"]["policy_actions"] for t in trials)],
        "motions_rejected_or_stopped": sum(t["machine_verdict"]["motions_stopped_or_rejected"] for t in trials),
        "arm_at_start_pose_at_the_end": sum(t["machine_verdict"]["arm_at_start_pose"] for t in trials),
        "every_run_paused_for_the_operators_confirmation": all(t["paused_for"] == "operator_confirmation" for t in trials),
        "summaries_written_by_the_model": sum(t["summary_source"] == "model" for t in trials),
        "summaries_replaced_by_the_template": sum(t["summary_source"] != "model" for t in trials),
        "override": next((t for t in trials if t["override_test"]), None) and {
            "without_a_reason": next(t for t in trials if t["override_test"])["override_test"]["without_a_reason"],
            "operator_verdict": next(t for t in trials if t["override_test"])["operator_verdict"],
            "machine_verdict": next(t for t in trials if t["override_test"])["machine_verdict"]["machine"]},
        "runs": trials}

    # A site with more clearance: the grasp descent of the same rollout is rejected in rehearsal.
    high = copy.deepcopy(spec)
    high["safety_profile"]["clearance_mm"] = 6.0
    env, sheet, twin, _ = prepare(5, site_=build_site(high, site.chain))
    run_id = f"reh_{int(time.time()) % 100000}"
    res = PT.start(env, sheet, op, run_id)
    out["d_trials"]["rehearsal_rejection"] = {
        "site": "site-a with the clearance over the pad raised to 6 mm", "outcome": res.outcome,
        "rejected_in": [r["stage"] for r in res.state["data"].get("dry_run", {}).get("rejections", [])],
        "rule": [v["rule"] for r in res.state["data"].get("dry_run", {}).get("rejections", []) for v in r["violations"]],
        "operator_was_asked": any(e["type"] == "approval_requested" for e in env.log.events(run_id)),
        "motions_commanded": commands(run_id)}
    log(f"d rehearsal rejection: {out['d_trials']['rehearsal_rejection']}")
    return out
