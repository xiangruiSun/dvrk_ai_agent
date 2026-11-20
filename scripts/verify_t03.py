"""Verification of task 03 (event-driven trial runtime) on the kinematic twin.

    python scripts/verify.py t03     ->  verification/t03_runtime.json

The approach paths are stretches of real recorded dVRK motion (ROSMA), mapped into the site.
Whether the needle is held is DECLARED per run, because the twin has no needle. Every fault is
INJECTED and labelled so.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vxagent.config import get_settings                                    # noqa: E402
from vxagent.data import episodes as ep_store                              # noqa: E402
from vxagent.llm.ollama import get_llm                                     # noqa: E402
from vxagent.pipelines import paths, trial                                 # noqa: E402
from vxagent.profiles import principal                                     # noqa: E402
from vxagent.robot import config as cfg                                    # noqa: E402
from vxagent.robot.kinematics import pose_distance                         # noqa: E402
from vxagent.robot.site import build_site, load_site                       # noqa: E402
from vxagent.robot.twin import KinematicTwin                               # noqa: E402
from vxagent.runtime.events import EventLog, rebuild                       # noqa: E402
from vxagent.runtime.graph import Limits, Runner                           # noqa: E402
from vxagent.store import Database                                         # noqa: E402
from vxagent.structured import proposals                                   # noqa: E402
from vxagent.structured.rules import InstrumentSet, RuleContext, validate_patch   # noqa: E402
from vxagent.structured.schemas import ConfigPatch                         # noqa: E402
from vxagent.tools.executor import decide_approval                         # noqa: E402


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def t03() -> dict[str, Any]:
    s = get_settings()
    site = load_site("site_a")
    work = s.runtime_dir / "t03"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    db_path = work / "t03.sqlite"
    db = Database(db_path)
    elog = EventLog(db)
    llm = get_llm()
    op = principal("op.lee", "operator", site.name)
    all_runs: list[str] = []

    out: dict[str, Any] = {"task": "t03_runtime", "ran": time.strftime("%Y-%m-%d %H:%M:%S"), "model": s.model,
                           "site": site.name, "backend": "twin", "control_rate_hz": 10.0,
                           "needle_held": "DECLARED per run: the twin has no needle",
                           "approach_paths": "stretches of recorded PSM1 motion from ROSMA (Zenodo 3932964, CC BY 4.0), "
                                             "mapped into the site by rotation, uniform scale and translation"}

    # ---------------------------------------------------------------- helpers
    def fresh() -> KinematicTwin:
        twin = site.twin()
        twin.enable()
        twin.home()
        return twin

    def approve(res: Any, yes: bool = True) -> int:
        aid = res.waiting_for["approval_id"]
        decide_approval(db, aid, yes, op.user_id)
        return aid

    def events_of(run_id: str) -> list[dict[str, Any]]:
        return elog.events(run_id)

    def digest(run_id: str, twin: Any) -> dict[str, Any]:
        evs = events_of(run_id)
        st = rebuild(evs)
        assert st is not None
        kinds: dict[str, int] = {}
        for e in evs:
            kinds[e["type"]] = kinds.get(e["type"], 0) + 1
        seconds = {e["data"]["node"]: e["data"]["seconds"] for e in evs if e["type"] == "node_finished"}
        d_mm = pose_distance(twin.state().measured_cp, site.poses["start"].T)[0] * 1000
        return {"run_id": run_id, "outcome": st["task"]["outcome"], "why": st["task"]["why"],
                "path": [f"{h['node']}:{h['route']}" for h in st["history"]],
                "events": len(evs), "events_by_type": kinds,
                "checkpoints": len(db.query("SELECT 1 FROM checkpoints WHERE run_id=?", (run_id,))),
                "seconds_per_node": seconds, "tip_travel_mm": st["usage"]["path_mm"],
                "motions": [{"label": a["label"], "status": a["status"], "path_mm": a["path_mm"]} for a in st["actions"]],
                "recovery": st["data"].get("recovery"),
                "arm_at_start_pose_at_end": d_mm <= 1.0, "arm_busy_at_end": twin.state().busy,
                "spans": {r["kind"]: r["n"] for r in db.query(
                    "SELECT kind, COUNT(*) AS n FROM spans WHERE run_id=? GROUP BY kind", (run_id,))}}

    def env_for(twin: Any, use_llm: bool = False, **kw: Any) -> Any:
        return trial.make_env(site, twin, db, llm=llm if use_llm else None, **kw)

    def go(env: Any, sheet: Any, held: bool, limits: Limits | None = None) -> tuple[Any, Runner, dict[str, Any]]:
        res, info = trial.start_trial(env, sheet, op, limits, held=held)
        all_runs.append(res.run_id)
        return res, Runner(trial.build_graph(), env, limits), info

    # ---------------------------------------------------------------- the recorded paths
    eps = ep_store.list_episodes()
    sources, stamp_note = [], []
    for e in eps:
        t, p = paths.positions(e)
        seg = paths.find_segment(t, p)
        st = ep_store.compute_stats(e)
        stamp_note.append({"episode": e.episode_id, "samples": st["samples"], "timestamps_moved_by_one_second":
                           st["timestamps_repaired"], "time_runs_forward": st["time_runs_forward"]})
        if seg:
            sources.append({"dataset": "ROSMA", "episode_id": e.episode_id, "arm": "PSM1", "first_sample": seg["first"],
                            "last_sample": seg["last"], "start_s": seg["start_s"], "end_s": seg["end_s"]})
    if not sources:
        raise SystemExit("no recorded stretch found: run scripts/fetch_rosma.py")

    def pick(i: int) -> dict[str, Any]:
        return sources[i % len(sources)]

    t01_file = ROOT / "verification" / "t01_tool_calling.json"
    unchanged = None
    if t01_file.exists():
        runs01 = json.loads(t01_file.read_text(encoding="utf-8"))["a_episode_questions"]["runs"]
        same = 0
        for r in runs01:
            st = ep_store.compute_stats(ep_store.get_episode(r["episode"]))
            now = (round(st["psm1_path_length_mm"] - st["psm2_path_length_mm"], 1) if r["question"] == "difference"
                   else st[r["question"]])
            same += int(now == r["truth"])
        unchanged = f"{same}/{len(runs01)}"
    out["recorded_time_stamps"] = {
        "note": "ROSMA writes milliseconds without leading zeros and, in some samples, a seconds field that is one "
                "sample late. The reader was corrected while this task was built.",
        "episodes": len(eps), "episodes_with_a_late_seconds_field": sum(x["timestamps_moved_by_one_second"] > 0 for x in stamp_note),
        "samples_moved_by_one_second": sum(x["timestamps_moved_by_one_second"] for x in stamp_note),
        "time_runs_forward_in_every_episode": all(x["time_runs_forward"] for x in stamp_note),
        "figures_of_task_01_unchanged_by_the_correction": unchanged, "per_episode": stamp_note}

    # ---------------------------------------------------------------- a. trials end to end
    rows = []
    for i, src in enumerate(sources):
        twin = fresh()
        env = env_for(twin, use_llm=True)
        held = i % 2 == 0
        t0 = time.time()
        res, runner, info = go(env, trial.make_sheet(site, op.user_id, src), held)
        waited_at = res.state["next"]
        seq_wait = twin.state().seq
        aid = approve(res)
        res = runner.resume(res.run_id)
        row = digest(res.run_id, twin)
        ap = db.one("SELECT * FROM approvals WHERE id=?", (aid,))
        row.update({"episode": src["episode_id"], "needle_held_declared": held, "path_fit": info,
                    "waited_at": waited_at, "robot_seq_while_waiting": seq_wait,
                    "approval": {"id": aid, "decided_by": ap["decided_by"], "status": ap["status"],
                                 "bound_to_robot_seq": ap["robot_seq"], "bound_to_envelope": ap["profile_version"]},
                    "summary": res.state["data"].get("summary"), "summary_source": res.state["data"].get("summary_source"),
                    "wall_seconds": round(time.time() - t0, 1),
                    "expected": "success" if held else "recovered"})
        row["as_expected"] = row["outcome"] == row["expected"] and row["arm_at_start_pose_at_end"]
        rows.append(row)
        log(f"a {src['episode_id']}: held {held} -> {row['outcome']} path {'>'.join(h.split(':')[0] for h in row['path'])} "
            f"({row['wall_seconds']} s, summary by {row['summary_source']})")
    out["a_trials"] = {
        "trials": len(rows), "recordings_without_a_usable_stretch": len(eps) - len(sources),
        "as_expected": sum(r["as_expected"] for r in rows),
        "success": sum(r["outcome"] == "success" for r in rows), "recovered": sum(r["outcome"] == "recovered" for r in rows),
        "other": [r["outcome"] for r in rows if r["outcome"] not in ("success", "recovered")],
        "arm_back_at_start_pose": sum(r["arm_at_start_pose_at_end"] for r in rows),
        "summaries_written_by_the_model": sum(r["summary_source"] == "model" for r in rows),
        "summaries_replaced_by_the_template": sum(r["summary_source"] != "model" for r in rows),
        "scale_of_the_path_fit_min_max": [min(r["path_fit"]["scale"] for r in rows), max(r["path_fit"]["scale"] for r in rows)],
        "largest_distance_from_the_straight_line_mm_min_max": [
            min(r["path_fit"]["largest_distance_from_the_straight_line_mm"] for r in rows),
            max(r["path_fit"]["largest_distance_from_the_straight_line_mm"] for r in rows)],
        "median_events": sorted(r["events"] for r in rows)[len(rows) // 2],
        "median_checkpoints": sorted(r["checkpoints"] for r in rows)[len(rows) // 2],
        "runs": rows}

    # ---------------------------------------------------------------- b. crash and resume, in separate processes
    sheet_file = work / "sheet.json"
    plant = work / "plant.json"

    def worker(*args: str) -> tuple[int, dict[str, Any] | None]:
        cmd = [sys.executable, "-m", "vxagent.pipelines.trial_worker", "--db", str(db_path), "--plant", str(plant), *args]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
        last = [ln for ln in r.stdout.strip().splitlines() if ln.startswith("{")]
        if r.returncode not in (0, 137):
            raise RuntimeError(f"worker failed ({r.returncode}): {r.stderr[-800:]}")
        return r.returncode, json.loads(last[-1]) if last else None

    def plant_state() -> dict[str, Any]:
        d = json.loads(plant.read_text(encoding="utf-8"))
        T = site.chain.fk(np.array(d["q"]))
        return {"seq": d["seq"], "enabled": d["enabled"], "homed": d["homed"],
                "mm_from_start": round(pose_distance(T, site.poses["start"].T)[0] * 1000, 3),
                "height_over_pad_mm": round(site.scene.height(T[:3, 3]) * 1000, 2), "jaw_closed": abs(d["jaw"]) < 1e-6}

    def crash_case(label: str, crash: str, held: bool, lost_while_down: bool = False,
                   controller_restarted: bool = False) -> dict[str, Any]:
        if plant.exists():
            plant.unlink()
        twin = KinematicTwin(site.chain, site.home_q, jaw=site.jaw_angles["open"], persist=plant)
        twin.enable()
        twin.home()
        run_id = f"crash_{label.replace(' ', '_')}_{int(time.time() * 1000) % 100000}"
        all_runs.append(run_id)
        src = pick(len(all_runs))
        sheet_file.write_text(trial.make_sheet(site, op.user_id, src).model_dump_json(), encoding="utf-8")
        node = crash.split(":")[0]
        before_go = node in ("preflight", "dry_run", "operator_go")
        start_args = ["--action", "start", "--run", run_id, "--sheet", str(sheet_file), "--operator", op.user_id,
                      "--held", "yes" if held else "no"]
        if before_go:
            code, _ = worker(*start_args, "--crash", crash)
        else:
            code, first = worker(*start_args)
            assert first and first["status"] == "waiting", first
            decide_approval(db, first["waiting_for"]["approval_id"], True, op.user_id)
            code, _ = worker("--action", "resume", "--run", run_id, "--crash", crash)
        row: dict[str, Any] = {"case": label, "killed_in": node, "at_control_tick": int(crash.split(":")[1]),
                               "exit_code_of_killed_process": code, "run_id": run_id}
        dead = plant_state()
        if lost_while_down:
            trial.declare_outcome(db, run_id, False, "verification: declared lost while the process was down")
        # A new process starts. It must mark the run and move nothing.
        boot_args = ["--action", "boot"] + (["--controller-restarted"] if controller_restarted else [])
        _, boot = worker(*boot_args)
        assert boot is not None
        after_boot = plant_state()
        status = db.one("SELECT status FROM runs WHERE run_id=?", (run_id,))["status"]
        _, nobody = worker("--action", "resume", "--run", run_id)
        row.update({"arm_when_killed": dead, "status_after_restart": status,
                    "marked_by_the_new_process": run_id in boot["marked_needs_reverification"],
                    "resume_without_an_operator": "refused" if nobody and "refused" in nobody else "ran",
                    "arm_moved_before_the_operator_resumed":
                        plant_state()["mm_from_start"] != dead["mm_from_start"] or plant_state()["jaw_closed"] != dead["jaw_closed"],
                    "controller_restarted_injected": controller_restarted, "needle_declared_lost_while_down": lost_while_down,
                    "arm_after_restart": after_boot})
        _, res = worker("--action", "resume", "--run", run_id, "--operator", op.user_id)
        assert res is not None
        hops = 0
        while res.get("status") == "waiting" and hops < 3:
            decide_approval(db, res["waiting_for"]["approval_id"], True, op.user_id)
            _, res = worker("--action", "resume", "--run", run_id, "--operator", op.user_id)
            assert res is not None
            hops += 1
        evs = events_of(run_id)
        cut = next(i for i, e in enumerate(evs) if e["type"] == "process_restarted")
        after = evs[cut + 1:]
        entered = [e["data"]["node"] for e in evs if e["type"] == "node_started"]
        st = rebuild(evs)
        assert st is not None
        row.update({
            "outcome": res.get("outcome"), "path": res.get("path"), "why": st["task"]["why"],
            "observed_before_anything_else_after_resume": [e["type"] for e in after[:2]] == ["resumed", "observed"],
            "precondition_failed_after_resume": next(([f["check"] for f in e["data"]["failed"]] for e in after
                                                      if e["type"] == "precondition_failed"), None),
            "times_the_killed_node_was_entered": entered.count(node),
            "place_entered_after_resume": any(e["type"] == "node_started" and e["data"]["node"] == "place" for e in after),
            "interrupted_motions": [a["label"] for a in st["actions"] if a["status"] == "interrupted"],
            "motions_after_resume": [e["data"]["label"] for e in after if e["type"] == "action_started"],
            "command_ids_unique": len({a["command_id"] for a in st["actions"]}) == len(st["actions"]),
            "recovery": st["data"].get("recovery"), "arm_at_end": plant_state()})
        log(f"b {label}: killed in {crash}, status {status}, outcome {row['outcome']}, "
            f"precondition {row['precondition_failed_after_resume']}")
        return row

    crashes = [
        crash_case("preflight", "preflight:0", True),
        crash_case("dry run", "dry_run:0", True),
        crash_case("waiting for the operator", "operator_go:0", True),
        crash_case("reach", "reach:6", True),
        crash_case("grasp", "grasp:2", True),
        crash_case("hold check", "hold_check:0", True),
        crash_case("place", "place:8", True),
        crash_case("recovery", "recover:6", False),
        crash_case("place, needle lost while down", "place:8", True, lost_while_down=True),
        crash_case("reach, controller restarted too", "reach:6", True, controller_restarted=True),
    ]
    out["b_crash_and_resume"] = {
        "how": "each process is ended with os._exit(137) at a chosen control tick; the arm's state is in a file "
               "outside the process, as a real arm's is",
        "cases": len(crashes),
        "marked_needs_reverification": sum(c["status_after_restart"] == "needs_reverification" for c in crashes),
        "resume_without_an_operator_refused": sum(c["resume_without_an_operator"] == "refused" for c in crashes),
        "arm_moved_before_the_operator_resumed": sum(c["arm_moved_before_the_operator_resumed"] for c in crashes),
        "looked_at_the_robot_first_after_resume": sum(c["observed_before_anything_else_after_resume"] for c in crashes),
        "finished_after_resume": sum(c["outcome"] is not None for c in crashes),
        "arm_at_start_pose_at_end": sum(c["arm_at_end"]["mm_from_start"] <= 1.0 for c in crashes),
        "runs": crashes}

    # ---------------------------------------------------------------- c. watchdog
    twin = fresh()
    gate = {"withhold": False}
    env = env_for(twin, feedback_gate=lambda n: not (gate["withhold"] and n > 5))

    def gate_events(event: dict[str, Any], state: dict[str, Any]) -> None:
        if event["type"] == "node_started":
            gate["withhold"] = event["data"]["node"] == "reach"       # INJECTED: only during reach

    env.on_event = gate_events
    res, runner, _ = go(env, trial.make_sheet(site, op.user_id, pick(0)), True)
    approve(res)
    res = runner.resume(res.run_id)
    evs = events_of(res.run_id)
    dog = next((e for e in evs if e["type"] == "watchdog"), None)
    act = next((e["data"] for e in evs if e["type"] == "action_result" and e["data"]["label"] == "reach"), {})
    row = digest(res.run_id, twin)
    row.update({"injected": "feedback of the reach motion is withheld after its fifth control step",
                "watchdog_fired": dog is not None, "watchdog_limit_s": act.get("watchdog_limit_s"),
                "feedback_silent_s_when_cancelled": act.get("feedback_silent_s"), "reach_action_status": act.get("status"),
                "control_steps_before_cancel": act.get("steps"),
                "planned_control_steps": res.state["data"]["dry_run"]["control_steps"]["reach"],
                "grasp_entered": any(e["type"] == "node_started" and e["data"]["node"] == "grasp" for e in evs)})
    out["c_watchdog"] = row
    log(f"c watchdog: fired {row['watchdog_fired']} after {row['feedback_silent_s_when_cancelled']} s, outcome {row['outcome']}")

    # ---------------------------------------------------------------- d. approvals bound to state
    stale: dict[str, Any] = {}

    def refusal(run_id: str) -> dict[str, Any] | None:
        return next((e["data"] for e in events_of(run_id) if e["type"] == "approval_refused"), None)

    def commands(run_id: str) -> int:
        return len(db.query("SELECT 1 FROM commands WHERE run_id=?", (run_id,)))

    # d1. The arm is moved by something else between the approval and its use.
    twin = fresh()
    env = env_for(twin)
    res, runner, _ = go(env, trial.make_sheet(site, op.user_id, pick(1)), True)
    approve(res)
    twin.perturb([0.002, 0.0, 0.0, 0.0, 0.0, 0.0])            # INJECTED
    res = runner.resume(res.run_id)
    ref = refusal(res.run_id)
    stale["arm_moved_after_approval"] = {
        "injected": "the arm is displaced by 0.002 rad of yaw after the operator approved",
        "approval_refused": ref is not None, "why": (ref or {}).get("why"),
        "approved_for_robot_seq": (ref or {}).get("approved_for_robot_seq"), "robot_seq_at_use": (ref or {}).get("robot_seq_now"),
        "status_after": res.status, "asked_again": res.status == "waiting", "motions_commanded": commands(res.run_id)}
    approve(res)
    res = runner.resume(res.run_id)
    stale["arm_moved_after_approval"].update({"after_a_new_approval": res.outcome,
                                              "arm_at_start_pose_at_end": digest(res.run_id, twin)["arm_at_start_pose_at_end"]})

    # d2. The envelope changes between the approval and its use, through an approved task-02 patch.
    twin = fresh()
    env = env_for(twin)
    res, runner, _ = go(env, trial.make_sheet(site, op.user_id, pick(2)), True)
    approve(res)
    site_path = s.sites_dir / "site_a.json"
    store = cfg.ConfigStore(work / "config", cfg.site_sources(site_path, s.dvrk_dir))
    target = "site/site_a.json"
    doc, _, _ = store.current(target)
    patch = ConfigPatch(target=target, limit_change="tighten", reason="The jaw touches the pad foam at 3 mm.",
                        evidence=["note:foam-contact"],
                        operations=[{"op": "replace", "path": "safety_profile/clearance_mm", "value": 4, "unit": "mm"}])
    v = validate_patch(patch, RuleContext(target=target, file_type="vx-site", doc=doc,
                                          schemas=cfg.SchemaSet(s.dvrk_dir / "share" / "schemas"),
                                          instruments=InstrumentSet(s.dvrk_dir / "share" / "tool"),
                                          evidence=frozenset(["note:foam-contact"])))
    pid = proposals.propose(db, principal("eng.kim", "service", site.name), site.name, patch, v, doc, "vx-site")
    proposals.decide(db, principal("saf.adler", "safety", site.name), pid, True)
    applied = proposals.apply(db, store, pid)
    new_doc, _, _ = store.current(target)
    env.site = build_site(dict(new_doc), site.chain)            # the site as the next process would load it
    res = runner.resume(res.run_id)
    evs = events_of(res.run_id)
    cut = next(i for i, e in enumerate(evs) if e["type"] == "approval_used" or e["type"] == "precondition_failed")
    stale["envelope_changed_after_approval"] = {
        "change": f"clearance over the pad 3 mm -> 4 mm, proposed by a service engineer and approved by the safety "
                  f"lead (proposal {pid}); envelope version {applied['envelope_version']}",
        "run_sheet_written_for_envelope": res.state["sheet"]["envelope_version"],
        "outcome": res.outcome, "stopped_by": evs[cut]["type"],
        "requirement_that_failed": [f["check"] for f in evs[cut]["data"].get("failed", [])],
        "at_node": evs[cut]["data"].get("node"),
        "approval_given_before_the_change_was_used": any(e["type"] == "approval_used" for e in evs),
        "operator_asked_again": any(e["type"] == "approval_requested" for e in evs[cut:]),
        "motions_commanded": commands(res.run_id), "path": [f"{h['node']}:{h['route']}" for h in res.state["history"]]}

    # d3. An approval that is used after its time limit.
    twin = fresh()
    env = env_for(twin, approval_ttl_s=0.5)
    res, runner, _ = go(env, trial.make_sheet(site, op.user_id, pick(3)), True)
    approve(res)
    time.sleep(0.8)
    res = runner.resume(res.run_id)
    ref = refusal(res.run_id)
    stale["approval_older_than_its_limit"] = {"limit_s": 0.5, "used_after_s": 0.8, "approval_refused": ref is not None,
                                              "why": (ref or {}).get("why"), "asked_again": res.status == "waiting",
                                              "motions_commanded": commands(res.run_id)}
    out["d_stale_approvals"] = stale
    log("d stale approvals: " + ", ".join(f"{k}: {v.get('why') or v.get('stopped_by')}" for k, v in stale.items()))

    # ---------------------------------------------------------------- e. stops, rejections, limits, faults
    e_cases: dict[str, Any] = {}

    # e1. The operator's stop during the descent. Recovery was agreed before the run, but a stop is not covered.
    twin = fresh()
    env = env_for(twin)
    count = {"n": 0, "armed": False}

    def stop_tick(dt: float) -> None:
        if count["armed"]:
            count["n"] += 1
            if count["n"] == 8:
                env.stop_event.set()                       # INJECTED: the operator presses stop
        time.sleep(dt)

    env.tick = stop_tick
    env.on_event = lambda ev, st: count.update(armed=True) if ev["type"] == "node_started" and ev["data"]["node"] == "reach" else None
    res, runner, _ = go(env, trial.make_sheet(site, op.user_id, pick(4), recovery_preapproved=True), True)
    approve(res)
    res = runner.resume(res.run_id)
    stopped_at = twin.state()
    held_position = {"mm_from_start": round(pose_distance(stopped_at.measured_cp, site.poses["start"].T)[0] * 1000, 2),
                     "height_over_pad_mm": round(site.scene.height(stopped_at.measured_cp[:3, 3]) * 1000, 2)}
    waiting_for = dict(res.waiting_for or {})
    time.sleep(0.5)
    moved_while_waiting = twin.state().seq != stopped_at.seq
    approve(res)
    res = runner.resume(res.run_id)
    row = digest(res.run_id, twin)
    row.update({"injected": "the operator's stop is set on the eighth control tick of the reach",
                "arm_stopped_at": held_position, "waited_for": waiting_for.get("node") or waiting_for.get("kind"),
                "recovery_was_preapproved_for_the_run": True, "recovery_still_needed_an_approval": res.state["data"]["recovery"]["preapproved"] is False,
                "arm_moved_while_waiting_for_the_operator": moved_while_waiting})
    e_cases["operator_stop_during_reach"] = row

    # e2. The operator says no at the go decision.
    twin = fresh()
    env = env_for(twin)
    res, runner, _ = go(env, trial.make_sheet(site, op.user_id, pick(5)), True)
    approve(res, yes=False)
    res = runner.resume(res.run_id)
    row = digest(res.run_id, twin)
    row["motions_commanded"] = commands(res.run_id)
    e_cases["operator_rejects_the_go"] = row

    # e3. Nobody answers within the time limit.
    twin = fresh()
    env = env_for(twin, approval_wait_s=0.5)
    res, runner, _ = go(env, trial.make_sheet(site, op.user_id, pick(5)), True)
    time.sleep(0.8)
    res = runner.resume(res.run_id)
    row = digest(res.run_id, twin)
    row.update({"wait_limit_s": 0.5, "motions_commanded": commands(res.run_id)})
    e_cases["nobody_answers_the_go"] = row

    # e4. A step limit that runs out in the middle of the trial.
    twin = fresh()
    env = env_for(twin)
    lim = Limits(max_steps=5)
    res, runner, _ = go(env, trial.make_sheet(site, op.user_id, pick(6)), True, lim)
    approve(res)
    res = runner.resume(res.run_id)
    row = digest(res.run_id, twin)
    row.update({"limit": "5 node entries", "steps_used": res.state["usage"]["steps"]})
    e_cases["step_limit_mid_trial"] = row

    # e5. A limit on tip travel that runs out during the trial.
    twin = fresh()
    env = env_for(twin)
    lim = Limits(max_path_mm=30.0)
    res, runner, _ = go(env, trial.make_sheet(site, op.user_id, pick(7)), True, lim)
    approve(res)
    res = runner.resume(res.run_id)
    row = digest(res.run_id, twin)
    row.update({"limit": "30 mm of tip travel", "checked": "before each node, so the motion that crosses the limit finishes"})
    e_cases["path_length_limit"] = row

    # e6. The arm accepts set points but does not follow, during the reach only.
    twin = fresh()
    env = env_for(twin)

    def follow_events(ev: dict[str, Any], st: dict[str, Any]) -> None:
        if ev["type"] == "node_started":
            twin.inject_dropped_commands(ev["data"]["node"] == "reach")     # INJECTED
    env.on_event = follow_events
    res, runner, _ = go(env, trial.make_sheet(site, op.user_id, pick(8)), True)
    approve(res)
    res = runner.resume(res.run_id)
    row = digest(res.run_id, twin)
    row.update({"injected": "set points are accepted but the arm does not move, during the reach",
                "reach": res.state["data"].get("reach")})
    e_cases["arm_does_not_follow_during_reach"] = row

    # e7. The robot reports a fault during the place motion. Recovery cannot run on a faulted arm.
    twin = fresh()
    env = env_for(twin)
    count2 = {"n": 0, "armed": False}

    def fault_tick(dt: float) -> None:
        if count2["armed"]:
            count2["n"] += 1
            if count2["n"] == 6:
                twin.inject_fault("amplifier fault (injected)")            # INJECTED
        time.sleep(dt)
    env.tick = fault_tick
    env.on_event = lambda ev, st: count2.update(armed=True) if ev["type"] == "node_started" and ev["data"]["node"] == "place" else None
    res, runner, _ = go(env, trial.make_sheet(site, op.user_id, pick(9)), True)
    approve(res)
    res = runner.resume(res.run_id)
    row = digest(res.run_id, twin)
    row.update({"injected": "the arm reports a fault on the sixth control tick of the place motion",
                "motions_commanded_after_the_fault": sum(1 for e in events_of(res.run_id)[
                    next(i for i, e in enumerate(events_of(res.run_id)) if e["type"] == "robot_fault"):]
                    if e["type"] == "action_started"),
                "fault_at_end": twin.state().fault})
    e_cases["robot_fault_during_place"] = row

    # e8. Calibration invalid at pre-flight (as after an applied base-frame change). The model explains.
    twin = fresh()
    env = env_for(twin, use_llm=True)
    proposals.set_calibration(db, site.name, False, "base frame of PSM1 changed by an applied configuration change")
    res, runner, _ = go(env, trial.make_sheet(site, op.user_id, pick(10)), True)
    proposals.set_calibration(db, site.name, True, "calibrated again (verification)")
    row = digest(res.run_id, twin)
    row.update({"failed_checks": [f["check"] for f in res.state["data"]["preflight"]["failed"]],
                "motions_commanded": commands(res.run_id), "explanation": res.state["data"].get("explanation"),
                "explanation_source": res.state["data"].get("explanation_source")})
    e_cases["calibration_invalid_at_preflight"] = row

    # e9. A path whose third waypoint is 1 mm over the pad. INJECTED path, not a recording.
    twin = fresh()
    env = env_for(twin)
    sheet = trial.make_sheet(site, op.user_id, None)
    run_id = f"trial_dryrun_{int(time.time())}"
    trial.prepare_path(db, site, sheet, run_id)
    low = site.poses["pregrasp"].T.copy()
    low[:3, 3] -= site.scene.surface_normal * 0.007
    db.execute("INSERT OR REPLACE INTO kv(k,v) VALUES (?,?)",
               (f"path:{run_id}", json.dumps({"info": {"source": "injected"}, "waypoints": [
                   site.poses["hover_needle"].T.tolist(), low.tolist(), site.poses["pregrasp"].T.tolist()]})))
    trial.declare_outcome(db, run_id, True, op.user_id)
    res = Runner(trial.build_graph(), env).start(run_id, sheet.model_dump(), op)
    all_runs.append(run_id)
    row = digest(run_id, twin)
    row.update({"injected": "an approach path with a waypoint 1 mm over the pad (clearance is 3 mm)",
                "rejections": res.state["data"]["dry_run"]["rejections"], "motions_commanded": commands(run_id)})
    e_cases["dry_run_rejects_the_path"] = row
    out["e_stops_limits_faults"] = e_cases
    for k, r in e_cases.items():
        log(f"e {k}: {r['outcome']} path {'>'.join(h.split(':')[0] for h in r['path'])}")

    # ---------------------------------------------------------------- f. state from the event log
    checked, equal, final_equal = 0, 0, 0
    for run_id in all_runs:
        evs = events_of(run_id)
        for cp in db.query("SELECT * FROM checkpoints WHERE run_id=? ORDER BY n", (run_id,)):
            checked += 1
            equal += int(json.loads(cp["state"]) == rebuild(evs[:cp["events_n"]]))
        final_equal += int(Runner.load(db, run_id) == rebuild(evs))
    out["f_state_from_events"] = {"runs": len(all_runs), "checkpoints_compared": checked,
                                  "checkpoints_equal_to_the_state_rebuilt_from_events": equal,
                                  "runs_whose_loaded_state_equals_the_rebuilt_state": final_equal,
                                  "events_in_all_runs": sum(len(events_of(r)) for r in all_runs)}
    log(f"f rebuild: {equal}/{checked} checkpoints equal, {final_equal}/{len(all_runs)} runs")
    return out
