"""Verification of task 04 (diagnostic reasoning and planning).

    python scripts/verify.py t04     ->  verification/t04_reasoning_and_planning.json

Part A: five reasoning strategies on runs of the twin with one INJECTED fault each, and four of
them on closed dVRK issue threads whose cause a maintainer stated (hand-labelled).
Part B: service goals through the planner and executor on the twin.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knowledge_service import start_knowledge                              # noqa: E402
from vxagent.agent.planner import run_goal                                 # noqa: E402
from vxagent.agent.strategies import STRATEGIES, diagnose                  # noqa: E402
from vxagent.config import get_settings                                    # noqa: E402
from vxagent.data import episodes as ep_store                              # noqa: E402
from vxagent.diagnostics.episode import CAUSES, make_contract, make_fault, run_episode   # noqa: E402
from vxagent.diagnostics.tools import CHECKS, DESCRIPTIONS, calibration_residual           # noqa: E402
from vxagent.llm.ollama import get_llm                                     # noqa: E402
from vxagent.pipelines import paths                                        # noqa: E402
from vxagent.profiles import principal                                     # noqa: E402
from vxagent.robot.kinematics import pose_distance                         # noqa: E402
from vxagent.robot.site import load_site                                   # noqa: E402
from vxagent.runtime.trace import Tracer                                   # noqa: E402
from vxagent.store import Database                                         # noqa: E402
from vxagent.structured import proposals                                   # noqa: E402
from vxagent.tools.builtin import register_builtin                         # noqa: E402
from vxagent.tools.registry import ToolContext                             # noqa: E402
from vxagent.tools.service import register_service, store_calibration_record   # noqa: E402

LABELS = ROOT / "evals" / "t04_thread_labels.json"
SMOKE = bool(os.environ.get("VX_SMOKE"))      # a short run to check the script itself; never the evidence
PER_CAUSE = 1 if SMOKE else 3
# The check in which each fault leaves its mark (see tests/test_reasoning.py for the proof on a made-up arm).
SIGNATURE = {"calibration_offset": "calibration_residual", "calibration_axis_flip": "calibration_residual",
             "action_unit_scale": "policy_ranges", "start_outside_training_range": "policy_ranges",
             "joint_limit_tightened": "safety_rejections", "tracking_error": "compare_poses",
             "perception_dropout": "perception_quality"}
PLAN_TOOLS = ["get_robot_state", "list_named_poses", "check_named_target", "enable_and_home", "move_to_named_pose",
              "set_jaw", "read_calibration_record", "calibration_residual"]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2] if len(xs) % 2 else round((xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2, 1)


def tally(rows: list[dict[str, Any]], strategies: tuple[str, ...]) -> dict[str, Any]:
    out = {}
    for s in strategies:
        mine = [r["by_strategy"][s] for r in rows if s in r["by_strategy"]]
        out[s] = {"cases": len(mine), "cause_correct": sum(m["correct"] for m in mine),
                  "family_correct": sum(m.get("family_correct", m["correct"]) for m in mine),
                  "errors": sum(m["error"] is not None for m in mine),
                  "median_calls": median([m["calls"] for m in mine]), "median_tokens": median([m["tokens"] for m in mine]),
                  "median_seconds": median([round(m["latency_ms"] / 1000, 1) for m in mine]),
                  "median_checks": median([len(m["checks"]) for m in mine])}
    return out


def t04() -> dict[str, Any]:
    s = get_settings()
    site = load_site("site_a")
    llm = get_llm()
    out: dict[str, Any] = {"task": "t04_reasoning_and_planning" + ("_smoke" if SMOKE else ""), "ran": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "model": s.model, "site": site.name, "backend": "twin"}
    client, server = start_knowledge(site)

    def retrieve(query: str) -> list[dict[str, Any]]:
        try:
            body = client.post("/v1/search", {"query": query[:400], "k": 3, "sources": ["doc"]})
            return [{"id": r["id"], "text": r.get("text", "")} for r in body.get("results", [])]
        except Exception:                                  # the critique then runs without documentation
            return []

    # ---------------------------------------------------------------- A1. injected faults
    stretches = []
    for e in ep_store.list_episodes():
        t, p = paths.positions(e)
        seg = paths.find_segment(t, p)
        if seg:
            stretches.append((e.episode_id, p[seg["first"]:seg["last"] + 1]))
    contract = make_contract(site, [p for _, p in stretches])
    nominal = [run_episode(site, p, contract, None, i) for i, (_, p) in enumerate(stretches)]
    candidates = {k: v["what"] for k, v in CAUSES.items()}
    rng = np.random.default_rng(20261120)
    rows = []
    for ci, cause in enumerate(list(CAUSES)[:2] if SMOKE else CAUSES):
        for k in range(PER_CAUSE):
            eid, pts = stretches[(ci * PER_CAUSE + k) % len(stretches)]
            fault = make_fault(cause, rng)
            rec = run_episode(site, pts, contract, fault, seed=100 * ci + k)
            measured = {name: fn(rec) for name, fn in CHECKS.items()}
            row: dict[str, Any] = {"injected_cause": cause, "family": CAUSES[cause]["family"], "fault": fault,
                                   "recording": eid, "outcome": rec["outcome"], "symptom": rec["symptom"],
                                   "measurements": measured, "by_strategy": {}}
            for strat in STRATEGIES + ("eliminate_all_checks",):
                d = diagnose("eliminate" if strat == "eliminate_all_checks" else strat, llm, rec["symptom"], candidates,
                             checks=DESCRIPTIONS, run_check=lambda name, rec=rec: CHECKS[name](rec), retrieve=retrieve,
                             budget=len(CHECKS) if strat == "eliminate_all_checks" else 3)
                r = d.as_dict()
                r["correct"] = d.cause == cause
                r["family_correct"] = d.cause is not None and CAUSES[d.cause]["family"] == CAUSES[cause]["family"]
                row["by_strategy"][strat] = r
            rows.append(row)
            log(f"A {cause} #{k}: " + ", ".join(f"{st}={'Y' if row['by_strategy'][st]['correct'] else 'n'}" for st in row["by_strategy"])
                + f" (eliminate checks {row['by_strategy']['eliminate']['checks']})")
    symptoms: dict[str, set[str]] = {}
    for r in rows:
        symptoms.setdefault(r["outcome"], set()).add(r["injected_cause"])
    elim = [r["by_strategy"]["eliminate"] for r in rows]
    out["a_injected_faults"] = {
        "how": "one fault per run, INJECTED into a deployment pipeline on the twin that replays a stretch of recorded "
               "dVRK motion; the symptom text is written by code from what an operator could see",
        "cases": len(rows), "causes": len(CAUSES), "runs_per_cause": PER_CAUSE, "check_budget_of_eliminate": 3,
        "fault_free_runs_that_reach_the_needle": f"{sum(n['outcome'] == 'completed' and n['tip_to_grasp_point_mm'] < 1 for n in nominal)}/{len(nominal)}",
        "causes_behind_each_kind_of_ending": {k: sorted(v) for k, v in symptoms.items()},
        "chance_of_a_guess": round(1 / len(CAUSES), 3),
        "by_strategy": tally(rows, STRATEGIES + ("eliminate_all_checks",)),
        "eliminate_all_checks": {
            "what": "the same strategy allowed to run all five checks, so that choosing a check and reading its result can be told apart",
            "narrowed_to_one": sum(r["by_strategy"]["eliminate_all_checks"]["detail"].get("narrowed_to_one", False) for r in rows),
            "states_rejected_by_the_rules": sum(r["by_strategy"]["eliminate_all_checks"]["detail"].get("states_rejected_by_the_rules", 0) for r in rows),
            "correct_by_cause": {c: f"{sum(r['by_strategy']['eliminate_all_checks']['correct'] for r in rows if r['injected_cause'] == c)}/{PER_CAUSE}"
                                 for c in CAUSES if any(r["injected_cause"] == c for r in rows)}},
        "eliminate": {"narrowed_to_one": sum(e["detail"].get("narrowed_to_one", False) for e in elim),
                      "states_rejected_by_the_rules": sum(e["detail"].get("states_rejected_by_the_rules", 0) for e in elim),
                      "first_check_chosen": {t: sum(1 for e in elim if e["checks"][:1] == [t]) for t in CHECKS},
                      "the_signature_check_was_among_those_run": sum(SIGNATURE[r["injected_cause"]] in r["by_strategy"]["eliminate"]["checks"] for r in rows),
                      "correct_when_it_was": sum(r["by_strategy"]["eliminate"]["correct"] for r in rows
                                                 if SIGNATURE[r["injected_cause"]] in r["by_strategy"]["eliminate"]["checks"]),
                      "correct_by_cause": {c: f"{sum(r['by_strategy']['eliminate']['correct'] for r in rows if r['injected_cause'] == c)}/{PER_CAUSE}"
                                           for c in CAUSES if any(r["injected_cause"] == c for r in rows)}},
        "reflect": {"answer_changed_after_review": sum(r["by_strategy"]["reflect"]["detail"].get("changed_after_review", False) for r in rows),
                    "changed_to_correct": sum(1 for r in rows if r["by_strategy"]["reflect"]["detail"].get("changed_after_review")
                                              and r["by_strategy"]["reflect"]["correct"]),
                    "changed_to_wrong": sum(1 for r in rows if r["by_strategy"]["reflect"]["detail"].get("changed_after_review")
                                            and r["by_strategy"]["reflect"]["detail"].get("first") == r["injected_cause"]),
                    "runs_with_passages": sum(bool(r["by_strategy"]["reflect"]["detail"].get("passages")) for r in rows)},
        "tot": {"all_three_lines_agree": sum(r["by_strategy"]["tot"]["detail"].get("lines_agree", False) for r in rows),
                "a_line_had_the_cause_but_another_was_picked": sum(
                    1 for r in rows if r["injected_cause"] in r["by_strategy"]["tot"]["detail"].get("lines", [])
                    and not r["by_strategy"]["tot"]["correct"])},
        "runs": rows}

    # ---------------------------------------------------------------- A2. labelled threads
    spec = json.loads(LABELS.read_text(encoding="utf-8"))
    threads = {}
    for f in (s.data_dir / "sources" / "issues").glob("*.json"):
        for t in json.loads(f.read_text(encoding="utf-8")):
            threads[(t["repo"].split("/")[1], int(t["number"]))] = t
    trows = []
    text_strategies = tuple(x for x in STRATEGIES if x != "eliminate")
    for repo, number, label, _why in (spec["threads"][:2] if SMOKE else spec["threads"]):
        t = threads.get((repo, number))
        if t is None:
            continue
        symptom = f"{t['title']}\n{(t['body'] or '')[:1500]}"
        row = {"thread": f"{repo}#{number}", "label": label, "by_strategy": {}}
        for strat in text_strategies:
            d = diagnose(strat, llm, symptom, spec["classes"], retrieve=retrieve)
            r = d.as_dict()
            r["correct"] = d.cause == label
            row["by_strategy"][strat] = r
        trows.append(row)
        log(f"A thread {repo}#{number} [{label}]: " + ", ".join(
            f"{st}={row['by_strategy'][st]['cause']}" for st in text_strategies))
    counts: dict[str, int] = {}
    for r in trows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    out["a_labelled_threads"] = {
        "threads": len(trows), "labels_file": str(LABELS.relative_to(ROOT)).replace("\\", "/"),
        "what_the_model_saw": "the title and the opening post, not the replies",
        "labels": counts, "always_the_most_common_label": f"{max(counts.values())}/{len(trows)}",
        "eliminate": "not run: a thread has no telemetry to check",
        "by_strategy": tally(trows, text_strategies),
        "reflect_answer_changed_after_review": sum(r["by_strategy"]["reflect"]["detail"].get("changed_after_review", False) for r in trows),
        "runs": trows}
    server.stop()

    # ---------------------------------------------------------------- B. planning
    db = Database()
    tracer = Tracer(db)
    registry = register_service(register_builtin())
    who = principal("op.verify", "operator", site.name)
    proposals.set_calibration(db, site.name, True, "calibrated (verification)")
    cal = run_episode(site, stretches[0][1], contract, None, 1)["calibration"]
    store_calibration_record(db, site.name, {**cal, "method": "point pairs, camera to robot base"})
    cal_truth = calibration_residual({"calibration": cal})["residual_mm"]

    def goal(label: str, text: str, ready: bool = True, prepare: Any = None, expect: str = "completed",
             expect_pose: str | None = None) -> dict[str, Any]:
        twin = site.twin()
        if ready:
            twin.enable()
            twin.home()
        ctx = ToolContext(principal=who, db=db, tracer=tracer, robot=twin, site=site, run_id=f"plan_{uuid.uuid4().hex[:10]}")
        if prepare:
            prepare(twin, ctx)
        tracer.start_run("planner", text, who.user_id, who.role, site.name, run_id=ctx.run_id)
        res = run_goal(llm, text, registry, ctx, PLAN_TOOLS, on_approval=lambda wait: True)
        tracer.end_run(ctx.run_id, res["status"], res["answer"])
        first = res["rounds"][0]
        last_plan = next((a.get("plan") for a in reversed(first["planning_attempts"]) if a.get("plan")), None)
        row = {"label": label, "goal": text, "status": res["status"], "expected_status": expect, "why": res.get("why"),
               "planning_attempts": [a["stage"] for r in res["rounds"] for a in r["planning_attempts"]],
               "planning_errors_first_attempt": first["planning_attempts"][0]["errors"],
               "plan": last_plan, "uses_a_reference": any(isinstance(v, str) and v.startswith("$")
                                                          for st in (last_plan or []) for v in st["arguments"].values()),
               "dry_run": [{k: r.get(k) for k in ("needed", "clean")} | {"failed": [x for x in r["steps"] if not x["ok"]]}
                           for r in (rd.get("dry_run") for rd in res["rounds"]) if r],
               "steps_executed": [(x["id"], x["tool"], x["ok"], x["error_kind"]) for rd in res["rounds"]
                                  for x in rd.get("execution", {}).get("steps", [])],
               "approvals_given": sum(rd.get("execution", {}).get("approvals", 0) for rd in res["rounds"]),
               "replans": res["replans"], "motions_executed": len(db.query(
                   "SELECT 1 FROM commands WHERE run_id=? AND status='done'", (ctx.run_id,))),
               "answer": res["answer"], "answer_source": res["answer_source"], "tokens": res["tokens"], "seconds": res["seconds"],
               "arm_busy_at_end": twin.state().busy}
        if expect_pose:
            row["final_error_to_expected_pose_mm"] = round(pose_distance(twin.state().measured_cp, site.poses[expect_pose].T)[0] * 1000, 3)
        row["as_expected"] = res["status"] == expect and (not expect_pose or row["final_error_to_expected_pose_mm"] < 0.1)
        log(f"B {label}: {res['status']} attempts {row['planning_attempts']} steps {[x[1] for x in row['steps_executed']]}")
        return row

    goals = [
        goal("bring up, move, report", "The arm has just been powered on. Bring it up, move it to hover_needle and "
             "report its state.", ready=False, expect_pose="hover_needle"),
        goal("second step needs the first step's output", "Read the site's calibration record and report its residual "
             "on the held-out points.", expect_pose="start"),
        goal("two motions asked for at once", "Move to left_edge and close the jaw.", expect_pose="left_edge"),
        goal("three moves in order", "Go to hover_needle, then pregrasp, then back to start.", expect_pose="start"),
        goal("impossible from the observed state", "Move the arm to hover_needle.", ready=False,
             prepare=lambda twin, ctx: twin.inject_power(False), expect="no_valid_plan", expect_pose="start"),
        goal("destination the safety filter rejects", "Withdraw the instrument to the retract pose.",
             expect="rejected_in_dry_run", expect_pose="start"),
    ]
    # A motion that fails once (INJECTED), so that one re-plan from the observed state is needed.
    twin_box: dict[str, Any] = {}

    def fail_once(twin: Any, ctx: ToolContext) -> None:
        twin.inject_dropped_commands(True)
        twin_box["twin"] = twin
        original = ctx.tracer.on_event

        def on_event(ev: dict[str, Any]) -> None:
            if ev["kind"] == "tool" and ev["name"] == "move_to_named_pose" and not ev["ok"]:
                twin.inject_dropped_commands(False)
            if original:
                original(ev)
        ctx.tracer = Tracer(db, on_event=on_event)
    goals.append(goal("a motion fails once (injected), one re-plan", "Move the arm to left_edge and report where it is.",
                      prepare=fail_once, expect_pose="left_edge"))
    cal_row = goals[1]
    cal_row["residual_computed_by_code_mm"] = cal_truth["mean"]
    cal_row["residual_in_the_answer"] = str(cal_truth["mean"]) in (cal_row["answer"] or "")
    out["b_planning"] = {"goals": len(goals), "as_expected": sum(g["as_expected"] for g in goals),
                         "plans_valid_on_first_attempt": sum(g["planning_attempts"][0] == "ok" for g in goals),
                         "tools_offered": PLAN_TOOLS, "runs": goals}
    return out
