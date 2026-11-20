"""Verification of task 05 (runbooks as skills).

    python scripts/verify.py t05     ->  verification/t05_tools_vs_skills.json

The same service requests three ways on the twin: atomic tools with the procedures written in
the prompt, runbooks run by code, and runbooks found through find_tools. The needle is
DECLARED per run (the twin has none).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knowledge_service import start_knowledge                              # noqa: E402
from vxagent.agent.guards import grounding_check                           # noqa: E402
from vxagent.agent.loop import AgentConfig, run_agent                      # noqa: E402
from vxagent.config import get_settings                                    # noqa: E402
from vxagent.diagnostics.episode import calibration                        # noqa: E402
from vxagent.diagnostics.tools import calibration_residual                 # noqa: E402
from vxagent.llm.ollama import get_llm                                     # noqa: E402
from vxagent.profiles import DEMO, principal                               # noqa: E402
from vxagent.robot.kinematics import pose_distance                         # noqa: E402
from vxagent.robot.needle import DeclaredNeedle                            # noqa: E402
from vxagent.robot.site import load_site                                   # noqa: E402
from vxagent.runtime.trace import Tracer                                   # noqa: E402
from vxagent.skills.index import default_library, register_bench, register_skills   # noqa: E402
from vxagent.store import Database                                         # noqa: E402
from vxagent.structured import proposals                                   # noqa: E402
from vxagent.tools.builtin import register_builtin                         # noqa: E402
from vxagent.tools.executor import decide_approval                         # noqa: E402
from vxagent.tools.knowledge import register_knowledge                     # noqa: E402
from vxagent.tools.registry import ToolContext                             # noqa: E402
from vxagent.tools.service import register_service, store_calibration_record   # noqa: E402

SMOKE = bool(os.environ.get("VX_SMOKE"))
VERDICT = re.compile(r"VERDICT:\s*([a-z_ ]+)", re.I)
ATOMIC = ["get_robot_state", "list_named_poses", "check_named_target", "rehearse_move", "enable_and_home",
          "move_to_named_pose", "set_jaw", "wait_action", "cancel_action", "list_episodes", "episode_summary",
          "read_signal_window", "calculator", "read_calibration_record", "calibration_residual", "observe_needle", "hold",
          "search_docs", "search_threads", "search_by_log_line"]
RULES = ("Rules:\n1. Everything you say about the robot must come from a tool result in this conversation.\n"
         "2. Before any motion call get_robot_state and pass its state_seq as based_on_seq. One motion per response.\n"
         "3. If a check of a procedure fails, do what the procedure says for that case and report which check failed.\n"
         "4. Never skip a step or a check of a procedure. Never invent a result.\n"
         "5. End with one line: VERDICT: <one of the words the request asks for>.")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2] if len(xs) % 2 else round((xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2, 1)


def subsequence(needed: list[str], seen: list[str]) -> bool:
    it = iter(seen)
    return all(any(x == n for x in it) for n in needed)


def t05() -> dict[str, Any]:
    s = get_settings()
    site = load_site("site_a")
    llm = get_llm()
    db = Database()
    tracer = Tracer(db)
    who = principal("op.verify", "operator", site.name)
    client, server = start_knowledge(site)
    registry = register_knowledge(register_bench(register_service(register_builtin())))
    library = default_library(registry)
    catalogue = register_skills(registry, library, ATOMIC)
    cal = calibration(None, site.scene.workspace_center, np.random.default_rng(11))
    record = {"id": cal["id"], "T_est": cal["T_est"].tolist(), "seen_m": cal["seen_m"].tolist(),
              "touched_m": cal["touched_m"].tolist(), "method": "point pairs, camera to robot base"}
    store_calibration_record(db, site.name, record)
    proposals.set_calibration(db, site.name, True, "calibrated (verification)")
    residual = calibration_residual({"calibration": record})["residual_mm"]["mean"]

    index_text = "\n".join(f"- {e['name']}: {e['summary']} Use it: {e['when_to_use']}" for e in library.index())
    procedures = "\n\n".join(rb.prose() for rb in library.books.values())
    head = "You assist the operator of a da Vinci Research Kit (dVRK) arm. " + DEMO + "\n"
    arms = {
        "atomic": AgentConfig(name="atomic", max_iterations=24, token_budget=120000, answer_check=grounding_check,
                              tool_names=ATOMIC, system_prompt=head + RULES + "\n\nThe lab's procedures. Follow the one "
                              "that fits the request, step by step, with the tools:\n" + procedures),
        "runbooks": AgentConfig(name="runbooks", max_iterations=8, token_budget=60000, answer_check=grounding_check,
                                tool_names=["run_runbook", "load_runbook", "get_robot_state"],
                                system_prompt=head + RULES + "\n\nThe lab's procedures are runbooks. Choose the one that "
                                "fits the request and run it with run_runbook; it returns every step and check. "
                                "Runbooks:\n" + index_text),
        "runbooks_with_search": AgentConfig(name="runbooks_with_search", max_iterations=8, token_budget=60000,
                                            answer_check=grounding_check, tool_names=["find_tools"], dynamic_tools=True,
                                            system_prompt=head + RULES + "\n\nYou start with one tool, find_tools. "
                                            "Search for what the request needs; the matches become available on your "
                                            "next turn. Prefer a runbook when one fits."),
    }

    requests: list[dict[str, Any]] = [
        {"id": "bring up", "text": "The controller has just been switched on. Bring the arm up. End with VERDICT: ready or "
         "VERDICT: not ready.", "bench": {"ready": False}, "runbook": "bring_up", "verdict": "ready", "pose": "start",
         "needs": ["get_robot_state", "enable_and_home", "check_named_target", "move_to_named_pose"]},
        {"id": "rig check", "text": "Check the rig before a trial. End with VERDICT: ok or VERDICT: not ok.", "bench": {},
         "runbook": "rig_check", "verdict": "ok", "pose": "start",
         "needs": ["get_robot_state", "check_named_target", "check_named_target", "check_named_target",
                   "read_calibration_record", "calibration_residual", "observe_needle"]},
        {"id": "calibration", "text": "Is the calibration still valid? End with VERDICT: valid or VERDICT: not valid.",
         "bench": {}, "runbook": "calibration_validity", "verdict": "valid", "pose": "start",
         "needs": ["read_calibration_record", "calibration_residual"]},
        {"id": "calibration, invalidated", "text": "Is the calibration still valid? End with VERDICT: valid or VERDICT: "
         "not valid.", "bench": {"calibration_valid": False}, "runbook": "calibration_validity", "verdict": "not valid",
         "pose": "start", "needs": ["read_calibration_record"]},
        {"id": "retention, needle holds", "text": "The open jaw is at the grasp pose over the needle. Verify that the "
         "needle is held. End with VERDICT: retained or VERDICT: not retained.", "bench": {"at": "pregrasp"},
         "runbook": "verify_grasp_retention", "verdict": "retained", "pose": "hover_needle", "jaw": "closed",
         "needs": ["observe_needle", "set_jaw", "observe_needle", "move_to_named_pose", "observe_needle", "hold", "observe_needle"]},
        {"id": "retention, slips on the lift", "text": "The open jaw is at the grasp pose over the needle. Verify that "
         "the needle is held. End with VERDICT: retained or VERDICT: not retained.",
         "bench": {"at": "pregrasp", "needle": {"slips_on_lift": True}}, "runbook": "verify_grasp_retention",
         "verdict": "not retained", "pose": "start", "jaw": "open",
         "needs": ["observe_needle", "set_jaw", "observe_needle", "move_to_named_pose", "observe_needle", "set_jaw", "move_to_named_pose"]},
        {"id": "retention, slips while held", "text": "The open jaw is at the grasp pose over the needle. Verify that "
         "the needle is held. End with VERDICT: retained or VERDICT: not retained.",
         "bench": {"at": "pregrasp", "needle": {"slips_while_holding": True}}, "runbook": "verify_grasp_retention",
         "verdict": "not retained", "pose": "start", "jaw": "open",
         "needs": ["observe_needle", "set_jaw", "observe_needle", "move_to_named_pose", "observe_needle", "hold",
                   "observe_needle", "set_jaw", "move_to_named_pose"]},
        {"id": "return to start", "text": "Put the arm back where a trial begins. End with VERDICT: at start or VERDICT: "
         "not at start.", "bench": {"at": "left_edge", "jaw": "closed"}, "runbook": "return_to_start",
         "verdict": "at start", "pose": "start", "jaw": "open", "needs": ["set_jaw", "move_to_named_pose"]},
    ]
    if SMOKE:
        requests = requests[:2]

    def bench(ready: bool = True, at: str | None = None, jaw: str | None = None, needle: dict[str, bool] | None = None,
              calibration_valid: bool = True) -> tuple[ToolContext, Any]:
        twin = site.twin()
        if ready:
            twin.enable()
            twin.home()
        if at:
            twin.set_joints(site.poses[at].q)
        if jaw:
            twin._jaw = site.jaw_angles[jaw]
        ctx = ToolContext(principal=who, db=db, tracer=tracer, robot=twin, site=site, run_id=f"run_{uuid.uuid4().hex[:12]}")
        ctx.extras["needle"] = DeclaredNeedle(site, **(needle or {})).attach(twin)
        ctx.extras["knowledge"] = client
        proposals.set_calibration(db, site.name, calibration_valid, "set for this run (verification)")
        return ctx, twin

    def one(arm: str, req: dict[str, Any]) -> dict[str, Any]:
        cfg = arms[arm]
        ctx, twin = bench(**req["bench"])
        tracer.start_run(cfg.name, req["text"], who.user_id, who.role, site.name, run_id=ctx.run_id)
        t0 = time.time()
        state = run_agent(llm, registry, ctx, req["text"], cfg)
        approvals = 0
        while state.status == "awaiting_approval" and approvals < 12:
            decide_approval(db, state.pending_approval["approval_id"], True, "operator.verify")
            approvals += 1
            state = run_agent(llm, registry, ctx, config=cfg, state=state)
        tracer.end_run(ctx.run_id, state.status, state.answer)
        end = twin.state()
        model_tools = [t["tool"] for t in state.tool_log]
        ran = [r["runbook"] for r in ctx.extras.get("runbook_runs", [])]
        inner = [r["name"] for r in db.query("SELECT name FROM spans WHERE run_id=? AND kind='tool' ORDER BY id", (ctx.run_id,))]
        said = VERDICT.findall(state.answer or "")
        verdict = said[-1].strip().lower() if said else None
        pose_ok = pose_distance(end.measured_cp, site.poses[req["pose"]].T)[0] < 1e-4
        jaw_ok = "jaw" not in req or abs(end.jaw - site.jaw_angles[req["jaw"]]) < 1e-6
        if arm == "atomic":
            checks_done = subsequence(req["needs"], [t for t, ok in zip(model_tools, [x["ok"] for x in state.tool_log]) if ok])
        else:
            checks_done = req["runbook"] in ran and all(r["status"] != "not_started" for r in ctx.extras.get("runbook_runs", []))
        row = {"arm": arm, "request": req["id"], "status": state.status, "stop_reason": state.stop_reason,
               "completed": state.status == "completed", "every_required_check_performed": checks_done,
               "verdict": verdict, "verdict_expected": req["verdict"], "verdict_correct": verdict == req["verdict"],
               "final_state_correct": bool(pose_ok and jaw_ok and not end.busy),
               "model_calls": state.iterations, "tool_calls_by_the_model": len(model_tools),
               "tool_calls_executed": len(inner), "tokens": state.total_tokens, "seconds": round(time.time() - t0, 1),
               "tool_errors": sum(1 for t in state.tool_log if not t["ok"]), "approvals_given": approvals,
               "schema_chars_in_prompt": state.schema_chars, "system_prompt_chars": len(cfg.system_prompt),
               "tools_the_model_called": model_tools, "runbooks_run": ran,
               "needle_declared": ctx.extras["needle"].declared(), "needle_log": ctx.extras["needle"].log,
               "answer": state.answer}
        if arm == "runbooks_with_search":
            searches = ctx.extras.get("searches", [])
            row["searches"] = searches
            row["search_returned_the_right_runbook"] = any(req["runbook"] in sr["returned"] for sr in searches)
        row["all_good"] = bool(row["completed"] and checks_done and row["verdict_correct"] and row["final_state_correct"])
        log(f"{arm} | {req['id']}: good {row['all_good']} checks {checks_done} verdict {verdict} state {row['final_state_correct']} "
            f"calls {state.iterations}/{len(model_tools)} ({row['seconds']} s)")
        return row

    runs = [one(arm, req) for arm in arms for req in requests]
    by_arm = {}
    for arm in arms:
        mine = [r for r in runs if r["arm"] == arm]
        by_arm[arm] = {"requests": len(mine), "completed": sum(r["completed"] for r in mine),
                       "every_required_check_performed": sum(r["every_required_check_performed"] for r in mine),
                       "verdict_correct": sum(r["verdict_correct"] for r in mine),
                       "final_state_correct": sum(r["final_state_correct"] for r in mine),
                       "all_four": sum(r["all_good"] for r in mine),
                       "median_model_calls": median([r["model_calls"] for r in mine]),
                       "median_tool_calls_by_the_model": median([r["tool_calls_by_the_model"] for r in mine]),
                       "median_tokens": median([r["tokens"] for r in mine]), "median_seconds": median([r["seconds"] for r in mine]),
                       "tool_errors": sum(r["tool_errors"] for r in mine),
                       "schema_chars_in_prompt": max(r["schema_chars_in_prompt"] for r in mine),
                       "system_prompt_chars": mine[0]["system_prompt_chars"]}
        if arm == "runbooks_with_search":
            by_arm[arm]["search_returned_the_right_runbook"] = sum(r["search_returned_the_right_runbook"] for r in mine)

    # Selection: one turn, which tool does the model reach for first, with 10 tools offered and with all.
    picks = [("Where is the arm right now?", {"get_robot_state"}), ("Which poses does this site have names for?", {"list_named_poses"}),
             ("Would the safety filter accept the pose pregrasp?", {"check_named_target"}),
             ("Rehearse a move to left_edge without moving anything.", {"rehearse_move"}),
             ("Which recordings are available here?", {"list_episodes"}),
             ("How long is the recording X01_Pea_on_a_Peg_01?", {"episode_summary"}),
             ("What is 17.5 times 3.2?", {"calculator"}), ("What is the id of the calibration record?", {"read_calibration_record"}),
             ("Is the needle in the jaw right now?", {"observe_needle"}), ("Keep the arm still for three seconds.", {"hold"}),
             ("What does the documentation say about homing a PSM?", {"search_docs"}),
             ("Has anyone reported a problem with the jaw not closing?", {"search_threads"})]
    if SMOKE:
        picks = picks[:3]
    small = ["get_robot_state", "list_named_poses", "check_named_target", "rehearse_move", "list_episodes",
             "episode_summary", "calculator", "read_calibration_record", "observe_needle", "hold", "search_docs",
             "search_threads"]
    wide = ATOMIC + ["find_tools", "load_runbook", "run_runbook"]
    sel = []
    for size, names in (("12 tools", small), (f"{len(wide)} tools", wide)):
        schemas = registry.schemas(who, names)
        right = 0
        detail = []
        for q, want in picks:
            resp = llm.chat([{"role": "system", "content": head + "Call the one tool that answers the request."},
                             {"role": "user", "content": q}], tools=schemas, temperature=0.0)
            got = resp.tool_calls[0].name if resp.tool_calls else None
            right += int(got in want)
            detail.append({"request": q, "expected": sorted(want), "called": got})
        sel.append({"catalogue": size, "requests": len(picks), "first_call_is_the_right_tool": right,
                    "schema_chars": len(json.dumps(schemas)), "wrong": [d for d in detail if d["called"] not in d["expected"]]})
        log(f"selection with {size}: {right}/{len(picks)}")
    hits = []
    for req in requests:
        found = [h["name"] for h in catalogue.search(req["text"].split(" End with")[0])]
        hits.append({"request": req["id"], "runbook": req["runbook"], "returned": found, "hit": req["runbook"] in found})
    server.stop()

    cited = sorted({st.source for rb in library.books.values() for st in rb.steps})
    doc_ids = [c for c in cited if c.startswith("doc:")]
    try:
        from vxrag.store import CorpusStore
        store = CorpusStore()
        known = {u.unit_id for u in store.units()}
        found_docs = [d for d in doc_ids if d in known]
    except Exception as exc:                                   # the corpus store is not there: say so, do not guess
        found_docs, known = [], None
        log(f"could not open the corpus store: {exc}")
    return {"task": "t05_tools_vs_skills" + ("_smoke" if SMOKE else ""), "ran": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": s.model, "site": site.name, "backend": "twin", "needle": "DECLARED per run: the twin has no needle",
            "runbooks": [{"name": rb.name, "version": rb.version, "content_hash": rb.content_hash, "steps": len(rb.steps),
                          "checks": sum(len(st.checks) for st in rb.steps), "tier": rb.tier} for rb in library.books.values()],
            "sources_cited_by_steps": {"documentation_sections": doc_ids,
                                       "found_in_the_corpus": None if known is None else f"{len(found_docs)}/{len(doc_ids)}",
                                       "not_checkable": [c for c in cited if not c.startswith("doc:")]},
            "calibration_residual_computed_by_code_mm": residual,
            "atomic_catalogue_size": len(ATOMIC), "requests": len(requests), "by_arm": by_arm,
            "selection_of_a_single_tool": sel,
            "search_without_a_model": {"requests": len(hits), "right_runbook_in_the_results": sum(h["hit"] for h in hits), "detail": hits},
            "runs": runs}
