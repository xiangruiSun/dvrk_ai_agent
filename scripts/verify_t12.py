"""Verification of task 12 (service engineer and safety lead) with the real model and the real
dVRK configuration files.

    python scripts/verify.py t12     ->  verification/t12_service_access.json

The camera is DECLARED and the camera move is INJECTED, so the calibration fit has a known
answer. That shows the procedure and its bookkeeping, not accuracy on hardware.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_bench import Bench, ask, log, route_matrix, strip, who             # noqa: E402
from vxagent import access                                                 # noqa: E402
from vxagent.config import get_settings                                    # noqa: E402
from vxagent.llm.base import ToolCall                                      # noqa: E402
from vxagent.memory import site as site_memory                             # noqa: E402
from vxagent.service import calibration as cal                             # noqa: E402
from vxagent.service import changes, diagnosis, report, tickets            # noqa: E402
from vxagent.skills import runbook as rbk                                  # noqa: E402
from vxagent.structured import proposals                                   # noqa: E402
from vxagent.structured.schemas import ConfigPatch                         # noqa: E402
from vxagent.tools.executor import AwaitingApproval, decide_approval, execute_tool   # noqa: E402
from vxagent.tools.service import load_calibration_record                  # noqa: E402

SMOKE = bool(os.environ.get("VX_SMOKE"))
OFFSET_MM = [4.0, -3.0, 3.5]                # INJECTED camera move
SCALE = [1.012, 1.0, 1.0]                   # INJECTED: the camera's first axis reads 1.2 % long
ERROR_LINE = "Error: a bytes-like object is required, not 'str'"
_CITE = re.compile(r"\[(doc|thread|site|changelog|schema):[^\]]+\]")


def t12() -> dict[str, Any]:
    s = get_settings()
    bench = Bench("t12", sites=("site_a", "site_b"))
    node = bench.nodes["site-a"]
    node.robot.enable()
    node.robot.home()
    op, eng, eng2, saf, saf_b = who("op.lee"), who("eng.kim"), who("eng.oyelaran"), who("saf.adler"), who("saf.novak")
    out: dict[str, Any] = {"task": "t12_service_access" + ("_smoke" if SMOKE else ""), "ran": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "model": s.model, "site": node.name, "backend": "twin",
                           "camera": "DECLARED: what the camera sees is computed from a declared transform with 0.2 mm noise",
                           "injected": {"camera_moved_by_mm": OFFSET_MM, "camera_scale": SCALE, "event_recorded": "ecm_moved"}}
    try:
        # ---------------------------------------------------------------- a. a ticket from a failed pre-flight to closure
        flow: dict[str, Any] = {}
        ctx = node.ctx(eng, node.new_run_id())
        rec0 = load_calibration_record(node.db, node.name)
        res = execute_tool(node.registry, ToolCall("r0", "calibration_residual", {"calibration_id": rec0["id"]}), ctx)   # becomes a site fact
        flow["before"] = {"calibration_in_use": rec0["id"], "residual_on_its_own_held_out_points_mm": res.content["mean_mm"],
                          "site_facts_active": len(site_memory.facts(node.db, eng, node.name)["active"])}
        cal.inject_camera_move(node, OFFSET_MM, SCALE)
        site_memory.record_event(node.db, node.name, "ecm_moved", op.user_id, "INJECTED for the verification")
        trial = node.start_trial(op, 0, use_llm=False)
        failed = (trial.state["data"].get("preflight") or {}).get("failed") or []
        tid = tickets.open_ticket(node.db, node.name, tickets.preflight_symptom(failed),
                                  "failed_preflight", op.user_id, {"failed_checks": failed, "run": trial.run_id})
        flow["ticket"] = {"ticket_id": tid, "source": "failed_preflight", "trial_outcome": trial.state["data"].get("outcome"),
                          "failed_checks": [f["check"] for f in failed], "repair_named": sorted({f["repair"] for f in failed}),
                          "motions_commanded": len(node.db.query("SELECT 1 FROM commands WHERE run_id=?", (trial.run_id,)))}
        log(f"a ticket {tid}: {flow['ticket']['failed_checks']}")

        diag = []
        for method in (("eliminate",) if SMOKE else ("eliminate", "team")):
            f = diagnosis.diagnose_ticket(node, eng, tid, method)
            diag.append({"method": method, "cause_named": f["cause"], "injected_cause": "calibration_offset",
                         "correct": f["cause"] == "calibration_offset", "narrowed_to_one": f["narrowed_to_one"],
                         "checks_run": f["checks_run"], "model_calls": f["calls"], "tokens": f["tokens"], "seconds": f["seconds"],
                         "fault_report": f.get("fault_report"), "evidence_resolves_to_ticket_records": f["evidence_resolves"]})
            log(f"a diagnosis {method}: {f['cause']} narrowed {f['narrowed_to_one']} resolves {f['evidence_resolves']}")
        flow["diagnosis"] = diag

        # the calibration procedure: the engineer runs it, the operator approves the motion
        run_id = node.new_run_id()
        ectx = node.ctx(eng, run_id)
        node.tracer.start_run("runbook", "collect_calibration_samples", eng.user_id, eng.role, node.name, run_id=run_id)
        access.begin_run(node.db, run_id, eng.user_id, node.name, ["sweep_calibration_poses"])
        approvals, seq0, t0 = [], node.robot.state().seq, time.time()
        while True:
            try:
                rb = rbk.run(node.library, "collect_calibration_samples", {}, node.registry, ectx, "cal-verification")
                break
            except AwaitingApproval as wait:
                try:
                    from vxagent import chat
                    chat.decide(node, eng, wait.approval_id, True)
                    self_approved = True
                except Exception:
                    self_approved = False
                decide_approval(node.db, wait.approval_id, True, op.user_id)
                approvals.append({"tool": wait.tool, "engineer_could_approve_it": self_approved, "approved_by": op.user_id})
        access.end_run(node.db, run_id)
        node.tracer.end_run(run_id, "completed")
        draft = cal.load_draft(node.db, node.name)
        world = cal.world(node)["T_base_camera"]
        h = draft["held_out_residual"]
        seen = np.array(draft["seen_m"])
        found = cal.difference(np.array(rec0["T_est"]), np.array(draft["T_est"]), seen)
        truth = cal.difference(world, np.array(draft["T_est"]), seen / np.array(SCALE))
        flow["calibration"] = {
            "runbook": rb["runbook"], "status": rb["status"], "steps": [(r["id"], r["ok"], [c["passed"] for c in r["checks"]]) for r in rb["steps"]],
            "approvals": approvals, "robot_state_changed": node.robot.state().seq != seq0, "seconds": round(time.time() - t0, 1),
            "samples": draft["samples"], "held_out": len(draft["touched_m"]),
            "held_out_residual_mm": {"record_in_use": h["record_in_use"], "rigid_fit": h["rigid"], "rigid_fit_plus_affine": h["rigid_plus_affine"]},
            "shift_between_the_old_record_and_the_fit_mm": found["shift_in_the_workspace_mm"],
            "injected_shift_mm": round(float(np.linalg.norm(OFFSET_MM)), 3),
            "rigid_fit_against_the_declared_truth_mm": truth["shift_in_the_workspace_mm"],
            "draft_status_after_the_fit": draft["status"], "record_in_use_after_the_fit": load_calibration_record(node.db, node.name)["id"],
            "acceptance_mm": cal.ACCEPT_MM}
        log(f"a calibration: {rb['status']}, held-out old {h['record_in_use']['mean_mm']} rigid {h['rigid']['mean_mm']} "
            f"affine {h['rigid_plus_affine']['mean_mm']} mm; shift found {found['shift_in_the_workspace_mm']} mm")

        # a question to the copilot while both records exist
        chat_rows = []
        if not SMOKE:
            r = ask(node, "eng.kim", "Compare the calibration in use with the new draft: residuals and how far apart they are.")
            chat_rows.append({**strip(r), "what": "a comparison of two calibration records",
                              "expected_tool_called": "compare_calibrations" in r["tools_called"]})
            log(f"b compare: {r['tools_called']} -> {r['stop_reason']}")

        # a motion approval issued before the change (used in part d)
        orun = node.new_run_id()
        octx = node.ctx(op, orun)
        node.tracer.start_run("probe", "approval before the change", op.user_id, op.role, node.name, run_id=orun)
        access.begin_run(node.db, orun, op.user_id, node.name, ["move_to_named_pose"])
        seq = execute_tool(node.registry, ToolCall("l", "get_robot_state", {}), octx).content["state_seq"]
        move = ToolCall("m", "move_to_named_pose", {"named_pose": "hover_needle", "based_on_seq": seq})
        try:
            execute_tool(node.registry, move, octx)
            old_approval = None
        except AwaitingApproval as wait:
            old_approval = wait.approval_id
            decide_approval(node.db, old_approval, True, op.user_id)

        prop = changes.propose_calibration(node, eng, "Camera arm was moved (event ecm_moved); recalibrated on the declared poses.", tid)
        flow["proposal"] = {"proposal_id": prop["proposal_id"], "kind": "calibration", "draft": prop["draft"],
                            "record_in_use_after_proposing": load_calibration_record(node.db, node.name)["id"]}
        decided = changes.decide(node, saf, prop["proposal_id"], True)
        rows = node.preflight_rows(op.user_id)
        tickets.attach(node.db, tid, "verification", {"what": "pre-flight after activation", "checks": len(rows),
                                                      "passed": sum(r["ok"] for r in rows)}, eng.user_id)
        flow["activation"] = {"approved_by": saf.user_id, "status": decided["status"], "event": decided["applied"]["event"],
                              "record_in_use": load_calibration_record(node.db, node.name)["id"],
                              "calibration_valid": bool(proposals.calibration(node.db, node.name)["valid"]),
                              "preflight_after": f"{sum(r['ok'] for r in rows)}/{len(rows)} checks pass",
                              "failed_after": [r["check"] for r in rows if not r["ok"]]}
        rep = report.close_with_report(node, eng, tid)
        flow["service_report"] = {"summary_written_by": rep["summary_source"], "attempts": rep["summary_attempts"],
                                  "problems_by_attempt": rep["summary_log"], "summary": rep["summary"],
                                  "not_verified": rep["not_verified"], "records_used": len(rep["records_used"]),
                                  "ticket_status": tickets.get(node.db, tid)["status"]}
        log(f"a activation: {flow['activation']}; report by {rep['summary_source']} after {rep['summary_attempts']} attempt(s)")
        out["a_ticket_to_closure"] = flow

        # ---------------------------------------------------------------- d. after activation
        facts = site_memory.facts(node.db, eng, node.name)
        octx.site = node.site
        res = None
        try:
            res = execute_tool(node.registry, move, octx)
            after = "executed" if res.ok else res.error_kind
        except AwaitingApproval:
            after = "asked for again"
        out["d_after_activation"] = {
            "site_facts": {"active": [f["text"] for f in facts["active"]],
                           "expired": [{"text": f["text"], "because": f["expired_because"]} for f in facts["expired"]]},
            "old_calibration_fact_expired": any(rec0["id"] in f["text"] for f in facts["expired"]),
            "approval_issued_before_the_change": {"approval_id": old_approval,
                                                  "status_now": node.db.one("SELECT status FROM approvals WHERE id=?", (old_approval,))["status"],
                                                  "motion_requested_with_it": after,
                                                  "motions_executed": len(node.db.query("SELECT 1 FROM commands WHERE run_id=? AND status='done'", (orun,)))}}
        access.end_run(node.db, orun)
        log(f"d after: {out['d_after_activation']['old_calibration_fact_expired']}, old approval {after}")

        # ---------------------------------------------------------------- b. questions through the chat path
        if not SMOKE:
            r = ask(node, "eng.kim", f"I get this line in the terminal: {ERROR_LINE}  What do the dVRK threads say about it?")
            chat_rows.append({**strip(r), "what": "a real error line searched in the support threads",
                              "expected_tool_called": "search_by_log_line" in r["tools_called"] or "search_threads" in r["tools_called"],
                              "answer_cites_a_passage": bool(_CITE.search(r["answer"]))})
            log(f"b log line: {r['tools_called']} cites {chat_rows[-1]['answer_cites_a_passage']}")
            r = ask(node, "eng.kim", "In our site file, what is safety_profile/near_distance_mm set to, what does it do, and "
                                     "which direction makes it tighter?")
            chat_rows.append({**strip(r), "what": "a configuration key explained from the file and its rules",
                              "expected_tool_called": bool({"read_config", "describe_config_paths"} & set(r["tools_called"])),
                              "answer_states_the_value": "20" in r["answer"]})
            log(f"b config key: {r['tools_called']} -> {r['stop_reason']}")
            r = ask(node, "eng.kim", "The phantom is softer than expected. File a change that lowers the near-step limit in our site "
                                     f"file from 2 mm to 1.5 mm, for ticket {tid}.")
            pend = changes.queue(node.db, node.name)
            chat_rows.append({**strip(r), "what": "a change proposed through the copilot",
                              "expected_tool_called": "propose_config_patch" in r["tools_called"],
                              "proposals_waiting_for_the_safety_lead": len(pend),
                              "limit_in_force_unchanged": abs(node.site.profile.near_step_m - 0.002) < 1e-9})
            log(f"b propose: {r['tools_called']} waiting {len(pend)}")
        out["b_questions"] = {"asked": len(chat_rows), "expected_tool_called": sum(x["expected_tool_called"] for x in chat_rows), "rows": chat_rows}

        # ---------------------------------------------------------------- c. the two-person rule
        tid2 = tickets.open_ticket(node.db, node.name, "Descent near the pad is too fast on the softer phantom.", "operator", op.user_id,
                                   {"observed": "INJECTED for the verification"})
        ev = tickets.records(node.db, tid2)[0]["record"]

        def patch(value: float, limit: str) -> ConfigPatch:
            return ConfigPatch.model_validate({"target": node.site_target, "limit_change": limit, "evidence": [ev],
                                               "reason": "Softer phantom: step size near the pad reviewed.",
                                               "operations": [{"op": "replace", "path": "safety_profile/near_step_mm", "value": value, "unit": "mm"}]})
        rule = []

        def attempt(label: str, fn: Any, expect: str) -> Any:
            try:
                got = fn()
                outcome = "accepted"
            except proposals.NotPermitted as exc:
                got, outcome = None, f"refused: {exc.kind}"
            rule.append({"case": label, "outcome": outcome, "expected": expect, "as_expected": outcome == expect})
            return got

        for p in changes.queue(node.db, node.name):          # what the copilot filed above is rejected first, so the file is as it was
            changes.decide(node, saf, p["proposal_id"], False, "superseded by the verification's own proposal")
        attempt("an operator proposing a change", lambda: changes.propose_patch(node, op, patch(1.0, "tighten"), tid2), "refused: role")
        first = attempt("the engineer proposing a tighter limit", lambda: changes.propose_patch(node, eng, patch(1.0, "tighten"), tid2), "accepted")
        pid = first["proposal_id"]
        attempt("the proposer approving their own change", lambda: changes.decide(node, eng, pid, True), "refused: role")
        attempt("another engineer approving it", lambda: changes.decide(node, eng2, pid, True), "refused: role")
        attempt("the safety lead of another site approving it", lambda: changes.decide(node, saf_b, pid, True), "refused: site")
        both = type(saf)("eng.kim", "safety", saf.scopes, site=node.name, tiers=saf.tiers)       # the proposer, had they the approver's role
        attempt("the proposer approving with the safety lead's role", lambda: changes.decide(node, both, pid, True), "refused: self_approval")
        applied = attempt("the safety lead approving it", lambda: changes.decide(node, saf, pid, True), "accepted")
        attempt("approving it a second time", lambda: changes.decide(node, saf, pid, True), "refused: state")
        loose = attempt("the engineer proposing a looser limit", lambda: changes.propose_patch(node, eng, patch(1.8, "loosen"), tid2), "accepted")
        attempt("the safety lead approving the looser limit without a reason", lambda: changes.decide(node, saf, loose["proposal_id"], True),
                "refused: reason_required")
        if not SMOKE:                                         # the safety lead's copilot, which can only read
            from vxagent import chat
            r = ask(node, "saf.adler", f"Explain proposal {loose['proposal_id']}: what does it change, and does it loosen a safety limit?")
            offered = [t.name for t in node.registry.allowed_for(saf, chat.config_for(saf).tool_names)]
            out["f_safety_copilot"] = {**strip(r), "expected_tool_called": "read_change" in r["tools_called"],
                                       "says_it_loosens": "loosen" in r["answer"].lower(), "tools_offered": offered,
                                       "tools_offered_that_write_or_move": [t.name for t in node.registry.allowed_for(saf, chat.config_for(saf).tool_names)
                                                                            if t.tier != "observe" or t.side_effect],
                                       "proposal_status_after_the_question": changes.view(node.db, loose["proposal_id"])["status"]}
            log(f"f safety copilot: {r['tools_called']} -> {r['stop_reason']}, says it loosens {out['f_safety_copilot']['says_it_loosens']}")
        attempt("the same with a written reason",
                lambda: changes.decide(node, saf, loose["proposal_id"], True, "Phantom C measured on site; 1.8 mm reviewed with the engineer."), "accepted")
        out["c_two_person_rule"] = {"cases": len(rule), "as_expected": sum(r["as_expected"] for r in rule), "rows": rule,
                                    "after_the_tighter_limit": {"event": applied["applied"]["event"], "envelope_version": applied["applied"]["envelope_version"],
                                                                "approvals_ended": applied["applied"]["approvals_ended"]},
                                    "limit_in_force_now_mm": round(node.site.profile.near_step_m * 1000, 3),
                                    "envelope_version_now": str(node.site.profile.version),
                                    "versions_of_the_site_file": node.store.versions(node.site_target),
                                    "loosened_proposal_marked": changes.view(node.db, loose["proposal_id"])["loosens_a_limit"],
                                    "refusals_audited": len(node.db.query("SELECT 1 FROM audit WHERE allowed=0 AND action LIKE 'config:%'"))}
        log(f"c two-person rule: {out['c_two_person_rule']['as_expected']}/{len(rule)}; limit now {out['c_two_person_rule']['limit_in_force_now_mm']} mm")

        # ---------------------------------------------------------------- e. what the engineer cannot do, and the pages by role
        console, passwords = bench.console()
        e = bench.client(console, passwords, "eng.kim")
        cannot = {"start a trial (POST /api/trials)": e.post("/api/trials", json={}).status_code,
                  "decide a change (POST /api/changes/1/decision)": e.post("/api/changes/1/decision", json={"approve": True}).status_code,
                  "see or decide approval cards (GET /api/approvals)": e.get("/api/approvals").status_code,
                  "give a support grant (POST /api/grants)": e.post("/api/grants", json={}).status_code}
        ctx = node.ctx(eng, node.new_run_id())
        access.begin_run(node.db, ctx.run_id, eng.user_id, node.name, ["move_to_named_pose"])
        seq = execute_tool(node.registry, ToolCall("l", "get_robot_state", {}), ctx).content["state_seq"]
        mv = execute_tool(node.registry, ToolCall("m", "move_to_named_pose", {"named_pose": "start", "based_on_seq": seq}), ctx)
        cannot["move the arm to a named pose (tool, inside a run)"] = mv.error_kind
        users = {"operator": "op.lee", "service": "eng.kim", "safety": "saf.adler", "support": "sup.wong"}
        pages = {}
        for role, u in users.items():
            c = bench.client(console, passwords, u)
            pages[role] = {p: c.get(p).status_code for p in ("/api/tickets", f"/api/tickets/{tid}", "/api/changes", "/api/config",
                                                               "/api/calibration", "/api/overrides", "/api/grants")}
        out["e_roles"] = {"the_engineer_cannot": cannot, "pages_by_role": pages, "matrix": route_matrix(bench, console, passwords, users)}
        log(f"e engineer cannot: {cannot}")
    finally:
        bench.close()
    return out
