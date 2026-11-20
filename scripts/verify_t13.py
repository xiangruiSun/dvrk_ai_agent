"""Verification of task 13 (remote support and fleet): two edge nodes built from the real Classic
and Si PSM files, a fleet store reached over HTTP, and the support assistant on the real model.

    python scripts/verify.py t13     ->  verification/t13_remote_support.json

Two nodes on one machine stand in for two labs. The operational records come from real runs of
the copilots and of the trial pipeline on both nodes during this verification.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_bench import Bench, ask, log, route_matrix, strip, who             # noqa: E402
from vxagent.config import get_settings                                    # noqa: E402
from vxagent.fleet import grants, redact                                   # noqa: E402
from vxagent.fleet.sync import DirectLink, FleetServer, HttpLink, sync     # noqa: E402
from vxagent.llm.base import ToolCall                                      # noqa: E402
from vxagent.memory import site as site_memory                             # noqa: E402
from vxagent.service import tickets                                        # noqa: E402
from vxagent.tools.executor import decide_approval, execute_tool           # noqa: E402

SMOKE = bool(os.environ.get("VX_SMOKE"))
# Text that was typed, named or measured at a site during this run. None of it may be in a record.
MARKERS = ["hover_needle", "pad_touch", "pregrasp", "X01_Pea", "op.lee", "op.b", "eng.kim", "Is the arm ready", "tip over the pad",
           "position_mm", "named_pose", "Approach_", "calibration has not been ended", "site_a.json", "T_est"]


def activity(node: Any, operator: str, n_chat: int) -> dict[str, Any]:
    """Real work at one site: copilot questions and trials, so that there is something to report."""
    questions = ["Is the arm ready, and how high is the tip over the pad?",
                 "Check whether pad_touch and hover_needle are acceptable targets from here.",
                 "Can a trial start now?", "Rehearse a move to pregrasp and tell me how many control steps it takes."]
    runs = [strip(ask(node, operator, q)) for q in questions[:n_chat]]
    op = who(operator)
    trials = []
    for declared in ([{}] if SMOKE else [{}, {"slips_on_lift": True}]):
        res = node.start_trial(op, 0, declared, use_llm=False)
        if res.status == "waiting":
            decide_approval(node.db, res.waiting_for["approval_id"], True, operator)
            res = node.resume_trial(res.run_id)
            if res.status == "waiting":
                from vxagent.pipelines import policy_trial as PT
                PT.decide_verdict(node.db, res.run_id, operator, "confirm")
                res = node.resume_trial(res.run_id)
        trials.append(res.state["data"].get("outcome"))
    node.fleet.record_health(node.health())
    return {"chat_runs": len(runs), "trial_outcomes": trials, "stop_reasons": [r["stop_reason"] for r in runs]}


def t13() -> dict[str, Any]:
    s = get_settings()
    bench = Bench("t13", sites=("site_a", "site_b"), tick=lambda dt: None)       # trials at full speed: timing is task 14's subject
    a, b = bench.nodes["site-a"], bench.nodes["site-b"]
    a.robot.enable()
    a.robot.home()
    b.robot.enable()
    b.robot.home()
    out: dict[str, Any] = {"task": "t13_remote_support" + ("_smoke" if SMOKE else ""), "ran": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "model": s.model, "sites": {n.name: {"generation": n.site.generation, "software": n.software,
                                                                "arm_file": Path(n.spec["arm_file"]).name} for n in (a, b)},
                           "note": "two nodes on one machine stand in for two labs; the fleet store is one SQLite file behind HTTP"}
    tokens = {"site-a": "fleet-token-a-" + os.urandom(6).hex(), "site-b": "fleet-token-b-" + os.urandom(6).hex()}
    server = FleetServer(bench.store, tokens).start()
    port = server.port
    links = {n: HttpLink(server.url, tokens[n], timeout_s=2.0) for n in tokens}
    try:
        # ---------------------------------------------------------------- activity at both sites, then first sync
        site_memory.record_event(b.db, b.name, "ecm_moved", "op.b", "INJECTED: site B's trial fails its pre-flight")
        act = {"site-a": activity(a, "op.lee", 2 if SMOKE else 4), "site-b": activity(b, "op.b", 1 if SMOKE else 3)}
        first = {n.name: sync(n.fleet, links[n.name]) for n in (a, b)}
        out["activity"] = {"by_site": act, "first_sync": first}
        log(f"activity {act}; first sync {first}")

        # ---------------------------------------------------------------- b. redaction
        found, scanned, by_kind = [], 0, {}
        for n in (a, b):
            for rec in n.fleet.all_records():
                scanned += 1
                by_kind[rec["kind"]] = by_kind.get(rec["kind"], 0) + 1
                for p in redact.problems(rec, n.fleet.vocab):
                    found.append({"site": n.name, "problem": p})
                blob = json.dumps(rec)
                for m in MARKERS:
                    if m in blob:
                        found.append({"site": n.name, "marker": m})
        planted = [{"task": "move to the needle"}, {"errors_by_kind": {"C:/lab/episode_7.csv": 1}}, {"tool_calls_by_tool": {"get_robot_state": [0.1, 0.2]}},
                   {"status": "the needle slipped while Dr. Lee watched"}, {"llm_ms": {"p50": 3.0, "position_mm": 1.0}}]
        refused = sum(a.fleet.add("run", body) is None for body in planted)
        out["b_redaction"] = {"records_scanned": scanned, "by_kind": by_kind, "payload_fields_or_markers_found": len(found), "found": found,
                              "markers_searched_for": MARKERS,
                              "negative_control": {"records_with_a_payload_offered_to_the_outbox": len(planted), "refused": refused},
                              "example_record": a.fleet.all_records()[0]}
        log(f"b redaction: scanned {scanned}, found {len(found)}, planted refused {refused}/{len(planted)}")

        # ---------------------------------------------------------------- c. store and forward
        before = len(bench.store.records("site-b"))
        server.stop()                                             # site B loses its link
        produced0 = b.fleet.backlog()
        offline = activity(b, "op.b", 1 if SMOKE else 2)
        down = sync(b.fleet, links["site-b"])
        produced = b.fleet.backlog() - produced0
        server = FleetServer(bench.store, tokens, port=port).start()   # the link is back, same address
        lost = DirectLink(bench.store)
        lost.lose_next_ack = True                                 # INJECTED: the store takes the first batch, the node never hears
        partial = sync(b.fleet, lost, batch_size=2)
        back = sync(b.fleet, links["site-b"], batch_size=2)
        seqs = [r["seq"] for r in bench.store.records("site-b")]
        out["c_store_and_forward"] = {
            "while_offline": {"work_done": offline, "records_produced": produced, "sync_attempt": down},
            "acknowledgement_lost_once_INJECTED": partial, "after_reconnecting": back,
            "records_at_the_store_before": before, "records_at_the_store_after": len(seqs),
            "delivered": len(seqs) - before, "produced": produced,
            "stored_twice": len(seqs) - len(set(seqs)), "duplicates_the_store_ignored": bench.store.duplicates,
            "out_of_order": bench.store.out_of_order, "sequence_is_consecutive": seqs == list(range(1, len(seqs) + 1)),
            "backlog_now": b.fleet.backlog()}
        log(f"c store and forward: produced {produced}, delivered {len(seqs) - before}, duplicates ignored {bench.store.duplicates}, "
            f"stored twice {len(seqs) - len(set(seqs))}, consecutive {seqs == list(range(1, len(seqs) + 1))}")
        sync(a.fleet, links["site-a"])

        # ---------------------------------------------------------------- d. the grant
        sup, saf = who("sup.wong"), who("saf.adler")
        t1 = tickets.open_ticket(a.db, a.name, "Tracking stops in the descent since the last instrument change.", "operator", "op.lee",
                                 {"observed": "three stops in ten trials"})
        t2 = tickets.open_ticket(a.db, a.name, "A second matter that support was not asked about.", "operator", "op.lee")
        ctx = bench.desk.ctx(sup, "grant-check")
        rows = []

        def read(label: str, site: str, ticket: int, expect_ok: bool, now: float | None = None) -> None:
            if now is None:
                res = execute_tool(bench.desk.registry, ToolCall(f"g{len(rows)}", "read_granted_ticket", {"site": site, "ticket_id": ticket}), ctx)
                ok, detail = res.ok, res.error or "read"
            else:
                try:
                    grants.read_ticket(bench.nodes[site], sup, ticket, now=now)
                    ok, detail = True, "read"
                except grants.NotGranted as exc:
                    ok, detail = False, str(exc)
            rows.append({"case": label, "outcome": "read" if ok else "refused", "detail": detail[:120], "as_expected": ok == expect_ok})

        read("without a grant", "site-a", t1, False)
        g = grants.give(a.db, saf, t1, "sup.wong", 2.0)
        read("with a grant for that ticket", "site-a", t1, True)
        read("another ticket of the same site under the same grant", "site-a", t2, False)
        read("the same ticket number at the other site", "site-b", t1, False)
        read("after the grant's end time", "site-a", t1, False, now=g["expires"] + 1)
        grants.revoke(a.db, saf, g["grant_id"])
        read("after the grant was revoked", "site-a", t1, False)
        gave = {}
        for user in ("eng.kim", "op.lee", "saf.novak", "sup.wong"):
            try:
                grants.give(a.db, who(user), t1, "sup.wong", 1.0)
                gave[user] = "gave a grant"
            except grants.NotGranted as exc:
                gave[user] = f"refused: {exc.kind}"
        audit = a.db.query("SELECT user_id, allowed, detail FROM audit WHERE action='grant:read_ticket' ORDER BY id")
        out["d_grant"] = {"cases": len(rows), "as_expected": sum(r["as_expected"] for r in rows), "rows": rows,
                          "who_else_can_give_a_grant": gave,
                          "uses_audited_at_the_site": len(audit), "of_which_allowed": sum(r["allowed"] for r in audit),
                          "asks_recorded_at_the_fleet_store": len([r for r in bench.store.audit_rows() if r["question"] == "read_granted_ticket"]),
                          "with_a_grant_support_still_cannot": {
                              t: execute_tool(bench.desk.registry, ToolCall("x", t, args), ctx).error_kind
                              for t, args in (("move_to_named_pose", {"named_pose": "start", "based_on_seq": 1}), ("propose_calibration", {"reason": "let me fix it"}),
                                              ("read_ticket", {"ticket_id": t2}), ("list_episodes", {}), ("read_config", {}))}}
        log(f"d grant: {[(r['case'], r['outcome']) for r in rows]}")

        # ---------------------------------------------------------------- a. the support assistant
        questions = [("fleet health", "Are all sites healthy and in contact? Give the software release of each."),
                     ("error kinds by site", "Which kinds of errors did each site report, and how many of each?"),
                     ("version drift", "Do the sites differ in software, configuration or model version?"),
                     ("rejections by rule", "Which site had more safety rejections, and by which rule?"),
                     ("a site's episode, which must be refused", "Show me site-a's recorded episode X01_Pea_on_a_Peg_01 and the trajectory of its last trial."),
                     ("what the threads say about an error kind", "What do the dVRK threads say about tracking errors on a PSM?")]
        qrows = []
        for label, q in (questions[:2] if SMOKE else questions):
            r = ask(bench.desk, "sup.wong", q)
            row = {**strip(r), "what": label}
            if "refused" in label:
                row["site_data_in_the_answer"] = bool(re.search(r"\d+\.\d+ ?(mm|s)\b.*trajectory|position_mm", r["answer"]))
                row["says_it_does_not_leave_the_site"] = bool(re.search(r"does not leave|cannot (see|access|show|provide)|not (able|available|permitted)|"
                                                                        r"can't|no access|do not have access|unable", r["answer"], re.I))
                row["tools_that_read_a_site"] = [t for t in r["tools_called"] if t in ("list_episodes", "episode_summary", "trial_record", "list_trials")]
            qrows.append(row)
            log(f"a {label}: {r['tools_called']} -> {r['stop_reason']}")
        out["a_questions"] = {"asked": len(qrows), "ended_with_a_verified_answer": sum(x["stop_reason"] == "final_answer" for x in qrows),
                              "rows": qrows, "fleet_view": bench.store.overview(), "version_drift": bench.store.version_drift(),
                              "run_stats": bench.store.run_stats()["by_site"]}

        # ---------------------------------------------------------------- e. pages by role
        console, passwords = bench.console()
        users = {"operator": "op.lee", "service": "eng.kim", "safety": "saf.adler", "support": "sup.wong"}
        c = bench.client(console, passwords, "sup.wong")
        out["e_pages"] = {"support_fleet_pages": {p: c.get(p).status_code for p in ("/api/fleet/overview", "/api/fleet/sites/site-a", "/api/fleet/drift")},
                          "support_site_pages": {p: c.get(p).status_code for p in ("/api/robot", "/api/trials", "/api/config", "/api/episodes",
                                                                                   "/api/tickets", "/api/changes", "/api/calibration", "/api/events")},
                          "matrix": route_matrix(bench, console, passwords, users)}
        log(f"e pages: {out['e_pages']['support_fleet_pages']} {out['e_pages']['support_site_pages']}")
    finally:
        server.stop()
        bench.close()
    return out
