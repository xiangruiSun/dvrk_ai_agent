"""Service engineer and safety lead (task 12) on made-up data. What must not regress: a fit is a
draft until an approved change activates it, the proposer never approves, the engineer never
moves the arm except through the calibration poses with an operator's approval, and applying a
change ends what was known and approved before it.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from console_bench import client_for, make_console, make_node, ready, who
from vxagent import access, chat
from vxagent.diagnostics.episode import CAUSES, T_BC
from vxagent.fleet import grants
from vxagent.llm.base import ToolCall
from vxagent.llm.scripted import ScriptedLLM, say
from vxagent.memory import site as site_memory
from vxagent.service import calibration as cal
from vxagent.service import changes, diagnosis, report, tickets
from vxagent.skills import runbook as rbk
from vxagent.structured import proposals
from vxagent.structured.schemas import ConfigPatch
from vxagent.tools.executor import AwaitingApproval, decide_approval, execute_tool
from vxagent.tools.service import load_calibration_record

ENG, ENG2, SAF, SAF_B, OP = "eng.kim", "eng.oyelaran", "saf.adler", "saf.novak", "op.lee"


def sweep(node, by=ENG, approve_as=OP):
    """Run the calibration runbook as the engineer, with the operator approving the motion."""
    eng = who(by)
    run_id = node.new_run_id()
    ctx = node.ctx(eng, run_id)
    access.begin_run(node.db, run_id, eng.user_id, node.name, ["sweep_calibration_poses"])

    def approve(wait):
        chat_card = chat.card(node, wait.approval_id)
        assert chat_card["tool"] == "sweep_calibration_poses"
        return True
    out = rbk.run(node.library, "collect_calibration_samples", {}, node.registry, ctx, "cal1",
                  on_approval=approve if approve_as else None)
    access.end_run(node.db, run_id)
    return out


# ------------------------------------------------------------------ the fit

def test_the_rigid_fit_recovers_a_known_transform_and_the_affine_step_absorbs_a_scale_error():
    rng = np.random.default_rng(3)
    touched = rng.uniform(-0.03, 0.03, size=(15, 3)) + [0.0, 0.0, -0.12]
    T = T_BC.copy()
    T[:3, 3] += [0.006, 0.0, 0.0]
    seen = (touched - T[:3, 3]) @ T[:3, :3]
    res = cal.fit([{"seen_m": s.tolist(), "touched_m": t.tolist()} for s, t in zip(seen, touched)], T_BC)
    d = cal.difference(T, np.array(res["T_est"]))
    assert d["translation_mm"] < 1e-6 and d["rotation_deg"] < 1e-5
    assert res["held_out_residual"]["rigid"]["max_mm"] < 1e-6 and res["held_out"] == 5 and res["used_for_the_fit"] == 10
    assert 5.5 < res["held_out_residual"]["record_in_use"]["mean_mm"] < 6.5
    scaled = seen * [1.02, 1.0, 1.0]
    res = cal.fit([{"seen_m": s.tolist(), "touched_m": t.tolist()} for s, t in zip(scaled, touched)])
    h = res["held_out_residual"]
    assert h["rigid_plus_affine"]["mean_mm"] < h["rigid"]["mean_mm"] / 5
    with pytest.raises(ValueError):
        cal.fit([{"seen_m": [0, 0, 0], "touched_m": [0, 0, 0]}] * 4)


def test_the_engineer_sweeps_the_calibration_poses_only_with_an_operators_approval(tmp_path):
    node = ready(make_node(tmp_path))
    cal.inject_camera_move(node, [6.0, 0.0, 0.0])
    eng = who(ENG)
    run_id = node.new_run_id()
    ctx = node.ctx(eng, run_id)
    look = execute_tool(node.registry, ToolCall("l", "get_robot_state", {}), ctx).content["state_seq"]
    call = ToolCall("s", "sweep_calibration_poses", {"based_on_seq": look})
    assert execute_tool(node.registry, call, ctx).error_kind == "unauthorized"            # not inside a run
    access.begin_run(node.db, run_id, eng.user_id, node.name, ["sweep_calibration_poses"])
    with pytest.raises(AwaitingApproval) as wait:
        execute_tool(node.registry, call, ctx)
    with pytest.raises(chat.NotAllowed):                       # the engineer cannot approve their own motion
        chat.decide(node, eng, wait.value.approval_id, True)
    move = ToolCall("m", "move_to_named_pose", {"named_pose": "hover_needle", "based_on_seq": look})
    assert execute_tool(node.registry, move, ctx).error_kind == "unauthorized"            # no robot:move at all
    decide_approval(node.db, wait.value.approval_id, True, OP)
    res = execute_tool(node.registry, call, ctx)
    assert res.ok and res.content["samples"] == 12 and res.content["returned_to"] == "start"
    assert len(cal.load_samples(node.db, node.name)["samples"]) == 12


def test_a_fit_is_a_draft_and_becomes_the_calibration_only_by_an_approved_change(tmp_path):
    node = ready(make_node(tmp_path))
    old = load_calibration_record(node.db, node.name)
    cal.inject_camera_move(node, [6.0, 0.0, 0.0])
    site_memory.record_event(node.db, node.name, "ecm_moved", OP, "bumped")
    assert not proposals.calibration(node.db, node.name)["valid"]
    out = sweep(node)
    fit = next(r for r in out["steps"] if r["id"] == "fit")
    assert out["status"] == "completed" and fit["checks"][0]["passed"]
    draft = cal.load_draft(node.db, node.name)
    assert draft["status"] == "draft" and load_calibration_record(node.db, node.name)["id"] == old["id"]
    h = draft["held_out_residual"]
    assert h["rigid"]["mean_mm"] < 1.0 and 5.0 < h["record_in_use"]["mean_mm"] < 7.0
    found = cal.difference(np.array(old["T_est"]), np.array(draft["T_est"]), draft["seen_m"])
    assert abs(found["shift_in_the_workspace_mm"] - 6.0) < 0.5
    p = changes.propose_calibration(node, who(ENG), "Camera arm was moved; recalibrated on 12 poses.")
    assert load_calibration_record(node.db, node.name)["id"] == old["id"]                # still nothing changed
    for user, kind in ((ENG, "role"), (ENG2, "role"), (SAF_B, "site"), (OP, "role")):
        with pytest.raises(proposals.NotPermitted) as e:
            changes.decide(node, who(user), p["proposal_id"], True)
        assert e.value.kind == kind
    done = changes.decide(node, who(SAF), p["proposal_id"], True)
    assert done["status"] == "applied" and done["applied"]["event"] == "calibration_redone"
    assert load_calibration_record(node.db, node.name)["id"] == draft["id"]
    assert proposals.calibration(node.db, node.name)["valid"] and cal.load_draft(node.db, node.name)["status"] == "active"
    assert all(r["ok"] for r in node.preflight_rows(OP)), [r for r in node.preflight_rows(OP) if not r["ok"]]


def test_the_safety_lead_who_proposed_cannot_be_the_one_who_approves(tmp_path):
    node = ready(make_node(tmp_path))
    sweep(node)
    both = access.principal_of({"user_id": "both", "role": "service", "site": node.name, "display_name": ""})
    both = type(both)(both.user_id, "safety", both.scopes | {"config:approve"}, site=node.name, tiers=both.tiers)
    p = changes.propose_calibration(node, both, "I propose and I would like to approve it too.")
    with pytest.raises(proposals.NotPermitted) as e:
        changes.decide(node, both, p["proposal_id"], True)
    assert e.value.kind == "self_approval"


# ------------------------------------------------------------------ configuration changes

def patch(node, value=1.0, limit="tighten", evidence=("ticket:1/evidence/1",)):
    return ConfigPatch.model_validate({"target": node.site_target, "limit_change": limit, "evidence": list(evidence),
                                       "reason": "Phantom B is softer; smaller steps near the pad.",
                                       "operations": [{"op": "replace", "path": "safety_profile/near_step_mm", "value": value, "unit": "mm"}]})


def test_a_change_is_validated_proposed_approved_and_then_ends_what_came_before(tmp_path):
    llm = ScriptedLLM([say("unused")])
    node = ready(make_node(tmp_path, llm=llm))
    tid = tickets.open_ticket(node.db, node.name, "Grasp descends too fast near the pad.", "operator", OP, {"note": "seen twice"})
    bad = changes.propose_patch(node, who(ENG), patch(node, value=9.0, limit="loosen"), tid)
    assert bad["proposed"] is False and bad["stage"] == "physical"
    wrong_evidence = changes.propose_patch(node, who(ENG), patch(node, evidence=("ticket:9/evidence/9",)), tid)
    assert wrong_evidence["proposed"] is False
    with pytest.raises(proposals.NotPermitted):
        changes.propose_patch(node, who(OP), patch(node), tid)
    ok = changes.propose_patch(node, who(ENG), patch(node), tid)
    assert ok["proposed"] and ok["limit_effect"] == "tighten" and node.site.profile.near_step_m == pytest.approx(0.002)
    # an approval to move, asked for before the change
    op = who(OP)
    run_id = node.new_run_id()
    ctx = node.ctx(op, run_id)
    access.begin_run(node.db, run_id, OP, node.name, ["move_to_named_pose"])
    node.tracer.start_run("t", "t", OP, "operator", node.name, run_id=run_id)
    look = execute_tool(node.registry, ToolCall("l", "get_robot_state", {}), ctx).content["state_seq"]
    move = ToolCall("m", "move_to_named_pose", {"named_pose": "hover_needle", "based_on_seq": look})
    with pytest.raises(AwaitingApproval) as wait:
        execute_tool(node.registry, move, ctx)
    decide_approval(node.db, wait.value.approval_id, True, OP)
    queue = changes.queue(node.db, node.name)
    assert queue[0]["proposer"] == ENG and queue[0]["evidence"] == ["ticket:1/evidence/1"] and "poses" in queue[0]["effect_on_the_twin"]
    done = changes.decide(node, who(SAF), ok["proposal_id"], True)
    assert done["applied"]["event"] == "limits_changed" and done["applied"]["approvals_ended"] == 1
    assert node.site.profile.near_step_m == pytest.approx(0.001) and str(node.site.profile.version) == "2"
    ctx.site = node.site
    with pytest.raises(AwaitingApproval):                      # the old approval is gone: the motion is asked for again
        execute_tool(node.registry, move, ctx)
    assert node.db.one("SELECT status FROM approvals WHERE id=?", (wait.value.approval_id,))["status"] == "expired"
    assert [r["kind"] for r in tickets.records(node.db, tid)] == ["evidence", "proposal", "change_applied"]


def test_a_loosened_limit_needs_the_safety_leads_written_reason(tmp_path):
    node = make_node(tmp_path)
    tickets.open_ticket(node.db, node.name, "Steps near the pad are too small for phantom C.", "operator", OP, {"n": 1})
    changes.decide(node, who(SAF), changes.propose_patch(node, who(ENG), patch(node, 1.0))["proposal_id"], True)
    p = changes.propose_patch(node, who(ENG), patch(node, 1.8, "loosen"))
    assert p["limit_effect"] == "loosen" and changes.view(node.db, p["proposal_id"])["loosens_a_limit"]
    with pytest.raises(proposals.NotPermitted) as e:
        changes.decide(node, who(SAF), p["proposal_id"], True)
    assert e.value.kind == "reason_required"
    assert changes.decide(node, who(SAF), p["proposal_id"], True, "Phantom C measured on site; 1.8 mm reviewed.")["status"] == "applied"


def test_the_engineer_cannot_approve_activate_or_run_a_trial(tmp_path):
    con = make_console(tmp_path)
    node = ready(con.nodes["site-a"])
    eng = client_for(con, ENG)
    assert eng.post("/api/trials", json={}).status_code == 403
    assert eng.post("/api/changes/1/decision", json={"approve": True}).status_code == 403
    assert eng.get("/api/approvals").status_code == 403 and eng.post("/api/grants", json={}).status_code == 403
    sweep(node)
    r = eng.post("/api/changes", json={"kind": "calibration", "reason": "Recalibrated after the camera arm moved."})
    assert r.status_code == 200 and r.json()["proposed"]
    saf = client_for(con, SAF)
    assert saf.post("/api/changes", json={"kind": "calibration", "reason": "I would like to propose this."}).status_code == 403
    q = saf.get("/api/changes").json()["changes"]
    assert q[0]["kind"] == "calibration" and q[0]["proposer"] == ENG
    assert saf.post(f"/api/changes/{q[0]['proposal_id']}/decision", json={"approve": True}).json()["status"] == "applied"
    assert client_for(con, "saf.novak").post(f"/api/changes/{q[0]['proposal_id']}/decision", json={"approve": True}).status_code in (404, 409)


# ------------------------------------------------------------------ diagnosis and the service report

def eliminate_script(cause="calibration_offset"):
    others = [c for c in CAUSES if c != cause]
    first = {"drops": [], "supports": [], "ranking": list(CAUSES), "next_check": "calibration_residual", "conclusion": None}
    second = {"drops": [{"cause": c, "result": "r1"} for c in others], "supports": [{"cause": cause, "result": "r1"}],
              "ranking": [cause], "next_check": None, "conclusion": cause}
    return [say(json.dumps(first)), say(json.dumps(second))]


def test_a_diagnosis_is_attached_to_the_ticket_as_a_fault_report_with_resolvable_evidence(tmp_path):
    node = ready(make_node(tmp_path, llm=ScriptedLLM(eliminate_script())))
    cal.inject_camera_move(node, [6.0, 0.0, 0.0])
    tid = tickets.open_ticket(node.db, node.name, "Pre-flight failed: the calibration was ended by an event.", "failed_preflight", OP)
    with pytest.raises(PermissionError):
        diagnosis.diagnose_ticket(node, who(OP), tid)
    with pytest.raises(PermissionError):
        diagnosis.diagnose_ticket(node, who(SAF), tid)
    f = diagnosis.diagnose_ticket(node, who(ENG), tid)
    assert f["cause"] == "calibration_offset" and f["narrowed_to_one"] and f["evidence_resolves"]
    rep = f["fault_report"]
    assert rep["cause_class"] == "calibration" and rep["next_check"] == "calibration_validity" and rep["confidence"] == "medium"
    ev = tickets.records(node.db, tid, "evidence")
    assert rep["evidence"][0]["source_id"] == ev[0]["record"] and ev[0]["data"]["result"]["residual_mm"]["mean"] > 5


def test_the_service_report_uses_only_the_tickets_records_and_refuses_an_ungrounded_summary(tmp_path):
    node = make_node(tmp_path)
    tid = tickets.open_ticket(node.db, node.name, "Pre-flight failed: calibration ended by an event.", "failed_preflight", OP)
    tickets.attach(node.db, tid, "finding", {"cause": "calibration_offset", "method": "eliminate", "narrowed_to_one": True}, ENG)
    built = report.build(node.db, tid)
    assert built["finding"]["cause"] == "calibration_offset" and any("no verification run" in x for x in built["not_verified"])
    assert report.problems("The residual fell to 0.31 mm and the repair was verified.", built) == [
        "the number 0.31 is not in the report", "the summary says something was verified, but the report has no verification run"]
    assert report.problems("The cause was calibration_offset. The repair has not been verified.", built) == []
    llm = ScriptedLLM([say("Fixed. Residual is now 0.2 mm, verified."), say("The finding is calibration_offset. Nothing was verified on a robot; "
                                                                           "no verification run is attached.")])
    with pytest.raises(PermissionError):
        report.close_with_report(node, who(OP), tid, llm)
    r = report.close_with_report(node, who(ENG), tid, llm)
    assert r["summary_source"] == "model" and r["summary_attempts"] == 2 and tickets.get(node.db, tid)["status"] == "closed"
    tid2 = tickets.open_ticket(node.db, node.name, "Another symptom of ten characters.", "operator", OP)
    r2 = report.close_with_report(node, who(ENG), tid2, ScriptedLLM([say("All verified at 9.9 mm.")]))
    assert r2["summary_source"].startswith("template") and r2["summary_attempts"] == 3 and "No finding" in r2["summary"]


# ------------------------------------------------------------------ support grants

def test_a_grant_is_for_one_ticket_for_a_while_and_is_checked_and_audited_at_the_site(tmp_path):
    node = make_node(tmp_path)
    sup = who("sup.wong")
    t1 = tickets.open_ticket(node.db, node.name, "Intermittent tracking stops in descent.", "operator", OP, {"n": 1})
    t2 = tickets.open_ticket(node.db, node.name, "Another matter that support must not see.", "operator", OP)
    with pytest.raises(grants.NotGranted):
        grants.read_ticket(node, sup, t1)
    for user in (ENG, OP, SAF_B):
        with pytest.raises(grants.NotGranted):
            grants.give(node.db, who(user), t1, "sup.wong", 2)
    g = grants.give(node.db, who(SAF), t1, "sup.wong", 2, now=1000.0)
    assert grants.read_ticket(node, sup, t1, now=1000.0 + 3600)["records"][0]["kind"] == "evidence"
    with pytest.raises(grants.NotGranted):
        grants.read_ticket(node, sup, t2, now=1000.0 + 3600)                              # another ticket, same grant
    with pytest.raises(grants.NotGranted) as e:
        grants.read_ticket(node, sup, t1, now=1000.0 + 2 * 3600 + 1)                      # after it has expired
    assert "expired" in str(e.value)
    g2 = grants.give(node.db, who(SAF), t1, "sup.wong", 2)
    assert grants.read_ticket(node, sup, t1)["access"].startswith("read-only")
    grants.revoke(node.db, who(SAF), g2["grant_id"])
    with pytest.raises(grants.NotGranted) as e:
        grants.read_ticket(node, sup, t1)
    assert "revoked" in str(e.value) and g["grant_id"] != g2["grant_id"]
    rows = node.db.query("SELECT allowed FROM audit WHERE action='grant:read_ticket' ORDER BY id")
    assert [r["allowed"] for r in rows] == [0, 1, 0, 0, 1, 0]
    with pytest.raises(grants.NotGranted):
        grants.give(node.db, who(SAF), t1, "sup.wong", 500)
