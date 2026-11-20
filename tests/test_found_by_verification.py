"""Faults that the verification on real data exposed in tasks 11 to 15, each pinned by a test
on made-up data so that it cannot come back unnoticed."""
from __future__ import annotations

import json

from console_bench import make_node, ready, who
from vxagent import chat
from vxagent.agent.guards import grounding_check
from vxagent.agent.loop import AgentState
from vxagent.llm.base import ToolCall
from vxagent.llm.scripted import ScriptedLLM, call, say
from vxagent.service import calibration as cal
from vxagent.tools.executor import execute_tool


def state_with(task: str, answer: str, tool: str = "", result: dict | None = None, extra_user: str = "") -> AgentState:
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": task}]
    if tool:
        messages.append({"role": "tool", "name": tool, "tool_call_id": "c", "content": json.dumps({"ok": True, "result": result})})
    if extra_user:
        messages.append({"role": "user", "content": extra_user})
    st = AgentState(run_id="r", task=task, messages=messages)
    st.answer_candidate = answer
    return st


def test_a_value_the_person_stated_in_the_request_may_be_repeated_but_not_one_from_the_guards_own_correction(tmp_path):
    ctx = make_node(tmp_path).ctx(who("eng.kim"), "r")
    asked = "Lower the near-step limit from 2 mm to 1.5 mm."
    assert grounding_check(state_with(asked, "A proposal to go from 2 mm to 1.5 mm was filed."), ctx) == []
    assert grounding_check(state_with(asked, "It is now 0.7 mm."), ctx)
    corrected = state_with("How far is it?", "It is 9.3 mm.", extra_user="You wrote 9.3 mm, but no tool returned that value.")
    assert grounding_check(corrected, ctx)                     # the correction does not make the number true


def test_a_number_can_be_checked_with_its_unit_when_the_key_that_carries_it_ends_in_the_unit(tmp_path):
    node = ready(make_node(tmp_path))
    ctx = node.ctx(who("eng.kim"), "r")
    res = execute_tool(node.registry, ToolCall("c", "read_config", {"target": node.site_target, "path": "safety_profile/near_distance_mm"}), ctx)
    assert res.ok and res.content["near_distance_mm"] == res.content["value"] == 20.0
    ok = state_with("What is it set to?", "It is set to 20.0 mm.", "read_config", res.content)
    assert grounding_check(ok, ctx) == []
    samples = [{"seen_m": [0.01 * i, 0.02 * (i % 3), 0.005 * (i % 4)], "touched_m": [0.01 * i + 0.006, 0.02 * (i % 3), 0.005 * (i % 4)]} for i in range(12)]
    cal.store_samples(node.db, node.name, samples, "eng.kim")
    assert execute_tool(node.registry, ToolCall("f", "fit_calibration", {}), ctx).ok
    cmp_ = execute_tool(node.registry, ToolCall("k", "compare_calibrations", {}), ctx)
    keys = set(cmp_.content["record_in_use"]) | set(cmp_.content["draft"]) | set(cmp_.content["difference"])
    assert all(k.endswith(("_mm", "_deg")) for k in keys if any(ch.isdigit() for ch in str(cmp_.content)) and ("mean" in k or "shift" in k or "rotation" in k))


def test_an_empty_reply_is_sent_back_and_is_never_an_answer(tmp_path):
    llm = ScriptedLLM([say(""), call("get_robot_state"), say("The arm is enabled and homed.")])
    node = ready(make_node(tmp_path, llm=llm))
    out = chat.start(node, who("op.lee"), "Is the arm ready?")
    assert out["status"] == "completed" and out["answer"].startswith("The arm") and len(llm.calls) == 3
    assert "Your reply was empty" in llm.calls[1][-1]["content"]
    node.llm = ScriptedLLM([say(""), say("  "), say("")])
    out = chat.start(node, who("op.lee"), "Is the arm ready?")
    assert out["status"] == "failed" and out["stop_reason"] == "empty_reply" and "empty reply" in out["answer"]


def test_a_failed_pre_flight_check_is_named_as_a_requirement_with_what_was_found(tmp_path):
    node = make_node(tmp_path)                                  # the arm is not enabled
    res = execute_tool(node.registry, ToolCall("p", "trial_preflight", {}), node.ctx(who("op.lee"), "r"))
    row = res.content["failed"][0]
    assert set(row) == {"problem", "repair_runbook"} and row["problem"].startswith("the arm is not ready") and "enabled False" in row["problem"]


def test_after_a_restart_the_site_in_force_is_the_stored_version_not_the_published_file(tmp_path):
    from test_service_access import patch
    from vxagent.service import changes, tickets
    node = make_node(tmp_path)
    tickets.open_ticket(node.db, node.name, "Steps near the pad are too large for the soft phantom.", "operator", "op.lee", {"n": 1})
    changes.decide(node, who("saf.adler"), changes.propose_patch(node, who("eng.kim"), patch(node, 1.0))["proposal_id"], True)
    assert str(node.site.profile.version) == "2"
    again = make_node(tmp_path)                                # a new process: the site file on disk is still the published one
    assert str(again.site.profile.version) == "1" and again.adopt_stored_site()
    assert str(again.site.profile.version) == "2" and abs(again.site.profile.near_step_m - 0.001) < 1e-12
    assert make_node(tmp_path / "fresh").adopt_stored_site() is False
