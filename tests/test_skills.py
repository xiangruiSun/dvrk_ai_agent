"""Runbooks as skills. What must not regress: a runbook's checks are run by code and cannot be
skipped, its inner motions still need the operator and the caller's own permissions, a changed
runbook has to reproduce its recorded outcomes, and a tool that a search did not return is not
offered.
"""
from __future__ import annotations

import json

import pytest

from runbook_bench import CASES, TRACES, make_bench, outcome, run_case
from vxagent.agent.loop import AgentConfig, run_agent
from vxagent.llm.base import LLMResponse, ToolCall, Usage
from vxagent.llm.scripted import ScriptedLLM, say
from vxagent.robot.kinematics import pose_distance
from vxagent.skills import runbook as rbk
from vxagent.tools.executor import AwaitingApproval, decide_approval, execute_tool
from vxagent.tools.registry import Principal


@pytest.mark.parametrize("name", list(CASES))
def test_every_runbook_reproduces_its_reference_trace(name):
    result, _, _ = run_case(name)
    recorded = json.loads((TRACES / f"{name}.json").read_text(encoding="utf-8"))
    assert outcome(result) == recorded["outcome"], (
        f"{recorded['runbook']} no longer reproduces its reference outcome. If the change was reviewed, record the "
        f"traces again with scripts/record_runbook_traces.py")


def test_a_runbook_with_a_check_removed_does_not_reproduce_its_trace():
    reg, ctx, _, library, _ = make_bench(at="pregrasp", needle={"slips_on_lift": True})
    text = library.get("verify_grasp_retention").text
    weakened = text.replace("      - {field: height_over_pad_mm, above_step: gripped.height_over_pad_mm, by: 5.0, why: "
                            '"the needle must have come up with the jaw; that the arm moved is not retention"}\n', "")
    weakened = weakened.replace("  - id: lifted\n    tool: observe_needle\n    checks:\n      - {field: held, equals: true}\n",
                                "  - id: lifted\n    tool: observe_needle\n    checks: []\n")
    assert weakened != text
    library.books["verify_grasp_retention"] = rbk.parse(weakened, reg, set(library.books))
    result = rbk.run(library, "verify_grasp_retention", {}, reg, ctx, "weak", on_approval=lambda w: True)
    recorded = json.loads((TRACES / "verify_grasp_retention__slips_on_lift.json").read_text(encoding="utf-8"))
    assert outcome(result) != recorded["outcome"]
    assert library.get("verify_grasp_retention").content_hash != recorded["content_hash"]


def test_the_recorded_hash_is_the_hash_of_the_file_as_it_is():
    _, _, _, library, _ = make_bench()
    for name, case in CASES.items():
        recorded = json.loads((TRACES / f"{name}.json").read_text(encoding="utf-8"))
        assert recorded["content_hash"] == library.get(case["runbook"]).content_hash, f"{case['runbook']} changed since its trace"


BASE = """name: demo
summary: s
when_to_use: w
tier: {tier}
version: "1"
requires: {requires}
{extra}steps:
{steps}"""


def book(steps: str, tier: str = "observe", requires: str = "[]", extra: str = "") -> str:
    return BASE.format(tier=tier, requires=requires, steps=steps, extra=extra)


@pytest.mark.parametrize("steps, kw, says", [
    ("  - {id: a, tool: teleport, source: doc:x}\n", {}, "no tool named 'teleport'"),
    ("  - {id: a, tool: get_robot_state, source: doc:x, checks: [{field: mood, equals: true}]}\n", {}, "returns no field 'mood'"),
    ("  - {id: a, tool: set_jaw, args: {jaw: open}, source: doc:x}\n", {}, "moves the arm but the runbook's tier is observe"),
    ("  - {id: a, tool: get_robot_state, source: doc:x}\n  - {id: b, tool: get_robot_state, source: doc:x, on_fail: 'goto:a'}\n",
     {}, "may only jump forward"),
    ("  - {id: a, tool: get_robot_state, source: doc:x, on_fail: recover}\n", {}, "names no recovery runbook"),
    ("  - {id: a, tool: get_robot_state, source: doc:x}\n", {"requires": "[levitating]"}, "is not a fact of the vocabulary"),
    ("  - {id: a, tool: check_named_target, args: {named_pose: $b.named_pose}, source: doc:x}\n", {}, "earlier step's result"),
    ("  - {id: a, tool: get_robot_state, source: doc:x, checks: [{field: enabled, equals: true, at_most: 1}]}\n", {}, "exactly one of"),
    ("  - {id: a, tool: get_robot_state}\n", {}, "source"),
])
def test_a_runbook_file_is_validated_when_it_is_loaded(steps, kw, says):
    reg, *_ = make_bench()
    with pytest.raises(rbk.RunbookError, match=says):
        rbk.parse(book(steps, **kw), reg, set())


def test_the_prose_form_states_every_check():
    _, _, _, library, _ = make_bench()
    prose = library.get("verify_grasp_retention").prose()
    assert "height_over_pad_mm is above gripped.height_over_pad_mm by at least 5" in prose
    assert prose.count("It must hold that") == 6 and "run procedure return_to_start" in prose


def test_a_runbook_does_nothing_when_its_requirements_are_not_met():
    reg, ctx, twin, library, _ = make_bench(ready=False)
    out = rbk.run(library, "return_to_start", {}, reg, ctx, "c1", on_approval=lambda w: True)
    assert out["status"] == "not_started" and out["unmet_requirements"] == ["enabled", "homed"] and out["steps"] == []
    assert not ctx.db.query("SELECT 1 FROM commands")


def test_an_inner_motion_waits_for_the_operator_and_the_run_goes_on_from_that_step():
    reg, ctx, twin, library, _ = make_bench(ready=False)
    with pytest.raises(AwaitingApproval) as wait:
        rbk.run(library, "bring_up", {}, reg, ctx, "c1")
    assert wait.value.tool == "enable_and_home" and not twin.state().enabled
    decide_approval(ctx.db, wait.value.approval_id, True, "op")
    with pytest.raises(AwaitingApproval) as wait:
        rbk.run(library, "bring_up", {}, reg, ctx, "c1")
    assert wait.value.tool == "move_to_named_pose" and twin.state().ready
    decide_approval(ctx.db, wait.value.approval_id, True, "op")
    out = rbk.run(library, "bring_up", {}, reg, ctx, "c1")
    assert out["status"] == "completed" and [s["id"] for s in out["steps"]] == ["look", "enable", "target", "to_start"]
    done = ctx.db.query("SELECT tool FROM commands WHERE status='done' ORDER BY created")
    assert [r["tool"] for r in done] == ["enable_and_home", "move_to_named_pose"]     # each executed once, not once per resume


def test_a_rejected_inner_motion_stops_the_runbook():
    reg, ctx, twin, library, _ = make_bench(ready=False)
    out = rbk.run(library, "bring_up", {}, reg, ctx, "c1", on_approval=lambda w: False)
    assert out["status"] == "stopped" and out["stopped_at"] == "enable" and out["steps"][-1]["error_kind"] == "rejected"
    assert not twin.state().enabled


def test_inner_steps_are_authorized_as_the_caller():
    reg, ctx, twin, library, _ = make_bench(at="left_edge", jaw="closed")
    ctx.principal = Principal("eng", "service", frozenset({"robot:read"}), site="site-a", tiers=frozenset({"observe", "simulate"}))
    out = rbk.run(library, "return_to_start", {}, reg, ctx, "c1", on_approval=lambda w: True)
    assert out["status"] == "stopped" and out["steps"][0]["error_kind"] == "unauthorized"
    assert not ctx.db.query("SELECT 1 FROM commands")


def test_failed_retention_runs_the_recovery_runbook_and_ends_at_start():
    result, twin, ctx = run_case("verify_grasp_retention__slips_on_lift")
    assert result["status"] == "recovered" and result["stopped_at"] == "lifted"
    assert [c["passed"] for c in result["steps"][-1]["checks"]] == [False, False]
    assert pose_distance(twin.state().measured_cp, ctx.site.poses["start"].T)[0] < 1e-6
    assert ctx.extras["needle"].log == ["the jaw closed on the needle", "the needle slipped out during the lift"]


def test_the_needle_is_only_taken_when_the_jaw_closes_at_the_grasp_pose():
    reg, ctx, twin, _, _ = make_bench(at="hover_needle")
    needle = ctx.extras["needle"]
    twin.begin_motion("j")
    twin.servo_jaw(ctx.site.jaw_angles["closed"], "j")
    twin.end_motion("j")
    assert not needle.held and needle.observe()["height_over_pad_mm"] == 0.0


# ------------------------------------------------------------------------------ the skill tools and the search

def test_find_tools_returns_runbooks_and_tools_and_offers_only_those():
    reg, ctx, _, _, cat = make_bench()
    names = [h["name"] for h in cat.search("check that the needle is held after the grasp")]
    assert names[0] == "verify_grasp_retention"
    assert "calibration_validity" in [h["name"] for h in cat.search("is the calibration still valid")[:3]]
    assert cat.search("xyzzy") == []
    res = execute_tool(reg, ToolCall("c1", "find_tools", {"query": "bring the arm up enable and home"}), ctx)
    assert res.ok and res.content["matches"][0]["name"] == "bring_up"
    assert "run_runbook" in ctx.extras["offered_tools"]


def test_load_runbook_returns_the_text_with_its_version_and_hash():
    reg, ctx, _, library, _ = make_bench()
    res = execute_tool(reg, ToolCall("c1", "load_runbook", {"name": "rig_check"}), ctx)
    assert res.ok and res.content["text"].startswith("name: rig_check") and res.content["content_hash"] == library.get("rig_check").content_hash
    assert execute_tool(reg, ToolCall("c2", "load_runbook", {"name": "nope"}), ctx).error_kind == "invalid_arguments"


def test_an_agent_runs_a_runbook_through_an_approval_and_gets_every_check_back():
    reg, ctx, twin, _, _ = make_bench(at="left_edge", jaw="closed")
    go = LLMResponse(tool_calls=[ToolCall("c1", "run_runbook", {"name": "return_to_start", "parameters": {}})], usage=Usage(1, 1))
    llm = ScriptedLLM([go, say("The arm is back at start.")])
    cfg = AgentConfig(tool_names=["run_runbook", "load_runbook"], max_iterations=4)
    state = run_agent(llm, reg, ctx, "put the arm back", cfg)
    approvals = 0
    while state.status == "awaiting_approval":
        decide_approval(ctx.db, state.pending_approval["approval_id"], True, "op")
        approvals += 1
        state = run_agent(llm, reg, ctx, config=cfg, state=state)
    assert state.status == "completed" and approvals == 2 and len(llm.calls) == 2
    out = ctx.extras["runbook_runs"][0]
    assert out["status"] == "completed" and out["checks_run"] == 2 and out["checks_failed"] == 0
    assert pose_distance(twin.state().measured_cp, ctx.site.poses["start"].T)[0] < 1e-6
    assert len(ctx.db.query("SELECT 1 FROM commands WHERE status='done'")) == 2


def test_with_dynamic_tools_only_what_the_search_returned_is_offered_next():
    reg, ctx, _, _, _ = make_bench()
    seen: list[list[str]] = []

    def step(messages, tools) -> LLMResponse:
        seen.append(sorted(t["function"]["name"] for t in tools or []))
        if len(seen) == 1:
            return LLMResponse(tool_calls=[ToolCall("c1", "find_tools", {"query": "is the calibration valid residual"})], usage=Usage(1, 1))
        if len(seen) == 2:
            return LLMResponse(tool_calls=[ToolCall("c2", "move_to_named_pose", {"named_pose": "start", "based_on_seq": 1})], usage=Usage(1, 1))
        return LLMResponse(content="done", usage=Usage(1, 1))

    cfg = AgentConfig(tool_names=["find_tools"], dynamic_tools=True, max_iterations=5)
    state = run_agent(ScriptedLLM([step]), reg, ctx, "check calibration", cfg)
    assert seen[0] == ["find_tools"]
    assert "calibration_residual" in seen[1] and "run_runbook" in seen[1] and "move_to_named_pose" not in seen[1]
    assert [t["error_kind"] for t in state.tool_log] == [None, "unknown_tool"]
    assert state.schema_chars > 0
