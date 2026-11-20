"""Diagnosis and planning. What must not regress: a candidate cause is never dropped without a
check result against it, a diagnostic tool measures and does not judge, and a plan is refused
before anything runs when a reference has the wrong type, a need is not met, two motions
share a level, or the steps form a cycle.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from vxagent.agent.facts import observe_facts
from vxagent.agent.planner import Plan, execute_plan, levels, make_plan, rehearse, run_goal, validate_plan
from vxagent.agent.strategies import HypothesisState, apply_update, check_state, diagnose, initial_state
from vxagent.diagnostics.episode import CAUSES, make_contract, make_fault, run_episode
from vxagent.diagnostics.tools import CHECKS, DESCRIPTIONS
from vxagent.llm.base import LLMError, LLMResponse, Usage
from vxagent.llm.scripted import ScriptedLLM, say
from vxagent.robot.kinematics import pose_distance
from vxagent.tools.registry import Principal
from vxagent.tools.service import register_service, store_calibration_record

CANDS = {"a": "cause a", "b": "cause b", "c": "cause c"}
TOOLS = ["get_robot_state", "check_named_target", "enable_and_home", "move_to_named_pose", "set_jaw",
         "read_calibration_record", "calibration_residual"]


def stretch() -> np.ndarray:
    t = np.linspace(0, 1, 40)
    return np.stack([0.025 * t, 0.003 * np.sin(5 * t), 0.004 * t], axis=1)


def steps(*rows) -> Plan:
    return Plan.model_validate({"steps": [{"id": i, "purpose": "p", "tool": t, "arguments": a, "depends_on": d}
                                          for i, t, a, d in rows]})


def errors(plan, env, facts=None):
    reg, ctx, _ = env
    return validate_plan(plan, register_service_once(reg), ctx.principal, set(TOOLS), facts or observe_facts(ctx))


def register_service_once(reg):
    if reg.get("read_calibration_record") is None:
        register_service(reg)
    return reg


# ------------------------------------------------------------------------------ hypothesis state

def state(**status) -> HypothesisState:
    s = initial_state(CANDS)
    for c in s.candidates:
        st, against = status.get(c.cause, ("open", []))
        c.status, c.evidence_against = st, against
    return s


def test_a_candidate_cannot_be_dropped_without_evidence_against_it():
    s = state(b=("dropped", []))
    s.next_check.tool = "t1"
    problems = check_state(s, CANDS, {}, ["t1", "t2"], [], 2)
    assert any("dropped without evidence" in p for p in problems)
    s = state(b=("dropped", ["check:1:t1"]))
    s.next_check.tool = "t2"
    assert check_state(s, CANDS, {"check:1:t1": {}}, ["t1", "t2"], ["t1"], 1) == []


def test_evidence_must_be_a_check_result_of_this_run():
    s = state(b=("dropped", ["check:9:made_up"]))
    s.next_check.tool = "t2"
    assert any("is not a check result" in p for p in check_state(s, CANDS, {"check:1:t1": {}}, ["t1", "t2"], ["t1"], 1))


def test_no_conclusion_while_several_candidates_are_open_and_checks_remain():
    s = state()
    s.conclusion = "a"
    assert any("still open" in p for p in check_state(s, CANDS, {}, ["t1"], [], 1))
    assert check_state(s, CANDS, {}, ["t1"], ["t1"], 0) == []           # no checks left: the ranked list decides
    s = state(b=("dropped", ["r"]), c=("dropped", ["r"]))
    s.conclusion = "b"
    assert any("not an open candidate" in p for p in check_state(s, CANDS, {"r": 1}, ["t1"], [], 1))


def test_a_check_is_not_run_twice_and_candidates_cannot_be_added_or_lost():
    s = state()
    s.next_check.tool = "t1"
    assert any("already run" in p for p in check_state(s, CANDS, {}, ["t1", "t2"], ["t1"], 1))
    s.candidates.pop()
    assert any("exactly these" in p for p in check_state(s, CANDS, {}, ["t1", "t2"], [], 1))


def _update(drops=(), supports=(), ranking=(), next_check=None, conclusion=None) -> LLMResponse:
    return LLMResponse(content=json.dumps({"drops": [{"cause": c, "result": r} for c, r in drops],
                                           "supports": [{"cause": c, "result": r} for c, r in supports],
                                           "ranking": list(ranking), "next_check": next_check, "conclusion": conclusion}),
                       usage=Usage(100, 50))


def test_the_model_reports_changes_and_only_code_writes_the_state():
    s0 = initial_state(CANDS)
    s1 = apply_update(s0, {"drops": [{"cause": "a", "result": "r1"}], "supports": [{"cause": "c", "result": "r1"}],
                           "ranking": ["c", "b", "a", "zzz"], "next_check": "t2", "conclusion": None})
    assert [c.cause for c in s1.candidates] == ["c", "b", "a"] and [c.status for c in s1.candidates] == ["open", "open", "dropped"]
    assert s1.candidates[2].evidence_against == ["r1"] and s1.candidates[0].evidence_for == ["r1"]
    assert all(c.status == "open" for c in s0.candidates)             # the earlier state is not touched


def test_eliminate_runs_the_chosen_check_and_rejects_a_premature_conclusion():
    ran = []
    llm = ScriptedLLM([_update(conclusion="a"),                         # nothing checked, three candidates open
                       _update(next_check="t2"),
                       _update(drops=[("a", "r1"), ("c", "r1")], ranking=["b"], conclusion="b")])
    d = diagnose("eliminate", llm, "it stopped", CANDS, checks={"t1": "one", "t2": "two"},
                 run_check=lambda name: ran.append(name) or {"value": 1}, budget=2)
    assert d.cause == "b" and ran == ["t2"] and d.checks == ["t2"]
    assert d.detail["states_rejected_by_the_rules"] == 1 and d.detail["narrowed_to_one"]
    assert d.detail["dropped_with_evidence"] == {"a": ["r1 t2"], "c": ["r1 t2"]}
    assert "still open" in llm.calls[1][-1]["content"]


def test_eliminate_spends_its_budget_and_then_takes_the_top_of_its_ranking():
    llm = ScriptedLLM([_update(next_check="t1"), _update(ranking=["c", "a", "b"], next_check="t2"),
                       _update(ranking=["b", "c", "a"])])
    d = diagnose("eliminate", llm, "it stopped", CANDS, checks={"t1": "one", "t2": "two"},
                 run_check=lambda name: {"value": 1}, budget=2)
    assert d.checks == ["t1", "t2"] and d.cause == "b" and not d.detail["narrowed_to_one"]


def test_the_text_strategies_use_the_stated_number_of_calls():
    one = say(json.dumps({"reasoning": "r", "cause": "b"}))
    assert diagnose("direct", ScriptedLLM([say('{"cause": "a"}')]), "s", CANDS).calls == 1
    assert diagnose("cot", ScriptedLLM([one]), "s", CANDS).cause == "b"
    review = say(json.dumps({"weaknesses": ["w"], "holds_up": False}))
    r = diagnose("reflect", ScriptedLLM([one, review, say(json.dumps({"reasoning": "r", "cause": "c"}))]), "s", CANDS,
                 retrieve=lambda q: [{"id": "doc:x", "text": "t"}])
    assert (r.calls, r.cause, r.detail["first"], r.detail["changed_after_review"], r.detail["passages"]) == (3, "c", "b", True, ["doc:x"])
    pick = say(json.dumps({"why": "w", "line": 2}))
    t = diagnose("tot", ScriptedLLM([say(json.dumps({"reasoning": "r", "cause": "a"})), one, one, pick]), "s", CANDS)
    assert (t.calls, t.cause, t.detail["lines"]) == (4, "b", ["a", "b", "b"])
    assert diagnose("direct", ScriptedLLM([LLMError("unavailable", "down")]), "s", CANDS).error.startswith("unavailable")


# ------------------------------------------------------------------------------ episodes and measurements

@pytest.fixture()
def contract(site):
    return make_contract(site, [stretch()])


def test_a_fault_free_run_reaches_the_needle(site, contract):
    rec = run_episode(site, stretch(), contract, None, 1)
    assert rec["outcome"] == "completed" and rec["tip_to_grasp_point_mm"] < 1.0 and not rec["stops"]


@pytest.mark.parametrize("cause", list(CAUSES))
def test_each_injected_fault_leaves_its_signature_in_one_measurement(site, contract, cause):
    rec = run_episode(site, stretch(), contract, make_fault(cause, np.random.default_rng(3)), 2)
    m = {name: fn(rec) for name, fn in CHECKS.items()}
    assert rec["outcome"] != "completed" or rec["tip_to_grasp_point_mm"] > 2.0
    signature = {
        "calibration_offset": m["calibration_residual"]["residual_mm"]["mean"] > 4 and m["calibration_residual"]["residual_mm"]["spread"] < 1,
        "calibration_axis_flip": min(m["calibration_residual"]["share_of_points_on_the_same_side_of_the_centre_per_axis"]) < 0.7,
        "action_unit_scale": not 0.5 < m["policy_ranges"].get("ratio_sent_to_policy", 1.0) < 2,
        "start_outside_training_range": m["policy_ranges"]["largest_distance_outside_the_range_mm"] > 8,
        "joint_limit_tightened": m["safety_rejections"]["by_rule"].get("joint_margin") == 1,
        "tracking_error": (m["compare_poses"].get("setpoint_vs_measured_mm") or {}).get("max", 0) > rec["tracking_stop_mm"],
        "perception_dropout": m["perception_quality"]["share_of_steps_with_a_valid_detection"] < 0.5,
    }
    assert signature[cause]
    assert set(DESCRIPTIONS) == set(CHECKS)
    assert not any("verdict" in k or "cause" in k for k in json.dumps(m))        # numbers and limits, never a verdict


def test_the_same_record_gives_the_same_measurement(site, contract):
    rec = run_episode(site, stretch(), contract, make_fault("tracking_error", np.random.default_rng(1)), 5)
    assert all(fn(rec) == fn(rec) for fn in CHECKS.values())
    again = run_episode(site, stretch(), contract, make_fault("tracking_error", np.random.default_rng(1)), 5)
    assert again["symptom"] == rec["symptom"]


# ------------------------------------------------------------------------------ plan validation

def test_a_sound_plan_passes_and_is_levelled(ready):
    plan = steps(("s1", "read_calibration_record", {}, []),
                 ("s2", "calibration_residual", {"calibration_id": "$s1.calibration_id"}, ["s1"]),
                 ("s3", "move_to_named_pose", {"named_pose": "hover_needle"}, []))
    assert errors(plan, ready) == []
    assert [[s.id for s in lv] for lv in levels(plan)] == [["s1", "s3"], ["s2"]]


def test_a_reference_of_the_wrong_type_is_refused(ready):
    plan = steps(("s1", "read_calibration_record", {}, []),
                 ("s2", "calibration_residual", {"calibration_id": "$s1.valid"}, ["s1"]))
    assert any("is boolean but calibration_residual takes string" in e for e in errors(plan, ready))


def test_a_reference_needs_the_field_and_the_dependency(ready):
    plan = steps(("s1", "read_calibration_record", {}, []),
                 ("s2", "calibration_residual", {"calibration_id": "$s1.serial"}, ["s1"]))
    assert any("has no field 'serial'" in e for e in errors(plan, ready))
    plan = steps(("s1", "read_calibration_record", {}, []),
                 ("s2", "calibration_residual", {"calibration_id": "$s1.calibration_id"}, []))
    assert any("does not depend on s1" in e for e in errors(plan, ready))


def test_an_unmet_precondition_is_an_error_before_anything_runs(env):
    plan = steps(("s1", "move_to_named_pose", {"named_pose": "hover_needle"}, []))
    assert any("needs 'enabled', which is false" in e for e in errors(plan, env))
    plan = steps(("s1", "enable_and_home", {}, []), ("s2", "move_to_named_pose", {"named_pose": "hover_needle"}, ["s1"]))
    assert errors(plan, env) == []                     # the first step makes it true


def test_a_fact_that_cannot_be_observed_does_not_count_as_true(ready):
    reg, ctx, _ = ready
    facts = {**observe_facts(ctx), "powered": None}
    plan = steps(("s1", "check_named_target", {"named_pose": "start"}, []))
    assert any("not known from observation" in e for e in errors(plan, ready, facts))


def test_two_motions_in_one_level_a_cycle_and_an_unknown_tool_are_refused(ready):
    plan = steps(("s1", "move_to_named_pose", {"named_pose": "left_edge"}, []), ("s2", "set_jaw", {"jaw": "closed"}, []))
    assert any("could run at the same time" in e for e in errors(plan, ready))
    plan = steps(("s1", "get_robot_state", {}, ["s2"]), ("s2", "get_robot_state", {}, ["s1"]))
    assert "the dependencies form a cycle" in errors(plan, ready)
    plan = steps(("s1", "teleport", {}, []))
    assert any("no tool named 'teleport'" in e for e in errors(plan, ready))
    plan = steps(("s1", "move_to_named_pose", {"named_pose": "start", "based_on_seq": 3}, []))
    assert any("do not give based_on_seq" in e for e in errors(plan, ready))


def test_a_role_without_the_motion_tier_cannot_plan_a_motion(ready):
    reg, ctx, _ = ready
    viewer = Principal("eng", "service", frozenset({"robot:read"}), site="site-a", tiers=frozenset({"observe"}))
    plan = steps(("s1", "move_to_named_pose", {"named_pose": "start"}, []))
    assert any("may not use move_to_named_pose" in e for e in validate_plan(plan, reg, viewer, set(TOOLS), observe_facts(ctx)))


# ------------------------------------------------------------------------------ rehearsal and execution

def test_a_destination_the_filter_rejects_is_caught_in_rehearsal_and_nothing_moves(ready):
    reg, ctx, twin = ready
    seq = twin.state().seq
    out = rehearse(steps(("s1", "move_to_named_pose", {"named_pose": "retract"}, [])), register_service_once(reg), ctx)
    assert out["needed"] and not out["clean"] and out["steps"][0]["error_kind"] == "safety_rejected"
    assert twin.state().seq == seq


def test_execution_looks_before_each_motion_and_passes_results_along(ready, site):
    reg, ctx, twin = ready
    register_service_once(reg)
    store_calibration_record(ctx.db, site.name, run_episode(site, stretch(), make_contract(site, [stretch()]), None, 1)["calibration"])
    plan = steps(("s1", "read_calibration_record", {}, []),
                 ("s2", "calibration_residual", {"calibration_id": "$s1.calibration_id"}, ["s1"]),
                 ("s3", "move_to_named_pose", {"named_pose": "hover_needle"}, ["s2"]))
    run = execute_plan(plan, reg, ctx, set(TOOLS), "p1", on_approval=lambda wait: True)
    assert run.status == "completed" and run.approvals == 1
    assert run.results["s2"]["calibration_id"] == run.results["s1"]["calibration_id"]
    assert pose_distance(twin.state().measured_cp, site.poses["hover_needle"].T)[0] < 1e-6


def test_a_state_that_does_not_match_the_declared_effect_stops_the_plan(env):
    reg, ctx, twin = env
    twin.home = twin.state                              # INJECTED: homing silently does nothing
    plan = steps(("s1", "enable_and_home", {}, []), ("s2", "move_to_named_pose", {"named_pose": "hover_needle"}, ["s1"]))
    run = execute_plan(plan, reg, ctx, set(TOOLS), "p1", on_approval=lambda wait: True)
    assert run.status == "state_mismatch" and run.stopped_at == "s1" and '"homed": false' in run.why
    assert [s["id"] for s in run.steps] == ["s1"]


def _plan_reply(*rows) -> LLMResponse:
    return say(steps(*rows).model_dump_json())


def test_planning_errors_go_back_to_the_planner_and_a_hopeless_goal_ends_without_a_plan(env):
    reg, ctx, twin = env
    bad = _plan_reply(("s1", "move_to_named_pose", {"named_pose": "hover_needle"}, []))
    good = _plan_reply(("s1", "enable_and_home", {}, []), ("s2", "move_to_named_pose", {"named_pose": "hover_needle"}, ["s1"]))
    llm = ScriptedLLM([bad, good])
    planned = make_plan(llm, "go to hover_needle", reg, ctx, TOOLS, observe_facts(ctx))
    assert planned.ok and [a["stage"] for a in planned.attempts] == ["validation", "ok"]
    assert "needs 'enabled'" in llm.calls[1][-1]["content"]
    twin.inject_power(False)
    out = run_goal(ScriptedLLM([good, good, good, say("Nothing was executed: the arm is not powered.")]), "go", reg, ctx, TOOLS)
    assert out["status"] == "no_valid_plan" and not ctx.db.query("SELECT 1 FROM commands")


def test_one_replan_after_a_failed_motion_and_no_more(ready, site):
    reg, ctx, twin = ready
    twin.inject_dropped_commands(True)                  # INJECTED
    move = _plan_reply(("s1", "move_to_named_pose", {"named_pose": "left_edge"}, []))
    out = run_goal(ScriptedLLM([move, move, say("The arm did not reach left_edge.")]), "go to left_edge", reg, ctx, TOOLS,
                   on_approval=lambda wait: True)
    assert out["status"] == "step_failed" and out["replans"] == 1 and len(out["rounds"]) == 2
    assert out["rounds"][1]["execution"]["steps"][0]["error_kind"] == "tracking_error"


def test_the_answer_is_replaced_when_it_contains_a_number_no_step_returned(ready):
    reg, ctx, _ = ready
    look = _plan_reply(("s1", "get_robot_state", {}, []))
    out = run_goal(ScriptedLLM([look, say("The arm is 77.7 mm over the pad."), say("It is 77.7 mm over the pad.")]),
                   "where is the arm", reg, ctx, TOOLS)
    assert out["status"] == "completed" and out["answer_source"].startswith("plain statement")
    assert "77.7" not in out["answer"]
