"""The agent core with a scripted model. These pin down what must never regress on a live
system: a motion based on an old look is refused, a motion is never retried or executed twice,
an approval is bound to the state it was given for, a rejected target is not clipped, and a
stopped run leaves nothing moving.
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest
from pydantic import BaseModel

from vxagent.agent.guards import grounding_check
from vxagent.agent.loop import AgentConfig, AgentState, run_agent, tool_messages
from vxagent.llm.base import LLMError, LLMResponse, ToolCall, Usage
from vxagent.llm.scripted import ScriptedLLM, call, say
from vxagent.robot.kinematics import pose_distance
from vxagent.tools.actions import ActionHandle, start_action
from vxagent.tools.executor import AwaitingApproval, decide_approval, execute_tool
from vxagent.tools.registry import Principal, Tool, ToolContext, ToolError, TransientToolError

CFG = AgentConfig(max_iterations=8)


def tc(name, cid="c1", **arguments):
    return ToolCall(id=cid, name=name, arguments=arguments)


def look(reg, ctx):
    return execute_tool(reg, tc("get_robot_state", "look"), ctx).content["state_seq"]


def approve_pending(ctx, approved=True):
    for row in ctx.db.query("SELECT id FROM approvals WHERE status='pending'"):
        decide_approval(ctx.db, row["id"], approved, "operator.test")


def execute_approved(reg, ctx, call_):
    """Execute a motion the way the console does: request, operator approves, execute."""
    with pytest.raises(AwaitingApproval):
        execute_tool(reg, call_, ctx)
    approve_pending(ctx)
    return execute_tool(reg, call_, ctx)


def run_with_approvals(llm, reg, ctx, task, cfg=CFG, approve=True, max_pauses=6):
    state = run_agent(llm, reg, ctx, task, cfg)
    pauses = 0
    while state.status == "awaiting_approval" and pauses < max_pauses:
        approve_pending(ctx, approve)
        state = run_agent(llm, reg, ctx, config=cfg, state=state)
        pauses += 1
    return state


def last_seq(messages):
    """The state_seq of the most recent tool observation, the way a model would read it."""
    for m in reversed(messages):
        if m["role"] == "tool":
            body = json.loads(m["content"])
            payload = body.get("result") or body.get("detail") or {}
            if isinstance(payload, dict) and "state_seq" in payload:
                return payload["state_seq"]
    raise AssertionError("no state_seq observed")


def then(name, **arguments):
    """A scripted step that quotes the state_seq of the latest observation."""
    def step(messages, tools):
        return call(name, based_on_seq=last_seq(messages), **arguments)
    return step


# ------------------------------------------------------------------------------ freshness

def test_motion_based_on_a_state_never_observed_is_refused(ready):
    reg, ctx, twin = ready
    res = execute_tool(reg, tc("move_to_named_pose", named_pose="left_edge", based_on_seq=twin.state().seq), ctx)
    assert not res.ok and res.error_kind == "stale_observation"


def test_motion_is_refused_when_the_robot_changed_after_the_look(ready):
    reg, ctx, twin = ready
    seq = look(reg, ctx)
    before = twin.state().measured_cp.copy()
    twin.perturb([0.02, 0, 0, 0, 0, 0])               # someone else moved the arm
    res = execute_tool(reg, tc("move_to_named_pose", named_pose="left_edge", based_on_seq=seq), ctx)
    assert not res.ok and res.error_kind == "stale_observation"
    assert str(twin.state().seq) not in res.observation()     # the current number is not handed back
    assert not np.allclose(twin.state().measured_cp, before)  # (only the perturbation moved it)
    assert ctx.db.query("SELECT * FROM commands") == []


def test_an_old_look_is_refused_even_if_nothing_changed(ready):
    reg, ctx, twin = ready
    seq = look(reg, ctx)
    ctx.observed[seq] -= ctx.freshness_limit_s + 1
    res = execute_tool(reg, tc("move_to_named_pose", named_pose="left_edge", based_on_seq=seq), ctx)
    assert not res.ok and res.error_kind == "stale_observation"


def test_agent_looks_again_after_a_stale_refusal(ready):
    reg, ctx, twin = ready

    def stale_then_fresh(messages, tools):
        twin.perturb([0.01, 0, 0, 0, 0, 0])           # the state changes between the look and the request
        return call("move_to_named_pose", named_pose="left_edge", based_on_seq=last_seq(messages))

    llm = ScriptedLLM([call("get_robot_state"), stale_then_fresh, call("get_robot_state"),
                       then("move_to_named_pose", named_pose="left_edge"), say("The arm is at left_edge.")])
    state = run_with_approvals(llm, reg, ctx, "go to left_edge")
    kinds = [t["error_kind"] for t in state.tool_log]
    assert kinds == [None, "stale_observation", None, None] and state.stop_reason == "final_answer"
    assert pose_distance(twin.state().measured_cp, ctx.site.poses["left_edge"].T)[0] < 1e-6


# ------------------------------------------------------------------------------ approval

def test_motion_pauses_for_the_operator_and_resumes(ready):
    reg, ctx, twin = ready
    llm = ScriptedLLM([call("get_robot_state"), then("move_to_named_pose", named_pose="hover_needle"),
                       say("Done.")])
    state = run_agent(llm, reg, ctx, "go to hover_needle", CFG)
    assert state.status == "awaiting_approval" and state.pending_approval["tool"] == "move_to_named_pose"
    assert pose_distance(twin.state().measured_cp, ctx.site.poses["start"].T)[0] < 1e-9     # nothing moved yet
    approve_pending(ctx)
    state = run_agent(llm, reg, ctx, config=CFG, state=state)
    assert state.stop_reason == "final_answer"
    assert pose_distance(twin.state().measured_cp, ctx.site.poses["hover_needle"].T)[0] < 1e-6
    cmd = ctx.db.one("SELECT * FROM commands")
    assert cmd["status"] == "done" and cmd["seq_after"] > cmd["seq_before"]


def test_rejected_approval_moves_nothing(ready):
    reg, ctx, twin = ready
    llm = ScriptedLLM([call("get_robot_state"), then("move_to_named_pose", named_pose="hover_needle"),
                       say("The operator did not approve the move.")])
    state = run_with_approvals(llm, reg, ctx, "go to hover_needle", approve=False)
    assert state.tool_log[-1]["error_kind"] == "rejected"
    assert pose_distance(twin.state().measured_cp, ctx.site.poses["start"].T)[0] < 1e-9


def test_approval_is_void_when_the_state_changed_before_it_was_used(ready):
    reg, ctx, twin = ready
    seq = look(reg, ctx)
    move = tc("move_to_named_pose", "m1", named_pose="left_edge", based_on_seq=seq)
    with pytest.raises(AwaitingApproval):
        execute_tool(reg, move, ctx)
    approve_pending(ctx)
    twin.perturb([0.01, 0, 0, 0, 0, 0])
    res = execute_tool(reg, move, ctx)
    assert not res.ok and res.error_kind == "stale_observation"


def test_approval_expires_after_its_time_limit(ready):
    reg, ctx, twin = ready
    seq = look(reg, ctx)
    move = tc("move_to_named_pose", "m1", named_pose="left_edge", based_on_seq=seq)
    with pytest.raises(AwaitingApproval):
        execute_tool(reg, move, ctx)
    approve_pending(ctx)
    ctx.db.execute("UPDATE approvals SET decided = decided - ?", (ctx.approval_ttl_s + 1,))
    res = execute_tool(reg, move, ctx)
    assert not res.ok and res.error_kind == "approval_expired"
    with pytest.raises(AwaitingApproval):             # asking again needs a new decision
        execute_tool(reg, move, ctx)


# ------------------------------------------------------------------------------ never twice

def test_the_same_motion_request_is_not_executed_twice(ready):
    reg, ctx, twin = ready
    seq = look(reg, ctx)
    move = tc("set_jaw", "j1", jaw="closed", based_on_seq=seq)
    assert execute_approved(reg, ctx, move).ok
    ctx.observed[seq] = time.time()
    twin._seq = seq                                   # even with the state number forced back
    res = execute_tool(reg, move, ctx)
    assert not res.ok and res.error_kind in ("duplicate_command", "approval_expired", "stale_observation")
    assert len(ctx.db.query("SELECT * FROM commands")) == 1


def test_registry_refuses_an_actuate_tool_that_could_be_retried_or_lacks_based_on_seq():
    class WithSeq(BaseModel):
        based_on_seq: int

    class NoSeq(BaseModel):
        x: int

    with pytest.raises(ValueError):
        Tool("bad", "d", NoSeq, lambda a, c: None, tier="actuate")
    with pytest.raises(ValueError):
        Tool("bad", "d", WithSeq, lambda a, c: None, tier="actuate", max_retries=1)


def test_reads_are_retried_and_motions_are_not(ready):
    reg, ctx, twin = ready
    counts = {"read": 0, "move": 0}

    class A(BaseModel):
        based_on_seq: int = 0

    def flaky_read(a, c):
        counts["read"] += 1
        if counts["read"] < 3:
            raise TransientToolError("archive 503")
        return {"ok": 1}

    def flaky_move(a, c):
        counts["move"] += 1
        raise TransientToolError("controller busy")

    reg.register(Tool("flaky_read", "d", A, flaky_read))
    reg.register(Tool("flaky_move", "d", A, flaky_move, tier="actuate", requires_approval=False))
    r = execute_tool(reg, tc("flaky_read"), ctx, sleep=lambda s: None)
    assert r.ok and r.attempts == 3
    m = execute_tool(reg, tc("flaky_move", "m", based_on_seq=look(reg, ctx)), ctx, sleep=lambda s: None)
    assert not m.ok and m.attempts == 1 and counts["move"] == 1


def test_two_calls_with_a_motion_in_one_response_are_all_refused(ready):
    reg, ctx, twin = ready
    seq = look(reg, ctx)
    both = LLMResponse(tool_calls=[ToolCall("a", "get_robot_state", {}),
                                   ToolCall("b", "move_to_named_pose", {"named_pose": "left_edge", "based_on_seq": seq})],
                       usage=Usage(10, 10))
    state = run_agent(ScriptedLLM([both, say("I could not do both at once.")]), reg, ctx, "look and move", CFG)
    assert [t["error_kind"] for t in state.tool_log] == ["not_parallel", "not_parallel"]
    assert ctx.db.query("SELECT * FROM approvals") == [] and state.stop_reason == "final_answer"


# ------------------------------------------------------------------------------ safety

def test_rejected_target_is_not_clipped_and_nothing_moves(ready):
    reg, ctx, twin = ready
    before = twin.state()
    res = execute_approved(reg, ctx, tc("move_to_named_pose", "m1", named_pose="retract", based_on_seq=look(reg, ctx)))
    assert not res.ok and res.error_kind == "safety_rejected"
    assert res.content["violations"][0]["rule"] == "workspace"
    assert np.allclose(twin.state().measured_cp, before.measured_cp) and not twin.state().busy


def test_three_safety_rejections_in_a_row_stop_the_run(ready):
    reg, ctx, twin = ready
    steps = []
    for name in ("retract", "pad_touch", "wrist_limit", "retract"):
        steps += [call("get_robot_state"), then("move_to_named_pose", named_pose=name)]
    state = run_with_approvals(ScriptedLLM(steps + [say("wrap up")]), reg, ctx, "try everything",
                               AgentConfig(max_iterations=20))
    assert state.stop_reason == "safety_rejections"
    assert [t["error_kind"] for t in state.tool_log if t["tool"] == "move_to_named_pose"] == ["safety_rejected"] * 3


def test_motion_aborts_when_the_arm_does_not_follow_and_reports_where_it_is(ready):
    reg, ctx, twin = ready
    twin.inject_dropped_commands()                    # INJECTED fault
    res = execute_approved(reg, ctx, tc("move_to_named_pose", "m1", named_pose="left_edge", based_on_seq=look(reg, ctx)))
    assert not res.ok and res.error_kind == "tracking_error" and res.attempts == 1
    assert "measured" in res.content and res.content["busy"] is False
    assert ctx.db.one("SELECT status FROM commands")["status"] == "tracking_error"


def test_robot_fault_stops_the_run(ready):
    reg, ctx, twin = ready

    def fault_then_move(messages, tools):
        seq = last_seq(messages)
        twin.inject_fault("amplifier fault")          # INJECTED fault; it also changes the state
        ctx.note_observed(twin.state().seq)           # pretend the model saw it, to reach the tool
        return call("move_to_named_pose", named_pose="left_edge", based_on_seq=twin.state().seq) if seq else None

    llm = ScriptedLLM([call("get_robot_state"), fault_then_move, say("The arm reports a fault.")])
    state = run_with_approvals(llm, reg, ctx, "go to left_edge")
    assert state.stop_reason == "robot_fault" and state.tool_log[-1]["error_kind"] == "robot_fault"


def test_arm_not_ready_is_an_observation_with_a_hint(env):
    reg, ctx, twin = env
    res = execute_approved(reg, ctx, tc("move_to_named_pose", "m1", named_pose="left_edge", based_on_seq=look(reg, ctx)))
    assert not res.ok and res.error_kind == "robot_not_ready" and "enable_and_home" in res.hint


def test_enable_and_home_then_move(env):
    reg, ctx, twin = env
    llm = ScriptedLLM([call("get_robot_state"), then("enable_and_home"),
                       then("move_to_named_pose", named_pose="pregrasp"), say("The arm is at pregrasp.")])
    state = run_with_approvals(llm, reg, ctx, "bring up and go to pregrasp")
    assert state.stop_reason == "final_answer" and twin.state().homed
    assert pose_distance(twin.state().measured_cp, ctx.site.poses["pregrasp"].T)[0] < 1e-6


# ------------------------------------------------------------------------------ actions

def test_operator_stop_cancels_a_running_motion(ready):
    reg, ctx, twin = ready
    ctx.tick = lambda dt: time.sleep(0.05)
    llm = ScriptedLLM([call("get_robot_state"), then("move_to_named_pose", named_pose="left_edge"), say("done")])
    state = run_agent(llm, reg, ctx, "go to left_edge", CFG)
    approve_pending(ctx)
    threading.Timer(0.12, ctx.stop_event.set).start()
    state = run_agent(llm, reg, ctx, config=CFG, state=state)
    assert state.stop_reason == "operator_stop" and "position_mm" in state.answer
    assert state.tool_log[-1]["error_kind"] == "operator_stop"     # the interrupted call is on record
    assert state.messages[-1]["role"] == "tool"
    assert all(h.done for h in ctx.actions.values()) and not twin.state().busy
    d_m, _ = pose_distance(twin.state().measured_cp, ctx.site.poses["left_edge"].T)
    assert 0.001 < d_m < 0.030                        # it stopped part of the way


def test_slow_action_is_returned_as_running_and_can_be_canceled(ready):
    reg, ctx, twin = ready
    ctx.tick = lambda dt: time.sleep(0.05)
    reg.get("move_to_named_pose").wait_s = 0.1
    res = execute_approved(reg, ctx, tc("move_to_named_pose", "m1", named_pose="left_edge", based_on_seq=look(reg, ctx)))
    assert res.ok and res.content["status"] == "running" and "action_id" in res.content
    out = execute_tool(reg, tc("cancel_action", "c", action_id=res.content["action_id"]), ctx)
    assert out.ok and out.content["status"] == "canceled" and not twin.state().busy
    assert "measured" in out.content


def test_wait_action_returns_the_final_result(ready):
    reg, ctx, twin = ready
    ctx.tick = lambda dt: time.sleep(0.02)
    reg.get("move_to_named_pose").wait_s = 0.05
    res = execute_approved(reg, ctx, tc("move_to_named_pose", "m1", named_pose="left_edge", based_on_seq=look(reg, ctx)))
    assert res.content["status"] == "running"
    out = execute_tool(reg, tc("wait_action", "w", action_id=res.content["action_id"]), ctx)
    assert out.ok and out.content["status"] == "succeeded" and out.content["final_error_mm"] < 0.01


def test_a_run_that_hits_a_limit_cancels_what_is_still_running(ready):
    reg, ctx, twin = ready
    ctx.tick = lambda dt: time.sleep(0.05)
    reg.get("move_to_named_pose").wait_s = 0.05
    llm = ScriptedLLM([call("get_robot_state"), then("move_to_named_pose", named_pose="left_edge"), say("wrap")])
    cfg = AgentConfig(max_iterations=2)
    state = run_with_approvals(llm, reg, ctx, "go", cfg)
    assert state.stop_reason == "max_iterations"
    assert all(h.status == "canceled" for h in ctx.actions.values()) and not twin.state().busy


def test_action_body_failure_does_not_hang_the_waiter():
    def body(h: ActionHandle):
        raise RuntimeError("boom")
    h = start_action("x", body)
    assert h.wait(2.0) and h.status == "aborted" and h.reason_kind == "failed"


# ------------------------------------------------------------------------------ authorization and arguments

def test_a_principal_without_the_actuate_tier_cannot_move(ready):
    reg, ctx, twin = ready
    ctx.principal = Principal("viewer", "operator", frozenset({"robot:read", "robot:move"}), site="site-a",
                              tiers=frozenset({"observe"}))
    res = execute_tool(reg, tc("move_to_named_pose", named_pose="left_edge", based_on_seq=look(reg, ctx)), ctx)
    assert not res.ok and res.error_kind == "unauthorized"
    assert "move_to_named_pose" not in {t.name for t in reg.allowed_for(ctx.principal)}
    assert ctx.db.one("SELECT allowed FROM audit ORDER BY id DESC")["allowed"] == 0


def test_unknown_tool_and_unknown_pose_and_free_coordinates(ready):
    reg, ctx, twin = ready
    assert execute_tool(reg, tc("teleport"), ctx).error_kind == "unknown_tool"
    seq = look(reg, ctx)
    r = execute_approved(reg, ctx, tc("move_to_named_pose", "m1", named_pose="over_there", based_on_seq=seq))
    assert r.error_kind == "invalid_arguments" and "hover_needle" in r.hint
    r = execute_tool(reg, tc("move_to_named_pose", "m2", named_pose="start", based_on_seq=seq, x_mm=12.0), ctx)
    assert r.error_kind == "invalid_arguments"        # there is no way to pass a coordinate


def test_a_result_over_the_size_limit_is_refused_not_truncated(ready):
    reg, ctx, twin = ready

    class A(BaseModel):
        pass

    reg.register(Tool("dump", "d", A, lambda a, c: {"rows": list(range(5000))}))
    res = execute_tool(reg, tc("dump"), ctx)
    assert not res.ok and res.error_kind == "result_too_large"
    json.loads(res.observation())                     # still valid JSON for the model


# ------------------------------------------------------------------------------ limits and the model

def test_infinite_loop_is_detected(ready):
    reg, ctx, twin = ready
    state = run_agent(ScriptedLLM([call("list_named_poses")]), reg, ctx, "loop", AgentConfig(max_iterations=20))
    assert state.stop_reason == "loop_detected"


def test_step_limit_gives_a_wrap_up_that_includes_the_state_read_by_code(ready):
    reg, ctx, twin = ready
    llm = ScriptedLLM([call("get_robot_state"), call("list_named_poses"), say("Stopped; the arm is at start.")])
    state = run_agent(llm, reg, ctx, "task", AgentConfig(max_iterations=2))
    assert state.stop_reason == "max_iterations" and state.answer
    assert "read by the system" in llm.calls[-1][-1]["content"]


def test_model_failure_ends_the_run_cleanly(ready):
    reg, ctx, twin = ready
    state = run_agent(ScriptedLLM([LLMError("unavailable", "down")]), reg, ctx, "task", CFG)
    assert state.status == "failed" and state.stop_reason == "llm_error"


def test_consecutive_tool_failures_stop_the_run(ready):
    reg, ctx, twin = ready
    steps = [call("episode_summary", episode_id=f"nope_{i}") for i in range(5)]
    state = run_agent(ScriptedLLM(steps), reg, ctx, "task", AgentConfig(max_iterations=10))
    assert state.stop_reason == "tool_failures"


# ------------------------------------------------------------------------------ grounding guard

def guard_state(answer, observations):
    msgs = [{"role": "tool", "name": n, "content": json.dumps(o)} for n, o in observations]
    return AgentState(run_id="r", task="t", messages=msgs, answer_candidate=answer)


STATE_OBS = ("get_robot_state", {"ok": True, "result": {"state_seq": 4, "enabled": True, "homed": False,
                                                       "measured": {"position_mm": [0.0, 0.0, -113.5]}}})
EP_OBS = ("episode_summary", {"ok": True, "result": {"duration_s": 119.916, "psm1_path_length_mm": 2212.4,
                                                     "psm1_peak_speed_mm_per_s": 103.1, "sample_rate_hz": 50.0}})


@pytest.mark.parametrize("answer,ok", [
    ("The episode lasted 119.9 s and PSM1 travelled 2212.4 mm.", True),
    ("The episode lasted 120 seconds.", True),
    ("Peak speed was 103.1 mm/s at 50 Hz.", True),
    ("The episode lasted 125 s.", False),
    ("PSM1 travelled 2300 mm.", False),
    ("Peak speed was 103.1 mm.", False),                 # right number, wrong unit
])
def test_guard_checks_numbers_with_their_unit(answer, ok):
    assert (not grounding_check(guard_state(answer, [EP_OBS]), None)) is ok


def test_guard_accepts_a_figure_the_calculator_produced():
    calc = ("calculator", {"ok": True, "result": {"expression": "2212.4/119.916", "value": 18.449581}})
    assert not grounding_check(guard_state("Mean speed was 18.4 mm/s.", [EP_OBS, calc]), None)
    assert grounding_check(guard_state("Mean speed was 18.4 mm/s.", [EP_OBS]), None)


def test_guard_checks_statements_about_the_robot_state():
    assert not grounding_check(guard_state("The arm is enabled.", [STATE_OBS]), None)
    assert grounding_check(guard_state("The arm is homed.", [STATE_OBS]), None)
    assert not grounding_check(guard_state("The arm is not homed.", [STATE_OBS]), None)
    assert grounding_check(guard_state("The arm is enabled.", []), None)       # never looked


def test_guard_reads_values_from_a_failed_motion_too():
    obs = ("move_to_named_pose", {"ok": False, "error_kind": "safety_rejected", "error": "x", "hint": "y",
                                  "detail": {"violations": [{"rule": "workspace", "value_mm": 60.0, "limit_mm": 40.0}]}})
    assert not grounding_check(guard_state("Rejected: 60 mm is outside the 40 mm workspace.", [obs]), None)


def test_guard_rejects_an_announcement_and_an_invented_citation():
    assert grounding_check(guard_state("I will now move the arm to start.", [STATE_OBS]), None)
    assert grounding_check(guard_state("See [doc:calibration#3].", [STATE_OBS]), None)


def test_ungrounded_answer_is_sent_back_then_flagged(ready):
    reg, ctx, twin = ready
    cfg = AgentConfig(max_iterations=8, answer_check=grounding_check)
    llm = ScriptedLLM([call("get_robot_state"), say("The tip is 500 mm above the pad.")])
    state = run_agent(llm, reg, ctx, "how high is the tip?", cfg)
    assert state.stop_reason == "final_answer_unverified" and "Not verified" in state.answer
    assert state.answer_repairs == 2 and len(tool_messages(state)) == 1


# ------------------------------------------------------------------------------ rehearsal and reads

def test_rehearsal_reports_without_moving(ready):
    reg, ctx, twin = ready
    before = twin.state()
    ok = execute_tool(reg, tc("rehearse_move", named_pose="pregrasp"), ctx).content
    bad = execute_tool(reg, tc("rehearse_move", "r2", named_pose="retract"), ctx).content
    assert ok["would_complete"] and ok["control_steps"] > 0 and ok["final_error_mm"] < 0.01
    assert not bad["would_complete"] and bad["violations"][0]["rule"] == "workspace"
    assert twin.state().seq == before.seq


def test_calculator_is_arithmetic_only(ready):
    reg, ctx, twin = ready
    assert execute_tool(reg, tc("calculator", expression="sqrt(15**2 + 10**2 + 8**2)"), ctx).content["value"] == 19.723083
    assert execute_tool(reg, tc("calculator", "c2", expression="__import__('os').getcwd()"), ctx).error_kind == "invalid_arguments"


def test_every_tool_call_is_traced_with_the_robot_sequence_number(ready):
    reg, ctx, twin = ready
    execute_approved(reg, ctx, tc("move_to_named_pose", "m1", named_pose="left_edge", based_on_seq=look(reg, ctx)))
    span = [s for s in ctx.tracer.events(ctx.run_id) if s["name"] == "move_to_named_pose"][-1]
    assert span["seq_end"] > span["seq_start"] and span["ok"] == 1
