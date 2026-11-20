"""Memory. What must not regress: a site fact is written by code from a tool result and stops
being offered as current when an event ends it; one site never reads another's; a credential
is never stored; an old look at the robot is never replayed to the model as if it were now.
"""
from __future__ import annotations

import json

import pytest

from runbook_bench import make_bench
from vxagent.agent.loop import AgentConfig, run_agent
from vxagent.data.similar import FEATURES, EpisodeIndex
from vxagent.llm.base import LLMResponse, ToolCall, Usage
from vxagent.llm.scripted import ScriptedLLM, say
from vxagent.memory import site as site_memory
from vxagent.memory import stores
from vxagent.memory.working import fit_context
from vxagent.profiles import principal
from vxagent.store import Database
from vxagent.tools.executor import decide_approval, execute_tool
from vxagent.tools.memory import fact_sink, register_memory

OP_A, OP_B = principal("op.a", "operator", "site-a"), principal("op.b", "operator", "site-b")
SUPPORT = principal("sup", "support", "")
RESIDUAL = {"calibration_id": "cal-1", "mean_mm": 0.31, "max_mm": 0.6, "acceptance_mm": 1.0, "within_acceptance": True}


# ------------------------------------------------------------------------------ site memory

def test_a_fact_is_written_from_a_tool_result_and_only_from_one_that_has_a_builder():
    db = Database(":memory:")
    fid = site_memory.record_from_result(db, "site-a", "calibration_residual", RESIDUAL, "calibration_residual in run r1")
    assert fid == 1 and site_memory.record_from_result(db, "site-a", "get_robot_state", {"enabled": True}, "x") is None
    assert site_memory.record_from_result(db, "site-a", "calibration_residual", "the residual is 0.1 mm", "x") is None
    f = site_memory.facts(db, OP_A, "site-a")["active"][0]
    assert f["value"]["mean_mm"] == 0.31 and f["source"] == "calibration_residual in run r1" and "0.31 mm mean" in f["text"]


def test_the_same_measurement_is_not_stored_twice_and_a_new_one_replaces_the_old():
    db = Database(":memory:")
    a = site_memory.record_from_result(db, "site-a", "calibration_residual", RESIDUAL, "r1")
    assert site_memory.record_from_result(db, "site-a", "calibration_residual", RESIDUAL, "r2") == a
    site_memory.record_from_result(db, "site-a", "calibration_residual", {**RESIDUAL, "mean_mm": 0.5}, "r3")
    active = site_memory.facts(db, OP_A, "site-a")["active"]
    assert len(active) == 1 and active[0]["value"]["mean_mm"] == 0.5


def test_an_event_expires_the_facts_it_ends_and_keeps_them_for_the_record():
    db = Database(":memory:")
    site_memory.record_from_result(db, "site-a", "calibration_residual", RESIDUAL, "r1")
    assert site_memory.record_event(db, "site-a", "software_updated", "eng")["facts_expired"] == []
    out = site_memory.record_event(db, "site-a", "ecm_moved", "eng", "camera arm repositioned")
    assert out["facts_expired"] == [1]
    f = site_memory.facts(db, OP_A, "site-a")
    assert f["active"] == [] and f["expired"][0]["expired_because"] == "ecm_moved"
    block = site_memory.context_block(db, OP_A, "site-a")
    assert "still valid" not in block and "EXPIRED" in block and "because of the event ecm_moved" in block
    with pytest.raises(ValueError):
        site_memory.record_event(db, "site-a", "moon_phase", "eng")


def test_site_memory_is_not_visible_to_another_site_or_to_support():
    db = Database(":memory:")
    site_memory.record_from_result(db, "site-a", "calibration_residual", RESIDUAL, "r1")
    for who in (OP_B, SUPPORT):
        with pytest.raises(site_memory.NotVisible):
            site_memory.facts(db, who, "site-a")
        assert site_memory.context_block(db, who, "site-a") == ""
    assert site_memory.facts(db, OP_B, "site-b") == {"active": [], "expired": []}


def test_the_executor_writes_the_fact_and_the_recall_tool_respects_the_site():
    reg, ctx, _, _, _ = make_bench()
    register_memory(reg)
    ctx.extras["on_result"] = fact_sink(ctx)
    rec = execute_tool(reg, ToolCall("c1", "read_calibration_record", {}), ctx)
    execute_tool(reg, ToolCall("c2", "calibration_residual", {"calibration_id": rec.content["calibration_id"]}), ctx)
    got = execute_tool(reg, ToolCall("c3", "recall_site_memory", {"subject": "calibration"}), ctx)
    assert got.ok and len(got.content["active"]) == 1 and got.content["active"][0]["source"].startswith("calibration_residual in run")
    ctx.principal = principal("sup", "support", "")
    ctx.principal = type(ctx.principal)("sup", "support", frozenset({"robot:read"}), site="")
    assert execute_tool(reg, ToolCall("c4", "recall_site_memory", {}), ctx).error_kind == "unauthorized"


# ------------------------------------------------------------------------------ user memory and sessions

def test_user_memory_is_deduplicated_bounded_and_per_user():
    db = Database(":memory:")
    assert stores.remember(db, "u1", "Give lengths in millimetres.")["stored"]
    assert not stores.remember(db, "u1", "give lengths in  MILLIMETRES")["stored"]
    assert stores.memories(db, "u2") == []
    for i in range(stores.MAX_MEMORIES):
        out = stores.remember(db, "u1", f"preference number {i}")
    assert len(stores.memories(db, "u1")) == stores.MAX_MEMORIES and out["dropped_to_make_room"] == "Give lengths in millimetres."
    assert "Give lengths in millimetres." not in stores.memories(db, "u1")


@pytest.mark.parametrize("text", ["my password is hunter2", "the API key for the console is abc", "token ghp_abcdefghijklmnop1234",
                                  "use QWxhZGRpbjpvcGVuIHNlc2FtZSBhbmQgbW9yZQ1234 to log in"])
def test_credentials_are_refused(text):
    db = Database(":memory:")
    with pytest.raises(stores.Refused):
        stores.remember(db, "u1", text)
    assert stores.memories(db, "u1") == []


def test_the_remember_tool_is_the_only_writer_and_reports_a_refusal():
    reg, ctx, _, _, _ = make_bench()
    register_memory(reg)
    ok = execute_tool(reg, ToolCall("c1", "remember", {"text": "Answer in short sentences."}), ctx)
    bad = execute_tool(reg, ToolCall("c2", "remember", {"text": "my password is hunter2"}), ctx)
    assert ok.ok and bad.error_kind == "refused" and stores.memories(ctx.db, ctx.principal.user_id) == ["Answer in short sentences."]


def test_sessions_and_memories_survive_a_restart(tmp_path):
    path = tmp_path / "m.sqlite"
    db = Database(path)
    stores.save_session(db, "u1", "s1", [{"role": "user", "content": "hello"}])
    stores.remember(db, "u1", "Prefers degrees.")
    site_memory.record_from_result(db, "site-a", "calibration_residual", RESIDUAL, "r1")
    db.close()
    again = Database(path)
    assert stores.load_session(again, "u1", "s1") == [{"role": "user", "content": "hello"}]
    assert stores.load_session(again, "u2", "s1") == [] and stores.list_sessions(again, "u1") == ["s1"]
    assert stores.memories(again, "u1") == ["Prefers degrees."]
    assert len(site_memory.facts(again, OP_A, "site-a")["active"]) == 1


# ------------------------------------------------------------------------------ working context

def look(seq: int) -> dict:
    return {"role": "tool", "name": "get_robot_state", "tool_call_id": f"c{seq}", "content": json.dumps({"state_seq": seq, "x": "y" * 300})}


def test_only_the_latest_look_at_the_robot_reaches_the_model():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}, look(3),
            {"role": "tool", "name": "calculator", "tool_call_id": "k", "content": "{\"value\": 4}"}, look(7)]
    out = fit_context(msgs, 100000, state_now='{"state_seq": 9}')
    assert "out of date" in out[3]["content"] and json.loads(out[5]["content"])["state_seq"] == 7
    assert out[4]["content"] == '{"value": 4}'
    assert out[2]["content"].startswith("For reference, the robot's state right now, read by the system")
    assert out[1]["content"] == "task" and len(out) == len(msgs) + 1
    assert json.loads(msgs[2]["content"])["state_seq"] == 3              # the history itself is not changed


def test_a_small_budget_shortens_old_observations_and_keeps_the_task():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "the task"}]
    for i in range(12):
        msgs.append({"role": "tool", "name": "episode_summary", "tool_call_id": str(i), "content": "z" * 800})
    out = fit_context(msgs, 600)
    assert out[1]["content"] == "the task" and out[2]["content"].startswith("[earlier tool episode_summary output shortened]")
    assert out[-1]["content"] == "z" * 800


def test_with_injection_the_model_is_shown_the_state_and_may_base_a_motion_on_it():
    reg, ctx, twin, _, _ = make_bench()
    seen = []

    def step(messages, tools) -> LLMResponse:
        shown = next(m["content"] for m in messages if "read by the system" in m["content"])
        seen.append(shown)
        if len(seen) == 1:
            seq = json.loads(shown.split("): ", 1)[1])["state_seq"]
            return LLMResponse(tool_calls=[ToolCall("c1", "set_jaw", {"jaw": "closed", "based_on_seq": seq})], usage=Usage(1, 1))
        return LLMResponse(content="The jaw is closed.", usage=Usage(1, 1))

    cfg = AgentConfig(tool_names=["set_jaw", "get_robot_state"], inject_robot_state=True, max_iterations=4)
    llm = ScriptedLLM([step])
    state = run_agent(llm, reg, ctx, "close the jaw", cfg)
    decide_approval(ctx.db, state.pending_approval["approval_id"], True, "op")
    state = run_agent(llm, reg, ctx, config=cfg, state=state)
    assert state.status == "completed" and state.tool_log[0]["ok"] and state.context_tokens_sent > 0
    assert len(seen) == 2 and all("right now, read by the system" in s for s in seen)
    assert not any("right now" in (m.get("content") or "") for m in state.messages)    # injected, never stored
    assert len(state.injected_states) == 2


# ------------------------------------------------------------------------------ similar episodes

def test_neighbours_come_with_the_features_that_made_them_near():
    table = {}
    for i, (name, base) in enumerate([("X01_Short_01", 40.0), ("X02_Short_01", 44.0), ("X01_Long_01", 200.0), ("X02_Long_01", 210.0)]):
        table[name] = {f: base * (1 + 0.01 * k) + i * 0.1 for k, f in enumerate(FEATURES)}
    index = EpisodeIndex(table=table)
    near = index.nearest("X01_Short_01", k=2)
    assert near[0]["episode_id"] == "X02_Short_01" and near[0]["distance"] < near[1]["distance"]
    assert len(near[0]["closest_in"]) == 3 and set(near[0]["closest_in"]) <= set(FEATURES)
    assert index.task("X01_Long_01") == "Long"
