"""Edge deployment (task 14) on made-up data. What must not regress: the language model is never
called while an action is executing, a late or missing perception answer fails closed, the
gateway obeys only the agent process on this machine, nothing leaves the lab network, and each
missing part degrades the system instead of breaking it.
"""
from __future__ import annotations

import time

import httpx
import pytest

from console_bench import make_node, ready, who
from vxagent import chat
from vxagent.config import EgressBlocked, assert_egress_allowed
from vxagent.llm.base import LLMError, ToolCall
from vxagent.llm.ollama import OllamaClient
from vxagent.llm.router import ROUTE, Router, choose
from vxagent.llm.scripted import ScriptedLLM, say
from vxagent.perception.client import PerceptionClient, PerceptionUnavailable
from vxagent.perception.worker import Worker, WorkerServer
from vxagent.robot.backend import BackendError
from vxagent.robot.gateway import Gateway, GatewayBackend
from vxagent.robot.timing import TickMeter, model_calls_inside_actions
from vxagent.tools.executor import decide_approval, execute_tool


@pytest.fixture()
def worker():
    server = WorkerServer(Worker(scenes=[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]], source="made up", work_ms=5.0)).start()
    yield server
    try:
        server.stop()
    except OSError:
        pass


def run_trial(node, needle=None):
    op = who("op.lee")
    res = node.start_trial(op, 0, needle, use_llm=node.llm is not None)
    if res.status != "waiting":
        return res
    decide_approval(node.db, res.waiting_for["approval_id"], True, op.user_id)
    return node.resume_trial(res.run_id)


# ------------------------------------------------------------------ latency classes

def test_no_model_call_happens_while_an_action_is_executing(tmp_path):
    llm = ScriptedLLM([say("The trial ended with the machine verdict success.")])
    node = ready(make_node(tmp_path, llm=llm))
    meter = TickMeter(lambda dt: None)
    node.tick = meter
    res = run_trial(node)
    assert res.waiting_for["kind"] == "operator_confirmation"
    from vxagent.pipelines import policy_trial as PT
    PT.decide_verdict(node.db, res.run_id, "op.lee", "confirm")
    res = node.resume_trial(res.run_id)
    assert res.status == "finished"
    inside = model_calls_inside_actions(node.db, res.run_id)
    assert inside["actions"] >= 5 and inside["model_calls"] == len(llm.calls) >= 1 and inside["model_calls_inside_an_action"] == 0
    assert len(meter.stamps) > 50


def test_the_tick_meter_reports_period_and_jitter_and_leaves_out_pauses():
    m = TickMeter(lambda dt: None)
    t = [0.0]
    for gap in (0.1, 0.1, 0.12, 0.1, 5.0, 0.1, 0.1):
        t[0] += gap
        m.stamps.append(t[0])
        m.asked.append(0.1)
    rep = m.report(0.1)
    assert rep["periods"] == 5 and rep["max_ms"] == pytest.approx(120.0) and rep["worst_late_ms"] == pytest.approx(20.0)


# ------------------------------------------------------------------ perception as another process

def test_the_worker_queues_requests_and_a_late_answer_is_unavailable(worker):
    c = PerceptionClient(worker.url, time_limit_s=0.5)
    out = c()
    assert out["visible"] and out["computed"] is False and out["needle_pose"][:3] == [1.0, 2.0, 3.0] and out["latency_s"] < 0.5
    assert c.health()["reachable"] and c.health()["scenes"] == 1
    httpx.post(worker.url + "/control", json={"delay_ms": 700})                     # INJECTED slowness
    with pytest.raises(PerceptionUnavailable):
        c()
    httpx.post(worker.url + "/control", json={"delay_ms": 0})
    time.sleep(0.8)
    assert c()["visible"]
    worker.stop()
    with pytest.raises(PerceptionUnavailable):
        c()
    assert c.health()["reachable"] is False and c.failures == 2


def test_without_perception_pre_flight_fails_closed_and_nothing_moves(tmp_path, worker):
    node = ready(make_node(tmp_path, perception=PerceptionClient(worker.url)))
    assert all(r["ok"] for r in node.preflight_rows("op.lee"))
    worker.stop()
    res = node.start_trial(who("op.lee"), 0, use_llm=False)
    failed = res.state["data"]["preflight"]["failed"]
    assert res.state["data"]["outcome"] == "preflight_failed" and [f["repair"] for f in failed] == ["rig_check"]
    assert node.db.query("SELECT 1 FROM commands") == [] and node.health()["perception"]["reachable"] is False


def test_perception_lost_during_the_grasp_stops_the_stage_and_during_place_falls_back(tmp_path):
    for lose_at, expect in (("grasp", "perception_unavailable"), ("place", None)):
        node = ready(make_node(tmp_path / lose_at))
        calls = {"n": 0}

        def look(lose_at=lose_at, calls=calls):
            calls["n"] += 1
            stage = {1: "preflight", 2: "grasp", 3: "place"}.get(calls["n"], "later")
            if stage == lose_at:
                raise PerceptionUnavailable("the worker is gone")
            return {"visible": True, "latency_s": 0.01}
        node.perception = look
        res = run_trial(node)
        d = res.state["data"]
        if expect:
            assert d["grasp"]["reason"] == expect and "recover" in [h["node"] for h in res.state["history"]]
            assert d["verdict"]["machine"] != "success"
        else:
            assert d["place"]["needle_pose_source"].startswith("carried") and d["verdict"]["machine"] == "success"


# ------------------------------------------------------------------ the gateway

def test_the_gateway_obeys_only_the_agent_process_on_an_allowed_host(tmp_path):
    node = make_node(tmp_path)
    gw = Gateway(node.site.twin(), "agent-token").start()
    try:
        robot = GatewayBackend(gw.url, "agent-token", node.site)
        assert robot.name == "twin@gateway" and not robot.state().enabled
        robot.enable()
        robot.home()
        assert robot.state().ready and robot.clone().state().ready
        with pytest.raises(BackendError):
            GatewayBackend(gw.url, "another-token", node.site).enable()
        assert gw.refused[-1]["why"] == "not the agent process's token"
    finally:
        gw.stop()
    closed = Gateway(node.site.twin(), "agent-token", command_hosts=("10.0.0.5",), state_hosts=("10.0.0.5",)).start()
    try:
        assert httpx.get(closed.url + "/state").status_code == 403
        r = httpx.post(closed.url + "/command", json={"op": "enable"}, headers={"Authorization": "Bearer agent-token"})
        assert r.status_code == 403 and not closed.backend.state().enabled
        assert [x["why"] for x in closed.refused] == ["host not on the state allowlist", "host not on the command allowlist"]
    finally:
        closed.stop()


def test_a_node_runs_a_motion_through_the_gateway_like_through_the_twin(tmp_path):
    node = make_node(tmp_path)
    gw = Gateway(node.robot, "agent-token").start()
    try:
        remote = GatewayBackend(gw.url, "agent-token", node.site)
        remote.enable()
        remote.home()
        from vxagent.tools.registry import SYSTEM, ToolContext
        ctx = ToolContext(principal=SYSTEM, db=node.db, tracer=node.tracer, run_id="gw", robot=remote, site=node.site, tick=lambda dt: None)
        seq = execute_tool(node.registry, ToolCall("l", "get_robot_state", {}), ctx).content["state_seq"]
        call = ToolCall("m", "move_to_named_pose", {"named_pose": "hover_needle", "based_on_seq": seq})
        with pytest.raises(Exception):
            execute_tool(node.registry, call, ctx)
        row = node.db.one("SELECT id FROM approvals ORDER BY id DESC LIMIT 1")
        decide_approval(node.db, row["id"], True, "op.lee")
        res = execute_tool(node.registry, call, ctx)
        assert res.ok and res.content["final_error_mm"] < 0.01
        assert execute_tool(node.registry, ToolCall("s", "get_robot_state", {}), ctx).content["backend"] == "twin@gateway"
    finally:
        gw.stop()
    with pytest.raises(BackendError) as e:
        remote.state()
    assert e.value.kind == "robot_fault"


# ------------------------------------------------------------------ egress and model routing

def test_nothing_leaves_the_lab_network(monkeypatch):
    monkeypatch.setenv("VXAGENT_PROFILE", "onprem")
    monkeypatch.setenv("VXAGENT_ALLOWED_HOSTS", "perception.lab, fleet.lab")
    for ok in ("http://127.0.0.1:11434", "http://localhost:8300/x", "http://perception.lab:8401", "https://fleet.lab/v1/ingest"):
        assert_egress_allowed(ok)
    for bad in ("https://api.example.com/v1/chat", "http://perception.lab.evil.org", "http://192.168.1.20:11434"):
        with pytest.raises(EgressBlocked):
            assert_egress_allowed(bad)
    sent = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: sent.append(a))
    with pytest.raises(LLMError) as e:
        OllamaClient(base_url="https://llm.example.com").chat([{"role": "user", "content": "the lab's unpublished result"}])
    assert e.value.kind == "egress_blocked" and sent == []
    assert OllamaClient(base_url="https://llm.example.com").health()["reachable"] is False
    with pytest.raises(PerceptionUnavailable):
        PerceptionClient("http://gpu.example.com:8401")()
    assert sent == []


def test_the_route_of_a_step_is_fixed_in_code(tmp_path):
    large, small = ScriptedLLM([say('{"choice": "service"}')]), ScriptedLLM([say('{"choice": "operator"}')])
    r = Router(large, small)
    assert r.for_step("diagnose") is large and r.for_step("copilot") is large and r.for_step("choose_runbook") is small
    assert set(ROUTE.values()) == {"small", "large"}
    with pytest.raises(KeyError):
        r.for_step("something_new")
    out = choose(small, "Pick.", {"operator": "a", "service": "b"}, "move the arm")
    assert out["choice"] == "operator" and out["tokens"] > 0
    assert choose(ScriptedLLM([say("operator, I think")]), "Pick.", {"operator": "a"}, "x")["choice"] is None
    assert choose(ScriptedLLM([LLMError("unreachable", "down")]), "Pick.", {"operator": "a"}, "x")["choice"] is None


# ------------------------------------------------------------------ degraded modes and start-up

def test_with_the_model_server_down_a_trial_still_runs_and_its_summary_is_a_template(tmp_path):
    node = ready(make_node(tmp_path, llm=ScriptedLLM([LLMError("unreachable", "connection refused")])))
    res = run_trial(node)
    from vxagent.pipelines import policy_trial as PT
    PT.decide_verdict(node.db, res.run_id, "op.lee", "confirm")
    res = node.resume_trial(res.run_id)
    d = res.state["data"]
    assert res.status == "finished" and d["verdict"]["machine"] == "success" and d["summary_source"].startswith("template")
    assert chat.start(node, who("op.lee"), "How did it go?")["answer"] == chat.UNAVAILABLE
    rb = execute_tool(node.registry, ToolCall("r", "run_runbook", {"name": "rig_check", "parameters": {}}), node.ctx(who("op.lee"), "rb"))
    assert rb.ok and rb.content["status"] == "completed"


def test_without_the_retrieval_service_a_search_says_so_and_robot_questions_still_work(tmp_path, monkeypatch):
    monkeypatch.setenv("VXRAG_URL", "http://127.0.0.1:9")
    node = ready(make_node(tmp_path))
    ctx = node.ctx(who("op.lee"), "r")
    res = execute_tool(node.registry, ToolCall("s", "search_docs", {"query": "potentiometer calibration"}), ctx)
    assert not res.ok and "do not answer from memory" in (res.hint or "")
    assert execute_tool(node.registry, ToolCall("g", "get_robot_state", {}), ctx).ok


def test_after_a_restart_the_arm_is_disabled_and_a_moving_run_must_be_looked_at(tmp_path):
    from vxagent.store import Database
    db = Database(tmp_path / "node.sqlite")
    node = ready(make_node(tmp_path, db=db))
    res = node.start_trial(who("op.lee"), 0, use_llm=False)
    assert res.status == "waiting"
    db.execute("UPDATE runs SET status='running' WHERE run_id=?", (res.run_id,))       # the process died while it was moving
    again = make_node(tmp_path, db=db)                         # a new process on the same database
    assert again.needs_reverification == [res.run_id] and not again.robot.state().enabled
    assert again.health()["runs_needing_reverification"] == 1
