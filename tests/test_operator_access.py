"""Operator access (task 11) on made-up data with a scripted model. What must not regress: the
site comes from the login and from nowhere else, a console operator can only move the arm inside
a run of their own and after their approval, no operator tool touches configuration, the stop
does not go through the model, and every answer carries its trace.
"""
from __future__ import annotations

import json

import pytest

from console_bench import PASSWORDS, client_for, make_console, make_node, ready, who
from vxagent import access, chat
from vxagent.llm.base import LLMError, ToolCall
from vxagent.llm.scripted import ScriptedLLM, call, say
from vxagent.service import tickets
from vxagent.store import Database
from vxagent.tools.executor import AwaitingApproval, execute_tool
from vxagent.web.app import route_table


def seq_then(tool: str, **args):
    """A scripted step: call `tool` with based_on_seq taken from the last look at the robot."""
    def step(messages, tools):
        seq = next(json.loads(m["content"])["result"]["state_seq"] for m in reversed(messages)
                   if m["role"] == "tool" and m.get("name") == "get_robot_state")
        return call(tool, based_on_seq=seq, **args)
    return step


# ------------------------------------------------------------------ login and sessions

def test_login_sets_a_signed_http_only_cookie_and_wrong_passwords_lock_the_account(tmp_path):
    con = make_console(tmp_path)
    c = client_for(con)
    assert c.get("/api/me").status_code == 401
    r = c.post("/api/login", json={"user_id": "op.lee", "password": PASSWORDS["op.lee"]})
    cookie = r.headers["set-cookie"].lower()
    assert r.status_code == 200 and "httponly" in cookie and "samesite=strict" in cookie
    assert r.json()["site"] == "site-a" and "password" not in r.text
    assert c.get("/api/me").json()["role"] == "operator"
    other = client_for(con)
    for _ in range(access.MAX_FAILURES):
        assert other.post("/api/login", json={"user_id": "eng.kim", "password": "nope"}).status_code == 401
    assert other.post("/api/login", json={"user_id": "eng.kim", "password": PASSWORDS["eng.kim"]}).status_code == 423
    assert other.post("/api/login", json={"user_id": "nobody", "password": "x"}).status_code == 401
    rows = con.db.query("SELECT allowed, detail FROM audit WHERE action='login'")
    assert sum(r["allowed"] for r in rows) == 1 and any(r["detail"] == "locked" for r in rows)


def test_a_forged_or_expired_session_is_not_a_session(tmp_path):
    con = make_console(tmp_path)
    c = client_for(con)
    good = access.sign("op.lee", con.secret)
    body, mac = good.rsplit(".", 1)
    for bad in (access.sign("saf.adler", "another-secret"), body + "." + "0" * len(mac), "garbage",
                access.sign("op.lee", con.secret, ttl_s=-1)):
        c.cookies.set("vx_session", bad)
        assert c.get("/api/me").status_code == 401
    c.cookies.set("vx_session", good)
    assert c.get("/api/me").status_code == 200


def test_role_and_site_are_read_from_the_user_table_on_every_request(tmp_path):
    con = make_console(tmp_path)
    c = client_for(con, "op.lee")
    assert c.get("/api/robot").json()["site"] == "site-a"
    con.db.execute("UPDATE users SET role='safety' WHERE user_id='op.lee'")      # the cookie is unchanged
    assert c.get("/api/approvals").status_code == 403
    assert c.get("/api/changes").status_code == 200


def test_every_route_answers_only_the_roles_it_declares(tmp_path):
    con = make_console(tmp_path)
    users = {"operator": "op.lee", "service": "eng.kim", "safety": "saf.adler", "support": "sup.wong"}
    anonymous = client_for(con)
    wrong = []
    for role, user in users.items():
        c = client_for(con, user)
        for r in route_table():
            path = r["path"].replace("{run_id}", "none").replace("{ticket_id}", "999").replace("{approval_id}", "999") \
                .replace("{proposal_id}", "999").replace("{grant_id}", "999").replace("{site}", "site-a")
            if path in ("/api/chat", "/api/stop") or path.endswith("/diagnose"):
                continue                                   # exercised by their own tests
            status = (c.get(path) if r["method"] == "GET" else c.post(path, json={})).status_code
            if (status == 403) != (role not in r["roles"]) and not (status == 403 and "tickets/site-a" in path):
                wrong.append((role, r["method"], path, status))
            assert (anonymous.get(path) if r["method"] == "GET" else anonymous.post(path, json={})).status_code == 401
    assert not wrong, wrong
    assert con.db.query("SELECT 1 FROM audit WHERE allowed=0 AND action LIKE 'http:%'")


def test_an_operator_sees_their_own_site_and_cannot_name_another(tmp_path):
    con = make_console(tmp_path)
    a, b = client_for(con, "op.lee"), client_for(con, "op.b")
    assert a.get("/api/robot").json()["site"] == "site-a" and b.get("/api/robot").json()["site"] == "site-b"
    assert a.get("/api/robot?site=site-b").json()["site"] == "site-a"             # a site in the request changes nothing
    for path in ("/api/tickets", "/api/config", "/api/changes", "/api/fleet/overview", "/api/grants", "/api/tickets/1"):
        assert a.get(path).status_code == 403, path


# ------------------------------------------------------------------ tiers and scopes in the executor

def test_no_operator_tool_reads_or_changes_configuration(tmp_path):
    node = make_node(tmp_path)
    op = who("op.lee")
    ctx = node.ctx(op, "run_x")
    offered = {t.name for t in node.registry.allowed_for(op, chat.config_for(op).tool_names)}
    for name, args in (("read_config", {}), ("describe_config_paths", {"target": "site"}), ("propose_calibration", {"reason": "because I want it"}),
                       ("fit_calibration", {}), ("read_change", {"proposal_id": 1}), ("list_tickets", {}), ("fleet_overview", {})):
        assert name not in offered
        res = execute_tool(node.registry, ToolCall("c", name, args), ctx)
        assert not res.ok and res.error_kind == "unauthorized", name
    refused = node.db.query("SELECT action FROM audit WHERE allowed=0 AND user_id='op.lee'")
    assert {"tool:read_config", "tool:propose_calibration"} <= {r["action"] for r in refused}


def test_actuate_is_held_per_run_and_ends_with_the_run(tmp_path):
    node = ready(make_node(tmp_path))
    op = who("op.lee")
    assert "actuate" not in op.tiers and op.per_run
    ctx = node.ctx(op, "run_1")
    look = execute_tool(node.registry, ToolCall("l", "get_robot_state", {}), ctx)
    move = ToolCall("m", "move_to_named_pose", {"named_pose": "hover_needle", "based_on_seq": look.content["state_seq"]})
    outside = execute_tool(node.registry, move, ctx)
    assert outside.error_kind == "unauthorized" and "inside a run" in outside.error
    access.begin_run(node.db, "run_1", "op.lee", node.name, ["move_to_named_pose"])
    with pytest.raises(AwaitingApproval):                    # inside the run it may be asked for, and then needs the approval
        execute_tool(node.registry, move, ctx)
    jaw = execute_tool(node.registry, ToolCall("j", "set_jaw", {"jaw": "closed", "based_on_seq": look.content["state_seq"]}), ctx)
    assert jaw.error_kind == "unauthorized" and "not started with set_jaw" in jaw.error
    other = node.ctx(who("op.b"), "run_1")
    other.note_observed(look.content["state_seq"])
    assert execute_tool(node.registry, move, other).error_kind == "unauthorized"       # another operator's run
    access.end_run(node.db, "run_1")
    after = execute_tool(node.registry, move, ctx)
    assert after.error_kind == "unauthorized" and "ended" in after.error
    assert node.db.query("SELECT 1 FROM commands") == []


def test_a_runbook_for_the_service_engineer_is_refused_to_an_operator(tmp_path):
    node = ready(make_node(tmp_path))
    op = who("op.lee")
    ctx = node.ctx(op, "run_rb")
    res = execute_tool(node.registry, ToolCall("r", "run_runbook", {"name": "collect_calibration_samples", "parameters": {}}), ctx)
    assert not res.ok and res.error_kind == "unauthorized"
    assert [r["name"] for r in node.library.index("operator")] == ["bring_up", "calibration_validity", "return_to_start", "rig_check",
                                                                   "verify_grasp_retention"]


# ------------------------------------------------------------------ the chat path

def test_an_answer_comes_back_with_its_trace(tmp_path):
    llm = ScriptedLLM([call("get_robot_state"), say("The arm is enabled and homed.")])
    node = ready(make_node(tmp_path, llm=llm))
    out = chat.start(node, who("op.lee"), "Is the arm ready?")
    assert out["status"] == "completed" and out["stop_reason"] == "final_answer"
    tr = out["trace"]
    assert tr["model_calls"] == 2 and tr["tool_calls"] == 1 and tr["tokens"] > 0 and tr["stop_reason"] == "final_answer"
    tool = next(s for s in tr["steps"] if s["kind"] == "tool")
    assert tool["name"] == "get_robot_state" and tool["output"]["enabled"] is True and "latency_ms" in tool
    assert "site-a" in llm.calls[0][0]["content"] and "rig_check" in llm.calls[0][0]["content"]
    assert node.fleet.backlog() == 1                         # one operational record, for the fleet


def test_a_motion_pauses_for_the_operators_card_and_only_an_operator_of_the_site_decides(tmp_path):
    llm = ScriptedLLM([call("get_robot_state"), seq_then("move_to_named_pose", named_pose="hover_needle"), call("get_robot_state"),
                       say("The arm is at hover_needle.")])
    node = ready(make_node(tmp_path, llm=llm))
    seq0 = node.robot.state().seq
    out = chat.start(node, who("op.lee"), "Move to hover_needle.")
    card = out["approval"]
    assert out["status"] == "awaiting_approval" and node.robot.state().seq == seq0
    assert card["tool"] == "move_to_named_pose" and card["to"]["named_pose"] == "hover_needle" and card["profile"] == "trial v1"
    assert card["rehearsal"]["would_complete"] is True and 0 < card["seconds_left"] <= chat.APPROVAL_WAIT_S
    assert card["bound_to"]["robot_state_seq"] == seq0 and card["from"]["state_seq"] == seq0
    for user in ("eng.kim", "saf.adler", "op.b"):
        with pytest.raises(chat.NotAllowed):
            chat.decide(node, who(user), card["approval_id"], True)
    assert node.robot.state().seq == seq0
    done = chat.decide(node, who("op.lee"), card["approval_id"], True)
    assert done["status"] == "completed" and done["decision"] == "approved" and node.robot.state().seq > seq0
    assert node.db.one("SELECT status FROM active_runs WHERE run_id=?", (done["run_id"],))["status"] == "ended"
    assert node.db.one("SELECT decided_by FROM approvals WHERE id=?", (card["approval_id"],))["decided_by"] == "op.lee"


def test_a_card_that_has_expired_does_not_move_the_arm(tmp_path):
    llm = ScriptedLLM([call("get_robot_state"), seq_then("move_to_named_pose", named_pose="hover_needle"), say("It was not approved.")])
    node = ready(make_node(tmp_path, llm=llm))
    out = chat.start(node, who("op.lee"), "Move to hover_needle.")
    node.db.execute("UPDATE approvals SET created=created-? WHERE id=?", (chat.APPROVAL_WAIT_S + 5, out["approval"]["approval_id"]))
    seq0 = node.robot.state().seq
    done = chat.decide(node, who("op.lee"), out["approval"]["approval_id"], True)
    assert done["decision"] == "expired" and node.robot.state().seq == seq0 and node.db.query("SELECT 1 FROM commands WHERE status='done'") == []


def test_stop_does_not_go_through_the_model(tmp_path):
    node = ready(make_node(tmp_path))

    def press_stop(messages, tools):
        node.stop_all("op.lee")                              # the console's button, pressed while the model is thinking
        return call("get_robot_state")
    node.llm = ScriptedLLM([press_stop, say("never reached")])
    out = chat.start(node, who("op.lee"), "Check the arm.")
    assert out["status"] == "stopped" and out["stop_reason"] == "operator_stop" and len(node.llm.calls) == 1
    assert node.db.query("SELECT 1 FROM audit WHERE action='stop'")


def test_without_a_model_the_copilot_says_so_and_the_console_still_reads_the_robot(tmp_path):
    node = ready(make_node(tmp_path, llm=None))
    out = chat.start(node, who("op.lee"), "Is the arm ready?")
    assert out["status"] == "failed" and out["answer"] == chat.UNAVAILABLE
    node.llm = ScriptedLLM([LLMError("unreachable", "connection refused")])
    assert chat.start(node, who("op.lee"), "Is the arm ready?")["answer"] == chat.UNAVAILABLE
    con = make_console(tmp_path / "c")
    assert client_for(con, "op.lee").get("/api/robot").status_code == 200


def test_a_ticket_is_opened_for_the_service_engineer_and_its_evidence_is_not_the_operators_to_read(tmp_path):
    llm = ScriptedLLM([call("trial_preflight"), call("request_service", symptom="Pre-flight fails: the arm is not enabled."),
                       say("I opened a ticket for the service engineer.")])
    node = make_node(tmp_path, llm=llm)                      # the arm is not enabled: pre-flight fails
    out = chat.start(node, who("op.lee"), "Can I start a trial? If not, ask the engineer.")
    pre = next(s for s in out["trace"]["steps"] if s["name"] == "trial_preflight")["output"]
    assert pre["can_start"] is False and pre["failed"][0]["repair_runbook"] == "bring_up"
    t = tickets.list_site(node.db, node.name)
    assert len(t) == 1 and t[0]["source"] == "failed_preflight" and t[0]["opened_by"] == "op.lee"
    assert tickets.records(node.db, t[0]["ticket_id"], "evidence")[0]["data"]["failed_checks"]
    res = execute_tool(node.registry, ToolCall("x", "read_ticket", {"ticket_id": t[0]["ticket_id"]}), node.ctx(who("op.lee"), "r2"))
    assert res.error_kind == "unauthorized"


def test_memory_is_per_operator_and_the_session_carries_the_last_turns(tmp_path):
    llm = ScriptedLLM([call("remember", text="Answer in millimetres."), say("Noted."), say("Hello again.")])
    node = make_node(tmp_path, llm=llm)
    chat.start(node, who("op.lee"), "Remember that I want millimetres.")
    chat.start(node, who("op.lee"), "Hello")
    system, *rest = llm.calls[-1]
    assert "Answer in millimetres." in system["content"]
    assert [m["content"] for m in rest[:2]] == ["Remember that I want millimetres.", "Noted."]
    other = ScriptedLLM([say("Hi.")])
    node.llm = other
    chat.start(node, who("op.b"), "Hello")
    assert "millimetres" not in other.calls[0][0]["content"] and len(other.calls[0]) == 2


def test_a_trace_is_only_shown_to_the_person_whose_run_it_is(tmp_path):
    con = make_console(tmp_path, llm=ScriptedLLM([say("Nothing to report.")]))
    a = client_for(con, "op.lee")
    run_id = a.post("/api/chat", json={"message": "Anything to report?"}).json()["run_id"]
    assert a.get(f"/api/runs/{run_id}").status_code == 200
    con.db.execute("INSERT INTO users(user_id,role,site,display_name,salt,pw_hash) SELECT 'op.two',role,site,display_name,salt,pw_hash "
                   "FROM users WHERE user_id='op.lee'")
    PASSWORDS["op.two"] = PASSWORDS["op.lee"]
    assert client_for(con, "op.two").get(f"/api/runs/{run_id}").status_code == 403


# ------------------------------------------------------------------ a trial from the console

def test_a_trial_runs_from_the_console_with_the_operators_go_and_confirmation(tmp_path):
    con = make_console(tmp_path)
    node = ready(con.nodes["site-a"])
    c = client_for(con, "op.lee")
    assert client_for(con, "eng.kim").post("/api/trials", json={}).status_code == 403
    t = c.post("/api/trials", json={}).json()
    assert t["waiting_for"]["kind"] == "approval" and node.db.query("SELECT 1 FROM commands") == []
    t = c.post(f"/api/trials/{t['run_id']}/go", json={"approve": True, "wait_s": 30}).json()
    assert t["waiting_for"]["kind"] == "operator_confirmation" and t["verdict"]["machine"] == "success"
    assert c.post(f"/api/trials/{t['run_id']}/verdict", json={"decision": "override", "to": "failed", "reason": "", "wait_s": 30}).json()["waiting_for"]["kind"] == "operator_confirmation"
    t = c.post(f"/api/trials/{t['run_id']}/verdict", json={"decision": "override", "to": "failed",
                                                          "reason": "The needle was not seated in the jaw.", "wait_s": 30}).json()
    assert t["status"] == "finished" and t["verdict"]["machine"] == "success" and t["operator_verdict"]["decision"] == "override"
    assert any(e["what"] == "approval_used" for e in t["timeline"]) and t["needle"] == "DECLARED"
    over = client_for(con, "saf.adler").get("/api/overrides").json()["overridden_verdicts"]
    assert over[0]["machine_verdict"] == "success" and over[0]["operator_verdict"] == "failed"
    assert node.fleet.all_records()[-1]["kind"] == "trial"


def test_a_trial_that_fails_pre_flight_moves_nothing_and_can_become_a_ticket(tmp_path):
    con = make_console(tmp_path)
    node = con.nodes["site-a"]                               # not enabled
    t = client_for(con, "op.lee").post("/api/trials", json={"open_ticket": True}).json()
    assert t["outcome"] == "preflight_failed" and t["repairs"] == ["bring_up"] and t["ticket_id"] == 1
    assert node.db.query("SELECT 1 FROM commands") == []
    assert node.fleet.all_records()[-1]["preflight_failed_by_check"] == {"arm": 1}


def test_the_event_stream_has_ids_and_resumes_after_the_last_one_seen(tmp_path):
    con = make_console(tmp_path)
    node = ready(con.nodes["site-a"])
    c = client_for(con, "op.lee")
    first = c.get("/api/events?once=1").text
    ids = [int(line[4:]) for line in first.splitlines() if line.startswith("id: ")]
    assert ids and ids == sorted(ids)
    node.bus.publish("robot", {"seq": 99})
    again = c.get("/api/events?once=1", headers={"Last-Event-ID": str(ids[-1])}).text
    assert [int(line[4:]) for line in again.splitlines() if line.startswith("id: ")] == [ids[-1] + 1]
    assert client_for(con, "sup.wong").get("/api/events?once=1").status_code == 403


def test_the_database_holds_no_password(tmp_path):
    con = make_console(tmp_path)
    blob = json.dumps(con.db.query("SELECT * FROM users"))
    assert all(pw not in blob for pw in PASSWORDS.values())
    assert isinstance(Database(":memory:"), Database)
