"""Remote support and the fleet (task 13) on made-up data. What must not regress: nothing but
counts, kinds and timings reaches the outbox, a node that is offline loses nothing and delivers
nothing twice, and the support assistant has no tool that reads a site's work.
"""
from __future__ import annotations

import json

import pytest

from console_bench import client_for, make_console, make_node, ready, who
from vxagent import chat
from vxagent.fleet import redact
from vxagent.fleet.desk import FleetDesk
from vxagent.fleet.store import FleetStore
from vxagent.fleet.sync import DirectLink, FleetServer, HttpLink, sync
from vxagent.llm.base import ToolCall
from vxagent.llm.scripted import ScriptedLLM, call, say
from vxagent.service import tickets
from vxagent.store import Database
from vxagent.tools.executor import execute_tool


def chat_once(node, text="Is the arm ready?"):
    node.llm = ScriptedLLM([call("get_robot_state"), call("check_named_target", named_pose="pad_touch"), say("Done.")])
    return chat.start(node, who("op.lee"), text)


# ------------------------------------------------------------------ redaction

def test_a_record_for_the_fleet_holds_counts_kinds_and_timings_and_nothing_else(tmp_path):
    node = ready(make_node(tmp_path))
    out = chat_once(node, "SECRET-TASK-TEXT move near X01_Pea_on_a_Peg_01")
    assert out["status"] == "completed"
    rec = node.fleet.all_records()[-1]
    assert rec["kind"] == "run" and rec["tool_calls_by_tool"] == {"get_robot_state": 1, "check_named_target": 1}
    assert rec["llm_calls"] == 3 and rec["llm_ms"]["count"] == 3 and rec["role"] == "operator"
    blob = json.dumps(node.fleet.all_records())
    for leak in ("SECRET-TASK-TEXT", "X01_Pea", "Done.", "pad_touch", "op.lee", "position_mm", "named_pose"):
        assert leak not in blob
    assert redact.problems(rec, node.fleet.vocab) == []


def test_the_outbox_refuses_a_record_that_carries_a_payload(tmp_path):
    node = make_node(tmp_path)
    box = node.fleet
    bad = [{"task": "move to the needle"}, {"llm_calls": 1, "errors_by_kind": {"C:/lab/data/episode_7.csv": 1}},
           {"tool_calls_by_tool": {"get_robot_state": [0.1, 0.2, 0.3]}}, {"status": "the operator said the needle slipped"},
           {"llm_ms": {"p50": 3.0, "position_mm": 1.0}}, {"model": "qwen3:14b; answer follows: hello there"},
           {"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}}]
    for body in bad:
        assert box.add("run", body) is None, body
    assert box.backlog() == 0 and box.refused == len(bad)
    assert box.add("run", {"llm_calls": 2, "errors_by_kind": {"timeout": 1}, "model": "qwen3:14b", "status": "completed"}) == 1
    assert len(node.db.query("SELECT 1 FROM audit WHERE action='outbox:refused'")) == len(bad)


def test_an_unknown_kind_is_reported_as_other_not_as_its_text(tmp_path):
    node = make_node(tmp_path)
    assert redact.known("needle_slipped_while_dr_lee_watched", node.fleet.vocab) == "other"
    assert redact.known("safety_rejected", node.fleet.vocab) == "safety_rejected"


# ------------------------------------------------------------------ store and forward

def test_an_offline_node_keeps_its_records_and_delivers_each_once_in_order(tmp_path):
    node = ready(make_node(tmp_path))
    store = FleetStore()
    link = DirectLink(store)
    chat_once(node)
    assert sync(node.fleet, link)["delivered"] == 1
    link.up = False
    for _ in range(4):
        chat_once(node)                                      # the node goes on working without the fleet
    node.fleet.record_health(node.health())
    off = sync(node.fleet, link)
    assert off["link"] == "down" and off["delivered"] == 0 and off["backlog"] == 5
    link.up, link.lose_next_ack = True, True                 # the store gets a batch but the node never hears
    lost = sync(node.fleet, link, batch_size=2)
    assert lost["link"] == "down" and node.fleet.backlog() == 5 and len(store.records("site-a")) == 3
    again = sync(node.fleet, link, batch_size=2)
    assert again["link"] == "up" and again["backlog"] == 0 and again["duplicates_ignored_by_the_store"] == 2
    seqs = [r["seq"] for r in store.records("site-a")]
    assert seqs == [1, 2, 3, 4, 5, 6] and store.duplicates == 2 and store.out_of_order == 0
    assert store.ingest("site-a", [{"seq": 9, "kind": "run", "body": {}}]) == {"accepted": 0, "duplicates": 0, "out_of_order": 1, "ack": 6}


def test_the_fleet_store_over_http_takes_a_sites_records_only_with_that_sites_token(tmp_path, monkeypatch):
    monkeypatch.setenv("VXAGENT_PROFILE", "onprem")
    node = ready(make_node(tmp_path))
    chat_once(node)
    store = FleetStore()
    server = FleetServer(store, {"site-a": "token-a", "site-b": "token-b"}).start()
    try:
        assert sync(node.fleet, HttpLink(server.url, "token-b"))["link"] == "down"       # another site's token
        assert store.records() == []
        assert sync(node.fleet, HttpLink(server.url, "token-a"))["delivered"] == 1
    finally:
        server.stop()
    chat_once(node)
    down = sync(node.fleet, HttpLink(server.url, "token-a", timeout_s=0.5))              # the store is gone
    assert down["link"] == "down" and down["backlog"] == 1
    blocked = sync(node.fleet, HttpLink("http://fleet.example.org", "token-a"))
    assert blocked["link"] == "down" and blocked["backlog"] == 1


# ------------------------------------------------------------------ the fleet's view and the assistant

def fleet_of_two(tmp_path, llm=None):
    a, b = ready(make_node(tmp_path, "site-a")), make_node(tmp_path, "site-b")
    b.fleet.meta["software"] = "2.4.0"
    store = FleetStore()
    for n in (a, b):
        chat_once(n)
        n.fleet.record_health(n.health())
        sync(n.fleet, DirectLink(store))
    return a, b, store, FleetDesk(store, Database(":memory:"), llm=llm, edges={"site-a": a, "site-b": b})


def test_the_fleet_view_is_built_from_what_the_sites_sent(tmp_path):
    a, b, store, desk = fleet_of_two(tmp_path)
    ov = store.overview()
    assert [s["site"] for s in ov["sites"]] == ["site-a", "site-b"] and all(s["in_contact"] for s in ov["sites"])
    assert store.site_health("site-a")["health"]["robot_ready"] is True and store.site_health("site-b")["health"]["robot_ready"] is False
    drift = store.version_drift()
    assert drift["drift"] and set(drift["differs"]) == {"software"}
    stats = store.run_stats()["by_site"]
    assert stats["site-a"]["runs"] == 1 and stats["site-a"]["tool_calls"] == 2
    assert store.overview(now=ov["sites"][0]["last_contact_s_ago"] + 10 ** 10)["sites"][0]["in_contact"] is False


def test_the_support_assistant_has_no_tool_that_reads_a_sites_work(tmp_path):
    a, b, store, desk = fleet_of_two(tmp_path)
    sup = who("sup.wong")
    offered = {t.name for t in desk.registry.allowed_for(sup, chat.config_for(sup).tool_names)}
    assert offered == set(chat.config_for(sup).tool_names)
    ctx = desk.ctx(sup, "r1")
    for name, args in (("list_episodes", {}), ("episode_summary", {"episode_id": "X01_Pea_on_a_Peg_01"}), ("get_robot_state", {}),
                       ("read_config", {}), ("list_trials", {}), ("read_ticket", {"ticket_id": 1}), ("propose_calibration", {"reason": "let me in please"}),
                       ("move_to_named_pose", {"named_pose": "start", "based_on_seq": 1}), ("recall_site_memory", {})):
        res = execute_tool(desk.registry, ToolCall("c", name, args), ctx)
        assert not res.ok and res.error_kind in ("unauthorized", "failed"), name
        assert name not in offered
    ok = execute_tool(desk.registry, ToolCall("o", "ops_run_stats", {"site": "site-a"}), ctx)
    assert ok.ok and ok.content["by_site"]["site-a"]["runs"] == 1
    assert execute_tool(desk.registry, ToolCall("o2", "site_health", {"site": "site-z"}), ctx).error_kind == "invalid_arguments"
    assert [r["question"] for r in store.audit_rows()] == ["ops_run_stats", "site_health"]            # the fleet end of the audit


def test_a_granted_ticket_is_read_through_the_edge_node_and_only_while_the_grant_holds(tmp_path):
    from vxagent.fleet import grants
    a, b, store, desk = fleet_of_two(tmp_path)
    sup = who("sup.wong")
    tid = tickets.open_ticket(a.db, "site-a", "Tracking stops in descent since Tuesday.", "operator", "op.lee", {"n": 1})
    ctx = desk.ctx(sup, "r1")
    read = ToolCall("g", "read_granted_ticket", {"site": "site-a", "ticket_id": tid})
    assert execute_tool(desk.registry, read, ctx).error_kind == "unauthorized"
    g = grants.give(a.db, who("saf.adler"), tid, "sup.wong", 1)
    got = execute_tool(desk.registry, ToolCall("g2", "read_granted_ticket", {"site": "site-a", "ticket_id": tid}), ctx)
    assert got.ok and got.content["symptom"].startswith("Tracking stops")
    assert execute_tool(desk.registry, ToolCall("g3", "read_granted_ticket", {"site": "site-b", "ticket_id": tid}), ctx).error_kind == "unauthorized"
    grants.revoke(a.db, who("saf.adler"), g["grant_id"])
    assert execute_tool(desk.registry, ToolCall("g4", "read_granted_ticket", {"site": "site-a", "ticket_id": tid}), ctx).error_kind == "unauthorized"
    assert len(a.db.query("SELECT 1 FROM audit WHERE action='grant:read_ticket'")) == 3 and b.db.query(
        "SELECT 1 FROM audit WHERE action='grant:read_ticket' AND allowed=1") == []


def test_the_assistant_answers_from_counts(tmp_path):
    llm = ScriptedLLM([call("version_drift"), say("site-a and site-b run different software releases.")])
    a, b, store, desk = fleet_of_two(tmp_path, llm)
    out = chat.start(desk, who("sup.wong"), "Do the sites run the same software?")
    assert out["status"] == "completed" and out["trace"]["tool_calls"] == 1
    assert "site" not in llm.calls[0][0]["content"].split("Rules:")[0].lower() or "You are working at" not in llm.calls[0][0]["content"]


def test_fleet_pages_are_for_support_and_site_pages_are_not(tmp_path):
    con = make_console(tmp_path)
    for n in con.nodes.values():
        n.fleet.record_health(n.health())
        sync(n.fleet, DirectLink(con.desk.store))
    sup = client_for(con, "sup.wong")
    assert [s["site"] for s in sup.get("/api/fleet/overview").json()["sites"]] == ["site-a", "site-b"]
    assert sup.get("/api/fleet/sites/site-a").json()["health"]["backend"] == "twin"
    assert sup.get("/api/fleet/drift").status_code == 200
    for path in ("/api/robot", "/api/trials", "/api/config", "/api/episodes", "/api/tickets", "/api/changes", "/api/approvals"):
        assert sup.get(path).status_code == 403, path
    assert sup.get("/api/fleet/tickets/site-a/1").status_code == 403
    for user in ("op.lee", "eng.kim", "saf.adler"):
        assert client_for(con, user).get("/api/fleet/overview").status_code == 403
    assert len(con.desk.store.audit_rows()) == 4


def test_health_says_what_runs_without_saying_what_was_done(tmp_path):
    node = ready(make_node(tmp_path))
    h = node.health()
    assert h["robot"] == {"reachable": True, "ready": True, "fault": False} and h["perception"]["reachable"] and h["outbox_backlog"] == 0
    assert node.fleet.record_health(h) == 1 and node.fleet.all_records()[0]["robot_ready"] is True
    with pytest.raises(KeyError):
        _ = node.fleet.all_records()[0]["task"]
