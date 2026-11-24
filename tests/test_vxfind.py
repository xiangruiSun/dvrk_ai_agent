"""vxfind: the training-data finder. Hermetic: made-up tables, a fake site, a scripted model.
No search engine, no network beyond loopback, no downloaded data."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vxagent.llm.base import LLMResponse
from vxagent.tools.registry import Principal
from vxfind.agent import answer_problems, listed_ids, pipeline, render
from vxfind.catalog.parse import parse_row
from vxfind.eval import answers as ANS
from vxfind.eval import feedback as FB
from vxfind.eval import fresh as FR
from vxfind.eval import gate as GATE
from vxfind.eval import harness as H
from vxfind.eval import meta as META
from vxfind.eval import metrics as M
from vxfind.eval import queries as Q
from vxfind.eval.families import family_of, is_bimanual
from vxfind.mcp.access import AccessDenied, AccessStore
from vxfind.mcp.hub import HubSite, InjectedSite, RateLimited, RateLimiter, verdict
from vxfind.search.engine import Engine, Hit, SearchConfig

CARD = """---
license: apache-2.0
tags:
- LeRobot
---
This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description

Teleoperated demonstrations of an arm stacking three wooden cubes on a white table, recorded in our lab for the onboarding course.

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{"codebase_version": "v3.0", "robot_type": "so101_follower", "total_episodes": 120, "total_frames": 36000, "total_tasks": 1,
 "fps": 30, "features": {"action": {"dtype": "float32", "shape": [6], "names": ["shoulder_pan.pos", "gripper.pos"]},
 "observation.state": {"dtype": "float32", "shape": [6]},
 "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3]},
 "observation.images.top": {"dtype": "video", "shape": [480, 640, 3]},
 "episode_index": {"dtype": "int64", "shape": [1]}}}
```
"""


def row(table_id="lab/stack_cubes", card=CARD):
    return parse_row(table_id, table_id.split("/")[0], "2026-10-01T00:00:00Z", "2026-09-01T00:00:00Z", 5, 1, ["LeRobot"], card)


# ---------------------------------------------------------------------------- catalog

def test_a_card_becomes_a_record_with_schema_and_a_persons_description():
    r = row()
    assert r["robot_type"] == "so101_follower" and r["fps"] == 30 and r["total_episodes"] == 120
    assert r["n_cameras"] == 2 and r["action_dim"] == 6 and r["schema_status"] == "ok"
    assert "stacking three wooden cubes" in r["description"] and r["has_description"]
    assert "120 episodes" in r["summary"] and "wrist" in r["summary"]
    assert r["url"] == "https://huggingface.co/datasets/lab/stack_cubes"


def test_a_card_without_a_schema_block_is_kept_and_marked():
    r = row(card="---\ntags:\n- LeRobot\n---\nnothing here")
    assert r["schema_status"] == "absent" and r["robot_type"] is None and not r["has_description"]


def test_scratch_names_are_recognised():
    assert row("lab/test_123")["is_scratch"] and row("lab/record-tmp")["is_scratch"]
    assert not row("lab/stack_cubes")["is_scratch"]


def test_a_changed_card_changes_the_content_hash_and_only_a_schema_change_the_schema_hash():
    a, b = row(), row(card=CARD.replace("wooden cubes", "plastic cubes"))
    c = row(card=CARD.replace('"fps": 30', '"fps": 15'))
    assert a["content_hash"] != b["content_hash"] and a["schema_hash"] == b["schema_hash"]
    assert a["content_hash"] != c["content_hash"]


def test_robot_spellings_fall_into_families():
    assert family_of("so101_follower") == family_of("bi_so_follower") == family_of("so100") == "so_arm"
    assert family_of("Panda") == family_of("franka_panda") == "franka"
    assert family_of("ur5e") == "ur" and family_of("aloha") == "aloha" and family_of("lekiwi_client") == "lekiwi"
    assert family_of("mobileai_robot") is None and family_of(None) is None and family_of("course") is None
    assert is_bimanual("bi_so_follower") and is_bimanual("arx5_bimanual") and not is_bimanual("so101_follower")


# ---------------------------------------------------------------------------- metrics

def test_ranking_metrics_on_a_hand_worked_case():
    qrels = {"a": 3, "b": 2, "c": 0, "d": 1}
    ranked = ["c", "a", "x", "b"]
    assert M.precision_at(ranked, qrels, 4) == 0.5
    assert M.recall_at(ranked, qrels, 4) == 1.0 and M.recall_at(ranked, qrels, 2) == 0.5
    assert M.mrr_at(ranked, qrels) == 0.5 and M.success_at(ranked, qrels) == 1.0
    assert M.unjudged_at(ranked, qrels, 4) == 0.25
    ideal = 7 + 3 / M.math.log2(3) + 1 / M.math.log2(4)
    got = 7 / M.math.log2(3) + 3 / M.math.log2(5)
    assert M.ndcg_at(ranked, qrels, 10) == pytest.approx(got / ideal)


def test_a_query_with_nothing_relevant_has_no_recall_and_no_ndcg():
    assert M.recall_at(["a"], {"a": 1}) is None and M.ndcg_at(["a"], {"a": 0}) is None
    assert M.per_query({"q": ["a"]}, {"q": {"a": 0}}, "ndcg@10") == {}


def test_paired_comparison_sees_a_real_difference_and_not_noise():
    a = {f"q{i}": 0.5 for i in range(60)}
    better = {k: v + 0.2 for k, v in a.items()}
    same = {k: v + (0.01 if i % 2 else -0.01) for i, (k, v) in enumerate(a.items())}
    assert M.paired_bootstrap(a, better)["ci95"][0] > 0 and M.randomization_test(a, better) < 0.01
    assert M.randomization_test(a, same) > 0.3


def test_agreement_statistics():
    assert M.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert M.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == 0.0
    assert M.weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    assert M.weighted_kappa([0, 0, 3, 3], [0, 1, 2, 3]) > M.weighted_kappa([0, 0, 3, 3], [3, 3, 0, 0])
    assert M.kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0 and M.kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0


# ---------------------------------------------------------------------------- query sets

def rec(tid, robot, eps=100, fps=30.0, cams=("wrist", "top"), ver="v3.0"):
    return {"table_id": tid, "owner": tid.split("/")[0], "name": tid.split("/")[1], "robot_type": robot, "robot_family": family_of(robot),
            "total_episodes": eps, "fps": fps, "n_cameras": len(cams), "cameras": list(cams), "codebase_version": ver, "family": tid}


def test_attribute_grades_are_computed_from_the_record():
    spec = {"family": "franka", "constraints": {"min_episodes": 200, "camera": "wrist"}}
    assert Q.grade_attribute(spec, rec("a/x", "panda", eps=250)) == 3
    assert Q.grade_attribute(spec, rec("a/y", "franka", eps=50)) == 1
    assert Q.grade_attribute(spec, rec("a/z", "franka", eps=250, cams=("top",))) == 1
    assert Q.grade_attribute(spec, rec("a/w", "so101_follower", eps=250)) == 0
    assert Q.grade_attribute({"family": "so_arm", "constraints": {"bimanual": True}}, rec("a/b", "bi_so_follower")) == 3


def test_an_empty_table_is_never_a_match():
    """Found while calibrating the judge: tables with no episodes met every stated constraint."""
    spec = {"family": "openarm", "constraints": {"camera": "front", "format": "v3.0"}}
    assert Q.grade_attribute(spec, rec("a/full", "bi_openarm_follower", eps=40, cams=("front",))) == 3
    assert Q.grade_attribute(spec, rec("a/empty", "bi_openarm_follower", eps=0, cams=("front",))) == 1
    assert Q.grade_attribute(spec, rec("a/none", "bi_openarm_follower", eps=None, cams=("front",))) == 1


def test_function_words_are_kept_out_of_the_match_against_names():
    """Found by the first search on the real index: "a Franka arm opening a drawer" returned a table named "a"."""
    from vxfind.search.engine import content_words
    assert content_words("Are there any datasets of a Franka arm opening a drawer or a cabinet door?") == "Franka arm opening drawer cabinet door"
    eng = Engine(es=object())
    q = eng._text_query("I need a table", SearchConfig(), [], [])
    fields = [list(m["match"])[0] for m in q["bool"]["must"][0]["dis_max"]["queries"]]
    assert "name" not in fields and "description" in fields              # nothing left to match against identifiers
    q = eng._text_query("a drawer", SearchConfig(), [], [])
    by_field = {list(m["match"])[0]: m["match"][list(m["match"])[0]]["query"] for m in q["bool"]["must"][0]["dis_max"]["queries"]}
    assert by_field["name"] == "drawer" and by_field["description"] == "a drawer"
    old = eng._text_query("a drawer", SearchConfig(ident_stop=False), [], [])
    assert {m["match"][list(m["match"])[0]]["query"] for m in old["bool"]["must"][0]["dis_max"]["queries"]} == {"a drawer"}


def test_a_written_need_that_names_its_table_is_caught():
    r = {"owner": "weply-labs", "name": "put_knife_in_tray_varied"}
    assert Q.leaks("data from weply-labs with knives", r) == "names the owner"
    assert Q.leaks("I want put knife in tray varied data", r) == "contains the table's name"
    assert Q.leaks("placing a kitchen knife into a cutlery tray with two grippers", r) is None
    assert Q.leaks("the pickplace2024b data", {"owner": "abcde", "name": "pickplace2024b_run"}) == "contains the identifier pickplace2024b"
    assert Q.leaks("the so101 arm picking cubes", {"owner": "abcde", "name": "so101_pick_run"}) is None      # a robot's name is not an identifier


def test_a_query_keeps_its_split():
    assert Q.split_of("C001") == Q.split_of("C001") and {Q.split_of(f"C{i:03d}") for i in range(40)} == {"dev", "test"}


def test_scoring_uses_each_sets_own_truth():
    recs = {"p/a": rec("p/a", "panda", eps=300), "p/b": rec("p/b", "panda", eps=10), "s/c": rec("s/c", "so100")}
    queries = [{"qid": "A1", "kind": "A", "need": "", "spec": {"family": "franka", "constraints": {"min_episodes": 200}}, "matching_tables": 1},
               {"qid": "B1", "kind": "B", "need": "", "target": "s/c", "target_family": "famC"},
               {"qid": "C1", "kind": "C", "need": ""}]
    run = {"A1": {"ids": ["p/a", "p/b", "s/c"], "families": ["f1", "f2", "f3"], "deep": []},
           "B1": {"ids": ["p/a", "x/sib"], "families": ["f1", "famC"], "deep": []},
           "C1": {"ids": ["p/b", "p/a"], "families": ["f2", "f1"], "deep": []}}
    pq = H.per_query(queries, run, recs, {"C1": {"p/a": 3, "p/b": 0}})
    assert pq["A"]["fit@10"]["A1"] == 0.1 and pq["A"]["robot_right@10"]["A1"] == 0.2
    assert pq["B"]["mrr@10"]["B1"] == 0.5 and pq["B"]["hit@1"]["B1"] == 0.0        # a near-identical sibling counts
    assert pq["C"]["mrr@10"]["C1"] == 0.5 and pq["C"]["unjudged@10"]["C1"] == 0.0
    assert H.repeat_at(["f", "f", "g", None, "f"]) == pytest.approx(2 / 5)


# ---------------------------------------------------------------------------- the live check

def fake_site(table: dict):
    calls = []

    def get(url):
        tid = url.split("/api/datasets/")[1]
        calls.append(tid)
        return table.get(tid, (401, {}, None))
    return get, calls


def test_the_site_check_tells_live_moved_and_gone_apart():
    get, calls = fake_site({"o/live": (200, {}, {"lastModified": "2026-11-01T00:00:00.000Z", "sha": "abc", "gated": False}),
                            "o/old": (307, {"location": "/api/datasets/n/new"}, None),
                            "n/new": (200, {}, {"lastModified": "2026-11-02T00:00:00.000Z", "sha": "def", "gated": "manual"}),
                            "o/busy": (429, {}, None)})
    s = HubSite(get=get, limiter=RateLimiter(100, 300))
    assert s.check("o/live")["state"] == "live" and s.check("o/live")["from_cache"] is True
    m = s.check("o/old")
    assert m["state"] == "moved" and m["moved_to"] == "n/new" and m["url"].endswith("/n/new") and m["gated"] == "manual"
    assert s.check("o/none")["state"] == "gone"
    busy = s.check("o/busy")
    assert busy["state"] == "unknown" and "limiting" in busy["reason"]
    s.check("o/busy")
    assert calls.count("o/busy") == 2 and calls.count("o/live") == 1          # a fact is remembered, a failure to ask is not


def test_the_budget_for_live_checks_is_kept():
    now = [0.0]
    lim = RateLimiter(calls=3, per_s=10, clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
    for _ in range(3):
        lim.acquire()
    with pytest.raises(RateLimited):
        lim.acquire()
    get, calls = fake_site({})
    s = HubSite(get=get, limiter=lim, clock=lambda: now[0])
    assert s.check("o/x")["state"] == "unknown" and calls == []               # over budget: not asked, and not called gone
    now[0] = 11.0
    assert s.check("o/x")["state"] == "gone" and lim.remaining() == 2
    lim2 = RateLimiter(calls=1, per_s=5, clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
    lim2.acquire()
    lim2.acquire(max_wait_s=6)                                                # waits for the window instead of failing
    assert lim2.waited_s >= 4


def test_changed_since_indexed_compares_dates():
    live = {"table_id": "o/t", "state": "live", "url": "u", "last_modified": "2026-11-20T10:00:00Z"}
    assert verdict("2026-10-01T00:00:00Z", live)["changed_since_indexed"] is True
    assert verdict("2026-11-20T10:00:00Z", live)["changed_since_indexed"] is False
    gone = verdict("2026-10-01T00:00:00Z", {"table_id": "o/t", "state": "gone", "url": "u"})
    assert gone["usable"] is False and gone["url"] is None


# ---------------------------------------------------------------------------- access

def test_hidden_owners_and_the_mentors_decision():
    a = AccessStore(restricted={"secretlab": ["team:core"]})
    assert a.hidden_owners({"team:manip"}) == ["secretlab"] and a.hidden_owners({"team:core"}) == []
    assert not a.may_see("secretlab/t", {"team:manip"}) and a.may_see("open/t", {"team:manip"})
    r1 = a.request("ana", "newcomer", "open/t", "train a first policy")
    r2 = a.request("ana", "newcomer", "open/t", "train a first policy")
    assert r1["created_now"] and not r2["created_now"] and r2["request_id"] == r1["request_id"]
    with pytest.raises(AccessDenied):
        a.decide("ana", "newcomer", r1["request_id"], True)
    mine = a.request("li", "mentor", "open/t", "my own")
    with pytest.raises(AccessDenied):
        a.decide("li", "mentor", mine["request_id"], True)
    assert a.decide("li", "mentor", r1["request_id"], True)["use"] == "approved"
    assert a.status("ana", "open/t")["use"] == "approved" and a.status("bo", "open/t")["use"] == "not_requested"
    assert [r["allowed"] for r in a.audit_rows() if r["action"] == "decide"] == [0, 0, 1]


def test_servers_answer_a_hidden_table_exactly_like_a_missing_one(tmp_path):
    import sqlite3

    from vxagent.mcp_layer.client import Connection
    from vxfind.catalog import store
    from vxfind.mcp import servers as S
    db = tmp_path / "catalog.db"
    con = sqlite3.connect(db)
    con.executescript(store.SCHEMA)
    for tid in ("open/stack", "secretlab/stack"):
        r = row(tid)
        vals = {**r, "columns_json": json.dumps(r["columns"]), "cameras_json": json.dumps(r["cameras"]), "gated": "no", "revision": "r",
                "family": "f-" + tid, "family_size": 1, "is_head": True}
        con.execute(f"INSERT INTO tables VALUES ({','.join('?' * len(store.COLUMNS))})", [vals.get(c) for c in store.COLUMNS])
    con.execute("INSERT INTO meta VALUES ('published', '2026-11-23')")
    con.commit()
    con.close()
    asked = []
    site = SimpleNamespace(check=lambda t: asked.append(t) or {"table_id": t, "state": "live", "url": "u", "last_modified": "2026-12-01T00:00:00Z"})
    desk = S.Desk(store.Catalog(db), AccessStore(restricted={"secretlab": ["team:core"]}), site)
    tok = desk.tokens.issue(Principal("ana", "newcomer", frozenset({"team:manip"})))
    running = S.start(desk, indexed=lambda t: "2026-10-01T00:00:00Z")
    conns = {n: Connection(n, r.url, tok).connect() for n, r in running.items()}
    try:
        tools = {t["name"]: t for c in conns.values() for t in c.list_tools()}
        assert set(tools) == {"table_details", "table_columns", "related_tables", "check_tables", "access_status", "request_access",
                              "pending_requests", "decide_request"}
        assert tools["check_tables"]["input_schema"]["properties"]["table_ids"]["maxItems"] == 10
        seen = conns["vx-catalog"].call_tool("table_details", {"table_id": "open/stack"})
        assert seen["ok"] and seen["result"]["total_episodes"] == 120 and seen["result"]["cameras"] == ["wrist", "top"]
        hidden = conns["vx-catalog"].call_tool("table_details", {"table_id": "secretlab/stack"})
        missing = conns["vx-catalog"].call_tool("table_details", {"table_id": "nobody/nothing"})
        assert hidden == missing and hidden["error_kind"] == "not_found"
        chk = conns["vx-site"].call_tool("check_tables", {"table_ids": ["open/stack", "secretlab/stack"]})["result"]
        assert chk[0]["state"] == "live" and chk[0]["changed_since_indexed"] is True
        assert chk[1]["state"] == "gone" and asked == ["open/stack"]           # the site is never asked about a hidden table
        # found by the leak test on real data: a hidden id and a made-up id must get the same reply from the live check too
        two = conns["vx-site"].call_tool("check_tables", {"table_ids": ["secretlab/stack", "nobody/nothing"]})["result"]
        assert {k: v for k, v in two[0].items() if k != "table_id"} == {k: v for k, v in two[1].items() if k != "table_id"}
        assert asked == ["open/stack"]                                         # and the site was asked about neither
        assert conns["vx-access"].call_tool("request_access", {"table_id": "secretlab/stack", "purpose": "a probe of a hidden table"})["error_kind"] == "not_found"
        assert conns["vx-access"].call_tool("decide_request", {"request_id": "req_x", "approve": True})["error_kind"] == "unauthorized"
        too_many = conns["vx-site"].call_tool("check_tables", {"table_ids": [f"o/t{i}" for i in range(11)]})
        assert too_many["ok"] is False
        with pytest.raises(Exception):
            Connection("vx-site", running["vx-site"].url, "not-a-token").connect()
    finally:
        for c in conns.values():
            c.close()
        S.stop(running)


# ---------------------------------------------------------------------------- the read path

class Scripted:
    model = "scripted"

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages, tools=None, json_schema=None, temperature=0.0, timeout_s=None, max_tokens=None):
        self.seen.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return LLMResponse(content=r if isinstance(r, str) else json.dumps(r))


def hits(*ids, fam=None):
    return [Hit(t, i, 1.0, {"summary": f"summary of {t}", "family": (fam or {}).get(t, t), "url": "https://huggingface.co/datasets/" + t}) for i, t in enumerate(ids, 1)]


def test_reciprocal_rank_fusion_and_collapse():
    lists = {"bm25": [{"_id": "a", "_source": {"family": "f1"}}, {"_id": "b", "_source": {"family": "f1"}}, {"_id": "c", "_source": {"family": "f2"}}],
             "knn": [{"_id": "c", "_source": {"family": "f2"}}, {"_id": "a", "_source": {"family": "f1"}}]}
    fused = Engine.fuse(lists, 60)
    assert [h.table_id for h in fused] == ["a", "c", "b"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62) and fused[0].ranks == {"bm25": 1, "knn": 2}
    assert [h.table_id for h in Engine.collapse(fused)] == ["a", "c"]


def test_the_model_can_reorder_and_choose_but_cannot_add_a_table():
    eng = Engine(es=object(), llm_small=Scripted([{"order": [3, 3, 9, 1]}]), llm_large=Scripted([{"chosen": [{"number": 2, "reason": "it fits"}, {"number": 7, "reason": "made up"}, {"number": 2, "reason": "again"}], "missing": ""}]))
    cfg = SearchConfig(rerank="small", rerank_depth=3)
    out, note = eng.rerank("need", hits("a", "b", "c", "d"), cfg)
    assert [h.table_id for h in out] == ["c", "a", "b", "d"] and note == ""     # repeated and out-of-range numbers dropped, the rest keep their place
    chosen, missing, note = eng.select("need", hits("a", "b", "c"), SearchConfig(select=True))
    assert [h.table_id for h in chosen] == ["b"] and chosen[0].reason == "it fits"
    broken = Engine(es=object(), llm_small=Scripted([RuntimeError("down")]))
    same, note = broken.rerank("need", hits("a", "b"), cfg)
    assert [h.table_id for h in same] == ["a", "b"] and "order unchanged" in note


def test_a_reading_becomes_filters_only_when_asked_to_be_strict():
    reading = {"keywords": "stack cubes", "robot": "franka", "min_episodes": 200, "fps": 30, "cameras": 2, "format": "v3.0"}
    f, b = Engine._constraints(reading, hard=True)
    assert len(f) == 4 and {"range": {"total_episodes": {"gte": 200}}} in f
    assert b == [{"match": {"robot_type.text": {"query": "franka", "operator": "or"}}}]     # a robot's name boosts, it never filters
    f, b = Engine._constraints(reading, hard=False)
    assert f == [] and len(b) == 5 and Engine._constraints(None, True) == ([], [])
    # found by measuring the reading stage: generic words must not become the robot
    assert Engine._constraints({"robot": "any robot"}, True) == ([], [])
    assert Engine._constraints({"robot": "AgileX Piper arm"}, True)[1][0]["match"]["robot_type.text"]["query"] == "AgileX Piper"


# ---------------------------------------------------------------------------- the answer

def test_addresses_are_read_from_an_answer():
    text = "1. https://huggingface.co/datasets/lab/stack_cubes (120 episodes).\n2. See huggingface.co/datasets/o-2/Pick.Place-v2, and lab/stack_cubes again https://huggingface.co/datasets/lab/stack_cubes"
    assert listed_ids(text) == ["lab/stack_cubes", "o-2/Pick.Place-v2"]


def test_twenty_candidates_fit_the_reply_the_executor_accepts():
    """Found by the agent cases: twenty full-length candidates were refused as too large."""
    from vxagent.tools.executor import MAX_RESULT_CHARS
    from vxfind.agent import cards
    long = [Hit(f"some-long-owner-name/a_rather_long_table_name_number_{i:03d}", i, 1.0,
                {"summary": "word " * 200, "total_episodes": 1234, "fps": 30.0, "robot_type": "bi_so101_follower",
                 "last_modified": "2026-08-21T03:56:43Z", "family_size": 12, "gated": "manual"}) for i in range(1, 21)]
    out = cards(long)
    assert len(out) == 20 and len(json.dumps({"tables": out, "note": "x" * 300})) < MAX_RESULT_CHARS
    assert len(cards(long[:10])[0]["summary"]) > len(out[0]["summary"])        # ten candidates keep longer summaries
    assert all(c["table_id"] == h.table_id for c, h in zip(out, long))         # ids are never cut


def state_ctx(answer, found, checked):
    return SimpleNamespace(answer_candidate=answer), SimpleNamespace(extras={"found": found, "checked": checked})


def test_an_answer_is_held_to_what_the_tools_returned():
    u = "https://huggingface.co/datasets/"
    found = {"a/ok": {}, "a/gone": {}, "a/old": {}, "a/unchecked": {}}
    checked = {"a/ok": {"state": "live"}, "a/gone": {"state": "gone"}, "a/old": {"state": "moved", "moved_to": "b/new", "url": u + "b/new"}}
    assert answer_problems(*state_ctx(f"{u}a/ok and {u}b/new", found, checked)) == []
    p = answer_problems(*state_ctx(f"{u}a/ok {u}a/gone {u}a/old {u}a/unchecked {u}z/invented", found, checked))
    assert len(p) == 4 and "is gone" in p[0] and "has moved" in p[1] and "not checked live" in p[2] and "not returned by any tool" in p[3]
    many = " ".join(f"{u}a/t{i}" for i in range(11))
    assert any("at most 10" in x for x in answer_problems(*state_ctx(many, {f"a/t{i}": {} for i in range(11)}, {f"a/t{i}": {"state": "live"} for i in range(11)})))


def test_the_fixed_pipeline_drops_gone_tables_and_follows_moved_ones():
    class FakeEngine:
        def search(self, need, cfg, visible=None):
            ids = [f"o/t{i}" for i in range(cfg.k)]
            return SimpleNamespace(hits=hits(*ids), notes=[], timings_ms={"total": 1.0})
    site = InjectedSite(SimpleNamespace(check=lambda t: {"table_id": t, "state": "live", "url": "https://huggingface.co/datasets/" + t, "last_modified": None}),
                        {"o/t0": {"state": "gone"}, "o/t1": {"state": "moved", "moved_to": "n/t1"}})
    check = lambda ids: [verdict(None, site.check(t)) for t in ids]
    res = pipeline(FakeEngine(), "need", check, [], SearchConfig(k=10))
    shown = [t["table_id"] for t in res["tables"]]
    assert "o/t0" not in shown and "n/t1" in shown and "o/t1" not in shown and len(shown) == 10        # refilled to ten
    assert res["dropped"] == [{"table_id": "o/t0", "why": "INJECTED"}]
    text = render(res)
    assert "moved from o/t1" in text and "o/t0" in text.split("Left out")[1] and "mentor" in text
    # found by reading a real answer: a moved table whose new address is also in the list was shown twice
    twice = InjectedSite(site, {"o/t3": {"state": "moved", "moved_to": "o/t5"}})
    res2 = pipeline(FakeEngine(), "need", lambda ids: [verdict(None, twice.check(t)) for t in ids], [], SearchConfig(k=10))
    ids2 = [t["table_id"] for t in res2["tables"]]
    assert ids2.count("o/t5") == 1 and len(ids2) == len(set(ids2)) == 10
    text2 = render(res2)
    assert "moved from o/t3" in text2 and "o/t5" not in text2.split("Left out")[1]      # a duplicate is not reported as gone
    unverified = pipeline(FakeEngine(), "need", check, [], SearchConfig(k=10), verify=False)
    assert "o/t0" in [t["table_id"] for t in unverified["tables"]]
    assert render({"tables": [], "notes": ["missing: no MRI data"], "dropped": []}).startswith("Nothing in the catalog fits")


# ---------------------------------------------------------------------------- evaluating the evaluation

def test_a_planted_fault_is_seen_and_an_order_only_fault_leaves_set_metrics_alone():
    recs = {f"p/{i}": rec(f"p/{i}", "panda", eps=300) for i in range(40)} | {f"s/{i}": rec(f"s/{i}", "so100") for i in range(200)}
    queries = [{"qid": f"A{i}", "kind": "A", "need": "", "spec": {"family": "franka", "constraints": {"min_episodes": 200}}, "matching_tables": 40} for i in range(30)]
    run = {q["qid"]: {"ids": [f"p/{(i + j) % 40}" for j in range(8)] + [f"s/{i}", f"s/{i + 1}"] + [f"s/{i + j}" for j in range(2, 12)],
                      "families": [None] * 20, "deep": []} for i, q in enumerate(queries)}
    base = H.per_query(queries, run, recs, {})
    half = H.per_query(queries, META.degraded(run, "half_random", [f"s/{i}" for i in range(200)]), recs, {})
    seen = META.detected(H.compare(base, half))
    assert seen["A"]["seen_as_a_loss"] and seen["A"]["difference"] < -0.2
    shuffled = H.per_query(queries, META.degraded(run, "shuffled_top10", list(recs)), recs, {})
    assert shuffled["A"]["fit@10"] == base["A"]["fit@10"]
    assert META.power(base, half)["A"]["queries"] == 30


def test_judge_calibration_counts():
    c = META.calibrate([3, 3, 1, 0, 0, 1], [3, 1, 2, 0, 0, 1])
    assert c["confusion"] == {"true_pos": 1, "false_pos": 1, "false_neg": 1, "true_neg": 3}
    assert c["agreement"] == pytest.approx(4 / 6, abs=1e-4) and c["judge_precision"] == 0.5 and c["judge_recall"] == 0.5
    assert META.set_agreement({"x": {"A": 0.9, "B": 0.5}, "y": {"A": 0.5, "B": 0.6}, "z": {"A": 0.1, "B": 0.9}})["tau_A_B"] == -1.0


def test_numbers_in_a_reason_must_come_from_the_table_or_the_request():
    src = {"summary": "Dataset x. Recorded: robot panda, 30 fps, 120 episodes, 36000 frames.", "total_episodes": 120, "fps": 30.0}
    assert ANS.unsupported_numbers("Has 120 episodes at 30 fps.", src) == []
    assert ANS.unsupported_numbers("Has 500 episodes.", src) == ["500"]
    assert ANS.unsupported_numbers("You asked for 200 episodes; this has 120.", src, need="at least 200 episodes") == []
    assert ANS.unsupported_numbers("36k frames", src) == []
    # found on real reasons: digits inside the table's own name are not a claim ("..._48h" -> "a 48-hour run")
    assert ANS.unsupported_numbers("Part of a 48-hour run.", {**src, "table_id": "lab/ur7_b_into_a_48h"}) == []
    assert ANS.unsupported_numbers("Part of a 72-hour run.", {**src, "table_id": "lab/ur7_b_into_a_48h"}) == ["72"]


def test_abstention_is_counted_per_query_set():
    qs = [{"qid": "D1", "kind": "D", "why_no_answer": "absent"}, {"qid": "D2", "kind": "D", "why_no_answer": "size"}, {"qid": "C1", "kind": "C"}]
    run = {"D1": {"ids": [], "notes": ["missing: no MRI data"]}, "D2": {"ids": ["a/b"], "notes": []}, "C1": {"ids": ["a/b", "a/c"], "notes": []}}
    out = ANS.abstention(run, qs)
    assert out["D"]["showed_nothing"] == 1 and out["D"]["showed_nothing_absent"] == "1/1" and out["D"]["showed_nothing_size"] == "0/1"
    assert out["C"]["share_showed_nothing"] == 0.0 and out["C"]["mean_tables_shown"] == 2.0


def test_dead_addresses_are_counted_before_the_check_and_refilled_after():
    run = {"q1": {"ids": ["a/1", "a/2", "a/3"]}, "q2": {"ids": ["a/3", "a/4"]}}
    answers = {"a/1": {"state": "gone"}, "a/2": {"state": "moved", "last_modified": "2026-11-01T00:00:00Z"},
               "a/3": {"state": "live", "last_modified": "2026-11-20T00:00:00Z"}}
    s = FR.staleness(run, answers, {"a/3": "2026-10-01T00:00:00Z", "a/2": "2026-11-01T00:00:00Z"}, k=2)
    assert s["addresses_shown"] == 4 and s["not_checked"] == 1 and s["dead"] == 1 and s["moved"] == 1 and s["changed_since_indexed"] == 1
    assert s["answers_with_a_dead_address"] == 1 and s["share_dead"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["after_live_check"]["answers_still_full_after_refill"] == 2


def test_feedback_becomes_candidate_cases_not_truth(tmp_path):
    log = FB.FeedbackLog(tmp_path / "fb.jsonl")
    log.add("s1", "cube stacking with a koch arm", "shown", shown=["a/1", "a/2"])
    log.add("s1", "cube stacking with a koch arm", "opened", table_id="a/1", rank=1)
    log.add("s2", "MRI scans", "shown", shown=["a/3"])
    log.add("s2", "MRI scans", "nothing_fits")
    log.add("s3", "Franka drawer opening", "shown", shown=["a/4", "a/5"])
    log.add("s3", "Franka drawer opening", "opened", table_id="a/5", rank=2)
    log.add("s3", "Franka drawer opening", "not_useful", table_id="a/4", rank=1)
    with pytest.raises(ValueError):
        log.add("s3", "x", "liked")
    ev = log.events()
    on = FB.online(ev)
    assert on["questions"] == 3 and on["answers_with_nothing_opened"] == 1 and on["mean_rank_of_first_opened"] == 1.5
    cands = FB.triage(ev, known_needs={"franka drawer opening"})
    assert [c["need"] for c in cands] == ["MRI scans"] and cands[0]["status"].startswith("candidate")
    assert len(FB.triage(ev, set())) == 2


def test_the_gate_stops_a_loss_and_an_unmeasured_floor():
    rules = {"alpha": 0.05, "no_loss": {"A": {"metric": "fit@10", "max_drop": 0.05}}, "floors": {"dead_addresses_shown_after_live_check": 0}}
    base = {"A": {"fit@10": {f"q{i}": 0.6 for i in range(40)}}}
    worse = {"A": {"fit@10": {f"q{i}": 0.3 for i in range(40)}}}
    ok = GATE.decide(base, base, {"dead_addresses_shown_after_live_check": 0}, rules)
    assert ok["passed"]
    assert not GATE.decide(base, worse, {"dead_addresses_shown_after_live_check": 0}, rules)["passed"]
    assert not GATE.decide(base, base, {"dead_addresses_shown_after_live_check": 2}, rules)["passed"]
    assert not GATE.decide(base, base, {}, rules)["passed"]                    # a floor nobody measured does not pass
