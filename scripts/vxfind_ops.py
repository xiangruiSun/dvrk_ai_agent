"""The parts of the evaluation that touch the running system. Each stage writes verification/vxfind_<stage>.json.

    python scripts/vxfind_ops.py write_path    what ingestion, embedding, indexing and the incremental sync cost
    python scripts/vxfind_ops.py fresh         dead, moved and changed addresses, before and after the live check
    python scripts/vxfind_ops.py access        can a person see or learn of a table hidden from them
    python scripts/vxfind_ops.py agent         the agent on cases with a known right behaviour
    python scripts/vxfind_ops.py rollout       feedback -> candidate cases, and the release gate
    python scripts/vxfind_ops.py capacity      searches per second, latency, storage, growth

`fresh` and `agent` ask the hosting site about tables (public ids only, rate limited).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path

from _vxfind import EVALS, JUDGE_MODEL, ROOT, SMALL_MODEL, Q, engine, evidence, index_info, judge_missing, llm, log, read, records, serving

from vxfind.eval import agentcases as AC
from vxfind.eval import capacity as CAP
from vxfind.eval import feedback as FB
from vxfind.eval import fresh as FR
from vxfind.eval import gate as GATE
from vxfind.eval import harness as H
from vxfind.eval.judge import Judge
from vxfind.search.engine import describe

DATA = ROOT / "data" / "vxfind"
# The policy names real owners of public tables as "restricted". It is derived at run time and kept
# out of git: a committed file would label named third parties with something that is not true of them.
POLICY = DATA / "eval" / "access_policy.json"


def j(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ---------------------------------------------------------------------------- write path

def write_path() -> None:
    eng = engine()
    man = j(ROOT / "data" / "hub" / "MANIFEST.json")
    old, new = j(DATA / "old" / "ingest_stats.json"), j(DATA / "new" / "ingest_stats.json")
    idx, sync = j(DATA / "old" / "index_stats.json"), j(DATA / "new" / "sync_stats.json")
    st = eng.es.indices.stats(index=eng.index, metric="store,docs,segments")["_all"]["primaries"]
    try:                                    # is the engine's own rank fusion usable on this licence?
        eng.es.search(index=eng.index, size=1, retriever={"rrf": {"retrievers": [
            {"standard": {"query": {"match": {"summary": "franka drawer"}}}},
            {"standard": {"query": {"match": {"name": "drawer"}}}}]}})
        native = "available"
    except Exception as exc:
        native = f"refused ({type(exc).__name__}: {str(getattr(exc, 'message', exc))[:120]})"
    lic = eng.es.license.get()["license"]["type"]
    evidence("write_path", {
        "engine_rrf_on_this_licence": f"{native}; licence type: {lic}",
        "source": {"what": "daily dump of dataset cards with metadata, two revisions about a month apart, and the site's listing",
                   "snapshots": {k: {x: v[x] for x in ("revision", "published", "bytes")} for k, v in man.get("snapshots", {}).items()},
                   "dump_revisions_available": man.get("dump_revisions_available")},
        "spark_job": {"old": old, "new": new},
        "embedding": {"old": j(DATA / "old" / "embed_stats.json"), "new": j(DATA / "new" / "embed_stats.json")},
        "index_build": idx, "incremental_sync": sync,
        "index_now": {**index_info(eng.es), "store_mb": round(st["store"]["size_in_bytes"] / 1e6, 1), "segments": st["segments"]["count"],
                      "bytes_per_table": round(st["store"]["size_in_bytes"] / max(1, st["docs"]["count"]))},
        "catalog_database": {"file_mb": round((DATA / "catalog.db").stat().st_size / 1e6, 1) if (DATA / "catalog.db").exists() else None},
    })
    log("write_path evidence written")


# ---------------------------------------------------------------------------- freshness

def fresh(limit: int) -> None:
    from vxfind.mcp.hub import HubSite, RateLimiter
    eng = engine()
    recs = records()
    qs = Q.all_queries()
    run = H.run(eng, serving(), qs, tag="_all")
    shown = list(dict.fromkeys(t for v in run.values() for t in v["ids"][:10]))
    rng = random.Random(9)
    random_sample = rng.sample(sorted(recs), 300)
    todo = shown[:limit] + random_sample
    log(f"{len(shown)} distinct tables shown for {len(qs)} queries; checking {len(shown[:limit])} of them and 300 random tables")
    site = HubSite(limiter=RateLimiter(calls=420, per_s=300.0), max_wait_s=400.0, cache_ttl_s=0)
    t0 = time.time()
    answers = FR.check_live(site, todo, workers=3, log=log)
    took = time.time() - t0
    import pyarrow.parquet as pq
    meta = pq.read_table(DATA / "old" / "tables.parquet", columns=["table_id", "last_modified", "created"]).to_pylist()
    modified = {r["table_id"]: r["last_modified"] for r in meta}
    created = {r["table_id"]: r["created"] for r in meta}
    checked_run = {qid: v for qid, v in run.items()}
    stale = FR.staleness(checked_run, answers, modified)
    by_kind = {k: FR.staleness({q["qid"]: run[q["qid"]] for q in qs if q["kind"] == k}, answers, modified) for k in "ABCD"}
    rs = [answers[t] for t in random_sample if t in answers]
    dead_r = sum(a["state"] == "gone" for a in rs)
    p = dead_r / len(rs) if rs else 0.0
    half = 1.96 * (p * (1 - p) / len(rs)) ** 0.5 if rs else 0.0
    lat_ids = [t for t in shown if answers.get(t, {}).get("state") == "live"][:120]
    lat = FR.check_latency(lambda: HubSite(limiter=site.limiter, max_wait_s=400.0, cache_ttl_s=0), lat_ids)
    listing_ids = set()
    for f in sorted((ROOT / "data" / "hub" / "listing").glob("*.jsonl"))[-1:]:
        listing_ids = {json.loads(l)["id"] for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}
    man = j(ROOT / "data" / "hub" / "MANIFEST.json")
    moved = [a for a in answers.values() if a["state"] == "moved" and a.get("moved_to")]
    examples = {"moved_to_another_owner": sum(a["moved_to"].split("/")[0] != a["table_id"].split("/")[0] for a in moved),
                "renamed_under_the_same_owner": sum(a["moved_to"].split("/")[0] == a["table_id"].split("/")[0] for a in moved)}
    # Why so many are dead: does the source ever drop a table that was deleted? Count the dead
    # tables that the NEWER dump, a month later, still carries as if nothing had happened.
    from vxfind.catalog.store import Catalog
    cat = Catalog()
    dead_ids = [t for t in todo if answers.get(t, {}).get("state") == "gone"]
    still = sum(len(cat.many(dead_ids[i:i + 400])) for i in range(0, len(dead_ids), 400))
    ch = collections.Counter(pq.read_table(DATA / "new" / "changes.parquet", columns=["change"]).column("change").to_pylist())
    source = {"dead_tables_found": len(dead_ids), "of_which_still_in_the_newer_dump": still,
              "tables_the_newer_dump_dropped": ch.get("deleted", 0), "tables_the_newer_dump_added": ch.get("created", 0),
              "reading": "the dump never drops a table. Deleted and privatised tables stay in it with their last card, so dead entries "
                         "pile up in the source itself. The dead share measured here is how far an index built from this source is "
                         "from the live site; it is not the share that died in the month between the snapshot and the check."}
    evidence("fresh", {
        "index_built_from_catalog_of": man["snapshots"]["old"]["published"], "site_asked_on": time.strftime("%Y-%m-%d"),
        "serving_config": describe(serving()), "queries": len(qs), "distinct_tables_shown": len(shown),
        # The site's answers are kept on record and replayed; a later run asks only about tables it
        # has no answer for. The record's own time stamps say when, and over how long, the site was asked.
        "live_checks": {"tables_on_record": sum(t in answers for t in set(todo)), "budget": "420 requests per 300 s",
                        "first_answer": time.strftime("%Y-%m-%d %H:%M", time.localtime(min(a["checked_at"] for a in answers.values()))),
                        "last_answer": time.strftime("%Y-%m-%d %H:%M", time.localtime(max(a["checked_at"] for a in answers.values()))),
                        "minutes_from_first_to_last_answer_of_the_first_pass": round(
                            (sorted(a["checked_at"] for a in answers.values())[min(len(answers), 1790) - 1] - min(a["checked_at"] for a in answers.values())) / 60, 1),
                        "requests_in_this_run": site.requests,
                        "states": dict(collections.Counter(a["state"] for t, a in answers.items() if t in set(todo)))},
        "what_was_shown": stale, "by_query_set": by_kind, "the_source_does_not_report_deletions": source,
        "dead_share_by_year_created": FR.by_age(shown, answers, created),
        "random_tables_of_the_index": {"checked": len(rs), "dead": dead_r, "share_dead": round(p, 4), "ci95": [round(max(0, p - half), 4), round(p + half, 4)],
                                       "moved": sum(a["state"] == "moved" for a in rs)},
        "listing_as_a_cheap_signal": {
            "what": "is absence from the site's public listing a usable sign that a table is dead, without asking about each table",
            **_listing_signal(answers, listing_ids)},
        "seconds_added_to_one_answer": lat, "kinds_of_move": examples,
    })
    log("fresh evidence written")


def _listing_signal(answers: dict, listing: set) -> dict:
    if not listing:
        return {"listing": "not available"}
    tp = sum(a["state"] == "gone" and t not in listing for t, a in answers.items())
    fp = sum(a["state"] != "gone" and t not in listing for t, a in answers.items())
    fn = sum(a["state"] == "gone" and t in listing for t, a in answers.items())
    return {"checked_tables": len(answers), "not_in_listing": tp + fp, "dead_and_not_in_listing": tp, "alive_but_not_in_listing": fp,
            "dead_but_in_listing": fn, "precision": round(tp / (tp + fp), 4) if tp + fp else None, "recall": round(tp / (tp + fn), 4) if tp + fn else None}


# ---------------------------------------------------------------------------- access

def policy(eng, qs) -> dict:
    """The DECLARED policy: three owners are restricted to one group. They are chosen to be the
    owners that appear most often in results, so that a leak test has something to leak."""
    if POLICY.exists():
        return json.loads(POLICY.read_text(encoding="utf-8"))
    run = H.run(eng, serving(), qs, tag="_all")
    owners = collections.Counter(t.split("/", 1)[0] for v in run.values() for t in v["ids"][:10])
    pick = [o for o, _ in owners.most_common(3)]
    pol = {"restricted": {o: ["team:core"] for o in pick},
           "note": "DECLARED for this project. No real organisation's policy. Owners chosen because they appear most often in results."}
    POLICY.parent.mkdir(parents=True, exist_ok=True)
    POLICY.write_text(json.dumps(pol, indent=1), encoding="utf-8")
    return pol


def desk_and_servers(pol: dict, site=None):
    from vxagent.tools.registry import Principal
    from vxfind.catalog.store import Catalog
    from vxfind.mcp import servers as S
    from vxfind.mcp.access import AccessStore
    from vxfind.mcp.hub import HubSite, RateLimiter
    cat = Catalog()
    desk = S.Desk(cat, AccessStore(restricted=pol["restricted"]), site or HubSite(limiter=RateLimiter(calls=420), max_wait_s=30.0))
    people = {"newcomer": Principal("ana", "newcomer", frozenset({"team:manip"})),
              "insider": Principal("bo", "newcomer", frozenset({"team:manip", "team:core"})),
              "mentor": Principal("mentor_li", "mentor", frozenset({"team:manip", "team:core"}))}
    tokens = {k: desk.tokens.issue(p) for k, p in people.items()}
    import pyarrow.parquet as pq
    modified = dict(zip(*[pq.read_table(DATA / "old" / "tables.parquet", columns=[c]).column(c).to_pylist() for c in ("table_id", "last_modified")]))
    running = S.start(desk, indexed=lambda t: modified.get(t))
    return desk, running, people, tokens


def access() -> None:
    from vxagent.mcp_layer.client import Connection
    from vxfind.mcp import servers as S
    eng = engine()
    qs = Q.all_queries()
    pol = policy(eng, qs)
    restricted = set(pol["restricted"])
    open_run = H.run(eng, serving(), qs, tag="_all")
    would = sum(t.split("/", 1)[0] in restricted for v in open_run.values() for t in v["ids"][:10])
    q_with = sum(any(t.split("/", 1)[0] in restricted for t in v["ids"][:10]) for v in open_run.values())
    desk, running, people, tokens = desk_and_servers(pol)
    try:
        hidden = desk.access.hidden_owners(people["newcomer"].scopes)
        closed = H.run(eng, serving(), qs, visible={"hidden_owners": hidden}, tag="_outsider")
        leaked = sum(t.split("/", 1)[0] in restricted for v in closed.values() for t in v["ids"])
        deep_leaked = sum(t.split("/", 1)[0] in restricted for v in closed.values() for t in v["deep"])
        insider = H.run(eng, serving(), qs, visible={"hidden_owners": desk.access.hidden_owners(people["insider"].scopes)}, tag="_all")
        ins_seen = sum(t.split("/", 1)[0] in restricted for v in insider.values() for t in v["ids"][:10])
        # probes through the servers: a hidden table and a table that does not exist must be answered the same way
        hidden_ids = list(dict.fromkeys(t for v in open_run.values() for t in v["ids"][:10] if t.split("/", 1)[0] in restricted))[:25]
        fake_ids = [f"no-such-owner-{i}/no_such_table_{i}" for i in range(len(hidden_ids))]
        conns = {n: Connection(n, r.url, tokens["newcomer"]).connect() for n, r in running.items()}
        site_before = desk.site.requests
        probes = collections.Counter()
        same = 0
        for h, f in zip(hidden_ids, fake_ids):
            pairs = [(conns["vx-catalog"].call_tool("table_details", {"table_id": x}), conns["vx-catalog"].call_tool("table_columns", {"table_id": x}),
                      conns["vx-catalog"].call_tool("related_tables", {"table_id": x}),
                      conns["vx-access"].call_tool("request_access", {"table_id": x, "purpose": "probe of a hidden table"}),
                      conns["vx-site"].call_tool("check_tables", {"table_ids": [x]}),
                      conns["vx-access"].call_tool("access_status", {"table_ids": [x]})) for x in (h, f)]
            strip = lambda r, x: json.dumps(r, sort_keys=True).replace(x, "ID")
            for k, name in enumerate(("table_details", "table_columns", "related_tables", "request_access", "check_tables", "access_status")):
                probes[name] += 1
                same += strip(pairs[0][k], h) == strip(pairs[1][k], f)
        site_asked_for_hidden = desk.site.requests - site_before
        # the mentor's tools are not reachable with a newcomer's token, and nobody decides their own request
        open_id = next(t for v in open_run.values() for t in v["ids"] if t.split("/", 1)[0] not in restricted)
        req = conns["vx-access"].call_tool("request_access", {"table_id": open_id, "purpose": "train a first policy during onboarding"})
        rid = req["result"]["request_id"]
        own = conns["vx-access"].call_tool("decide_request", {"request_id": rid, "approve": True})
        queue = conns["vx-access"].call_tool("pending_requests", {})
        m = Connection("vx-access", running["vx-access"].url, tokens["mentor"]).connect()
        mreq = m.call_tool("request_access", {"table_id": open_id, "purpose": "the mentor's own request, for the self-approval probe"})
        self_dec = m.call_tool("decide_request", {"request_id": mreq["result"]["request_id"], "approve": True})
        ok_dec = m.call_tool("decide_request", {"request_id": rid, "approve": True})
        after = conns["vx-access"].call_tool("access_status", {"table_ids": [open_id]})
        try:
            Connection("vx-catalog", running["vx-catalog"].url, "not-a-token").connect()
            no_token = "connected"
        except Exception as exc:
            no_token = "refused: " + type(exc).__name__
        m.close()
        for c in conns.values():
            c.close()
        from vxfind.agent import ALLOW
        evidence("access", {
            "policy": {"restricted_owners": len(pol["restricted"]), "restricted_to": "one group", "note": pol["note"],
                       "owners_named_in": "data/vxfind/eval/access_policy.json (not in git)"}, "queries": len(qs),
            "without_the_rule": {"restricted_tables_in_top_10": would, "queries_with_one": q_with},
            "outsider": {"restricted_tables_in_any_result": leaked, "restricted_tables_among_the_first_100_candidates": deep_leaked},
            "insider_of_the_group": {"restricted_tables_in_top_10": ins_seen},
            "probes_by_an_outsider": {"hidden_tables_probed": len(hidden_ids), "tools_probed": dict(probes),
                                      "replies_identical_to_a_nonexistent_table": same, "replies_compared": sum(probes.values()),
                                      "requests_sent_to_the_site_for_hidden_or_fake_ids": site_asked_for_hidden},
            "approval": {"newcomer_decides_own_request": {k: own.get(k) for k in ("ok", "error_kind", "error")},
                         "newcomer_reads_the_queue": {k: queue.get(k) for k in ("ok", "error_kind")},
                         "mentor_decides_own_request": {k: self_dec.get(k) for k in ("ok", "error_kind", "error")},
                         "mentor_decides_newcomers_request": (ok_dec.get("result") or {}).get("use"),
                         "newcomer_status_after": after["result"][0]["use"]},
            "no_token": no_token,
            "tools_offered_to_the_assistant": ALLOW, "mentor_tools_offered_to_the_assistant": [t for ts in ALLOW.values() for t in ts if t in ("decide_request", "pending_requests")],
            "audit_rows": len(desk.access.audit_rows()),
        })
    finally:
        S.stop(running)
    log("access evidence written")


# ---------------------------------------------------------------------------- agent

def agent(models: list[str]) -> None:
    from vxfind.agent import pipeline, render
    from vxfind.mcp import servers as S
    from vxfind.mcp.hub import HubSite, RateLimiter, RecordedSite
    eng = engine(with_models=False)
    qs = Q.all_queries()
    pol = policy(eng, qs)
    restricted_owner = sorted(pol["restricted"])[0]
    recorded = FR.load_answers()
    live = HubSite(limiter=RateLimiter(calls=420), max_wait_s=30.0)

    class ReplayThenLive:
        requests = 0
        limiter = live.limiter

        def check(self, t):
            a = recorded.get(t)
            return {**a, "from_cache": False} if a else live.check(t)

    base = ReplayThenLive()
    desk, running, people, tokens = desk_and_servers(pol, site=base)
    urls = {n: r.url for n, r in running.items()}
    out = {}
    try:
        cases = AC.cases()
        for model in models:
            m = llm(model)
            results = []
            for c in cases:
                r = AC.run_case(c, m, eng, desk, urls, base, people["newcomer"], tokens["newcomer"], serving(), restricted_owner)
                results.append(r)
                log(f"{model} {c['id']}: {'pass' if r['passed'] else 'FAIL ' + str([k for k, v in r['properties'].items() if v is False])} "
                    f"{r['seconds']} s, tools {r['tools']}")
            out[model] = {"summary": AC.summary(results), "cases": results}
        # the fixed pipeline on the same needs, for cost: no model call when selection is off
        from vxagent.mcp_layer.client import Connection
        conn = Connection("vx-site", urls["vx-site"], tokens["newcomer"]).connect()
        check = lambda ids: conn.call_tool("check_tables", {"table_ids": ids})["result"]
        hidden = desk.access.hidden_owners(people["newcomer"].scopes)
        pt = []
        for c in cases[:10]:
            need = c["need"].format(restricted_owner=restricted_owner)
            faults = AC.faults_for(eng, need, c.get("inject", {}), hidden, serving())
            from vxfind.mcp.hub import InjectedSite
            desk.site = AC.AllUnknown() if "*" in c.get("inject", {}) else InjectedSite(base, faults)
            res = pipeline(eng, need, check, hidden, serving())
            gone = {t for t, f in faults.items() if f["state"] == "gone"}
            moved = {t for t, f in faults.items() if f["state"] == "moved"}
            pt.append({"id": c["id"], "tables": len(res["tables"]), "gone_shown": len(gone & {t["table_id"] for t in res["tables"]}),
                       "old_address_of_moved_shown": len(moved & {t["table_id"] for t in res["tables"]}),
                       "dropped": len(res["dropped"]), "ms": res["timings_ms"]["total"], "ms_live_check": res["timings_ms"]["live_check"]})
            desk.site = base
        conn.close()
        out["fixed_pipeline_same_needs"] = {"cases": pt, "gone_shown": sum(p["gone_shown"] for p in pt),
                                            "old_address_shown": sum(p["old_address_of_moved_shown"] for p in pt),
                                            "median_ms": sorted(p["ms"] for p in pt)[len(pt) // 2], "model_calls": 0}
        from vxfind.mcp.hub import verdict
        out["example_pipeline_answer"] = render(pipeline(eng, cases[0]["need"], lambda ids: [verdict(None, base.check(t)) for t in ids],
                                                         hidden, serving()))[:1800]
    finally:
        S.stop(running)
    evidence("agent", {"cases_file": "evals/vxfind/agent_cases.json", "serving_config": describe(serving()),
                       "world": "site answers recorded earlier are replayed; INJECTED faults per case; restricted owners DECLARED", **out})
    log("agent evidence written")


def agent_first_run() -> None:
    fr = read("agent_first_run")
    models = {}
    for m, v in fr.items():
        if isinstance(v, dict) and "summary" in v:
            models[m] = {"cases": v["summary"]["cases"], "passed": v["summary"]["passed"],
                         "ended_with_nothing_shown_and_no_live_check": sum(1 for c in v["cases"] if not c["listed"] and "check_tables" not in c["tools"]
                                                                           and c["id"] not in ("restricted_owner",))}
    evidence("agent", first_run={"evidence": "vxfind_agent_first_run.json", "models": models,
                                 "what_it_showed": "when the model asked for twenty candidates the search tool's reply was larger than the "
                                                   "executor accepts; the tool failed, the model asked again and gave up; and those runs "
                                                   "passed, because no property required an answer. The tool now fits its reply, and the "
                                                   "property listed_some was added."})
    log("agent first-run note written")


# ---------------------------------------------------------------------------- rollout

def rollout() -> None:
    eng = engine(with_models=True)          # one gate candidate reads the need with a model
    recs = records()
    judge = Judge(llm(JUDGE_MODEL))
    test, dev = Q.all_queries("test"), Q.all_queries("dev")
    qrels = judge.qrels(dev + test)
    run = H.run(eng, serving(), test, tag="_test")
    path = ROOT / "data" / "vxfind" / "eval" / "feedback_scripted.jsonl"
    path.unlink(missing_ok=True)
    fl = FB.FeedbackLog(path)
    rng = random.Random(17)
    # SCRIPTED sessions. There are no users. A script plays one: it opens what the evaluation's own
    # labels call relevant and marks some of what they call irrelevant. This exercises the log,
    # the online measures and the triage; it says nothing about how real people behave.
    for i, q in enumerate(test):
        sid = f"s{i:03d}"
        ids = run[q["qid"]]["ids"][:10]
        fams = run[q["qid"]]["families"][:10]
        fl.add(sid, q["need"], "shown", shown=ids, release="r1")
        if q["kind"] == "A":
            grade = lambda t, f: Q.grade_attribute(q["spec"], recs[t]) if t in recs else 0
        elif q["kind"] == "B":
            grade = lambda t, f: 3 if (t == q["target"] or f == q["target_family"]) else 1
        else:
            grade = lambda t, f: qrels.get(q["qid"], {}).get(t, 0)
        good = [(r, t) for r, (t, f) in enumerate(zip(ids, fams), 1) if grade(t, f) >= 2]
        for r, t in good[:3]:
            fl.add(sid, q["need"], "opened", table_id=t, rank=r, release="r1")
        for r, (t, f) in enumerate(zip(ids, fams), 1):
            if grade(t, f) == 0 and rng.random() < 0.4:
                fl.add(sid, q["need"], "not_useful", table_id=t, rank=r, release="r1")
        if not good:
            fl.add(sid, q["need"], "nothing_fits", release="r1")
            if rng.random() < 0.5:
                fl.add(sid, q["need"], "asked_again", release="r1")
    events = fl.events()
    known = {" ".join(q["need"].lower().split()) for q in dev}
    cands = FB.triage(events, known)
    (EVALS / "queries" / "F_candidates.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cands), encoding="utf-8")

    # the release gate on the locked test queries
    fr, acc, ag = read("fresh"), read("access"), read("agent")
    floors = {"dead_addresses_shown_after_live_check": (ag.get("fixed_pipeline_same_needs") or {}).get("gone_shown"),
              "restricted_tables_shown_to_outsiders": (acc.get("outsider") or {}).get("restricted_tables_in_any_result")}
    spq = H.per_query(test, run, recs, qrels)
    decisions = {}
    for name, cfg in (("rrf (plain fusion)", H.CONFIGS["rrf"]), ("bm25 (vectors switched off)", H.CONFIGS["bm25"]),
                      ("bm25_name_only (a planted regression)", H.CONFIGS["bm25_name_only"]),
                      ("serving with window 20", serving().but(name="serving_w20", window=20)),
                      ("serving + the need read by the large model, numbers as filters",
                       serving().but(name="serving+understand_hard_large", understand="hard", understand_model="large"))):
        r = H.run(eng, cfg, test, tag="_test")
        judge_missing(judge, eng, test, r)          # a candidate's new tables are judged before it is compared
        qrels = judge.qrels(dev + test)
        decisions[name] = GATE.decide(H.per_query(test, run, recs, qrels), H.per_query(test, r, recs, qrels), floors)
    decisions["serving against itself"] = GATE.decide(spq, spq, floors)
    evidence("rollout", {
        "scripted_sessions": {"what": "SCRIPTED. No real users. A script opens what the evaluation's labels call relevant.",
                              "events": len(events), "online_measures": FB.online(events)},
        "triage": {"candidate_cases": len(cands), "reasons": FB.label_mix(cands), "written_to": "evals/vxfind/queries/F_candidates.jsonl",
                   "status": "candidates only; none is in a scored query set until a person has read and labelled it",
                   "by_query_set": dict(collections.Counter(next(q["kind"] for q in test if q["need"] == c["need"]) for c in cands))},
        "gate": {"rules": GATE.rules(), "serving": describe(serving()), "floors_measured": floors, "decisions": decisions},
    })
    log("rollout evidence written")


# ---------------------------------------------------------------------------- capacity

def capacity() -> None:
    eng = engine()
    needs = [q["need"] for q in Q.all_queries()]
    cfgs = [H.CONFIGS["bm25"], H.CONFIGS["knn_nomic"], serving()]
    rows = []
    for cfg in cfgs:
        for w in (1, 4, 16, 32):
            r = CAP.load_test(eng, cfg, needs, w, seconds=12.0)
            rows.append(r)
            log(f"{cfg.name} x{w}: {r['per_second']} per second, p50 {r['p50_ms']} ms, p95 {r['p95_ms']} ms")
    cold = CAP.load_test(eng, serving(), needs, 4, seconds=12.0, warm_vectors=False)
    log(f"serving x4 with the query vector computed per search: {cold['per_second']} per second, p95 {cold['p95_ms']} ms")
    import pyarrow.parquet as pq
    months = collections.Counter(pq.read_table(DATA / "new" / "tables.parquet", columns=["created_month"]).column("created_month").to_pylist())
    t = pq.read_table(DATA / "new" / "tables.parquet", columns=["data_mb", "video_mb", "total_frames", "total_episodes"]).to_pydict()
    declared = sum((a or 0) + (b or 0) for a, b in zip(t["data_mb"], t["video_mb"]))
    st = eng.es.indices.stats(index=eng.index, metric="store,docs")["_all"]["primaries"]
    g = CAP.growth({k: v for k, v in months.items() if k})
    per_table = st["store"]["size_in_bytes"] / max(1, st["docs"]["count"])
    evidence("capacity", {
        "machine": "one workstation; the search engine (1 node, 1 shard, 4 GB heap), the load generator and the embedding model share it",
        "index": index_info(eng.es), "needs_cycled": len(needs), "load": rows, "query_vector_computed_per_search": cold,
        "client_connections": 64,
        "first_run": {"evidence": "vxfind_capacity_first_run.json",
                      "what_it_showed": "with the search client's default of 10 connections, callers queued inside the client: the throughput was the "
                                        "same, but the queueing showed up as a 99th percentile of seconds at 32 callers instead of as latency",
                      "serving_32_callers": next((r for r in read("capacity_first_run").get("load", []) if r["retriever"] == "serving" and r["workers"] == 32), None)},
        "storage": {"index_mb": round(st["store"]["size_in_bytes"] / 1e6, 1), "bytes_per_table": round(per_table),
                    "vector_bytes_per_table_raw": (768 + 1024) * 4,
                    "data_the_catalog_describes_tb": round(declared / 1e6, 1), "tables_declaring_size": sum(1 for a in t["data_mb"] if a),
                    "frames_described": sum(x or 0 for x in t["total_frames"])},
        "growth": {**g, "projected_index_mb_in_12_months": round(g["projected_tables_in_12_months"] * per_table / 1e6, 1)},
    })
    log("capacity evidence written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["write_path", "fresh", "access", "agent", "agent_first_run", "rollout", "capacity"])
    ap.add_argument("--limit", type=int, default=1500, help="fresh: how many shown tables to ask the site about")
    ap.add_argument("--models", default=f"{JUDGE_MODEL},{SMALL_MODEL}")
    a = ap.parse_args()
    if a.stage == "fresh":
        fresh(a.limit)
    elif a.stage == "agent":
        agent(a.models.split(","))
    else:
        {"write_path": write_path, "access": access, "rollout": rollout, "capacity": capacity, "agent_first_run": agent_first_run}[a.stage]()
