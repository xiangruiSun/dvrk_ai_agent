"""Release gates: what an index version has to pass before it may serve.

The thresholds are a policy and are written here, once. The values compared with them are
measured on the version itself.

  integrity   the snapshot hashes to what its manifest says; every cross-reference in the
              documentation points at a label that exists
  coverage    every public source is present, and none has shrunk to less than 90% of the
              version now serving (a source that failed to download would otherwise be
              "removed upstream" and silently disappear)
  quality     nDCG@10 on the development queries, built from the version's own units, is at
              least the floor
  regression  on the development queries that the version and the one now serving have in
              common (same question, same labelled units), the version is at most 0.02 below.
              The two versions are scored on the same queries: a number measured on one
              version's queries cannot be compared with a number measured on another's
  exclusion   no query got back a unit it had excluded
  isolation   a caller who may read only the public collection, or another site's, never
              receives a unit of a site's private collection
  latency     p95 per query within the budget

None of the gates calls a language model: they run in seconds, on every build.
"""
from __future__ import annotations

from typing import Any

from .evaluation.labels import build_queries
from .evaluation.runner import compare, run
from .retrieval import Index
from .store import PUBLIC, PUBLIC_SOURCES, units_hash

POLICY = {"ndcg_floor": 0.60, "ndcg_tolerance": 0.02, "coverage_share": 0.90, "latency_p95_ms": 500.0}


def isolation_probe(index: Index, strategy: str, k: int = 10) -> dict[str, Any]:
    """Ask for every private unit by its own words, as every kind of caller. The owner should
    get it; nobody else may."""
    private = [u for u in index.units.values() if u.collection != PUBLIC]
    collections = sorted({u.collection for u in private})
    leaks, searches, found_by_owner = 0, 0, 0
    for u in private:
        query = (u.title + " " + u.text)[:300]
        for other in [None, *[c for c in collections if c != u.collection]]:
            allowed = {PUBLIC} | ({other} if other else set())
            hits = index.search(query, strategy, k=k, collections=allowed)
            searches += 1
            leaks += sum(index.units[h.unit_id].collection not in allowed for h in hits)
        own = index.search(query, strategy, k=k, collections={PUBLIC, u.collection})
        found_by_owner += any(h.unit_id == u.unit_id for h in own)
    return {"private_units": len(private), "collections": len(collections), "searches_as_other_callers": searches,
            "private_units_leaked": leaks, "found_by_owner": found_by_owner}


def run_gates(index: Index, manifest: dict[str, Any], reference: dict[str, Any] | None = None,
              policy: dict[str, float] | None = None, reference_index: Index | None = None) -> dict[str, Any]:
    p = {**POLICY, **(policy or {})}
    strategy = manifest["config"]["strategy"]
    units = list(index.units.values())
    public = [u for u in units if u.source in PUBLIC_SOURCES]
    gates: list[dict[str, Any]] = []

    def gate(name: str, passed: bool, value: Any, limit: Any, detail: str = "") -> None:
        gates.append({"gate": name, "passed": bool(passed), "value": value, "limit": limit, "detail": detail})

    # integrity
    recomputed = units_hash(sorted((u.unit_id, u.content_hash) for u in units))
    labels = {lb for u in public for lb in u.labels}
    dangling = sum(1 for u in public for r in u.refs if r["target"] not in labels)
    gate("integrity", recomputed == manifest["snapshot_hash"] and dangling == 0,
         {"hash_matches": recomputed == manifest["snapshot_hash"], "dangling_references": dangling},
         {"hash_matches": True, "dangling_references": 0})

    # coverage
    counts = {s: sum(u.source == s for u in units) for s in PUBLIC_SOURCES}
    ref_counts = (reference or {}).get("units", {})
    short = [s for s in PUBLIC_SOURCES
             if counts[s] == 0 or counts[s] < p["coverage_share"] * ref_counts.get(s, 0)]
    gate("coverage", not short, counts,
         {s: round(p["coverage_share"] * ref_counts.get(s, 0)) or 1 for s in PUBLIC_SOURCES},
         "too few units: " + ", ".join(short) if short else "")

    # quality, exclusion, latency: one run over the development queries
    queries = build_queries(public)
    dev = [q for q in queries if q.split == "dev"]
    if dev and all(counts.values()):
        r = run(index, queries, strategy, "dev")
        gate("quality", r["ndcg@10"] >= p["ndcg_floor"], r["ndcg@10"], p["ndcg_floor"],
             f"{r['queries']} development queries built from this version")
        if reference_index is not None:
            ref_public = [u for u in reference_index.units.values() if u.source in PUBLIC_SOURCES]
            theirs = {q.qid: q for q in build_queries(ref_public) if q.split == "dev"}
            common = [q for q in dev if q.qid in theirs and theirs[q.qid].text == q.text
                      and theirs[q.qid].relevant == q.relevant and theirs[q.qid].exclude == q.exclude]
            if common:
                mine = run(index, common, strategy, "dev", latency_probe=0)
                ref = run(reference_index, common, reference["config"]["strategy"], "dev", latency_probe=0)
                d = compare(ref, mine)
                gate("regression", mine["ndcg@10"] >= ref["ndcg@10"] - p["ndcg_tolerance"], mine["ndcg@10"],
                     round(ref["ndcg@10"] - p["ndcg_tolerance"], 4),
                     f"{len(common)} queries common to both versions; serving version {ref['ndcg@10']}; paired "
                     f"difference {d['mean']} ({d['low']} to {d['high']})")
            else:
                gate("regression", False, None, None, "the two versions have no development query in common")
        gate("exclusion", r["gates"]["exclusion_respected"], r["gates"]["excluded_units_returned"], 0)
        gate("latency", (r["latency_ms"]["p95"] or 0) <= p["latency_p95_ms"], r["latency_ms"]["p95"], p["latency_p95_ms"])
        measured = {"dev_ndcg@10": r["ndcg@10"], "dev_recall@10": r["recall@10"], "dev_queries": r["queries"],
                    "latency_p95_ms": r["latency_ms"]["p95"]}
    else:
        gate("quality", False, None, p["ndcg_floor"], "no development queries can be built from this corpus")
        measured = {}

    iso = isolation_probe(index, strategy)
    gate("isolation", iso["private_units_leaked"] == 0, iso["private_units_leaked"], 0,
         f"{iso['searches_as_other_callers']} searches for {iso['private_units']} private units as other callers; "
         f"{iso['found_by_owner']} found by their owner")
    return {"passed": all(g["passed"] for g in gates), "gates": gates, "measured": measured,
            "suspect_units": sorted(index.suspect), "policy": p}
