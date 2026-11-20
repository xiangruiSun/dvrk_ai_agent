"""Running a retrieval configuration over a set of labelled queries, comparing configurations,
and choosing one by a rule that was written down before the numbers were seen.

Selection policy
    1. Gates. A run is rejected if any result is a unit the query excluded, or if it scored a
       query from the wrong split.
    2. Primary objective: mean nDCG@10 on the development queries.
    3. Credible improvement: a candidate replaces the baseline only if the 95% paired-bootstrap
       interval of its per-query nDCG@10 difference lies above zero.
    4. Resource limit: p95 latency per query within the stated budget.
    5. Otherwise the simpler baseline is kept.
The held-out test queries are scored once, for the selected configuration only.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..retrieval import Index, _parts
from .labels import CATEGORIES, Query
from .metrics import mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k, unlabelled_share


def run(index: Index, queries: list[Query], strategy: str, split: str, k: int = 10,
        latency_probe: int = 25) -> dict[str, Any]:
    qs = [q for q in queries if q.split == split]
    index.prepare(strategy)
    for part in _parts(strategy):                       # embed all query texts in one batch
        if part.startswith("dense:"):
            index._embedders[part[6:]].embed([q.text for q in qs], query=True)
    rows = []
    excluded_returned = 0
    for q in qs:
        hits = index.search(q.text, strategy, k=k, exclude_units=q.exclude)
        ranked = [h.unit_id for h in hits]
        excluded_returned += len(set(ranked) & q.exclude)
        rows.append({"qid": q.qid, "category": q.category, "generation": q.generation,
                     "ndcg": ndcg_at_k(ranked, q.relevant, k), "recall5": recall_at_k(ranked, q.relevant, 5),
                     "recall10": recall_at_k(ranked, q.relevant, 10), "mrr": mrr_at_k(ranked, q.relevant, k),
                     "unlabelled": unlabelled_share(ranked, q.relevant, k), "top": ranked[:3]})

    # Latency with the query-embedding cache switched off, so a dense strategy pays for its model call.
    lat = []
    embedders = [index._embedders[p[6:]] for p in _parts(strategy) if p.startswith("dense:")]
    for e in embedders:
        e.read_cache = False
    try:
        for q in qs[:latency_probe]:
            t0 = time.perf_counter()
            index.search(q.text, strategy, k=k, exclude_units=q.exclude)
            lat.append((time.perf_counter() - t0) * 1000)
    finally:
        for e in embedders:
            e.read_cache = True

    def mean(key: str, subset: list[dict[str, Any]]) -> float:
        return round(float(np.mean([r[key] for r in subset])), 4) if subset else 0.0

    def block(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {"queries": len(subset), "ndcg@10": mean("ndcg", subset), "recall@5": mean("recall5", subset),
                "recall@10": mean("recall10", subset), "mrr@10": mean("mrr", subset),
                "unlabelled@10": mean("unlabelled", subset)}

    return {"strategy": strategy, "chunking": index.chunking, "split": split, "k": k, **block(rows),
            "by_category": {c: block([r for r in rows if r["category"] == c]) for c in CATEGORIES},
            "latency_ms": {"mean": round(float(np.mean(lat)), 1) if lat else None,
                           "p95": round(float(np.percentile(lat, 95)), 1) if lat else None, "probe_queries": len(lat)},
            "gates": {"excluded_units_returned": excluded_returned, "exclusion_respected": excluded_returned == 0,
                      "split_isolation": all(q.split == split for q in qs)},
            "per_query": rows}


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    a = {r["qid"]: r["ndcg"] for r in baseline["per_query"]}
    b = {r["qid"]: r["ndcg"] for r in candidate["per_query"]}
    ids = sorted(set(a) & set(b))
    return paired_bootstrap([a[i] for i in ids], [b[i] for i in ids])


def select(runs: list[dict[str, Any]], baseline_name: str, latency_budget_ms: float = 500.0) -> dict[str, Any]:
    """Apply the selection policy to runs made on the development split."""
    name = lambda r: f"{r['strategy']} @ {r['chunking']}"          # noqa: E731
    base = next(r for r in runs if name(r) == baseline_name)
    decisions = []
    best, best_score = base, base["ndcg@10"]
    for r in runs:
        if r is base:
            continue
        d = compare(base, r)
        gates_ok = r["gates"]["exclusion_respected"] and r["gates"]["split_isolation"]
        credible = d["low"] > 0
        in_budget = (r["latency_ms"]["p95"] or 0) <= latency_budget_ms
        accepted = gates_ok and credible and in_budget
        decisions.append({"candidate": name(r), "ndcg@10": r["ndcg@10"], "difference_to_baseline": d,
                          "gates_passed": gates_ok, "credible_improvement": credible,
                          "p95_latency_ms": r["latency_ms"]["p95"], "within_latency_budget": in_budget,
                          "accepted": accepted})
        if accepted and r["ndcg@10"] > best_score:
            best, best_score = r, r["ndcg@10"]
    return {"baseline": baseline_name, "baseline_ndcg@10": base["ndcg@10"], "latency_budget_ms": latency_budget_ms,
            "decisions": decisions, "selected": name(best), "selected_strategy": best["strategy"],
            "selected_chunking": best["chunking"], "kept_baseline": best is base}
