"""Run search configurations over the query sets and score them.

A run is: one configuration, every query, the ranked ids it returned. Runs are stored, so that
scoring, pooling and comparing never repeat a search, and so that a number in a report can be
traced to the list it was computed from.

Scores, one headline per query set (see queries.py for why each set has its own):

  A   fit@10      share of the top 10 that satisfy every stated constraint (computed by code)
  B   mrr@10      1 / rank of the table the need was written from (or a near-identical sibling)
  C   ndcg@10     graded relevance from the judge over the pooled results; unjudged counts as 0
  D   (answers)   scored in answers.py: the system should say that nothing fits

A comparison of two configurations is paired: the same queries, the difference per query, an
interval from resampling queries and a randomization test.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..search.engine import Engine, SearchConfig, describe
from . import metrics as M
from .queries import grade_attribute

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "vxfind" / "eval" / "runs"
HEADLINE = {"A": "fit@10", "B": "mrr@10", "C": "ndcg@10"}

BASE = SearchConfig(name="rrf")
CONFIGS: dict[str, SearchConfig] = {c.name: c for c in [
    # which retriever
    BASE.but(name="bm25", retriever="bm25"),
    BASE.but(name="knn_nomic", retriever="knn"),
    BASE.but(name="knn_bge", retriever="knn", embedding="bge-m3"),
    BASE,
    BASE.but(name="rrf_bge", embedding="bge-m3"),
    # which text is searched (lexical only, so the effect is not hidden by the vectors)
    BASE.but(name="bm25_no_name", retriever="bm25", fields=(("description", 2.0), ("summary", 1.0), ("schema_text", 1.0), ("robot_type.text", 2.0))),
    BASE.but(name="bm25_no_description", retriever="bm25", fields=(("name", 3.0), ("summary", 1.0), ("schema_text", 1.0), ("robot_type.text", 2.0))),
    BASE.but(name="bm25_no_summary", retriever="bm25", fields=(("name", 3.0), ("description", 2.0), ("schema_text", 1.0), ("robot_type.text", 2.0))),
    BASE.but(name="bm25_no_schema", retriever="bm25", fields=(("name", 3.0), ("description", 2.0), ("summary", 1.0), ("robot_type.text", 2.0))),
    BASE.but(name="bm25_name_only", retriever="bm25", fields=(("name", 1.0),)),
    BASE.but(name="bm25_summary_only", retriever="bm25", fields=(("summary", 1.0),)),
    BASE.but(name="bm25_flat_weights", retriever="bm25", fields=(("name", 1.0), ("description", 1.0), ("summary", 1.0), ("schema_text", 1.0), ("robot_type.text", 1.0))),
    BASE.but(name="bm25_names_match_function_words", retriever="bm25", ident_stop=False),
    BASE.but(name="rrf_names_match_function_words", ident_stop=False),
    # fusion settings
    BASE.but(name="rrf_c10", rank_constant=10), BASE.but(name="rrf_c100", rank_constant=100),
    BASE.but(name="rrf_w20", window=20), BASE.but(name="rrf_w50", window=50), BASE.but(name="rrf_w200", window=200),
    # the catalog's mess
    BASE.but(name="rrf_no_collapse", collapse=False),
    BASE.but(name="rrf_scratch_0.5", demote_scratch=0.5), BASE.but(name="rrf_scratch_0.2", demote_scratch=0.2),
    BASE.but(name="rrf_described_1.5", boost_described=1.5), BASE.but(name="rrf_described_2", boost_described=2.0),
    BASE.but(name="rrf_scratch_0.5_described_1.5", demote_scratch=0.5, boost_described=1.5),
]}
LLM_CONFIGS: dict[str, SearchConfig] = {c.name: c for c in [
    BASE.but(name="rrf_understand_soft", understand="soft"), BASE.but(name="rrf_understand_hard", understand="hard"),
    BASE.but(name="rrf_understand_hard_large", understand="hard", understand_model="large"),
    BASE.but(name="rrf_rerank_small", rerank="small"), BASE.but(name="rrf_rerank_large", rerank="large"),
    BASE.but(name="rrf_select", select=True), BASE.but(name="rrf_rerank_large_select", rerank="large", select=True),
    BASE.but(name="rrf_understand_hard_large_select", understand="hard", understand_model="large", select=True),
]}


def config_key(cfg: SearchConfig, index: str) -> str:
    d = describe(cfg)
    d.pop("name")
    if d["understand"] == "off":                # a run that does not read the need does not depend on how it would be read
        d.pop("understand_model", None)
        d.pop("understand_version", None)
    return hashlib.sha256((json.dumps(d, sort_keys=True) + index).encode()).hexdigest()[:12]


def run(engine: Engine, cfg: SearchConfig, queries: list[dict[str, Any]], depth: int = 20, visible: dict[str, Any] | None = None,
        folder: Path = RUNS, log: Any = None, tag: str = "") -> dict[str, dict[str, Any]]:
    """qid -> {ids, families, deep, timings_ms, notes, reasons, reading}. Searches already stored are not repeated."""
    folder.mkdir(parents=True, exist_ok=True)
    real_index = sorted(engine.es.indices.get_alias(name=engine.index))[0] if engine.es.indices.exists_alias(name=engine.index) else engine.index
    path = folder / f"{cfg.name}{tag}.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    ck = config_key(cfg, real_index)
    if stored.get("config_key") != ck:
        stored = {"config_key": ck, "config": describe(cfg), "index": real_index, "queries": {}}
    out, new = stored["queries"], 0
    t0 = time.time()
    for q in queries:
        qk = hashlib.sha256(q["need"].encode()).hexdigest()[:12]
        if q["qid"] in out and out[q["qid"]].get("need_key") == qk:
            continue
        k = cfg.k if cfg.select else depth
        res = engine.search(q["need"], cfg.but(k=k), visible=visible)
        out[q["qid"]] = {"need_key": qk, "ids": res.ids(), "families": [h.source.get("family") for h in res.hits],
                         "deep": res.pool, "timings_ms": res.timings_ms, "notes": res.notes, "reading": res.reading,
                         "reasons": {h.table_id: h.reason for h in res.hits if h.reason}}
        new += 1
        if log and new % 50 == 0:
            log(f"  {cfg.name}: {new} searches, {time.time() - t0:.0f} s")
    if new:
        path.write_text(json.dumps(stored), encoding="utf-8")
    return {q["qid"]: out[q["qid"]] for q in queries}


def pool(runs: dict[str, dict[str, dict[str, Any]]], depth: int = 10) -> dict[str, list[str]]:
    """qid -> the union of every run's top `depth`, in a fixed order."""
    out: dict[str, list[str]] = {}
    for r in runs.values():
        for qid, v in r.items():
            seen = out.setdefault(qid, [])
            for t in v["ids"][:depth]:
                if t not in seen:
                    seen.append(t)
    return out


def repeat_at(families: list[str | None], k: int = 10) -> float:
    """Share of the top k that is a near-identical copy of a table already shown above it."""
    seen: set[str] = set()
    rep = 0
    top = families[:k]
    for f in top:
        if f and f in seen:
            rep += 1
        if f:
            seen.add(f)
    return rep / len(top) if top else 0.0


def per_query(queries: list[dict[str, Any]], r: dict[str, dict[str, Any]], recs: dict[str, dict[str, Any]],
              qrels: dict[str, dict[str, int]]) -> dict[str, dict[str, dict[str, float]]]:
    """kind -> metric -> qid -> value."""
    out: dict[str, dict[str, dict[str, float]]] = {"A": {}, "B": {}, "C": {}}

    def put(kind: str, metric: str, qid: str, v: float | None) -> None:
        if v is not None:
            out[kind].setdefault(metric, {})[qid] = float(v)

    for q in queries:
        v = r.get(q["qid"])
        if v is None:
            continue
        ids, kind = v["ids"], q["kind"]
        if kind == "A":
            g = {t: grade_attribute(q["spec"], recs[t]) for t in ids[:10] if t in recs}
            put("A", "fit@10", q["qid"], sum(x == 3 for x in g.values()) / 10)
            put("A", "robot_right@10", q["qid"], sum(x >= 1 for x in g.values()) / 10)
            ideal = M.dcg([3] * min(10, q["matching_tables"]))
            put("A", "ndcg@10", q["qid"], M.dcg([g.get(t, 0) for t in ids[:10]]) / ideal if ideal else None)
        elif kind == "B":
            rank = next((i for i, (t, f) in enumerate(zip(ids[:10], v["families"][:10]), 1)
                         if t == q["target"] or (f and f == q["target_family"])), None)
            put("B", "mrr@10", q["qid"], 1 / rank if rank else 0.0)
            for k in (1, 5, 10):
                put("B", f"hit@{k}", q["qid"], float(bool(rank and rank <= k)))
            deep = v.get("deep") or []
            put("B", "hit@100", q["qid"], float(q["target"] in deep or bool(rank)))
        elif kind == "C":
            qr = qrels.get(q["qid"], {})
            for m in ("ndcg@10", "precision@10", "success@10", "mrr@10", "unjudged@10"):
                put("C", m, q["qid"], M.METRICS[m](ids, qr))
        if kind in out:
            put(kind, "repeat@10", q["qid"], repeat_at(v["families"]))
    return out


def summarise(pq: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind, ms in pq.items():
        out[kind] = {}
        for m, vals in ms.items():
            xs = list(vals.values())
            lo, hi = M.bootstrap_ci(xs)
            out[kind][m] = {"mean": round(M.mean(xs), 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(xs)}
    return out


def compare(base: dict[str, dict[str, dict[str, float]]], other: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    """other minus base on each set's headline metric."""
    out = {}
    for kind, metric in HEADLINE.items():
        a, b = base.get(kind, {}).get(metric, {}), other.get(kind, {}).get(metric, {})
        if a and b:
            d = M.paired_bootstrap(a, b)
            d["p_randomization"] = round(M.randomization_test(a, b), 4)
            d["metric"] = metric
            out[kind] = d
    return out


def latency(r: dict[str, dict[str, Any]], stage: str = "total") -> dict[str, float]:
    xs = sorted(v["timings_ms"].get(stage, 0.0) for v in r.values() if stage in v["timings_ms"])
    if not xs:
        return {}
    pick = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"p50_ms": round(pick(0.5), 1), "p95_ms": round(pick(0.95), 1), "max_ms": round(xs[-1], 1), "n": len(xs)}
