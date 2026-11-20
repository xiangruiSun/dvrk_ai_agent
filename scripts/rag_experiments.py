"""Retrieval experiments on the real corpus, in three stages, then one run on the held-out queries.

    python scripts/rag_experiments.py

  E  tokenizers and embedding models, with chunking held at "section"
  C  chunking strategies, for the best lexical and the best dense retriever of stage E
  R  fusion of the best lexical and dense retrievers, at the best chunking of stage C

All three stages use the development queries. The selection policy (vxrag/evaluation/runner.py)
then picks one configuration, and only that configuration is scored on the test queries. The
choice is written to verification/rag_final_test.lock.json; a later run that would score the
test queries with a different configuration is refused.

Writes verification/rag_experiments.json. Labels come from the structure of the sources
(vxrag/evaluation/labels.py); no number here was typed by hand.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vxrag.evaluation.labels import build_queries                 # noqa: E402
from vxrag.evaluation.runner import compare, run, select          # noqa: E402
from vxrag.ingest import ingest                                   # noqa: E402
from vxrag.retrieval import Index                                 # noqa: E402
from vxrag.store import CorpusStore                               # noqa: E402

OUT = ROOT / "verification"
LOCK = OUT / "rag_final_test.lock.json"
BASELINE = ("bm25:plain", "fixed")
LEXICAL = ["bm25:plain", "bm25:ident"]
DENSE = ["dense:nomic-embed-text", "dense:bge-m3"]
CHUNKINGS = ["fixed", "section", "section_ctx"]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def slim(r: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in r.items() if k != "per_query"}


def main() -> int:
    store = CorpusStore()
    report = ingest(store)
    units = store.units()
    queries = build_queries(units)
    counts = Counter((q.category, q.split) for q in queries)
    log(f"corpus {report['units']} hash {report['corpus_hash']}; queries {dict(counts)}")
    indexes: dict[str, Index] = {}
    runs: dict[tuple[str, str], dict[str, Any]] = {}

    def go(strategy: str, chunking: str, split: str = "dev") -> dict[str, Any]:
        key = (strategy, chunking)
        if split == "dev" and key in runs:
            return runs[key]
        if chunking not in indexes:
            indexes[chunking] = Index(units, chunking)
        r = run(indexes[chunking], queries, strategy, split)
        log(f"  [{split}] {strategy} @ {chunking}: nDCG@10 {r['ndcg@10']} recall@10 {r['recall@10']} "
            f"p95 {r['latency_ms']['p95']} ms")
        if split == "dev":
            runs[key] = r
        return r

    best_of = lambda names, chunking: max(names, key=lambda s: go(s, chunking)["ndcg@10"])      # noqa: E731

    log("stage E: tokenizers and embedding models")
    base = go(*BASELINE)
    for s in LEXICAL + DENSE:
        go(s, "section")
    best_lex, best_dense = best_of(LEXICAL, "section"), best_of(DENSE, "section")

    log(f"stage C: chunking for {best_lex} and {best_dense}")
    for c in CHUNKINGS:
        go(best_lex, c)
        go(best_dense, c)
    best_chunking = max(CHUNKINGS, key=lambda c: max(go(best_lex, c)["ndcg@10"], go(best_dense, c)["ndcg@10"]))

    log(f"stage R: fusion at {best_chunking}")
    fused = [f"rrf:{best_lex}+{best_dense}", f"rrf:{best_lex}+{DENSE[0]}+{DENSE[1]}"]
    for s in fused:
        go(s, best_chunking)

    all_runs = list(runs.values())
    selection = select(all_runs, f"{BASELINE[0]} @ {BASELINE[1]}")
    log(f"selected: {selection['selected']}")

    # The held-out queries are scored once, for the selected configuration (and the baseline for reference).
    chosen = {"strategy": selection["selected_strategy"], "chunking": selection["selected_chunking"],
              "corpus_hash": report["corpus_hash"], "queries": len(queries)}
    if LOCK.exists():
        locked = json.loads(LOCK.read_text(encoding="utf-8"))
        if {k: locked.get(k) for k in chosen} != chosen:
            print(f"refused: the test queries were already scored for {locked}. Scoring them again for {chosen} "
                  f"would turn the test set into a tuning set. Delete {LOCK.name} only if the protocol itself "
                  "has changed, and say so in the report.", file=sys.stderr)
            return 1
    final = go(chosen["strategy"], chosen["chunking"], split="test")
    final_base = go(*BASELINE, split="test")
    LOCK.write_text(json.dumps({**chosen, "locked": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")

    worst = sorted(runs[(chosen["strategy"], chosen["chunking"])]["per_query"], key=lambda r: r["ndcg"])[:12]
    texts = {q.qid: q for q in queries}
    out = {
        "task": "rag_experiments", "ran": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus": {"units": report["units"], "hash": report["corpus_hash"], "validation": report["validation"],
                   "sources": {k: {kk: vv for kk, vv in v.items() if kk != "written"} for k, v in report["sources"].items()}},
        "queries": {"total": len(queries), "by_category_and_split": {f"{c}/{s}": n for (c, s), n in sorted(counts.items())},
                    "labels": "built by code from cross-references, thread replies and rare identifiers; weak labels"},
        "chunks": {c: len(ix.chunks) for c, ix in indexes.items()},
        "index_build_seconds": {c: ix.build_seconds for c, ix in indexes.items()},
        "stage_E": [slim(runs[(s, "section")]) for s in LEXICAL + DENSE],
        "stage_C": [slim(runs[(s, c)]) for s in (best_lex, best_dense) for c in CHUNKINGS],
        "stage_R": [slim(runs[(s, best_chunking)]) for s in (best_lex, best_dense, *fused)],
        "baseline": slim(base),
        "comparisons": {
            "ident_vs_plain_tokenizer": compare(runs[("bm25:plain", "section")], runs[("bm25:ident", "section")]),
            "section_ctx_vs_fixed_chunking": compare(runs[(best_lex, "fixed")], runs[(best_lex, "section_ctx")]),
            "fusion_vs_best_lexical": compare(runs[(best_lex, best_chunking)], runs[(fused[0], best_chunking)]),
            "fusion_vs_best_dense": compare(runs[(best_dense, best_chunking)], runs[(fused[0], best_chunking)]),
        },
        "selection": selection,
        "final_test": {"selected": slim(final), "baseline": slim(final_base),
                       "difference": compare(final_base, final)},
        "hardest_dev_queries_for_selected": [
            {"qid": w["qid"], "category": w["category"], "ndcg": round(w["ndcg"], 3),
             "query": texts[w["qid"]].text.split(chr(10))[0][:120], "expected": sorted(texts[w["qid"]].relevant, key=lambda u: -texts[w["qid"]].relevant[u])[:2],
             "returned_top3": w["top"]} for w in worst],
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "rag_experiments.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {OUT / 'rag_experiments.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
