"""The write path, stage 3: catalog records and their vectors into Elasticsearch.

    python -m vxfind.ingest.index build --snapshot old            a new index version, then switch the alias to it
    python -m vxfind.ingest.index sync  --from old --to new       apply what changed between two snapshots in place

An index version is named after the snapshot it was built from. Readers use the alias
`vxfind-tables`, so a new version is built beside the serving one and takes over in one alias
switch; the previous version stays until it is deleted, which is the rollback.

`sync` is the incremental path: it deletes what was deleted, and writes what was created or
changed, without rebuilding. It reports how long each took, which is the cost of keeping the
index close to the catalog.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "vxfind"
ES_URL = "http://127.0.0.1:9200"
ALIAS = "vxfind-tables"
MODELS = {"nomic-embed-text": "emb_nomic", "bge-m3": "emb_bge"}

SETTINGS: dict[str, Any] = {
    "number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1",
    "analysis": {
        # Dataset and column names are identifiers: so101_PickPlace-v2, observation.images.wrist.
        # They are split at punctuation, at case changes and between letters and digits, and the
        # unsplit form is kept as well, so both "so101" and "101" and "pick" find the table.
        "filter": {"ident_parts": {"type": "word_delimiter_graph", "preserve_original": True, "split_on_case_change": True,
                                   "split_on_numerics": True, "catenate_words": False}},
        "analyzer": {"ident": {"type": "custom", "tokenizer": "whitespace",
                               "char_filter": ["ident_punct"], "filter": ["ident_parts", "lowercase", "flatten_graph"]},
                     "ident_query": {"type": "custom", "tokenizer": "whitespace", "char_filter": ["ident_punct"],
                                     "filter": ["ident_parts", "lowercase"]}},
        "char_filter": {"ident_punct": {"type": "pattern_replace", "pattern": "[/.,;:()\\[\\]{}\"'`|]+", "replacement": " "}},
    },
}
MAPPINGS: dict[str, Any] = {
    "dynamic": "strict",
    "properties": {
        "table_id": {"type": "keyword"}, "owner": {"type": "keyword"}, "url": {"type": "keyword", "index": False},
        "name": {"type": "text", "analyzer": "ident", "search_analyzer": "ident_query", "fields": {"raw": {"type": "keyword"}}},
        "description": {"type": "text"},
        "summary": {"type": "text"},
        "schema_text": {"type": "text", "analyzer": "ident", "search_analyzer": "ident_query"},
        "robot_type": {"type": "keyword", "fields": {"text": {"type": "text", "analyzer": "ident", "search_analyzer": "ident_query"}}},
        "codebase_version": {"type": "keyword"}, "licence": {"type": "keyword"}, "gated": {"type": "keyword"},
        "tags": {"type": "keyword"}, "cameras": {"type": "keyword"}, "family": {"type": "keyword"},
        "fps": {"type": "float"}, "total_episodes": {"type": "long"}, "total_frames": {"type": "long"},
        "total_tasks": {"type": "long"}, "n_cameras": {"type": "integer"}, "action_dim": {"type": "integer"},
        "family_size": {"type": "integer"}, "downloads": {"type": "long"}, "likes": {"type": "long"},
        "has_description": {"type": "boolean"}, "is_scratch": {"type": "boolean"}, "is_head": {"type": "boolean"},
        "created": {"type": "date"}, "last_modified": {"type": "date"},
        "content_hash": {"type": "keyword"}, "schema_hash": {"type": "keyword"}, "revision": {"type": "keyword"},
        "emb_nomic": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"},
        "emb_bge": {"type": "dense_vector", "dims": 1024, "index": True, "similarity": "cosine"},
    },
}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def client(url: str = ES_URL) -> Any:
    from elasticsearch import Elasticsearch
    # The client keeps 10 connections per node unless told otherwise. FOUND BY THE LOAD TEST: with 32
    # callers, searches queued inside the client for a connection and a few waited seconds
    # (verification/vxfind_capacity_first_run.json). The pool is sized for the callers.
    return Elasticsearch(url, request_timeout=120, connections_per_node=64)


def schema_text(cols: list[dict[str, Any]]) -> str:
    out = []
    for c in cols:
        out.append(c["name"])
        out += c.get("names") or []
    return " ".join(out)[:4000]


def to_doc(r: dict[str, Any]) -> dict[str, Any]:
    cols = json.loads(r["columns_json"])
    doc = {"table_id": r["table_id"], "owner": r["owner"], "url": r["url"], "name": r["name"],
           "description": r["description"] or None, "summary": r["summary"], "schema_text": schema_text(cols),
           "robot_type": r["robot_type"], "codebase_version": r["codebase_version"], "licence": r["licence"],
           "gated": r["gated"], "tags": json.loads(r["tags_json"]),
           "cameras": [c.rsplit(".", 1)[-1] for c in json.loads(r["cameras_json"])], "family": r["family"],
           "fps": r["fps"], "total_episodes": r["total_episodes"], "total_frames": r["total_frames"],
           "total_tasks": r["total_tasks"], "n_cameras": r["n_cameras"], "action_dim": r["action_dim"],
           "family_size": r["family_size"], "downloads": r["downloads"], "likes": r["likes"],
           "has_description": r["has_description"], "is_scratch": r["is_scratch"], "is_head": r["is_head"],
           "created": r["created"], "last_modified": r["last_modified"], "content_hash": r["content_hash"],
           "schema_hash": r["schema_hash"], "revision": r["revision"]}
    return {k: v for k, v in doc.items() if v is not None}


def rows(snapshot: str, only: set[str] | None = None) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    import pyarrow.parquet as pq
    table = pq.read_table(OUT / snapshot / "tables.parquet")
    ids = json.loads((OUT / snapshot / "emb_ids.json").read_text(encoding="utf-8"))
    pos = {t: i for i, t in enumerate(ids)}
    vecs = {f: np.load(OUT / snapshot / f"emb_{m.replace(':', '_')}.npy", mmap_mode="r") for m, f in MODELS.items()
            if (OUT / snapshot / f"emb_{m.replace(':', '_')}.npy").exists()}
    for batch in table.to_batches(max_chunksize=2000):
        for r in batch.to_pylist():
            if only is not None and r["table_id"] not in only:
                continue
            i = pos[r["table_id"]]
            yield r, {f: v[i] for f, v in vecs.items()}


def actions(index: str, snapshot: str, only: set[str] | None = None) -> Iterator[dict[str, Any]]:
    for r, vec in rows(snapshot, only):
        doc = to_doc(r)
        for f, v in vec.items():
            doc[f] = v.tolist()
        yield {"_op_type": "index", "_index": index, "_id": r["table_id"], "_source": doc}


def build(snapshot: str, switch: bool = True) -> dict[str, Any]:
    from elasticsearch import helpers
    es = client()
    index = f"{ALIAS}-{snapshot}-{time.strftime('%Y%m%d%H%M%S')}"
    es.indices.create(index=index, settings=SETTINGS, mappings=MAPPINGS)
    t0 = time.time()
    ok = failed = 0
    for success, item in helpers.parallel_bulk(es, actions(index, snapshot), thread_count=4, chunk_size=500, raise_on_error=False):
        ok += success
        failed += not success
        if not success and failed <= 3:
            log(f"  failed: {json.dumps(item)[:300]}")
        if (ok + failed) % 20000 == 0:
            log(f"  {ok + failed} documents")
    t_bulk = time.time() - t0
    es.indices.put_settings(index=index, settings={"refresh_interval": "1s"})
    es.indices.refresh(index=index)
    t1 = time.time()
    es.indices.forcemerge(index=index, max_num_segments=1)
    t_merge = time.time() - t1
    st = es.indices.stats(index=index)["indices"][index]["primaries"]
    out = {"index": index, "snapshot": snapshot, "documents": ok, "failed": failed, "bulk_seconds": round(t_bulk, 1),
           "documents_per_second": round(ok / t_bulk, 1), "force_merge_seconds": round(t_merge, 1),
           "store_mb": round(st["store"]["size_in_bytes"] / 1e6, 1), "count": es.count(index=index)["count"]}
    if switch:
        out["alias"] = switch_alias(es, index)
    # A build that does not take over the alias is a copy (for example to try the sync on); its
    # figures go to their own file and the serving index's stay.
    (OUT / snapshot / ("index_stats.json" if switch else "index_stats_copy.json")).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def switch_alias(es: Any, index: str) -> dict[str, Any]:
    """Point the alias at one index, in one request. Returns what it pointed at before."""
    before = sorted(es.indices.get_alias(name=ALIAS)) if es.indices.exists_alias(name=ALIAS) else []
    acts = [{"remove": {"index": i, "alias": ALIAS}} for i in before] + [{"add": {"index": index, "alias": ALIAS}}]
    es.indices.update_aliases(actions=acts)
    return {"now": index, "before": before}


def sync(old: str, new: str, index: str | None = None) -> dict[str, Any]:
    """Apply the changes between two snapshots to the serving index, in place."""
    import pyarrow.parquet as pq
    from elasticsearch import helpers
    es = client()
    index = index or sorted(es.indices.get_alias(name=ALIAS))[0]
    ch = pq.read_table(OUT / new / "changes.parquet").to_pylist()
    deleted = [c["table_id"] for c in ch if c["change"] == "deleted"]
    upsert = {c["table_id"] for c in ch if c["change"] in ("created", "changed")}
    t0 = time.time()
    d_ok = sum(ok for ok, _ in helpers.streaming_bulk(es, ({"_op_type": "delete", "_index": index, "_id": i} for i in deleted),
                                                      chunk_size=1000, raise_on_error=False))
    t_del = time.time() - t0
    t1 = time.time()
    u_ok = sum(ok for ok, _ in helpers.parallel_bulk(es, actions(index, new, upsert), thread_count=4, chunk_size=500, raise_on_error=False))
    t_up = time.time() - t1
    es.indices.refresh(index=index)
    serving = sorted(es.indices.get_alias(name=ALIAS)) if es.indices.exists_alias(name=ALIAS) else []
    out = {"index": index, "is_the_serving_index": index in serving, "from": old, "to": new, "deleted": len(deleted),
           "deleted_ok": d_ok, "written": len(upsert), "written_ok": u_ok, "delete_seconds": round(t_del, 2),
           "write_seconds": round(t_up, 2), "count_after": es.count(index=index)["count"]}
    (OUT / new / "sync_stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Index catalog records into Elasticsearch.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--snapshot", required=True)
    b.add_argument("--no-switch", action="store_true")
    s = sub.add_parser("sync")
    s.add_argument("--from", dest="old", required=True)
    s.add_argument("--to", dest="new", required=True)
    s.add_argument("--index", help="the index to bring up to date (default: the serving one)")
    a = ap.parse_args()
    log(json.dumps(build(a.snapshot, not a.no_switch) if a.cmd == "build" else sync(a.old, a.new, a.index)))
