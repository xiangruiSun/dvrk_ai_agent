"""The catalog database: the record of every table, one row each, in SQLite.

    python -m vxfind.catalog.store build --snapshot new

This is the system of record the search index is built from, and it is deliberately a separate
store: the index is a copy made for searching and is allowed to lag. Here the database is loaded
from the newer snapshot and the index from the older one, so the two disagree in the way they do
in production (a table created last week is in the database and not yet in the index), and the
agent has to deal with it.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "vxfind"
DB = OUT / "catalog.db"

COLUMNS = ["table_id", "owner", "name", "url", "created", "last_modified", "downloads", "likes", "licence", "schema_status",
           "codebase_version", "robot_type", "fps", "total_episodes", "total_frames", "total_tasks", "data_mb", "video_mb",
           "n_columns", "n_cameras", "action_dim", "state_dim", "has_description", "is_scratch", "content_hash",
           "schema_hash", "summary", "description", "columns_json", "cameras_json", "gated", "revision", "family",
           "family_size", "is_head"]
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS tables ({", ".join(c + (" TEXT PRIMARY KEY" if c == "table_id" else "") for c in COLUMNS)});
CREATE INDEX IF NOT EXISTS tables_owner ON tables(owner);
CREATE INDEX IF NOT EXISTS tables_family ON tables(family);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


def build(snapshot: str, path: Path = DB) -> dict[str, Any]:
    import pyarrow.parquet as pq
    t0 = time.time()
    tmp = path.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript(SCHEMA)
    table = pq.read_table(OUT / snapshot / "tables.parquet", columns=COLUMNS)
    n = 0
    for batch in table.to_batches(max_chunksize=5000):
        rows = [tuple(r[c] for c in COLUMNS) for r in batch.to_pylist()]
        con.executemany(f"INSERT INTO tables VALUES ({','.join('?' * len(COLUMNS))})", rows)
        n += len(rows)
    manifest = json.loads((ROOT / "data" / "hub" / "MANIFEST.json").read_text(encoding="utf-8"))
    snap = manifest["snapshots"][snapshot]
    for k, v in {"snapshot": snapshot, "published": snap["published"], "revision": snap["revision"], "tables": n,
                 "built": time.strftime("%Y-%m-%dT%H:%M:%S")}.items():
        con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, str(v)))
    con.commit()
    con.close()
    path.unlink(missing_ok=True)
    tmp.rename(path)                                   # readers see the old file or the new one, never half of one
    return {"snapshot": snapshot, "tables": n, "seconds": round(time.time() - t0, 1), "file_mb": round(path.stat().st_size / 1e6, 1)}


class Catalog:
    """Read access to the catalog database."""

    def __init__(self, path: Path | str = DB):
        self._con = sqlite3.connect(str(path), check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._lock = threading.Lock()

    def _q(self, sql: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._con.execute(sql, args).fetchall()]

    def meta(self) -> dict[str, str]:
        return {r["k"]: r["v"] for r in self._q("SELECT k, v FROM meta")}

    def get(self, table_id: str) -> dict[str, Any] | None:
        rows = self._q("SELECT * FROM tables WHERE table_id = ?", (table_id,))
        return _typed(rows[0]) if rows else None

    def many(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}
        rows = self._q(f"SELECT * FROM tables WHERE table_id IN ({','.join('?' * len(ids))})", tuple(ids))
        return {r["table_id"]: _typed(r) for r in rows}

    def family(self, table_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rec = self.get(table_id)
        if rec is None:
            return []
        return [_typed(r) for r in self._q("SELECT * FROM tables WHERE family = ? AND table_id != ? ORDER BY "
                                           "CAST(total_episodes AS INTEGER) DESC LIMIT ?", (rec["family"], table_id, limit))]

    def by_owner(self, owner: str, limit: int = 50) -> list[dict[str, Any]]:
        return [_typed(r) for r in self._q("SELECT * FROM tables WHERE owner = ? ORDER BY last_modified DESC LIMIT ?", (owner, limit))]

    def count(self) -> int:
        return int(self._q("SELECT COUNT(*) AS n FROM tables")[0]["n"])


def _typed(r: dict[str, Any]) -> dict[str, Any]:
    out = dict(r)
    for k in ("columns_json", "cameras_json"):
        out[k[:-5]] = json.loads(out.pop(k) or "[]")
    for k in ("has_description", "is_scratch", "is_head"):
        out[k] = str(out[k]).lower() in ("1", "true")
    for k in ("total_episodes", "total_frames", "total_tasks", "n_columns", "n_cameras", "action_dim", "state_dim",
              "family_size", "downloads", "likes"):
        out[k] = int(float(out[k])) if out[k] not in (None, "") else None
    for k in ("fps", "data_mb", "video_mb"):
        out[k] = float(out[k]) if out[k] not in (None, "") else None
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the catalog database from a snapshot.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--snapshot", required=True)
    a = ap.parse_args()
    print(json.dumps(build(a.snapshot)))
