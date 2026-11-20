"""The authoritative store. Indexes are derived from it and can always be rebuilt.

A *unit* is what retrieval returns and what evaluation judges: a documentation section, the
question of an issue thread, the replies of an issue thread, one category of one release in the
change log, or one property of a configuration schema. Chunks are cut from units at index time
and are never the unit of an answer.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS units (
    unit_id TEXT PRIMARY KEY, doc_id TEXT, source TEXT, kind TEXT, ord INTEGER, title TEXT, path TEXT,
    text TEXT, url TEXT, generation TEXT, version TEXT, authority TEXT, labels TEXT, refs TEXT, meta TEXT,
    content_hash TEXT);
CREATE INDEX IF NOT EXISTS units_doc ON units(doc_id, ord);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, report TEXT);
CREATE TABLE IF NOT EXISTS changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, source TEXT, unit_id TEXT, change TEXT);
"""

PUBLIC_SOURCES = ("doc", "thread", "changelog", "schema")
SOURCES = PUBLIC_SOURCES + ("note",)          # note: a site's own notes, a private collection
PUBLIC = "public"
COLUMNS = ("unit_id", "doc_id", "source", "kind", "ord", "title", "path", "text", "url", "generation", "version",
           "authority", "labels", "refs", "meta", "content_hash", "collection")


@dataclass
class Unit:
    unit_id: str
    doc_id: str
    source: str                  # doc | thread | changelog | schema
    kind: str                    # section | question | replies | release | property
    ord: int
    title: str
    path: list[str]              # breadcrumb, outermost first
    text: str
    url: str = ""
    generation: str = "both"     # Classic | Si | both
    version: str = ""            # release a change-log unit belongs to
    authority: str = "official"  # official | site | maintainer | community
    labels: list[str] = field(default_factory=list)
    refs: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    collection: str = PUBLIC     # "public", or "site:<name>" for text only that site may read


class CorpusStore:
    def __init__(self, path: Path | str | None = None):
        self.path = str(path or get_settings().db_path)
        self._con = sqlite3.connect(self.path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.executescript(SCHEMA)
        if "collection" not in {r["name"] for r in self._con.execute("PRAGMA table_info(units)")}:
            with self._con:                      # a store made before private collections existed
                self._con.execute(f"ALTER TABLE units ADD COLUMN collection TEXT DEFAULT '{PUBLIC}'")

    def replace_source(self, source: str, units: Iterable[Unit]) -> dict[str, int]:
        """Make the stored units of one source equal to `units`. Returns what changed, so a
        repeated ingestion of unchanged files reports zero writes and a removed page is removed."""
        new = {u.unit_id: u for u in units}
        old = {r["unit_id"]: r["content_hash"] for r in
               self._con.execute("SELECT unit_id, content_hash FROM units WHERE source=?", (source,))}
        added = [u for k, u in new.items() if k not in old]
        changed = [u for k, u in new.items() if k in old and old[k] != u.content_hash]
        removed = [k for k in old if k not in new]
        at = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._con:
            self._con.executemany("DELETE FROM units WHERE unit_id=?", [(k,) for k in removed])
            self._con.executemany(
                f"INSERT OR REPLACE INTO units ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})",
                [(u.unit_id, u.doc_id, u.source, u.kind, u.ord, u.title, json.dumps(u.path), u.text, u.url,
                  u.generation, u.version, u.authority, json.dumps(u.labels), json.dumps(u.refs),
                  json.dumps(u.meta), u.content_hash, u.collection) for u in added + changed])
            # Change events: what an index has to redo, and nothing else.
            self._con.executemany(
                "INSERT INTO changes(at, source, unit_id, change) VALUES (?,?,?,?)",
                [(at, source, u.unit_id, "added") for u in added] + [(at, source, u.unit_id, "changed") for u in changed]
                + [(at, source, k, "removed") for k in removed])
        return {"added": len(added), "changed": len(changed), "removed": len(removed),
                "unchanged": len(new) - len(added) - len(changed)}

    def units(self, sources: Iterable[str] | None = None) -> list[Unit]:
        sql, params = "SELECT * FROM units", ()
        if sources is not None:
            sources = tuple(sources)
            sql += f" WHERE source IN ({','.join('?' * len(sources))})"
            params = sources
        rows = self._con.execute(sql + " ORDER BY source, doc_id, ord", params).fetchall()
        return [Unit(unit_id=r["unit_id"], doc_id=r["doc_id"], source=r["source"], kind=r["kind"], ord=r["ord"],
                     title=r["title"], path=json.loads(r["path"]), text=r["text"], url=r["url"],
                     generation=r["generation"], version=r["version"], authority=r["authority"],
                     labels=json.loads(r["labels"]), refs=json.loads(r["refs"]), meta=json.loads(r["meta"]),
                     content_hash=r["content_hash"], collection=r["collection"] or PUBLIC) for r in rows]

    def counts(self, sources: Iterable[str] | None = None) -> dict[str, int]:
        keep = None if sources is None else set(sources)
        return {r["source"]: r["n"] for r in
                self._con.execute("SELECT source, COUNT(*) AS n FROM units GROUP BY source")
                if keep is None or r["source"] in keep}

    def corpus_hash(self, public_only: bool = True) -> str:
        """Hash of the unit ids and content hashes. By default over the public sources, which is
        what the retrieval experiments run on and what their lock file records."""
        return units_hash((r["unit_id"], r["content_hash"]) for r in self._con.execute(
            "SELECT unit_id, content_hash FROM units" + (" WHERE collection=?" if public_only else "")
            + " ORDER BY unit_id", (PUBLIC,) if public_only else ()))

    def last_change_id(self) -> int:
        return int(self._con.execute("SELECT COALESCE(MAX(id), 0) FROM changes").fetchone()[0])

    def changes_since(self, change_id: int) -> list[dict[str, Any]]:
        return [dict(r) for r in self._con.execute(
            "SELECT id, at, source, unit_id, change FROM changes WHERE id>? ORDER BY id", (change_id,))]

    def log_run(self, report: dict[str, Any]) -> None:
        with self._con:
            self._con.execute("INSERT INTO runs(at, report) VALUES (?,?)",
                              (time.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(report)))

    def close(self) -> None:
        self._con.close()


def units_hash(pairs: Iterable[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for unit_id, content_hash in pairs:
        h.update(f"{unit_id}:{content_hash}\n".encode())
    return h.hexdigest()[:16]
