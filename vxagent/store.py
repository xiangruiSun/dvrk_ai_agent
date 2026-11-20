"""SQLite persistence shared by the runtime: traces, approvals, idempotency keys, one-use motion
command ids and the audit log. One file, WAL mode, safe for a web server's threads."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, agent TEXT, task TEXT, user_id TEXT, role TEXT, site TEXT, status TEXT,
    started REAL, ended REAL, answer TEXT, meta TEXT);
CREATE TABLE IF NOT EXISTS spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, kind TEXT, name TEXT, started REAL,
    latency_ms REAL, prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0,
    ok INTEGER DEFAULT 1, error_kind TEXT, error TEXT, attempts INTEGER DEFAULT 1,
    seq_start INTEGER, seq_end INTEGER, input TEXT, output TEXT);
CREATE INDEX IF NOT EXISTS spans_run ON spans(run_id, id);
CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, call_key TEXT, tool TEXT, arguments TEXT,
    requested_by TEXT, status TEXT, decided_by TEXT, created REAL, decided REAL,
    robot_seq INTEGER, profile_version TEXT);
CREATE INDEX IF NOT EXISTS approvals_run ON approvals(run_id, call_key);
CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT PRIMARY KEY, tool TEXT, result TEXT, created REAL);
CREATE TABLE IF NOT EXISTS commands (
    command_id TEXT PRIMARY KEY, run_id TEXT, tool TEXT, arguments TEXT, user_id TEXT,
    seq_before INTEGER, seq_after INTEGER, status TEXT, created REAL, ended REAL);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL, user_id TEXT, role TEXT, action TEXT,
    resource TEXT, allowed INTEGER, detail TEXT);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS proposals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, site TEXT, target TEXT, file_type TEXT, patch TEXT, patched TEXT,
    proposer TEXT, status TEXT, created REAL, decided_by TEXT, decided REAL, decision_reason TEXT,
    base_hash TEXT, limit_effect TEXT, invalidates_calibration INTEGER DEFAULT 0, effect TEXT,
    applied_version INTEGER);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, n INTEGER, type TEXT, data TEXT, at REAL,
    robot_seq INTEGER);
CREATE INDEX IF NOT EXISTS events_run ON events(run_id, n);
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, n INTEGER, next_node TEXT, state TEXT, at REAL,
    events_n INTEGER);
CREATE INDEX IF NOT EXISTS checkpoints_run ON checkpoints(run_id, n);
"""


class Database:
    def __init__(self, path: Path | str | None = None):
        self.path = str(path or get_settings().db_path)
        self._lock = threading.RLock()
        self._con = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._con.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._con.execute("PRAGMA journal_mode=WAL")
        self._con.executescript(SCHEMA)

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        with self._lock:
            cur = self._con.execute(sql, tuple(params))
            return cur.lastrowid or 0

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._con.execute(sql, tuple(params)).fetchall()]

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def audit(self, user_id: str, role: str, action: str, resource: str, allowed: bool, detail: str = "") -> None:
        self.execute("INSERT INTO audit(at,user_id,role,action,resource,allowed,detail) VALUES (?,?,?,?,?,?,?)",
                     (time.time(), user_id, role, action, resource, int(allowed), detail[:500]))

    def close(self) -> None:
        with self._lock:
            self._con.close()


_default: Database | None = None
_default_lock = threading.Lock()


def get_db() -> Database:
    global _default
    with _default_lock:
        if _default is None:
            _default = Database()
        return _default


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
