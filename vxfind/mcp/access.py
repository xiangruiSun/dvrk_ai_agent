"""Who may see a table, and who may use it for training.

Two rules, both DECLARED for this project (no real organisation's policy is modelled):

  seeing   Tables of a restricted owner are shown only to members of the groups that owner is
           restricted to. To everyone else they do not exist: not in search results, not in
           details, not in an error message.
  using    Using a table for training needs a request that a mentor has approved. Searching and
           reading descriptions needs no approval. A person never approves their own request,
           and the language model has no tool that approves anything.

A table the source site marks as gated needs, in addition, its owner's consent on that site.
This system cannot give that; it says so.
"""
from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  request_id TEXT PRIMARY KEY, table_id TEXT NOT NULL, user_id TEXT NOT NULL, purpose TEXT NOT NULL,
  status TEXT NOT NULL, created REAL NOT NULL, decided_by TEXT, decided_at REAL, note TEXT);
CREATE INDEX IF NOT EXISTS requests_user ON requests(user_id, table_id);
CREATE TABLE IF NOT EXISTS audit (at REAL, user_id TEXT, role TEXT, action TEXT, table_id TEXT, allowed INTEGER, detail TEXT);
"""


class AccessDenied(Exception):
    pass


class AccessStore:
    def __init__(self, path: Path | str = ":memory:", restricted: dict[str, list[str]] | None = None):
        self._con = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._con.row_factory = sqlite3.Row
        self._con.executescript(SCHEMA)
        self._lock = threading.Lock()
        self.restricted = restricted or {}            # owner -> groups that may see its tables

    def _q(self, sql: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._con.execute(sql, args).fetchall()]

    def audit(self, user_id: str, role: str, action: str, table_id: str, allowed: bool, detail: str = "") -> None:
        self._q("INSERT INTO audit VALUES (?,?,?,?,?,?,?)", (time.time(), user_id, role, action, table_id, int(allowed), detail[:300]))

    # ------------------------------------------------------------------ seeing
    def hidden_owners(self, groups: frozenset[str] | set[str]) -> list[str]:
        """Owners whose tables this person must not see."""
        return sorted(o for o, allowed in self.restricted.items() if not (set(allowed) & set(groups)))

    def may_see(self, table_id: str, groups: frozenset[str] | set[str]) -> bool:
        return table_id.split("/", 1)[0] not in self.hidden_owners(groups)

    # ------------------------------------------------------------------ using
    def status(self, user_id: str, table_id: str) -> dict[str, Any]:
        rows = self._q("SELECT * FROM requests WHERE user_id = ? AND table_id = ? ORDER BY created DESC LIMIT 1", (user_id, table_id))
        if not rows:
            return {"table_id": table_id, "use": "not_requested", "request_id": None}
        r = rows[0]
        return {"table_id": table_id, "use": r["status"], "request_id": r["request_id"], "decided_by": r["decided_by"],
                "note": r["note"]}

    def request(self, user_id: str, role: str, table_id: str, purpose: str) -> dict[str, Any]:
        """File a request. Asking twice for the same table returns the request already open."""
        cur = self.status(user_id, table_id)
        if cur["use"] in ("pending", "approved"):
            return {**cur, "created_now": False}
        rid = "req_" + uuid.uuid4().hex[:10]
        self._q("INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?)", (rid, table_id, user_id, purpose[:500], "pending", time.time(), None, None, None))
        self.audit(user_id, role, "request_use", table_id, True, purpose[:120])
        return {"table_id": table_id, "use": "pending", "request_id": rid, "created_now": True}

    def pending(self) -> list[dict[str, Any]]:
        return self._q("SELECT request_id, table_id, user_id, purpose, created FROM requests WHERE status = 'pending' ORDER BY created")

    def decide(self, mentor_id: str, role: str, request_id: str, approve: bool, note: str = "") -> dict[str, Any]:
        if role != "mentor":
            self.audit(mentor_id, role, "decide", request_id, False, "not a mentor")
            raise AccessDenied("only a mentor decides a request")
        rows = self._q("SELECT * FROM requests WHERE request_id = ?", (request_id,))
        if not rows:
            raise AccessDenied("no such request")
        r = rows[0]
        if r["user_id"] == mentor_id:
            self.audit(mentor_id, role, "decide", r["table_id"], False, "own request")
            raise AccessDenied("a person does not decide their own request")
        if r["status"] != "pending":
            return {"request_id": request_id, "table_id": r["table_id"], "use": r["status"], "changed": False}
        status = "approved" if approve else "refused"
        self._q("UPDATE requests SET status = ?, decided_by = ?, decided_at = ?, note = ? WHERE request_id = ?",
                (status, mentor_id, time.time(), note[:300], request_id))
        self.audit(mentor_id, role, "decide", r["table_id"], True, status)
        return {"request_id": request_id, "table_id": r["table_id"], "use": status, "changed": True}

    def audit_rows(self) -> list[dict[str, Any]]:
        return self._q("SELECT * FROM audit ORDER BY at")
