"""Short-term history and user memory.

Sessions: the turns of one conversation, keyed by user and session, so a conversation can be
picked up after a restart and one user never sees another's.

User memory: a few durable preferences per user. It is written only through the remember tool,
it is bounded and de-duplicated, and anything that looks like a credential is refused.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from ..store import Database, dumps

MAX_MEMORIES = 20
MAX_CHARS = 240

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    user_id TEXT, session_id TEXT, updated REAL, messages TEXT, PRIMARY KEY (user_id, session_id));
CREATE TABLE IF NOT EXISTS user_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, text TEXT, norm TEXT, created REAL);
CREATE INDEX IF NOT EXISTS user_memory_user ON user_memory(user_id);
"""
_SECRET = [re.compile(p, re.I) for p in (
    r"\b(pass(word|code|phrase)?|pwd|secret|token|api[ _-]?key|private key|credential)s?\b",
    r"\b[A-Za-z0-9+/_-]{28,}\b",                    # a long opaque string
    r"\b(sk|pk|ghp|vx|xox[abp])[_-][A-Za-z0-9_-]{12,}",
    r"-----BEGIN [A-Z ]+-----")]


class Refused(ValueError):
    pass


def _ready(db: Database) -> None:
    with db._lock:
        db._con.executescript(SCHEMA)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# ------------------------------------------------------------------------------ sessions

def save_session(db: Database, user_id: str, session_id: str, messages: list[dict[str, Any]]) -> None:
    _ready(db)
    db.execute("INSERT OR REPLACE INTO sessions(user_id,session_id,updated,messages) VALUES (?,?,?,?)",
               (user_id, session_id, time.time(), dumps(messages)))


def load_session(db: Database, user_id: str, session_id: str) -> list[dict[str, Any]]:
    _ready(db)
    row = db.one("SELECT messages FROM sessions WHERE user_id=? AND session_id=?", (user_id, session_id))
    return json.loads(row["messages"]) if row else []


def list_sessions(db: Database, user_id: str) -> list[str]:
    _ready(db)
    return [r["session_id"] for r in db.query("SELECT session_id FROM sessions WHERE user_id=? ORDER BY updated DESC", (user_id,))]


# ------------------------------------------------------------------------------ user memory

def remember(db: Database, user_id: str, text: str) -> dict[str, Any]:
    text = " ".join(text.split())
    if not text or len(text) > MAX_CHARS:
        raise Refused(f"a memory is one short statement of at most {MAX_CHARS} characters")
    if any(p.search(text) for p in _SECRET):
        raise Refused("this looks like a credential; credentials are never stored")
    _ready(db)
    norm = _norm(text)
    same = db.one("SELECT id FROM user_memory WHERE user_id=? AND norm=?", (user_id, norm))
    if same:
        return {"stored": False, "reason": "already remembered", "memory_id": same["id"]}
    count = db.one("SELECT COUNT(*) AS n FROM user_memory WHERE user_id=?", (user_id,))["n"]
    dropped = None
    if count >= MAX_MEMORIES:                        # bounded: the oldest one makes room
        oldest = db.one("SELECT id, text FROM user_memory WHERE user_id=? ORDER BY id LIMIT 1", (user_id,))
        db.execute("DELETE FROM user_memory WHERE id=?", (oldest["id"],))
        dropped = oldest["text"]
    mid = db.execute("INSERT INTO user_memory(user_id,text,norm,created) VALUES (?,?,?,?)", (user_id, text, norm, time.time()))
    return {"stored": True, "memory_id": mid, "dropped_to_make_room": dropped}


def memories(db: Database, user_id: str) -> list[str]:
    _ready(db)
    return [r["text"] for r in db.query("SELECT text FROM user_memory WHERE user_id=? ORDER BY id", (user_id,))]


def forget(db: Database, user_id: str, memory_id: int) -> bool:
    _ready(db)
    before = db.one("SELECT COUNT(*) AS n FROM user_memory WHERE user_id=? AND id=?", (user_id, memory_id))["n"]
    db.execute("DELETE FROM user_memory WHERE user_id=? AND id=?", (user_id, memory_id))
    return bool(before)
