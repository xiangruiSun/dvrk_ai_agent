"""Who is logged in, and what a run may move.

Users       a server-side table: role, site, password hash. It is read on every request, so a
            cookie never carries a role.
Sessions    a signed value naming a user and an expiry. Signed with a secret of this node.
Lock-out    repeated wrong passwords lock the account for a while. Logins are audited.
Runs        an operator holds observe and simulate. Actuate is held per run: a motion tool is
            accepted only inside a run of that operator that is still active, only if the run
            was registered with that tool, and then still needs the operator's approval.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

from .store import Database, dumps
from .tools.registry import Principal

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY, role TEXT, site TEXT, display_name TEXT, salt TEXT, pw_hash TEXT,
    failed INTEGER DEFAULT 0, locked_until REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS active_runs (
    run_id TEXT PRIMARY KEY, user_id TEXT, site TEXT, tools TEXT, status TEXT, started REAL, ended REAL);
"""
PBKDF2_ROUNDS = 200_000
MAX_FAILURES = 5
LOCK_S = 300.0
SESSION_S = 8 * 3600.0
_DUMMY_SALT = "00" * 16


def ready(db: Database) -> None:
    if not getattr(db, "_access_ready", False):
        db._con.executescript(SCHEMA)
        db._access_ready = True                                              # type: ignore[attr-defined]


# ------------------------------------------------------------------ users and passwords

def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), PBKDF2_ROUNDS).hex()


def add_user(db: Database, user_id: str, role: str, site: str, display_name: str, password: str) -> None:
    from .profiles import ROLE_SCOPES
    if role not in ROLE_SCOPES:
        raise ValueError(f"unknown role {role!r}")
    ready(db)
    salt = secrets.token_hex(16)
    db.execute("INSERT OR REPLACE INTO users(user_id,role,site,display_name,salt,pw_hash,failed,locked_until) "
               "VALUES (?,?,?,?,?,?,0,0)", (user_id, role, site, display_name, salt, hash_password(password, salt)))


def user(db: Database, user_id: str) -> dict[str, Any] | None:
    ready(db)
    row = db.one("SELECT user_id, role, site, display_name FROM users WHERE user_id=?", (user_id,))
    return row


class LoginRefused(Exception):
    def __init__(self, kind: str):                    # bad_credentials | locked
        super().__init__(kind)
        self.kind = kind


def login(db: Database, user_id: str, password: str, now: float | None = None) -> dict[str, Any]:
    """The user's row, or LoginRefused. An unknown user costs the same work as a known one and
    gets the same answer."""
    ready(db)
    now = time.time() if now is None else now
    row = db.one("SELECT * FROM users WHERE user_id=?", (user_id,))
    if row is not None and row["locked_until"] > now:
        db.audit(user_id, row["role"], "login", "session", False, "locked")
        raise LoginRefused("locked")
    given = hash_password(password, row["salt"] if row else _DUMMY_SALT)
    if row is None or not hmac.compare_digest(given, row["pw_hash"]):
        if row is not None:
            failed = row["failed"] + 1
            until = now + LOCK_S if failed >= MAX_FAILURES else 0
            db.execute("UPDATE users SET failed=?, locked_until=? WHERE user_id=?", (0 if until else failed, until, user_id))
        db.audit(user_id[:60], row["role"] if row else "-", "login", "session", False, "bad credentials")
        raise LoginRefused("bad_credentials")
    db.execute("UPDATE users SET failed=0, locked_until=0 WHERE user_id=?", (user_id,))
    db.audit(user_id, row["role"], "login", "session", True)
    return {k: row[k] for k in ("user_id", "role", "site", "display_name")}


# ------------------------------------------------------------------ signed session values

def sign(user_id: str, secret: str, ttl_s: float = SESSION_S, now: float | None = None) -> str:
    body = base64.urlsafe_b64encode(json.dumps({"u": user_id, "exp": (time.time() if now is None else now) + ttl_s,
                                                "n": secrets.token_hex(4)}).encode()).decode()
    mac = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{mac}"


def verify(token: str, secret: str, now: float | None = None) -> str | None:
    """The user id a session value names, or None if it is forged, damaged or expired."""
    try:
        body, mac = token.rsplit(".", 1)
        good = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(mac, good):
            return None
        data = json.loads(base64.urlsafe_b64decode(body.encode()))
        if float(data["exp"]) < (time.time() if now is None else now):
            return None
        return str(data["u"])
    except (ValueError, KeyError, TypeError):
        return None


def principal_of(row: dict[str, Any]) -> Principal:
    """The principal of a console session. An operator's actuate tier is held per run."""
    from .profiles import ROLE_SCOPES, ROLE_TIERS
    tiers = ROLE_TIERS[row["role"]]
    # The operator moves the arm and the service engineer sweeps the calibration poses, each only
    # inside a run of their own and only with an operator's approval.
    per_run = row["role"] in ("operator", "service")
    return Principal(row["user_id"], row["role"], ROLE_SCOPES[row["role"]], site=row["site"] or "",
                     tiers=tiers - {"actuate"}, display_name=row["display_name"] or row["user_id"], per_run=per_run)


# ------------------------------------------------------------------ actuate, per run

def begin_run(db: Database, run_id: str, user_id: str, site: str, tools: list[str]) -> None:
    ready(db)
    db.execute("INSERT OR REPLACE INTO active_runs(run_id,user_id,site,tools,status,started,ended) VALUES (?,?,?,?,?,?,NULL)",
               (run_id, user_id, site, dumps(sorted(tools)), "active", time.time()))


def end_run(db: Database, run_id: str) -> None:
    ready(db)
    db.execute("UPDATE active_runs SET status='ended', ended=? WHERE run_id=? AND status='active'", (time.time(), run_id))


def run_allows(db: Database, run_id: str, user_id: str, tool: str) -> tuple[bool, str]:
    ready(db)
    row = db.one("SELECT * FROM active_runs WHERE run_id=?", (run_id,))
    if row is None:
        return False, "a motion is only possible inside a run; this call belongs to none"
    if row["user_id"] != user_id:
        return False, "this run belongs to another operator"
    if row["status"] != "active":
        return False, "the run has ended; its permission to move the arm ended with it"
    if tool not in json.loads(row["tools"]):
        return False, f"this run was not started with {tool}"
    return True, ""


# ------------------------------------------------------------------ demonstration accounts

DEMO_ACCOUNTS = (
    ("op.lee", "operator", "site-a", "Operator, site A"),
    ("op.b", "operator", "site-b", "Operator, site B"),
    ("eng.kim", "service", "site-a", "Service engineer, site A"),
    ("eng.oyelaran", "service", "site-a", "Second service engineer, site A"),
    ("saf.adler", "safety", "site-a", "Safety lead, site A"),
    ("saf.novak", "safety", "site-b", "Safety lead, site B"),
    ("sup.wong", "support", "", "Remote support"),
)


def seed_demo(db: Database, passwords: dict[str, str] | None = None) -> dict[str, str]:
    """Create the demonstration accounts. Passwords are generated here unless given, and are
    returned once so the caller can write them where the person who runs the demo will look."""
    out = {}
    for user_id, role, site, name in DEMO_ACCOUNTS:
        pw = (passwords or {}).get(user_id) or secrets.token_urlsafe(9)
        add_user(db, user_id, role, site, name, pw)
        out[user_id] = pw
    return out
