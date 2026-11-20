"""Grants: how remote support gets to see the evidence of one ticket.

A grant is given by the safety lead of the ticket's site, to one support engineer, for one
ticket, read-only, with an end time. It lives in the site's own database and is checked there
on every call, so it cannot be widened from outside. It ends on its own, can be revoked, and
every use, allowed or refused, is audited at the site.
"""
from __future__ import annotations

import time
from typing import Any

from ..service import tickets
from ..store import Database

SCHEMA = """
CREATE TABLE IF NOT EXISTS grants (
    id INTEGER PRIMARY KEY AUTOINCREMENT, site TEXT, ticket_id INTEGER, grantee TEXT, granted_by TEXT,
    created REAL, expires REAL, revoked REAL, revoked_by TEXT);
"""
MAX_HOURS = 72.0


class NotGranted(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _ready(db: Database) -> None:
    if not getattr(db, "_grants_ready", False):
        db._con.executescript(SCHEMA)
        db._grants_ready = True                                              # type: ignore[attr-defined]


def give(db: Database, who: Any, ticket_id: int, grantee: str, hours: float, now: float | None = None) -> dict[str, Any]:
    _ready(db)
    t = tickets.get(db, ticket_id)
    resource = f"ticket:{ticket_id}"
    if "grant:manage" not in who.scopes and "*" not in who.scopes:
        db.audit(who.user_id, who.role, "grant:give", resource, False, "role")
        raise NotGranted("role", f"role {who.role!r} may not give support access")
    if t is None or t["site"] != who.site:
        db.audit(who.user_id, who.role, "grant:give", resource, False, "site")
        raise NotGranted("site", "that ticket is not a ticket of your site")
    if not 0 < hours <= MAX_HOURS:
        raise NotGranted("period", f"a grant lasts at most {MAX_HOURS:g} hours")
    now = time.time() if now is None else now
    gid = db.execute("INSERT INTO grants(site,ticket_id,grantee,granted_by,created,expires) VALUES (?,?,?,?,?,?)",
                     (t["site"], ticket_id, grantee, who.user_id, now, now + hours * 3600.0))
    db.audit(who.user_id, who.role, "grant:give", resource, True, f"grant {gid} to {grantee} for {hours:g} h")
    return {"grant_id": gid, "site": t["site"], "ticket_id": ticket_id, "grantee": grantee, "expires": now + hours * 3600.0,
            "access": "read-only, this ticket only"}


def revoke(db: Database, who: Any, grant_id: int, now: float | None = None) -> dict[str, Any]:
    _ready(db)
    g = db.one("SELECT * FROM grants WHERE id=?", (grant_id,))
    if g is None or g["site"] != who.site or ("grant:manage" not in who.scopes and "*" not in who.scopes):
        db.audit(who.user_id, who.role, "grant:revoke", f"grant:{grant_id}", False, "not a grant of this person's site")
        raise NotGranted("site", "that grant is not a grant of your site")
    db.execute("UPDATE grants SET revoked=?, revoked_by=? WHERE id=? AND revoked IS NULL",
               (time.time() if now is None else now, who.user_id, grant_id))
    db.audit(who.user_id, who.role, "grant:revoke", f"grant:{grant_id}", True)
    return {"grant_id": grant_id, "revoked": True}


def listing(db: Database, site: str, now: float | None = None) -> list[dict[str, Any]]:
    _ready(db)
    now = time.time() if now is None else now
    out = []
    for g in db.query("SELECT * FROM grants WHERE site=? ORDER BY id DESC", (site,)):
        state = "revoked" if g["revoked"] else ("expired" if g["expires"] <= now else "active")
        out.append({"grant_id": g["id"], "ticket_id": g["ticket_id"], "grantee": g["grantee"], "granted_by": g["granted_by"],
                    "expires": g["expires"], "state": state})
    return out


def check(db: Database, grantee: str, site: str, ticket_id: int, now: float | None = None) -> tuple[bool, str]:
    _ready(db)
    now = time.time() if now is None else now
    rows = db.query("SELECT * FROM grants WHERE site=? AND ticket_id=? AND grantee=? ORDER BY id DESC", (site, ticket_id, grantee))
    if not rows:
        return False, "no grant for this ticket"
    for g in rows:
        if g["revoked"] is None and g["expires"] > now:
            return True, f"grant {g['id']}"
    return False, "the grant was revoked" if rows[0]["revoked"] else "the grant has expired"


def read_ticket(node: Any, who: Any, ticket_id: int, now: float | None = None) -> dict[str, Any]:
    """A ticket's records for a support engineer, if this site has given a grant that holds now.
    Called at the edge node. Read-only: there is no write path behind a grant."""
    db = node.db
    ok, why = check(db, who.user_id, node.name, ticket_id, now)
    db.audit(who.user_id, who.role, "grant:read_ticket", f"{node.name}:ticket:{ticket_id}", ok, why)
    if not ok:
        raise NotGranted("no_grant", f"{why}; the safety lead of {node.name} decides who may see a ticket")
    t = tickets.get(db, ticket_id)
    if t is None:
        raise NotGranted("no_ticket", "no such ticket")
    recs = tickets.records(db, ticket_id)
    return {"site": node.name, "ticket_id": ticket_id, "symptom": t["symptom"], "source": t["source"], "status": t["status"],
            "records": [{"record": r["record"], "kind": r["kind"], "data": tickets._short(r["data"], 700)} for r in recs],
            "access": f"read-only, under {why}"}
