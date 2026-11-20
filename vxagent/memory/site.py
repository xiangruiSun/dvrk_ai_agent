"""Site memory: what is true of a site, for as long as it is true.

A fact about a site or its robot is written by code from a tool result, never from model text.
It records where it came from and which events end its validity. When such an event is
recorded the fact is marked expired: it is kept for the record, it is no longer offered as
current, and an agent asked about it can say that it has expired and why.

Facts are keyed by site. A principal of another site gets nothing, and remote support gets
nothing without a grant (task 13).
"""
from __future__ import annotations

import json
import time
from typing import Any

from ..store import Database, dumps

EVENTS = ("ecm_moved", "setup_joint_moved", "instrument_changed", "software_updated", "base_frame_changed",
          "calibration_redone", "limits_changed")

ENDS_CALIBRATION = ("ecm_moved", "setup_joint_moved", "instrument_changed", "base_frame_changed")

SCHEMA = """
CREATE TABLE IF NOT EXISTS site_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, site TEXT, subject TEXT, text TEXT, value TEXT, source TEXT,
    created REAL, ends_on TEXT, status TEXT DEFAULT 'active', expired_by INTEGER, expired REAL);
CREATE TABLE IF NOT EXISTS site_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, site TEXT, kind TEXT, at REAL, recorded_by TEXT, detail TEXT);
"""


class NotVisible(Exception):
    pass


def _ready(db: Database) -> None:
    with db._lock:
        db._con.executescript(SCHEMA)


def may_read(principal: Any, site: str) -> bool:
    if "*" in principal.scopes:
        return True
    return principal.site == site and principal.role in ("operator", "service", "safety")


# What a tool result becomes. Each builder returns (subject, text, value, events that end it) or None.
def _from_residual(result: dict[str, Any]) -> tuple[str, str, dict[str, Any], list[str]] | None:
    if "mean_mm" not in result or "calibration_id" not in result:
        return None
    text = (f"camera-to-robot calibration {result['calibration_id']}: residual on held-out points "
            f"{result['mean_mm']} mm mean, {result['max_mm']} mm max (acceptance {result['acceptance_mm']} mm)")
    return "calibration", text, {k: result[k] for k in ("calibration_id", "mean_mm", "max_mm", "acceptance_mm", "within_acceptance")}, \
        ["ecm_moved", "setup_joint_moved", "instrument_changed", "base_frame_changed", "calibration_redone"]


BUILDERS = {"calibration_residual": _from_residual}


def record_from_result(db: Database, site: str, tool: str, result: Any, source: str) -> int | None:
    """Called by the executor after a successful tool call. Model text never reaches this."""
    builder = BUILDERS.get(tool)
    made = builder(result) if builder and isinstance(result, dict) else None
    if made is None:
        return None
    _ready(db)
    subject, text, value, ends = made
    same = db.one("SELECT id FROM site_facts WHERE site=? AND subject=? AND status='active' AND value=?",
                  (site, subject, dumps(value)))
    if same:
        return same["id"]
    db.execute("UPDATE site_facts SET status='superseded', expired=? WHERE site=? AND subject=? AND status='active'",
               (time.time(), site, subject))
    return db.execute("INSERT INTO site_facts(site,subject,text,value,source,created,ends_on) VALUES (?,?,?,?,?,?,?)",
                      (site, subject, text, dumps(value), source, time.time(), dumps(ends)))


def record_event(db: Database, site: str, kind: str, recorded_by: str, detail: str = "") -> dict[str, Any]:
    """Something happened at the site. Every active fact that this kind of event ends is expired."""
    if kind not in EVENTS:
        raise ValueError(f"unknown event {kind!r}; events: {', '.join(EVENTS)}")
    _ready(db)
    now = time.time()
    eid = db.execute("INSERT INTO site_events(site,kind,at,recorded_by,detail) VALUES (?,?,?,?,?)",
                     (site, kind, now, recorded_by, detail[:300]))
    if kind in ENDS_CALIBRATION:                      # the record itself stops being usable, not only what was noted about it
        from ..structured.proposals import set_calibration
        set_calibration(db, site, False, f"the event {kind} was recorded for this site")
    elif kind == "calibration_redone":
        from ..structured.proposals import set_calibration
        set_calibration(db, site, True, "calibrated again")
    ended = []
    for f in db.query("SELECT id, ends_on FROM site_facts WHERE site=? AND status='active'", (site,)):
        if kind in json.loads(f["ends_on"]):
            db.execute("UPDATE site_facts SET status='expired', expired_by=?, expired=? WHERE id=?", (eid, now, f["id"]))
            ended.append(f["id"])
    return {"event_id": eid, "kind": kind, "facts_expired": ended}


def facts(db: Database, principal: Any, site: str, subject: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Active and expired facts of a site, for a principal who may read them."""
    if not may_read(principal, site):
        raise NotVisible(f"{principal.role} of {principal.site or 'no site'} may not read the memory of {site}")
    _ready(db)
    rows = db.query("SELECT f.*, e.kind AS ended_by, e.at AS ended_at FROM site_facts f LEFT JOIN site_events e "
                    "ON e.id=f.expired_by WHERE f.site=? AND f.status IN ('active','expired') ORDER BY f.id", (site,))
    out: dict[str, list[dict[str, Any]]] = {"active": [], "expired": []}
    for r in rows:
        if subject and r["subject"] != subject:
            continue
        item = {"fact_id": f"fact:{r['id']}", "subject": r["subject"], "text": r["text"], "value": json.loads(r["value"]),
                "source": r["source"], "recorded": time.strftime("%Y-%m-%d %H:%M", time.localtime(r["created"]))}
        if r["status"] == "expired":
            item.update({"expired_because": r["ended_by"],
                         "expired_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ended_at"] or r["expired"]))})
        out[r["status"]].append(item)
    return out


def context_block(db: Database, principal: Any, site: str) -> str:
    """What is put in front of the model at the start of a run. Expired facts are named as
    expired with the reason, so the model can say so, and are never presented as current."""
    try:
        f = facts(db, principal, site)
    except NotVisible:
        return ""
    lines = []
    if f["active"]:
        lines.append("Known about this site and still valid:")
        lines += [f"- [{x['fact_id']}] {x['text']} (recorded {x['recorded']})" for x in f["active"]]
    if f["expired"]:
        lines.append("No longer valid. Do not state these as current; if asked, say that they have expired and why:")
        lines += [f"- [{x['fact_id']}] {x['text']} EXPIRED on {x['expired_at']} because of the event {x['expired_because']}"
                  for x in f["expired"]]
    return "\n".join(lines)
