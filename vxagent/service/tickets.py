"""Service tickets: what an operator hands to the service engineer, and everything that is
attached to it on the way to closing it.

A ticket belongs to one site. Its records (the failed checks, the finding, the calibration
draft, the proposal, the verification run) are the only material the service report may use.
"""
from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..store import Database, dumps
from ..tools.registry import Tool, ToolContext, ToolError, ToolRegistry

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT, site TEXT, symptom TEXT, source TEXT, status TEXT, opened_by TEXT,
    opened REAL, closed REAL, closed_by TEXT);
CREATE TABLE IF NOT EXISTS ticket_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id INTEGER, kind TEXT, data TEXT, by TEXT, at REAL);
CREATE INDEX IF NOT EXISTS ticket_records_t ON ticket_records(ticket_id, id);
"""
SOURCES = ("operator", "failed_preflight", "failed_trial")
KINDS = ("evidence", "finding", "calibration_draft", "proposal", "change_applied", "verification", "report", "note")
TICKET_OPEN = frozenset({"ticket:open"})
TICKET_READ = frozenset({"ticket:read"})


def ready(db: Database) -> None:
    if not getattr(db, "_tickets_ready", False):
        db._con.executescript(SCHEMA)
        db._tickets_ready = True                                             # type: ignore[attr-defined]


def open_ticket(db: Database, site: str, symptom: str, source: str, opened_by: str,
                evidence: dict[str, Any] | None = None) -> int:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    ready(db)
    tid = db.execute("INSERT INTO tickets(site,symptom,source,status,opened_by,opened) VALUES (?,?,?,?,?,?)",
                     (site, symptom.strip()[:600], source, "open", opened_by, time.time()))
    if evidence:
        attach(db, tid, "evidence", evidence, opened_by)
    return tid


def preflight_symptom(failed: list[dict[str, Any]]) -> str:
    """The symptom of a ticket opened for a failed pre-flight: what is wrong, in the words the
    pre-flight wrote, not the names of the checks."""
    return ("Pre-flight failed: " + "; ".join(f.get("problem") or f.get("check", "") for f in failed))[:600]


def attach(db: Database, ticket_id: int, kind: str, data: dict[str, Any], by: str) -> int:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    ready(db)
    return db.execute("INSERT INTO ticket_records(ticket_id,kind,data,by,at) VALUES (?,?,?,?,?)",
                      (ticket_id, kind, dumps(data), by, time.time()))


def get(db: Database, ticket_id: int) -> dict[str, Any] | None:
    ready(db)
    return db.one("SELECT * FROM tickets WHERE id=?", (ticket_id,))


def records(db: Database, ticket_id: int, kind: str | None = None) -> list[dict[str, Any]]:
    ready(db)
    rows = db.query("SELECT * FROM ticket_records WHERE ticket_id=? ORDER BY id", (ticket_id,))
    out = [{"record": f"ticket:{ticket_id}/{r['kind']}/{r['id']}", "kind": r["kind"], "by": r["by"], "at": r["at"],
            "data": json.loads(r["data"])} for r in rows]
    return [r for r in out if kind is None or r["kind"] == kind]


def list_site(db: Database, site: str, status: str | None = None) -> list[dict[str, Any]]:
    ready(db)
    rows = db.query("SELECT * FROM tickets WHERE site=? ORDER BY id DESC", (site,))
    out = []
    for t in rows:
        if status and t["status"] != status:
            continue
        finding = records(db, t["id"], "finding")
        out.append({"ticket_id": t["id"], "symptom": t["symptom"], "source": t["source"], "status": t["status"],
                    "opened_by": t["opened_by"], "opened": t["opened"],
                    "finding": finding[-1]["data"].get("cause") if finding else None})
    return out


def close(db: Database, ticket_id: int, by: str, report: dict[str, Any]) -> None:
    attach(db, ticket_id, "report", report, by)
    db.execute("UPDATE tickets SET status='closed', closed=?, closed_by=? WHERE id=?", (time.time(), by, ticket_id))


def may_read(principal: Any, ticket: dict[str, Any]) -> bool:
    """The whole ticket with its evidence: the service engineer and the safety lead of its site."""
    if "*" in principal.scopes:
        return True
    return "ticket:read" in principal.scopes and principal.site == ticket["site"]


# ------------------------------------------------------------------ tools

class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RequestArgs(Args):
    symptom: str = Field(min_length=10, max_length=400, description="What is wrong or what is needed, in the "
                         "operator's words, with the failed check if there is one.")


class ListArgs(Args):
    status: str | None = Field(default=None, description="open or closed; leave out for all.")


class TicketArgs(Args):
    ticket_id: int = Field(ge=1)


def request_service(a: RequestArgs, ctx: ToolContext) -> dict[str, Any]:
    site = getattr(ctx.site, "name", "") or ctx.principal.site
    failed = ctx.extras.get("last_preflight_failed")
    tid = open_ticket(ctx.db, site, a.symptom, "failed_preflight" if failed else "operator", ctx.principal.user_id,
                      {"failed_checks": failed, "run": ctx.run_id} if failed else {"run": ctx.run_id})
    return {"ticket_id": tid, "status": "open", "site": site,
            "note": "the service engineer of this site will see it; nothing on the robot was changed"}


def list_tickets(a: ListArgs, ctx: ToolContext) -> dict[str, Any]:
    site = getattr(ctx.site, "name", "") or ctx.principal.site
    return {"site": site, "tickets": list_site(ctx.db, site, a.status)[:20]}


def read_ticket(a: TicketArgs, ctx: ToolContext) -> dict[str, Any]:
    t = get(ctx.db, a.ticket_id)
    if t is None or not may_read(ctx.principal, t):
        raise ToolError("unauthorized", f"ticket {a.ticket_id} is not a ticket of your site", hint="do not retry this tool")
    recs = records(ctx.db, a.ticket_id)
    return {"ticket_id": t["id"], "site": t["site"], "symptom": t["symptom"], "source": t["source"], "status": t["status"],
            "opened_by": t["opened_by"], "records": [{"record": r["record"], "kind": r["kind"], "data": _short(r["data"])} for r in recs]}


def _short(data: Any, limit: int = 900) -> Any:
    text = dumps(data)
    return data if len(text) <= limit else {"shortened": text[:limit] + " ..."}


def register_tickets(reg: ToolRegistry) -> ToolRegistry:
    reg.register(Tool("request_service", "Open a ticket for the site's service engineer. Use it when something needs "
                      "a change you are not allowed to make: a limit, a calibration, a configuration, a repair.",
                      RequestArgs, request_service, scopes=TICKET_OPEN, side_effect=True, timeout_s=5))
    reg.register(Tool("list_tickets", "The service tickets of this site with their status and finding.", ListArgs,
                      list_tickets, scopes=TICKET_READ, timeout_s=5))
    reg.register(Tool("read_ticket", "One ticket of this site with the records attached to it.", TicketArgs, read_ticket,
                      scopes=TICKET_READ, timeout_s=5))
    return reg
