"""The service report written when a ticket is closed (task 12 R6).

The report is a structure filled by code from the ticket's records. One model call writes its
summary sentence from that structure, and the sentence is accepted only if every number in it
is in the records and it names nothing as verified that was not. Otherwise a template is used
and the report says so. Three attempts at most.
"""
from __future__ import annotations

import re
from typing import Any

from ..llm.base import LLMError
from ..store import dumps
from . import tickets

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
MAX_ATTEMPTS = 3
SYSTEM = ("You write the summary of a service report for a surgical-robot research set-up, three sentences at most. "
          "Use only the report given. Do not add numbers, causes or advice that are not in it. Say what was not "
          "verified if the report lists anything under not_verified. Plain sentences, no lists.")


def build(db: Any, ticket_id: int) -> dict[str, Any]:
    """The report's facts, from the ticket's records and nothing else."""
    t = tickets.get(db, ticket_id)
    recs = tickets.records(db, ticket_id)
    last = lambda kind: next((r for r in reversed(recs) if r["kind"] == kind), None)      # noqa: E731
    finding, change, verification = last("finding"), last("change_applied"), last("verification")
    proposal = last("proposal")
    not_verified = ["anything on a physical robot: every motion and measurement was on the kinematic twin"]
    if finding is None:
        not_verified.append("the cause: no diagnosis was attached")
    elif not finding["data"].get("narrowed_to_one"):
        not_verified.append("the cause: the diagnosis did not narrow the candidates to one")
    if change is None:
        not_verified.append("a change: none was applied under this ticket")
    if verification is None:
        not_verified.append("the repair: no verification run was attached")
    return {
        "ticket_id": ticket_id, "site": t["site"], "symptom": t["symptom"], "source": t["source"],
        "finding": {"record": finding["record"], "cause": finding["data"].get("cause"), "method": finding["data"].get("method"),
                    "narrowed_to_one": finding["data"].get("narrowed_to_one")} if finding else None,
        "change_applied": {"record": change["record"], **{k: v for k, v in change["data"].items()
                                                          if k in ("kind", "proposal", "active_record", "target", "version", "event")}} if change else None,
        "proposal": {"record": proposal["record"], "proposal_id": proposal["data"].get("proposal_id")} if proposal else None,
        "verification": {"record": verification["record"], **verification["data"]} if verification else None,
        "not_verified": not_verified,
        "records_used": [r["record"] for r in recs],
    }


def problems(text: str, report: dict[str, Any]) -> list[str]:
    """Grounding of the summary against the report: numbers, and no claim of a verified repair
    when no verification run is attached."""
    allowed = [float(x) for x in _NUM.findall(dumps(report))]
    out = []
    for x in _NUM.findall(text):
        if not any(abs(float(x) - a) <= 0.0501 for a in allowed):
            out.append(f"the number {x} is not in the report")
    if report["verification"] is None and re.search(r"\b(verified|confirmed|passes|passed)\b", text, re.I) \
            and not re.search(r"\bnot (been )?(verified|confirmed)\b|\bno verification\b|\bunverified\b", text, re.I):
        out.append("the summary says something was verified, but the report has no verification run")
    cause = (report.get("finding") or {}).get("cause")
    if cause is None and re.search(r"\bcaused by\b|\bthe cause (is|was)\b", text, re.I):
        out.append("the summary names a cause, but the report has no finding")
    return out


def template(report: dict[str, Any]) -> str:
    f, c, v = report["finding"], report["change_applied"], report["verification"]
    parts = [f"Ticket {report['ticket_id']} at {report['site']}: {report['symptom']}"]
    parts.append(f"Finding: {f['cause']} (method {f['method']})." if f and f["cause"] else "No finding was recorded.")
    parts.append(f"Change applied: {c['kind']}, event {c.get('event')}." if c else "No change was applied.")
    parts.append("A verification run is attached." if v else "No verification run is attached.")
    return " ".join(parts)


def write(db: Any, ticket_id: int, llm: Any = None, tracer: Any = None) -> dict[str, Any]:
    report = build(db, ticket_id)
    text, source, attempts, log = template(report), "template", 0, []
    if llm is not None:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Report:\n" + dumps(report)}]
        for attempts in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = llm.chat(messages, temperature=0.0, max_tokens=260)
            except LLMError as exc:
                source = f"template (model {exc.kind})"
                break
            bad = problems(resp.content, report) if resp.content.strip() else ["the summary is empty"]
            log.append({"attempt": attempts, "problems": bad})
            if not bad:
                text, source = resp.content.strip(), "model"
                break
            messages += [{"role": "assistant", "content": resp.content},
                         {"role": "user", "content": "Not acceptable:\n" + "\n".join(f"{i}. {p}" for i, p in enumerate(bad, 1))
                          + "\nWrite the summary again."}]
        else:
            source = "template (the model's summary did not pass the grounding rules)"
    report.update({"summary": text, "summary_source": source, "summary_attempts": attempts, "summary_log": log,
                   "valid": True})
    return report


def close_with_report(node: Any, who: Any, ticket_id: int, llm: Any = None) -> dict[str, Any]:
    t = tickets.get(node.db, ticket_id)
    if t is None or not tickets.may_read(who, t) or "ticket:write" not in who.scopes:
        node.db.audit(who.user_id, who.role, "ticket:close", f"ticket:{ticket_id}", False, "not a ticket of this person's site")
        raise PermissionError(f"{who.user_id} may not close ticket {ticket_id}")
    report = write(node.db, ticket_id, llm if llm is not None else node.llm)
    tickets.close(node.db, ticket_id, who.user_id, report)
    return report
