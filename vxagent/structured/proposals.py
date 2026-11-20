"""Who may propose a change, who may approve it, and what happens when it is applied.

This is the third layer, and it is separate from the other two on purpose: a patch can be well
formed and physically sensible and still not be this person's to make.

    propose   a service engineer, for their own site
    approve   the safety lead of that site, who is not the proposer; a loosened limit needs a
              written reason from the approver
    apply     only an approved proposal, and only if the file is still the one it was checked
              against. The result is a new version; nothing is edited in place.
"""
from __future__ import annotations

import json
import time
from typing import Any

from ..robot.config import ConfigStore, doc_hash
from ..store import Database, dumps
from ..tools.registry import Principal
from .rules import Validation
from .schemas import ConfigPatch

MIN_REASON = 20


class NotPermitted(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _refuse(db: Database, who: Principal, action: str, resource: str, kind: str, message: str) -> NotPermitted:
    db.audit(who.user_id, who.role, action, resource, False, f"{kind}: {message}")
    return NotPermitted(kind, message)


def _has(who: Principal, scope: str) -> bool:
    return "*" in who.scopes or scope in who.scopes


def propose(db: Database, who: Principal, site: str, patch: ConfigPatch, validation: Validation, base_doc: Any,
            file_type: str, effect: dict[str, Any] | None = None) -> int:
    resource = f"{site}:{patch.target}"
    if not _has(who, "config:propose"):
        raise _refuse(db, who, "config:propose", resource, "role", f"role {who.role!r} may not propose configuration changes")
    if who.site != site:
        raise _refuse(db, who, "config:propose", resource, "site", f"{who.user_id} belongs to {who.site or 'no site'}, not {site}")
    if not validation.ok:
        raise _refuse(db, who, "config:propose", resource, "invalid",
                      f"the patch did not pass validation (stage {validation.stage})")
    pid = db.execute(
        "INSERT INTO proposals(site,target,file_type,patch,patched,proposer,status,created,base_hash,limit_effect,"
        "invalidates_calibration,effect) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (site, patch.target, file_type, patch.model_dump_json(), json.dumps(validation.patched), who.user_id,
         "proposed", time.time(), doc_hash(base_doc), validation.limit_effect, int(validation.invalidates_calibration),
         dumps(effect or {})))
    db.audit(who.user_id, who.role, "config:propose", resource, True, f"proposal {pid}")
    return pid


def get(db: Database, proposal_id: int) -> dict[str, Any]:
    row = db.one("SELECT * FROM proposals WHERE id=?", (proposal_id,))
    if row is None:
        raise KeyError(f"no proposal {proposal_id}")
    return row


def decide(db: Database, who: Principal, proposal_id: int, approve: bool, reason: str = "") -> dict[str, Any]:
    p = get(db, proposal_id)
    resource = f"{p['site']}:{p['target']}#{proposal_id}"
    if not _has(who, "config:approve"):
        raise _refuse(db, who, "config:approve", resource, "role", f"role {who.role!r} may not approve configuration changes")
    if who.site != p["site"]:
        raise _refuse(db, who, "config:approve", resource, "site", f"{who.user_id} belongs to {who.site or 'no site'}, not {p['site']}")
    if who.user_id == p["proposer"]:
        raise _refuse(db, who, "config:approve", resource, "self_approval", "the proposer cannot approve their own change")
    if p["status"] != "proposed":
        raise _refuse(db, who, "config:approve", resource, "state", f"the proposal is already {p['status']}")
    if approve and p["limit_effect"] == "loosen" and len(reason.strip()) < MIN_REASON:
        raise _refuse(db, who, "config:approve", resource, "reason_required",
                      "this change loosens a safety limit: approval needs a written reason")
    db.execute("UPDATE proposals SET status=?, decided_by=?, decided=?, decision_reason=? WHERE id=?",
               ("approved" if approve else "rejected", who.user_id, time.time(), reason.strip()[:500], proposal_id))
    db.audit(who.user_id, who.role, "config:approve", resource, True, "approved" if approve else "rejected")
    return get(db, proposal_id)


def apply(db: Database, store: ConfigStore, proposal_id: int) -> dict[str, Any]:
    """Write an approved proposal as the next version of its file."""
    p = get(db, proposal_id)
    if p["status"] != "approved":
        raise NotPermitted("state", f"proposal {proposal_id} is {p['status']}, not approved")
    _, now_hash, _ = store.current(p["target"])
    if now_hash != p["base_hash"]:
        db.execute("UPDATE proposals SET status='stale' WHERE id=?", (proposal_id,))
        raise NotPermitted("stale", "the file has changed since this proposal was checked; it must be proposed again")
    doc = json.loads(p["patched"])
    patch = json.loads(p["patch"])
    bumped = None
    if p["file_type"] == "vx-site" and any(o["path"].startswith("safety_profile/") for o in patch["operations"]):
        # The envelope has a version, and approvals to move are bound to it (task 03 R10).
        old = str(doc["safety_profile"]["version"])
        bumped = str(int(old) + 1) if old.isdigit() else old + ".1"
        doc["safety_profile"]["version"] = bumped
    version = store.commit(p["target"], doc, {"proposal": proposal_id, "proposer": p["proposer"],
                                             "approver": p["decided_by"], "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    db.execute("UPDATE proposals SET status='applied', applied_version=? WHERE id=?", (version, proposal_id))
    if p["invalidates_calibration"]:
        set_calibration(db, p["site"], False, f"configuration change, proposal {proposal_id} ({p['target']})")
        from ..memory.site import record_event            # what was noted about the old calibration expires too
        record_event(db, p["site"], "instrument_changed" if any(o["path"] == "tool" for o in patch["operations"])
                     else "base_frame_changed", p["decided_by"] or "", f"proposal {proposal_id}")
    return {"proposal": proposal_id, "target": p["target"], "version": version, "envelope_version": bumped,
            "calibration_invalidated": bool(p["invalidates_calibration"])}


# ------------------------------------------------------------------------------ calibration validity

def set_calibration(db: Database, site: str, valid: bool, why: str) -> None:
    db.execute("INSERT OR REPLACE INTO kv(k,v) VALUES (?,?)",
               (f"calibration:{site}", dumps({"valid": valid, "why": why, "at": time.time()})))


def calibration(db: Database, site: str) -> dict[str, Any]:
    row = db.one("SELECT v FROM kv WHERE k=?", (f"calibration:{site}",))
    return json.loads(row["v"]) if row else {"valid": True, "why": "no change recorded since calibration", "at": None}
