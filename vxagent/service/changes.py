"""Changes to what the robot may do: proposing them, the safety lead's queue, and what happens
when one is applied.

Two kinds of change go through the same proposal record and the same two-person rule:
    configuration   a ConfigPatch on one of the site's files (task 02)
    calibration     activating a calibration draft
Applying either records a site event, so that what the site remembered about the old state
expires, and ends every approval to move that was asked for before the change.
"""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from ..memory import site as site_memory
from ..robot import config as cfg
from ..store import Database, dumps
from ..structured import proposals
from ..structured.rules import RuleContext, validate_patch
from ..structured.schemas import ConfigPatch
from ..tools.registry import Principal
from ..tools.service import load_calibration_record, store_calibration_record
from . import calibration as cal
from . import tickets

CALIBRATION_TARGET = "calibration/camera-to-robot"


def evidence_ids(db: Database, site: str) -> frozenset[str]:
    """Record ids a change at this site may cite: the records of its tickets and its site facts."""
    ids = set()
    for t in tickets.list_site(db, site):
        ids.update(r["record"] for r in tickets.records(db, t["ticket_id"]))
    site_memory._ready(db)
    ids.update(f"fact:{r['id']}" for r in db.query("SELECT id FROM site_facts WHERE site=?", (site,)))
    return frozenset(ids)


def rule_context(node: Any, target: str) -> RuleContext:
    if node.store is None:
        raise proposals.NotPermitted("unavailable", "this node has no configuration store")
    doc, _, _ = node.store.current(target)
    return RuleContext(target=target, file_type=cfg.file_type(doc), doc=doc, schemas=node.schemas, instruments=node.instruments,
                       frames_declared=frozenset(node.spec.get("frames", [])), evidence=evidence_ids(node.db, node.name))


def propose_patch(node: Any, who: Principal, patch: ConfigPatch, ticket_id: int | None = None) -> dict[str, Any]:
    """Validate, compute the effect on the twin, and file the proposal. Changes nothing."""
    from ..structured.dryrun import dry_apply
    try:
        ctx = rule_context(node, patch.target)
    except KeyError as exc:
        return {"proposed": False, "stage": "target", "errors": [str(exc)]}
    v = validate_patch(patch, ctx)
    if not v.ok:
        node.db.audit(who.user_id, who.role, "config:propose", f"{node.name}:{patch.target}", False, f"invalid at {v.stage}")
        return {"proposed": False, "stage": v.stage, "errors": v.errors[:6]}
    if ctx.file_type == "vx-site":                      # the site's own file: compared on this node's arm
        from ..robot.site import build_site
        from ..structured.dryrun import _compare
        effect = _compare(build_site(dict(ctx.doc), node.site.chain), build_site(dict(v.patched), node.site.chain))
        effect.update({"applied_to": "the site's kinematic twin", "poses_expressed_in": "the arm's base frame"})
    elif node.dvrk_dir is not None:
        effect = dry_apply(ctx.file_type, ctx.doc, v.patched, node.spec, node.dvrk_dir, v.changes)
    else:
        effect = {"note": "not computed: the dVRK files are not on this node"}
    pid = proposals.propose(node.db, who, node.name, patch, v, ctx.doc, ctx.file_type, effect)
    if ticket_id is not None:
        tickets.attach(node.db, ticket_id, "proposal", {"proposal_id": pid, "kind": "configuration", "target": patch.target,
                                                        "changes": v.changes, "limit_effect": v.limit_effect}, who.user_id)
    return {"proposed": True, "proposal_id": pid, "status": "proposed", "limit_effect": v.limit_effect,
            "invalidates_calibration": v.invalidates_calibration, "changes": v.changes,
            "poses_that_move": effect.get("poses_that_move", []), "largest_move_mm": effect.get("largest_move_mm"),
            "note": "nothing was changed; the safety lead of this site decides"}


def propose_calibration(node: Any, who: Principal, reason: str, ticket_id: int | None = None) -> dict[str, Any]:
    """File the activation of the current calibration draft as a change for the safety lead."""
    db, site = node.db, node.name
    resource = f"{site}:{CALIBRATION_TARGET}"
    if "config:propose" not in who.scopes and "*" not in who.scopes:
        raise proposals._refuse(db, who, "config:propose", resource, "role", f"role {who.role!r} may not propose changes")
    if who.site != site:
        raise proposals._refuse(db, who, "config:propose", resource, "site", f"{who.user_id} does not belong to {site}")
    draft = cal.load_draft(db, site)
    if draft is None or draft.get("status") != "draft":
        raise proposals.NotPermitted("state", "there is no calibration draft to propose; run the calibration procedure first")
    old = load_calibration_record(db, site)
    diff = cal.difference(np.array(old["T_est"], dtype=float), np.array(draft["T_est"], dtype=float), draft["seen_m"]) if old else None
    effect = {"kind": "calibration", "draft": draft["id"], "replaces": old["id"] if old else None,
              "held_out_residual": draft["held_out_residual"], "difference_to_the_record_in_use": diff,
              "applied_to": "nothing on the twin moves; camera-seen positions map to different robot positions"}
    patch = {"target": CALIBRATION_TARGET, "kind": "calibration", "draft": draft["id"], "reason": reason.strip()[:400],
             "operations": [{"op": "replace", "path": "active_record", "value": draft["id"]}]}
    pid = db.execute(
        "INSERT INTO proposals(site,target,file_type,patch,patched,proposer,status,created,base_hash,limit_effect,"
        "invalidates_calibration,effect) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (site, CALIBRATION_TARGET, "calibration", dumps(patch), dumps(draft), who.user_id, "proposed", time.time(),
         old["id"] if old else "", "none", 0, dumps(effect)))
    db.audit(who.user_id, who.role, "config:propose", resource, True, f"proposal {pid}")
    if ticket_id is not None:
        tickets.attach(db, ticket_id, "proposal", {"proposal_id": pid, "kind": "calibration", "draft": draft["id"],
                                                   "held_out_residual": draft["held_out_residual"]}, who.user_id)
    return {"proposed": True, "proposal_id": pid, "status": "proposed", "draft": draft["id"],
            "note": "the draft is not in use; it becomes active only when the safety lead approves this change"}


def view(db: Database, proposal_id: int) -> dict[str, Any]:
    p = proposals.get(db, proposal_id)
    patch = json.loads(p["patch"])
    return {"proposal_id": p["id"], "site": p["site"], "target": p["target"], "kind": p["file_type"], "status": p["status"],
            "proposer": p["proposer"], "reason": patch.get("reason", ""), "evidence": patch.get("evidence", []),
            "operations": patch.get("operations", []), "limit_effect": p["limit_effect"],
            "loosens_a_limit": p["limit_effect"] == "loosen", "invalidates_calibration": bool(p["invalidates_calibration"]),
            "validation": "passed: allowlist, file schema and physical rules" if p["file_type"] != "calibration"
            else "fit judged on held-out samples", "effect_on_the_twin": json.loads(p["effect"] or "{}"),
            "decided_by": p["decided_by"], "decision_reason": p["decision_reason"], "applied_version": p["applied_version"]}


def queue(db: Database, site: str, status: str | None = "proposed") -> list[dict[str, Any]]:
    rows = db.query("SELECT id FROM proposals WHERE site=? ORDER BY id DESC", (site,))
    out = [view(db, r["id"]) for r in rows]
    return [v for v in out if status is None or v["status"] == status]


def expire_approvals(db: Database, site: str, why: str) -> int:
    """An approval to move was given for a state of the site. After a change it is that no longer."""
    rows = db.query("SELECT a.id FROM approvals a WHERE a.status IN ('pending','approved') AND (a.run_id IN "
                    "(SELECT run_id FROM runs WHERE site=?) OR a.run_id IN (SELECT run_id FROM active_runs WHERE site=?))",
                    (site, site))
    for r in rows:
        db.execute("UPDATE approvals SET status='expired' WHERE id=?", (r["id"],))
    if rows:
        db.audit("system", "system", "approvals:expire", site, True, f"{len(rows)} ended: {why}")
    return len(rows)


def apply_change(node: Any, proposal_id: int) -> dict[str, Any]:
    """Apply an approved proposal: a new file version or the new calibration record, the site
    event that goes with it, and the end of approvals asked for before it."""
    db = node.db
    p = proposals.get(db, proposal_id)
    if p["file_type"] == "calibration":
        if p["status"] != "approved":
            raise proposals.NotPermitted("state", f"proposal {proposal_id} is {p['status']}, not approved")
        draft = json.loads(p["patched"])
        current = load_calibration_record(db, node.name)
        if (current["id"] if current else "") != p["base_hash"]:
            db.execute("UPDATE proposals SET status='stale' WHERE id=?", (proposal_id,))
            raise proposals.NotPermitted("stale", "the calibration in use has changed since this was proposed")
        record = {k: draft[k] for k in ("id", "T_est", "affine", "seen_m", "touched_m", "method", "session")}
        store_calibration_record(db, node.name, record)
        db.execute("INSERT OR REPLACE INTO kv(k,v) VALUES (?,?)", (f"scene_session:{node.name}", json.dumps(draft["session"])))
        draft["status"] = "active"
        db.execute("INSERT OR REPLACE INTO kv(k,v) VALUES (?,?)", (f"caldraft:{node.name}", dumps(draft)))
        db.execute("UPDATE proposals SET status='applied', applied_version=1 WHERE id=?", (proposal_id,))
        event = site_memory.record_event(db, node.name, "calibration_redone", p["decided_by"] or "", f"proposal {proposal_id}")
        out = {"proposal": proposal_id, "kind": "calibration", "active_record": record["id"], "event": "calibration_redone",
               "facts_expired": event["facts_expired"]}
    else:
        res = proposals.apply(db, node.store, proposal_id)
        patch = json.loads(p["patch"])
        limits = any(o["path"].startswith("safety_profile/") for o in patch["operations"])
        kind = "limits_changed" if limits else "software_updated"
        event = site_memory.record_event(db, node.name, kind, p["decided_by"] or "", f"proposal {proposal_id} ({p['target']})")
        if p["target"] == getattr(node, "site_target", ""):
            node.reload_site()
        out = {**res, "kind": "configuration", "event": kind, "facts_expired": event["facts_expired"]}
    out["approvals_ended"] = expire_approvals(db, node.name, f"proposal {proposal_id} was applied")
    for t in tickets.list_site(db, node.name, "open"):
        if any(r["data"].get("proposal_id") == proposal_id for r in tickets.records(db, t["ticket_id"], "proposal")):
            tickets.attach(db, t["ticket_id"], "change_applied", {k: v for k, v in out.items() if k != "facts_expired"},
                           p["decided_by"] or "system")
    return out


def decide(node: Any, who: Principal, proposal_id: int, approve: bool, reason: str = "") -> dict[str, Any]:
    """The safety lead's decision. An approved change is applied at once."""
    row = proposals.decide(node.db, who, proposal_id, approve, reason)
    out: dict[str, Any] = {"proposal_id": proposal_id, "status": row["status"], "decided_by": who.user_id}
    if approve:
        out["applied"] = apply_change(node, proposal_id)
        out["status"] = "applied"
    return out


def overrides(db: Database, site: str) -> list[dict[str, Any]]:
    """Trials whose machine verdict an operator overrode, with both verdicts side by side."""
    from ..tools.trialtools import trials_of
    return [{"run_id": t["run_id"], "operator": t["operator"], "machine_verdict": t["machine_verdict"],
             "operator_verdict": t["operator_verdict"], "reason": t["override_reason"]}
            for t in trials_of(db, site) if t["operator_decision"] == "override"]
