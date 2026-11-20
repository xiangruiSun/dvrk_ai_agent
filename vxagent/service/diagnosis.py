"""Diagnosis on a ticket (task 12 R3).

The service engineer starts one of two methods on a ticket: the elimination strategy of task 04
or the triage team of task 08. Both work on a diagnostic run: a stretch of recorded motion
replayed through the deployment pipeline on a copy of the twin, with the site as it is now.
Every check that is run is attached to the ticket as a record, and the finding is stored as a
FaultReport whose evidence points at those records.

The model names the cause. Code writes the report around it: the class, the runbook for that
cause, the evidence ids and a confidence that only depends on how the method ended.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..diagnostics.episode import CAUSES, make_contract, run_episode
from ..diagnostics.tools import CHECKS, DESCRIPTIONS
from ..structured.schemas import FaultReport
from . import calibration as cal
from . import tickets

CLASS_OF = {"robot_calibration": "calibration", "policy_transfer": "configuration", "control_and_safety": "configuration",
            "visual_perception": "instrument"}
COMPONENT_OF = {"robot_calibration": "camera-to-robot calibration", "policy_transfer": "policy interface",
                "control_and_safety": "arm limits and tracking", "visual_perception": "needle perception"}


def stretches(node: Any, limit: int = 3) -> list[tuple[str, np.ndarray]]:
    """Stretches of recorded motion to replay. Without recordings on the node, one straight line."""
    out: list[tuple[str, np.ndarray]] = []
    try:
        from ..data import episodes as ep_store
        from ..pipelines import paths
        for e in ep_store.list_episodes(node.rosma_dir)[:8]:
            t, p = paths.positions(e)
            seg = paths.find_segment(t, p)
            if seg:
                out.append((e.episode_id, p[seg["first"]:seg["last"] + 1]))
            if len(out) >= limit:
                break
    except (FileNotFoundError, OSError):
        pass
    if not out:
        a, b = node.site.poses["start"].T[:3, 3], node.site.poses["pregrasp"].T[:3, 3]
        out.append(("straight line (no recordings on this node)", np.linspace(a, b, 30)))
    return out


def world_fault(node: Any) -> list[dict[str, Any]]:
    """What is wrong with the scenario's world right now, as the pipeline's fault description.
    Everything in it was INJECTED by a verification or a test; a normal node has none."""
    w = cal.world(node)
    faults = list(w.get("faults") or [])
    inj = w.get("injected")
    if inj and float(np.linalg.norm(inj["offset_mm"])) > 0 and not w.get("recalibrated"):
        faults.append({"cause": "calibration_offset", "mm": float(np.linalg.norm(inj["offset_mm"])), "direction": list(inj["offset_mm"])})
    return faults


def diagnostic_run(node: Any, seed: int = 0) -> dict[str, Any]:
    st = stretches(node)
    contract = make_contract(node.site, [p for _, p in st])
    rec = run_episode(node.site, st[0][1], contract, world_fault(node) or None, seed)
    rec["recording"] = st[0][0]
    return rec


def diagnose_ticket(node: Any, who: Any, ticket_id: int, method: str = "eliminate", llm: Any = None) -> dict[str, Any]:
    t = tickets.get(node.db, ticket_id)
    if t is None or not tickets.may_read(who, t) or "ticket:write" not in who.scopes:
        node.db.audit(who.user_id, who.role, "ticket:diagnose", f"ticket:{ticket_id}", False, "not a ticket of this person's site")
        raise PermissionError(f"{who.user_id} may not work on ticket {ticket_id}")
    if method not in ("eliminate", "team"):
        raise ValueError("method is eliminate or team")
    llm = llm or node.llm
    rec = diagnostic_run(node, seed=ticket_id)
    symptom = f"{t['symptom']} A diagnostic run on a copy of the twin: {rec['symptom']}"
    candidates = {k: v["what"] for k, v in CAUSES.items()}
    ids: dict[str, str] = {}

    def run_check(name: str) -> Any:
        result = CHECKS[name](rec)
        if name not in ids:
            rid = tickets.attach(node.db, ticket_id, "evidence", {"check": name, "on": f"diagnostic run replaying {rec['recording']}",
                                                                  "result": result}, who.user_id)
            ids[name] = f"ticket:{ticket_id}/evidence/{rid}"
        return result

    if method == "eliminate":
        from ..agent.strategies import diagnose
        d = diagnose("eliminate", llm, symptom, candidates, checks=DESCRIPTIONS, run_check=run_check, budget=len(CHECKS))
        cause, narrowed = d.cause, len(d.detail.get("open_at_the_end") or []) == 1
        cost = {"calls": d.calls, "tokens": d.tokens, "seconds": round(d.latency_ms / 1000, 1), "error": d.error}
        used = list(d.checks)
    else:
        from ..multi.team import run_design
        res = run_design("fan_out", llm, {"id": f"ticket-{ticket_id}", "symptom": symptom}, candidates, run_check,
                         {"calibration_valid": False, "grasp_verified": False})
        f = res.get("finding") or {}
        cause, narrowed = f.get("cause"), bool(f.get("evidence_for"))
        cost = {"calls": res["calls"], "tokens": res["tokens"], "seconds": res["seconds"], "error": None,
                "ledger_entries": res["entries"], "vetoes": res["vetoes"]}
        used = sorted({e["check"] for e in res["ledger"]})
    finding: dict[str, Any] = {"method": method, "cause": cause, "narrowed_to_one": narrowed, "checks_run": used, **cost}
    if cause is not None and used:
        from ..multi.team import RUNBOOK_FOR
        family = CAUSES[cause]["family"]
        report = FaultReport(symptom=t["symptom"], cause_class=CLASS_OF.get(family, "unknown"), component=COMPONENT_OF.get(family),
                             evidence=[{"source_id": ids[c], "locator": c} for c in used if c in ids],
                             next_check=RUNBOOK_FOR.get(cause, "escalate_to_engineer"),
                             confidence="medium" if narrowed else "low")
        finding["fault_report"] = report.model_dump()
    rid = tickets.attach(node.db, ticket_id, "finding", finding, who.user_id)
    finding["record"] = f"ticket:{ticket_id}/finding/{rid}"
    finding["evidence_resolves"] = all(any(r["record"] == e["source_id"] for r in tickets.records(node.db, ticket_id))
                                       for e in (finding.get("fault_report") or {}).get("evidence", []))
    return finding
