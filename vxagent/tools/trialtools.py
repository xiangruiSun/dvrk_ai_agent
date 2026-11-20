"""What the operator's copilot can do about trials: check whether one can start, rehearse it on
a copy of the twin, and read the records of trials that were run. Starting a trial, giving the
go and confirming a verdict are the operator's own buttons; no tool here does any of them."""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..store import Database
from .registry import Tool, ToolContext, ToolError, ToolRegistry

TRIAL = frozenset({"trial:run"})
TRIAL_READ = frozenset({"robot:read"})


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(Args):
    pass


class TrialArgs(Args):
    run_id: str = Field(description="Id of a trial, exactly as list_trials returns it.")


def _node(ctx: ToolContext) -> Any:
    node = ctx.extras.get("node")
    if node is None:
        raise ToolError("failed", "no edge node is connected to this session")
    return node


def trial_preflight(a: NoArgs, ctx: ToolContext) -> dict[str, Any]:
    try:
        rows = _node(ctx).preflight_rows(ctx.principal.user_id)
    except FileNotFoundError as exc:
        raise ToolError("failed", f"the policy's demonstrations are not on this node: {exc}")
    failed = [{"check": r["check"], "detail": r["detail"], "repair_runbook": r["repair"]} for r in rows if not r["ok"]]
    ctx.extras["last_preflight_failed"] = failed or None
    return {"can_start": not failed, "checks_run": len(rows), "checks_failed": len(failed), "failed": failed,
            "note": "a failed check is repaired by its runbook or by the service engineer; it is not worked around"}


def rehearse_trial(a: NoArgs, ctx: ToolContext) -> dict[str, Any]:
    try:
        return _node(ctx).rehearse_trial(ctx.principal.user_id)
    except FileNotFoundError as exc:
        raise ToolError("failed", f"the policy's demonstrations are not on this node: {exc}")


def trials_of(db: Database, site: str) -> list[dict[str, Any]]:
    """Trials of a site, newest first, from their last checkpoint."""
    out = []
    for r in db.query("SELECT run_id, MAX(id) AS id FROM checkpoints GROUP BY run_id ORDER BY id DESC"):
        row = db.one("SELECT state FROM checkpoints WHERE id=?", (r["id"],))
        st = json.loads(row["state"])
        sheet = st.get("sheet") or {}
        if sheet.get("site") != site:
            continue
        d = st.get("data") or {}
        v, o = d.get("verdict") or {}, d.get("operator_verdict") or {}
        out.append({"run_id": r["run_id"], "operator": sheet.get("operator"), "status": (st.get("task") or {}).get("status"),
                    "outcome": d.get("outcome"), "machine_verdict": v.get("machine"), "operator_decision": o.get("decision"),
                    "operator_verdict": o.get("verdict"), "override_reason": o.get("reason"),
                    "waiting_for": (st.get("waiting_for") or {}).get("kind")})
    return out


def trial_state(db: Database, run_id: str) -> dict[str, Any] | None:
    row = db.one("SELECT state FROM checkpoints WHERE run_id=? ORDER BY id DESC LIMIT 1", (run_id,))
    return json.loads(row["state"]) if row else None


def list_trials(a: NoArgs, ctx: ToolContext) -> dict[str, Any]:
    site = getattr(ctx.site, "name", "")
    rows = trials_of(ctx.db, site)
    return {"site": site, "count": len(rows), "trials": rows[:15]}


def trial_record(a: TrialArgs, ctx: ToolContext) -> dict[str, Any]:
    st = trial_state(ctx.db, a.run_id)
    if st is None or (st.get("sheet") or {}).get("site") != getattr(ctx.site, "name", ""):
        raise ToolError("invalid_arguments", f"no trial {a.run_id!r} at this site", hint="call list_trials for the ids")
    d = st["data"]
    rec = d.get("record")
    if rec is None:
        failed = (d.get("preflight") or {}).get("failed") or []
        return {"run_id": a.run_id, "finished": (st.get("task") or {}).get("status") == "finished", "outcome": d.get("outcome"),
                "stages_run": [h["node"] for h in st.get("history", [])], "failed_checks": failed,
                "needle": "declared, not observed"}
    return {"run_id": a.run_id, "outcome": d.get("outcome"), "rollout": rec.get("rollout"), "envelope": rec.get("envelope"),
            "stages_run": rec.get("stages_run"), "reach_error_mm": (rec.get("reach") or {}).get("error_mm"),
            "needle_rose_mm": rec.get("needle_rose_mm"), "machine_verdict": rec["machine_verdict"],
            "operator_verdict": rec.get("operator_verdict"), "tip_travel_mm": rec.get("tip_travel_mm"),
            "motions": rec.get("motions"), "needle": "declared, not observed"}


def register_trial_tools(reg: ToolRegistry) -> ToolRegistry:
    reg.register(Tool("trial_preflight", "Check whether a trial could start now: arm, instrument, calibration, scene "
                      "records, policy contract, training range and perception. Each failed check names the runbook "
                      "that repairs it. Moves nothing.", NoArgs, trial_preflight, scopes=TRIAL, timeout_s=30))
    reg.register(Tool("rehearse_trial", "Run the whole trial on a copy of the twin through the safety filter. The arm "
                      "does not move. Returns control steps per stage and any rejection.", NoArgs, rehearse_trial,
                      tier="simulate", scopes=TRIAL, timeout_s=60))
    reg.register(Tool("list_trials", "Trials run at this site with their machine verdict and the operator's decision.",
                      NoArgs, list_trials, scopes=TRIAL_READ, timeout_s=10))
    reg.register(Tool("trial_record", "The record of one trial of this site: stages, reach error, machine verdict, "
                      "operator's decision.", TrialArgs, trial_record, scopes=TRIAL_READ, timeout_s=10))
    return reg
