"""Service tools: the site's calibration record and its residual on held-out points.

A calibration record is kept per site. Reading it returns its id and whether it is still valid
(a change of instrument or base frame invalidates it, task 02 R10). Its residual is computed by
code from the held-out point pairs stored with the record.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..diagnostics.tools import calibration_residual as _residual
from ..store import Database, dumps
from ..structured.proposals import calibration as calibration_status
from .registry import Tool, ToolContext, ToolError, ToolRegistry

ROBOT_READ = frozenset({"robot:read"})


def store_calibration_record(db: Database, site: str, record: dict[str, Any]) -> None:
    """record: {id, T_est, seen_m, touched_m, method}. Written when a calibration is accepted."""
    db.execute("INSERT OR REPLACE INTO kv(k,v) VALUES (?,?)", (f"calrec:{site}", dumps(record)))


def load_calibration_record(db: Database, site: str) -> dict[str, Any] | None:
    row = db.one("SELECT v FROM kv WHERE k=?", (f"calrec:{site}",))
    return json.loads(row["v"]) if row else None


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(Args):
    pass


class CalibrationArgs(Args):
    calibration_id: str = Field(description="Id of a calibration record, as read_calibration_record returns it.")


def read_calibration_record(a: NoArgs, ctx: ToolContext) -> dict[str, Any]:
    site = getattr(ctx.site, "name", "")
    rec = load_calibration_record(ctx.db, site)
    if rec is None:
        raise ToolError("failed", f"no calibration record is stored for {site}", hint="a calibration has to be made first")
    status = calibration_status(ctx.db, site)
    return {"calibration_id": rec["id"], "site": site, "method": rec.get("method", ""), "frames": "robot base from camera",
            "held_out_points": len(rec["touched_m"]), "valid": bool(status["valid"]), "validity_note": status["why"]}


def calibration_residual(a: CalibrationArgs, ctx: ToolContext) -> dict[str, Any]:
    rec = load_calibration_record(ctx.db, getattr(ctx.site, "name", ""))
    if rec is None or rec["id"] != a.calibration_id:
        raise ToolError("invalid_arguments", f"no calibration record {a.calibration_id!r} at this site",
                        hint="call read_calibration_record for the id")
    out = _residual({"calibration": rec})
    r = out["residual_mm"]
    return {"calibration_id": rec["id"], "held_out_points": out["held_out_points"], "mean_mm": r["mean"],
            "max_mm": r["max"], "acceptance_mm": out["acceptance_mm"], "within_acceptance": r["mean"] <= out["acceptance_mm"]}


def register_service(reg: ToolRegistry) -> ToolRegistry:
    reg.register(Tool("read_calibration_record", "Read the site's current camera-to-robot calibration record: its id, "
                      "the method, and whether it is still valid.", NoArgs, read_calibration_record, scopes=ROBOT_READ,
                      timeout_s=5))
    reg.register(Tool("calibration_residual", "Residual of a calibration record on its held-out points, in mm, with "
                      "the acceptance limit. Needs the record's calibration_id.", CalibrationArgs, calibration_residual,
                      scopes=ROBOT_READ, timeout_s=10))
    return reg
