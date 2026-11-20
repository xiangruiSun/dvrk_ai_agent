"""What tools need and what they change, over a small fixed vocabulary of facts, and what
each tool's result contains. A plan is checked against these before anything runs.

A fact is True, False or None. None means it cannot be established by looking: a step that
needs such a fact is an error, not a guess.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..structured.proposals import calibration

FACTS = ("powered", "enabled", "homed", "calibration_valid", "instrument_matches", "needle_visible", "needle_held",
         "in_workspace")

# tool -> (facts it needs, facts it sets)
TOOL_FACTS: dict[str, tuple[tuple[str, ...], dict[str, bool]]] = {
    "get_robot_state": ((), {}),
    "list_named_poses": ((), {}),
    "check_named_target": (("powered",), {}),
    "rehearse_move": (("powered",), {}),
    "enable_and_home": (("powered",), {"enabled": True, "homed": True}),
    "move_to_named_pose": (("powered", "enabled", "homed"), {"in_workspace": True}),
    "set_jaw": (("powered", "enabled", "homed"), {}),
    "read_calibration_record": ((), {}),
    "calibration_residual": ((), {}),
    "calculator": ((), {}),
    "observe_needle": (("powered",), {}),
    "hold": ((), {}),
    "sweep_calibration_poses": (("powered", "enabled", "homed"), {"in_workspace": True}),
    "fit_calibration": ((), {}),
    "record_site_event": ((), {"calibration_valid": False}),
    "trial_preflight": ((), {}),
}

# tool -> fields of its result that a later step may refer to, with their JSON type
RESULT_FIELDS: dict[str, dict[str, str]] = {
    "get_robot_state": {"state_seq": "integer", "powered": "boolean", "enabled": "boolean", "homed": "boolean",
                        "busy": "boolean", "jaw_deg": "number", "site": "string", "arm": "string"},
    "list_named_poses": {"site": "string"},
    "check_named_target": {"named_pose": "string", "accepted": "boolean", "distance_mm": "number",
                           "rotation_deg": "number", "control_steps": "integer", "state_seq": "integer"},
    "rehearse_move": {"named_pose": "string", "would_complete": "boolean", "control_steps": "integer",
                      "expected_duration_s": "number"},
    "enable_and_home": {"status": "string", "enabled": "boolean", "homed": "boolean", "state_seq": "integer"},
    "move_to_named_pose": {"status": "string", "named_pose": "string", "final_error_mm": "number", "state_seq": "integer"},
    "set_jaw": {"status": "string", "jaw": "string", "jaw_deg": "number", "state_seq": "integer"},
    "read_calibration_record": {"calibration_id": "string", "site": "string", "method": "string", "valid": "boolean",
                                "held_out_points": "integer"},
    "calibration_residual": {"calibration_id": "string", "mean_mm": "number", "max_mm": "number",
                             "acceptance_mm": "number", "within_acceptance": "boolean"},
    "calculator": {"value": "number"},
    "observe_needle": {"visible": "boolean", "held": "boolean", "height_over_pad_mm": "number", "state_seq": "integer"},
    "hold": {"seconds": "number"},
    "sweep_calibration_poses": {"status": "string", "samples": "integer", "sweep_id": "string", "state_seq": "integer"},
    "fit_calibration": {"draft_id": "string", "samples": "integer", "held_out": "integer", "held_out_rigid_mean_mm": "number",
                        "held_out_with_affine_mean_mm": "number", "within_acceptance": "boolean"},
    "record_site_event": {"event": "string", "calibration_valid": "boolean"},
    "trial_preflight": {"can_start": "boolean", "checks_failed": "integer"},
}


def observe_facts(ctx: Any) -> dict[str, bool | None]:
    """The facts as they are now, read from the robot and the site's records."""
    out: dict[str, bool | None] = {f: None for f in FACTS}
    if ctx.robot is not None:
        s = ctx.robot.state()
        out.update({"powered": bool(s.powered) and s.fault is None, "enabled": bool(s.enabled), "homed": bool(s.homed)})
        if ctx.site is not None:
            off = np.abs(s.measured_cp[:3, 3] - ctx.site.scene.workspace_center)
            out["in_workspace"] = bool(off.max() <= ctx.site.profile.workspace_half_m + 1e-9)
    if ctx.site is not None:
        out["calibration_valid"] = bool(calibration(ctx.db, ctx.site.name)["valid"])
    needle = ctx.extras.get("needle")
    if callable(needle):
        seen = needle(ctx)
        out["needle_visible"], out["needle_held"] = bool(seen.get("visible")), bool(seen.get("held"))
    return out
