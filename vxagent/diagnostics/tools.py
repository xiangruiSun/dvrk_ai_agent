"""Diagnostic tools. Each one is a deterministic measurement over the record of a run: it
returns numbers and the limit they are judged against, never a verdict. The same record always
gives the same result, so a check is never worth running twice.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..tools.registry import Tool, ToolContext, ToolError, ToolRegistry

DIAG_READ = frozenset({"robot:read"})


def compare_poses(rec: dict[str, Any]) -> dict[str, Any]:
    """Commanded target against controller set point, and set point against measured pose."""
    rows = [s for s in rec["steps"] if "measured_mm" in s]
    if not rows:
        return {"steps_with_motion": 0, "note": "no step of this run reached the arm"}
    cmd = np.array([s["commanded_mm"] for s in rows])
    sp = np.array([s["setpoint_mm"] for s in rows])
    ms = np.array([s["measured_mm"] for s in rows])
    clip, track = np.linalg.norm(cmd - sp, axis=1), np.linalg.norm(sp - ms, axis=1)
    return {"steps_with_motion": len(rows),
            "commanded_vs_setpoint_mm": {"max": round(float(clip.max()), 3), "mean": round(float(clip.mean()), 3)},
            "setpoint_vs_measured_mm": {"max": round(float(track.max()), 3), "mean": round(float(track.mean()), 3),
                                        "first_step_over_threshold": next((rows[i]["step"] for i in range(len(rows))
                                                                           if track[i] > rec["tracking_stop_mm"]), None)},
            "tracking_stop_threshold_mm": rec["tracking_stop_mm"]}


def calibration_residual(rec: dict[str, Any]) -> dict[str, Any]:
    """The calibration record applied to held-out points: where it predicts the arm would touch
    against where the arm did touch."""
    cal = rec["calibration"]
    T, seen, touched = np.array(cal["T_est"]), np.array(cal["seen_m"]), np.array(cal["touched_m"])
    pred = seen @ T[:3, :3].T + T[:3, 3]
    err = (pred - touched) * 1000.0
    norm = np.linalg.norm(err, axis=1)
    a, b = pred - pred.mean(axis=0), touched - touched.mean(axis=0)
    agree = [round(float(np.mean(np.sign(a[:, i]) == np.sign(b[:, i]))), 2) for i in range(3)]
    return {"calibration_id": cal["id"], "held_out_points": len(norm),
            "residual_mm": {"mean": round(float(norm.mean()), 2), "max": round(float(norm.max()), 2),
                            "spread": round(float(norm.std()), 2)},
            "mean_error_vector_mm": [round(float(x), 2) for x in err.mean(axis=0)],
            "share_of_points_on_the_same_side_of_the_centre_per_axis": agree,
            "acceptance_mm": 1.0}


def policy_ranges(rec: dict[str, Any]) -> dict[str, Any]:
    """Observations and actions of the run against the range the policy was trained on."""
    c = rec["contract"]
    lo, hi = np.array(c["observation_min_mm"]), np.array(c["observation_max_mm"])
    obs = np.array([s["observation_mm"] for s in rec["steps"] if "observation_mm" in s])
    out: dict[str, Any] = {"training_observation_min_mm": c["observation_min_mm"],
                           "training_observation_max_mm": c["observation_max_mm"],
                           "training_action_step_max_mm": c["action_step_max_mm"], "action_unit_of_the_policy": c["action_unit"]}
    if len(obs):
        over = np.maximum(obs - hi, 0) + np.maximum(lo - obs, 0)
        out.update({"observation_at_first_step_mm": [round(float(x), 2) for x in obs[0]],
                    "share_of_steps_outside_the_training_range": round(float(np.mean(over.max(axis=1) > 0)), 2),
                    "largest_distance_outside_the_range_mm": round(float(over.max()), 2),
                    "outside_at_first_step": bool(over[0].max() > 0)})
    acts = [(s["action_norm_policy_unit"], s["commanded_step_mm"]) for s in rec["steps"] if "commanded_step_mm" in s
            and s["action_norm_policy_unit"] > 0]
    if acts:
        policy_mm = np.array([a for a, _ in acts]) * (1000.0 if c["action_unit"] == "m" else 1.0)
        sent_mm = np.array([b for _, b in acts])
        out.update({"policy_step_mm_median": round(float(np.median(policy_mm)), 3),
                    "step_sent_to_the_robot_mm_median": round(float(np.median(sent_mm)), 4),
                    "ratio_sent_to_policy": round(float(np.median(sent_mm / policy_mm)), 4)})
    return out


def safety_rejections(rec: dict[str, Any]) -> dict[str, Any]:
    """Targets the safety filter refused, by rule."""
    rej = [s for s in rec["stops"] if s["kind"] == "rejected_by_safety_filter"]
    by: dict[str, int] = {}
    for r in rej:
        by[r["rule"]] = by.get(r["rule"], 0) + 1
    return {"targets_rejected": len(rej), "by_rule": by,
            "first": {k: v for k, v in rej[0].items() if k != "kind"} if rej else None,
            "other_stops": [s["kind"] for s in rec["stops"] if s["kind"] != "rejected_by_safety_filter"]}


def perception_quality(rec: dict[str, Any]) -> dict[str, Any]:
    """Needle detection over the run: how often it was valid and how confident."""
    valid = [bool(s["detection_valid"]) for s in rec["steps"]]
    conf = [s["detection_confidence"] for s in rec["steps"]]
    longest = run = 0
    for v in valid:
        run = 0 if v else run + 1
        longest = max(longest, run)
    return {"steps": len(valid), "share_of_steps_with_a_valid_detection": round(float(np.mean(valid)), 2) if valid else None,
            "longest_gap_steps": longest, "mean_confidence": round(float(np.mean(conf)), 2) if conf else None,
            "steps_held_for_lack_of_a_target": sum(1 for s in rec["steps"] if s.get("held")),
            "gap_the_pipeline_tolerates_steps": 5}


CHECKS = {"compare_poses": compare_poses, "calibration_residual": calibration_residual, "policy_ranges": policy_ranges,
          "safety_rejections": safety_rejections, "perception_quality": perception_quality}
DESCRIPTIONS = {name: (fn.__doc__ or "").strip().replace("\n    ", " ") for name, fn in CHECKS.items()}


class RunArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run: str = Field(description="Id of the recorded run to examine.")


def _record(a: RunArgs, ctx: ToolContext) -> dict[str, Any]:
    runs = ctx.extras.get("diagnostic_runs") or {}
    if a.run not in runs:
        raise ToolError("invalid_arguments", f"no recorded run {a.run!r}", hint=f"runs: {', '.join(runs) or 'none'}")
    return runs[a.run]


def register_diagnostics(reg: ToolRegistry) -> ToolRegistry:
    for name, fn in CHECKS.items():
        reg.register(Tool(name, DESCRIPTIONS[name] + " Returns numbers, not a verdict.", RunArgs,
                          lambda a, ctx, fn=fn: fn(_record(a, ctx)), scopes=DIAG_READ, cacheable=True, timeout_s=10))
    return reg
