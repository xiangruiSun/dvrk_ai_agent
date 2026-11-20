"""Executing a Cartesian path on a robot backend, one target per control tick, with the safety
filter in front of every command and a tracking check behind it.

This module contains no language model and is never waited on by one: it is the control-rate
part of the system.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from .backend import RobotBackend, RobotState
from .kinematics import Chain, pose_distance, rotvec
from .safety import SafetyProfile, Scene, Violation, check_pose, check_step, check_tracking, plan_steps


class MotionStopped(Exception):
    """A path ended early. kind: safety_rejected | tracking_error | canceled"""

    def __init__(self, kind: str, message: str, violations: list[Violation], steps_done: int):
        super().__init__(message)
        self.kind, self.violations, self.steps_done = kind, violations, steps_done


def pose_report(T: np.ndarray, scene: Scene | None = None) -> dict[str, Any]:
    """A pose the way tools show it to the model: millimetres and degrees, with the unit in
    every key."""
    rv = rotvec(T[:3, :3])
    out: dict[str, Any] = {"position_mm": [round(float(x) * 1000, 2) for x in T[:3, 3]],
                           "rotation_vector_deg": [round(math.degrees(float(x)), 2) for x in rv]}
    if scene is not None:
        out["height_over_pad_mm"] = round(scene.height(T[:3, 3]) * 1000, 2)
    return out


def state_report(s: RobotState, scene: Scene | None = None) -> dict[str, Any]:
    return {"state_seq": s.seq, "powered": s.powered, "enabled": s.enabled, "homed": s.homed,
            "busy": s.busy, "fault": s.fault,
            "measured": pose_report(s.measured_cp, scene),
            "jaw_deg": round(math.degrees(s.jaw), 1)}


def check_goal(profile: SafetyProfile, scene: Scene, chain: Chain, T_goal: np.ndarray, seed_q: Any) -> list[Violation]:
    """Is the destination itself acceptable? Checked before anything moves."""
    q, ok = chain.ik(T_goal, seed_q)
    out = check_pose(profile, scene, chain, T_goal, q)
    if not ok:
        d_m, _ = pose_distance(T_goal, chain.fk(q))
        out.append(Violation("unreachable", d_m, 1e-5, "mm", "inverse kinematics cannot reach this pose"))
    return out


def execute_path(robot: RobotBackend, chain: Chain, profile: SafetyProfile, scene: Scene, T_goal: np.ndarray,
                 command_id: str, tick: Callable[[float], None], cancel: Callable[[], bool] = lambda: False,
                 feedback: Callable[..., None] | None = None) -> dict[str, Any]:
    """Move from the current set point to T_goal. Raises MotionStopped when a target is rejected,
    when the arm does not follow, or on cancel. The arm stays where it stopped."""
    start = robot.state()
    targets = plan_steps(profile, scene, start.setpoint_cp, T_goal)
    total = max(1, len(targets))
    current = start.setpoint_cp
    done = 0
    for T in targets:
        if cancel():
            raise MotionStopped("canceled", "the motion was canceled", [], done)
        q_next, _ = chain.ik(T, robot.state().q)
        bad = check_step(profile, scene, chain, current, T, q_next)
        if bad:
            raise MotionStopped("safety_rejected", f"target rejected by rule {bad[0].rule}", bad, done)
        s = robot.servo_cp(T, command_id)
        tracked = check_tracking(profile, T, s.setpoint_cp, s.measured_cp)
        if tracked:
            raise MotionStopped("tracking_error", f"the arm did not follow: {tracked[0].rule}", tracked, done + 1)
        done += 1
        current = T
        if feedback is not None:
            feedback(fraction=round(done / total, 3), state_seq=s.seq,
                     position_mm=[round(float(x) * 1000, 2) for x in s.measured_cp[:3, 3]])
        tick(1.0 / robot.rate_hz)
    return {"steps": done, "planned_steps": len(targets)}
