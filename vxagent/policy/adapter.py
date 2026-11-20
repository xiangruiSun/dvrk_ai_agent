"""The adapter between a policy's space and a site, generated from the contract.

Policy space (SurgicAI): a pose is [x, y, z, roll, pitch, yaw, jaw] in the arm's base frame,
translations in the contract's unit, the rotation R = Rz(yaw) Ry(pitch) Rx(roll). An action is
seven numbers in [-1, 1]; the commanded pose becomes pose + action * step_size.

Site space: 4 x 4 transforms in metres in the arm's base frame. The simulator uses the dVRK PSM
model, so the base frames agree in orientation; a site anchors the policy's poses by one
translation that puts the policy's goal on the site's grasp pose.

The adapter is round-trip tested: a pose taken to site space and back must be the same pose.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..structured.units import convert
from .contract import Contract


def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def R_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    pitch = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    if abs(math.cos(pitch)) < 1e-9:
        return 0.0, pitch, math.atan2(-R[0, 1], R[1, 1])
    return math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])


class Adapter:
    def __init__(self, contract: Contract, anchor_m: Any = (0.0, 0.0, 0.0)):
        self.c = contract
        self.anchor = np.array(anchor_m, dtype=float)                 # site position = policy position + anchor
        self._to_m = convert(1.0, contract.translation_unit, "m")

    @classmethod
    def anchored(cls, contract: Contract, policy_goal: Any, site_goal_T: np.ndarray) -> "Adapter":
        """The adapter that puts the policy's goal position on a pose of the site."""
        scale = convert(1.0, contract.translation_unit, "m")
        return cls(contract, site_goal_T[:3, 3] - np.asarray(policy_goal, dtype=float)[:3] * scale)

    def pose_to_site(self, pose: Any) -> np.ndarray:
        p = np.asarray(pose, dtype=float)
        T = np.eye(4)
        T[:3, :3] = rpy_to_R(*p[3:6])
        T[:3, 3] = p[:3] * self._to_m + self.anchor
        return T

    def pose_to_policy(self, T: np.ndarray, jaw: float = 0.0) -> np.ndarray:
        r, p, y = R_to_rpy(T[:3, :3])
        return np.array([*((T[:3, 3] - self.anchor) / self._to_m), r, p, y, jaw])

    def step(self, pose: Any, action: Any) -> np.ndarray:
        """The policy-space pose after one action."""
        a = np.clip(np.asarray(action, dtype=float), self.c.action_low, self.c.action_high)
        scale = np.array([self.c.step_translation] * 3 + [self.c.step_rotation_rad] * 3 + [self.c.step_jaw])
        return np.asarray(pose, dtype=float) + a * scale

    def observation(self, T_now: np.ndarray, T_goal: np.ndarray, jaw: float = 0.0, goal_jaw: float = 0.0) -> np.ndarray:
        cur, goal = self.pose_to_policy(T_now, jaw), self.pose_to_policy(T_goal, goal_jaw)
        return np.concatenate([cur, goal, goal - cur])

    def largest_step(self) -> tuple[float, float]:
        """(metres, radians) the largest single action can ask for."""
        a = max(abs(self.c.action_low), abs(self.c.action_high))
        return math.sqrt(3) * a * self.c.step_translation * self._to_m, math.sqrt(3) * a * self.c.step_rotation_rad


def angle_between_rpy(a: Any, b: Any) -> float:
    Ra, Rb = rpy_to_R(*np.asarray(a, dtype=float)[3:6]), rpy_to_R(*np.asarray(b, dtype=float)[3:6])
    c = max(-1.0, min(1.0, (np.trace(Ra @ Rb.T) - 1.0) / 2.0))
    return math.acos(c)
