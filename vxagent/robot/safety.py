"""The safety filter between any requested Cartesian target and the robot.

Default limits are the ones published for the deployment framework this project follows
(Sun, Wen, Liu and Munawar, "On Transferring Surgical Policies from Simulation to Reality",
Sec. 3.5): steps of at most 5 mm and 5 degrees in free space and 2 mm and 3 degrees within
20 mm of the needle or the phantom, a local workspace of +/- 40 mm, joint-limit margins,
clearance above the supporting surface, and a stop when the measured pose differs from the
requested target by more than 2 mm or 2 degrees.

A target that fails a rule is rejected. It is never shortened or moved to make it pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .kinematics import Chain, interpolate, pose_distance


@dataclass(frozen=True)
class SafetyProfile:
    name: str = "trial"
    version: str = "1"
    free_step_m: float = 0.005
    free_step_rad: float = math.radians(5.0)
    near_step_m: float = 0.002
    near_step_rad: float = math.radians(3.0)
    near_distance_m: float = 0.020
    workspace_half_m: float = 0.040
    track_pos_m: float = 0.002
    track_rot_rad: float = math.radians(2.0)
    # The paper states that joint-limit margins and surface clearance are enforced but gives no
    # figures; these two are declared per site.
    joint_margin_rad: float = math.radians(5.0)
    joint_margin_m: float = 0.005
    clearance_m: float = 0.003


@dataclass(frozen=True)
class Scene:
    """What the filter needs to know about the surroundings, in the arm's base frame."""
    workspace_center: np.ndarray                       # centre of the local workspace
    surface_point: np.ndarray                          # a point on the supporting surface (pad)
    surface_normal: np.ndarray                         # unit normal pointing away from the pad
    near_points: tuple[np.ndarray, ...] = field(default_factory=tuple)   # needle, phantom features

    def height(self, p: np.ndarray) -> float:
        return float(np.dot(p - self.surface_point, self.surface_normal))

    def is_near(self, p: np.ndarray, distance: float) -> bool:
        if self.height(p) <= distance:
            return True
        return any(float(np.linalg.norm(p - n)) <= distance for n in self.near_points)


@dataclass(frozen=True)
class Violation:
    rule: str            # step_position | step_rotation | workspace | clearance | joint_margin |
                         # unreachable | tracking_position | tracking_rotation | clipped
    value: float
    limit: float
    unit: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        """The violation as tools show it. The keys say what each number is, because a bare
        "value" was read by the model as a joint angle when it was the room left to a limit."""
        scale = 1000.0 if self.unit == "mm" else (180.0 / math.pi if self.unit == "deg" else 1.0)
        value_key, limit_key = _KEYS.get(self.rule, ("value", "limit"))
        return {"rule": self.rule, f"{value_key}_{self.unit}": round(self.value * scale, 3),
                f"{limit_key}_{self.unit}": round(self.limit * scale, 3), "detail": self.detail}


_KEYS = {"workspace": ("offset_from_workspace_centre", "workspace_half_width"),
         "clearance": ("height_over_pad", "required_clearance"),
         "joint_margin": ("room_left_to_joint_limit", "required_margin"),
         "step_position": ("step", "step_limit"), "step_rotation": ("step", "step_limit"),
         "tracking_position": ("error", "stop_threshold"), "tracking_rotation": ("error", "stop_threshold"),
         "clipped": ("difference", "tolerance"), "unreachable": ("residual", "tolerance")}


def joint_margins(chain: Chain, profile: SafetyProfile) -> np.ndarray:
    return np.array([profile.joint_margin_rad if j.kind == "revolute" else profile.joint_margin_m
                     for j in chain.joints])


def check_pose(profile: SafetyProfile, scene: Scene, chain: Chain, T: np.ndarray, q: Any) -> list[Violation]:
    """Rules that depend only on where a pose is: workspace, clearance, joint margins."""
    out: list[Violation] = []
    p = T[:3, 3]
    off = np.abs(p - scene.workspace_center)
    if float(off.max()) > profile.workspace_half_m + 1e-9:
        axis = "xyz"[int(off.argmax())]
        out.append(Violation("workspace", float(off.max()), profile.workspace_half_m, "mm",
                             f"{axis} is outside the local workspace"))
    h = scene.height(p)
    if h < profile.clearance_m - 1e-9:
        out.append(Violation("clearance", h, profile.clearance_m, "mm", "too close to the supporting surface"))
    q = np.asarray(q, dtype=float)
    margins = joint_margins(chain, profile)
    for j, qi, m in zip(chain.joints, q, margins):
        room = min(qi - j.qmin, j.qmax - qi)
        if room < m - 1e-9:
            unit = "deg" if j.kind == "revolute" else "mm"
            out.append(Violation("joint_margin", float(room), float(m), unit, f"joint {j.name}"))
    return out


def step_limits(profile: SafetyProfile, scene: Scene, T_from: np.ndarray, T_to: np.ndarray) -> tuple[float, float]:
    near = (scene.is_near(T_from[:3, 3], profile.near_distance_m)
            or scene.is_near(T_to[:3, 3], profile.near_distance_m))
    return (profile.near_step_m, profile.near_step_rad) if near else (profile.free_step_m, profile.free_step_rad)


def check_step(profile: SafetyProfile, scene: Scene, chain: Chain, T_from: np.ndarray, T_to: np.ndarray,
               q_to: Any) -> list[Violation]:
    """Everything that must hold for one Cartesian target sent at the control rate."""
    out: list[Violation] = []
    lim_m, lim_rad = step_limits(profile, scene, T_from, T_to)
    d_m, d_rad = pose_distance(T_from, T_to)
    where = "near the needle or phantom" if lim_m == profile.near_step_m else "free space"
    if d_m > lim_m + 1e-9:
        out.append(Violation("step_position", d_m, lim_m, "mm", where))
    if d_rad > lim_rad + 1e-9:
        out.append(Violation("step_rotation", d_rad, lim_rad, "deg", where))
    return out + check_pose(profile, scene, chain, T_to, q_to)


def check_tracking(profile: SafetyProfile, requested: np.ndarray, setpoint: np.ndarray,
                   measured: np.ndarray) -> list[Violation]:
    """After a command: was the request clipped by inverse kinematics, and did the arm follow?"""
    out: list[Violation] = []
    c_m, c_rad = pose_distance(requested, setpoint)
    if c_m > 1e-4 or c_rad > 1e-3:
        out.append(Violation("clipped", c_m, 1e-4, "mm", "the controller set point is not the requested target"))
    t_m, t_rad = pose_distance(requested, measured)
    if t_m > profile.track_pos_m:
        out.append(Violation("tracking_position", t_m, profile.track_pos_m, "mm"))
    if t_rad > profile.track_rot_rad:
        out.append(Violation("tracking_rotation", t_rad, profile.track_rot_rad, "deg"))
    return out


def plan_steps(profile: SafetyProfile, scene: Scene, T_from: np.ndarray, T_to: np.ndarray,
               max_steps: int = 2000) -> list[np.ndarray]:
    """Cartesian targets from T_from to T_to, one per control tick, each inside the step limit
    that applies where it is. This generates a path; it does not make an unacceptable goal
    acceptable, and every target is still checked by check_step when it is executed."""
    targets: list[np.ndarray] = []
    current = T_from
    for _ in range(max_steps):
        d_m, d_rad = pose_distance(current, T_to)
        if d_m < 1e-7 and d_rad < 1e-7:
            break
        candidate = T_to
        # Try a free-space step first. If that step would start or end near the needle or the
        # phantom, the smaller limit applies and the step is made again with it.
        for lim_m, lim_rad in ((profile.free_step_m, profile.free_step_rad),
                               (profile.near_step_m, profile.near_step_rad)):
            if d_m <= lim_m and d_rad <= lim_rad:
                s = 1.0
            else:
                s = 0.999 * min(lim_m / d_m if d_m > 0 else math.inf, lim_rad / d_rad if d_rad > 0 else math.inf)
            candidate = interpolate(current, T_to, s)
            if step_limits(profile, scene, current, candidate) == (lim_m, lim_rad):
                break
        targets.append(candidate)
        current = candidate
    return targets
