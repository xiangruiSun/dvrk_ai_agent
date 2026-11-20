"""Serial-chain kinematics from dVRK kinematic files.

A dVRK arm is described by two files: the arm (for a PSM: yaw, pitch, insertion) and the
instrument (roll, wrist pitch, wrist yaw, the jaw limits and a fixed tool-tip transform). Both
use modified Denavit-Hartenberg parameters, where the transform of joint i is

    Rx(alpha) . Tx(A) . Rz(theta) . Tz(D)

with the joint value added to theta (revolute) or D (prismatic) together with the joint's offset.
Everything here is SI: metres and radians, as in the files.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import jsonc


@dataclass(frozen=True)
class Joint:
    name: str
    alpha: float
    a: float
    theta: float
    d: float
    kind: str          # revolute | prismatic
    offset: float
    qmin: float
    qmax: float


def _dh(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(theta), math.sin(theta)
    return np.array([[ct, -st, 0.0, a],
                     [st * ca, ct * ca, -sa, -sa * d],
                     [st * sa, ct * sa, ca, ca * d],
                     [0.0, 0.0, 0.0, 1.0]])


class Chain:
    def __init__(self, joints: list[Joint], tooltip: np.ndarray | None = None,
                 jaw_limits: tuple[float, float] = (0.0, 1.0), source: dict[str, str] | None = None):
        self.joints = joints
        self.tooltip = np.eye(4) if tooltip is None else np.asarray(tooltip, dtype=float)
        self.jaw_limits = jaw_limits
        self.source = source or {}
        self.qmin = np.array([j.qmin for j in joints])
        self.qmax = np.array([j.qmax for j in joints])

    @property
    def n(self) -> int:
        return len(self.joints)

    def _frames(self, q: np.ndarray) -> list[np.ndarray]:
        T = np.eye(4)
        frames = []
        for j, qi in zip(self.joints, q):
            if j.kind == "revolute":
                T = T @ _dh(j.alpha, j.a, j.theta + j.offset + qi, j.d)
            else:
                T = T @ _dh(j.alpha, j.a, j.theta, j.d + j.offset + qi)
            frames.append(T)
        return frames

    def fk(self, q: Any) -> np.ndarray:
        """Tool-tip pose in the arm's base frame for joint values q."""
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n,):
            raise ValueError(f"expected {self.n} joint values, got {q.shape}")
        return self._frames(q)[-1] @ self.tooltip

    def jacobian(self, q: Any) -> np.ndarray:
        """Geometric Jacobian (6 x n) of the tool tip, expressed in the base frame."""
        q = np.asarray(q, dtype=float)
        frames = self._frames(q)
        p = (frames[-1] @ self.tooltip)[:3, 3]
        J = np.zeros((6, self.n))
        for i, (j, T) in enumerate(zip(self.joints, frames)):
            z, o = T[:3, 2], T[:3, 3]
            if j.kind == "revolute":
                J[:3, i] = np.cross(z, p - o)
                J[3:, i] = z
            else:
                J[:3, i] = z
        return J

    def clamp(self, q: Any) -> np.ndarray:
        return np.minimum(np.maximum(np.asarray(q, dtype=float), self.qmin), self.qmax)

    def within_limits(self, q: Any, margin: Any = 0.0) -> bool:
        q = np.asarray(q, dtype=float)
        m = np.asarray(margin, dtype=float)
        return bool(np.all(q >= self.qmin + m - 1e-12) and np.all(q <= self.qmax - m + 1e-12))

    def ik(self, target: np.ndarray, seed: Any, iterations: int = 200, damping: float = 1e-3,
           tol_m: float = 1e-7, tol_rad: float = 1e-6) -> tuple[np.ndarray, bool]:
        """Damped least-squares inverse kinematics from a seed. Joint limits are respected by
        clamping after every step, so the result may not reach the target: the caller compares
        fk(result) with what it asked for, which is how clipping is detected."""
        q = self.clamp(seed)
        for _ in range(iterations):
            T = self.fk(q)
            e = np.concatenate([target[:3, 3] - T[:3, 3], rotvec(target[:3, :3] @ T[:3, :3].T)])
            if np.linalg.norm(e[:3]) < tol_m and np.linalg.norm(e[3:]) < tol_rad:
                return q, True
            J = self.jacobian(q)
            dq = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(6), e)
            step = np.linalg.norm(dq)
            if step > 0.5:
                dq *= 0.5 / step
            q = self.clamp(q + dq)
        T = self.fk(q)
        ok = (np.linalg.norm(target[:3, 3] - T[:3, 3]) < 1e-5
              and angle_between(target[:3, :3], T[:3, :3]) < 1e-4)
        return q, bool(ok)


def load_chain(arm_file: Path | str, instrument_file: Path | str) -> Chain:
    arm, tool = jsonc.load(arm_file), jsonc.load(instrument_file)
    joints = []
    for spec in arm["DH"]["joints"] + tool["DH"]["joints"]:
        joints.append(Joint(name=spec["name"], alpha=float(spec["alpha"]), a=float(spec["A"]),
                            theta=float(spec["theta"]), d=float(spec["D"]), kind=spec["type"],
                            offset=float(spec["offset"]), qmin=float(spec["qmin"]), qmax=float(spec["qmax"])))
    for conv in (arm["DH"].get("convention"), tool["DH"].get("convention")):
        if conv != "modified":
            raise ValueError(f"only modified DH is supported, file says {conv!r}")
    jaw = tool.get("jaw") or {}
    return Chain(joints, tooltip=np.array(tool.get("tooltip_offset", np.eye(4)), dtype=float),
                 jaw_limits=(float(jaw.get("qmin", 0.0)), float(jaw.get("qmax", 1.0))),
                 source={"arm": Path(arm_file).name, "instrument": Path(instrument_file).name})


# ----------------------------------------------------------------------------- rotations

def rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis times angle) of a rotation matrix."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    angle = math.acos(c)
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-6:          # near a half turn the antisymmetric part vanishes
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        k = int(np.argmax(axis))
        signs = np.sign(R[k, :] + R[:, k])
        signs[k] = 1.0
        return axis * signs * angle
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * math.sin(angle)) * angle


def from_rotvec(v: Any) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def angle_between(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.linalg.norm(rotvec(Ra @ Rb.T)))


def interpolate(Ta: np.ndarray, Tb: np.ndarray, s: float) -> np.ndarray:
    """Pose a fraction s of the way from Ta to Tb: straight line in position, constant axis in
    rotation."""
    T = np.eye(4)
    T[:3, 3] = Ta[:3, 3] + s * (Tb[:3, 3] - Ta[:3, 3])
    T[:3, :3] = from_rotvec(s * rotvec(Tb[:3, :3] @ Ta[:3, :3].T)) @ Ta[:3, :3]
    return T


def pose_distance(Ta: np.ndarray, Tb: np.ndarray) -> tuple[float, float]:
    """(metres, radians) between two poses."""
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), angle_between(Ta[:3, :3], Tb[:3, :3])
