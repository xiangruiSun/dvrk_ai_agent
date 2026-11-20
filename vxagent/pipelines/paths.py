"""Approach paths taken from recorded dVRK motion.

A stretch of a real recording (the PSM1 tool tip in ROSMA, 50 Hz) is used as the shape of an
approach: where a person's hand actually went, with its curvature and hesitations. The stretch
is mapped into the site by a rotation, a uniform scale and a translation that put its first
sample on the pose the arm starts from and its last sample on the pose it must reach. Nothing
else is changed. The orientation is turned evenly from the first pose to the last.

What this is not: the recording was made on another robot doing another task, so the mapped
path says nothing about suturing. It gives the runtime and the safety filter a path that was
not drawn with a ruler.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..data.episodes import Episode, load
from ..robot.kinematics import from_rotvec, interpolate


def positions(ep: Episode, arm: str = "PSM1") -> tuple[np.ndarray, np.ndarray]:
    t, cols = load(ep)
    p = np.stack([cols[f"{arm}_position_{a}"] for a in "xyz"], axis=1)
    return t, p


def find_segment(t: np.ndarray, p: np.ndarray, net_mm: tuple[float, float] = (24.0, 34.0), max_s: float = 3.0,
                 stride: int = 5) -> dict[str, Any] | None:
    """The first stretch of at most max_s seconds whose net displacement lies in net_mm. With
    the default range the scale needed to fit a 28 mm approach stays between 0.8 and 1.2."""
    n = len(t)
    for i in range(0, n, stride):
        j_end = int(np.searchsorted(t, t[i] + max_s))
        if j_end <= i + 2:
            continue
        d = np.linalg.norm(p[i + 1:j_end] - p[i], axis=1) * 1000.0
        hits = np.nonzero((d >= net_mm[0]) & (d <= net_mm[1]))[0]
        if len(hits):
            j = i + 1 + int(hits[0])
            seg = p[i:j + 1]
            return {"start_s": round(float(t[i]), 3), "end_s": round(float(t[j]), 3), "samples": int(j - i + 1),
                    "net_mm": round(float(d[hits[0]]), 2),
                    "path_mm": round(float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()) * 1000.0, 2),
                    "first": i, "last": j}
    return None


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s, c = float(np.linalg.norm(axis)), float(np.dot(a, b))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        other = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, other)
        return from_rotvec(axis / np.linalg.norm(axis) * math.pi)
    return from_rotvec(axis / s * math.atan2(s, c))


def fit_path(points: np.ndarray, T_from: np.ndarray, T_to: np.ndarray, spacing_m: float = 0.002
             ) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Waypoints from T_from to T_to that follow the shape of `points` (N x 3, metres)."""
    a, b = points[-1] - points[0], T_to[:3, 3] - T_from[:3, 3]
    scale = float(np.linalg.norm(b) / np.linalg.norm(a))
    R = _rotation_between(a, b)
    mapped = T_from[:3, 3] + scale * (points - points[0]) @ R.T
    mapped[-1] = T_to[:3, 3]
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(mapped, axis=0), axis=1))])
    total = float(arc[-1])
    keep, last = [], 0.0
    for i in range(1, len(mapped)):
        if arc[i] - last >= spacing_m or i == len(mapped) - 1:
            keep.append(i)
            last = float(arc[i])
    way = []
    for i in keep:
        T = interpolate(T_from, T_to, float(arc[i] / total) if total > 0 else 1.0)
        T[:3, 3] = mapped[i]
        way.append(T)
    way[-1] = T_to.copy()
    line = b / np.linalg.norm(b)
    off = mapped - T_from[:3, 3]
    side = off - np.outer(off @ line, line)
    info = {"scale": round(scale, 3), "waypoints": len(way), "mapped_path_mm": round(total * 1000.0, 2),
            "straight_line_mm": round(float(np.linalg.norm(b)) * 1000.0, 2),
            "largest_distance_from_the_straight_line_mm": round(float(np.linalg.norm(side, axis=1).max()) * 1000.0, 2)}
    return way, info
