"""A site: one arm with its instrument, its named poses, its surroundings and its safety profile.

The arm's kinematics come from the dVRK project's own files. Everything a lab decides for
itself (where "start" is, where the pad and the needle are, which poses have names) is declared
in sites/<name>.json, in millimetres and degrees, and converted here.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_settings
from .kinematics import Chain, from_rotvec, load_chain
from .safety import SafetyProfile, Scene
from .twin import KinematicTwin


@dataclass
class NamedPose:
    name: str
    description: str
    T: np.ndarray
    q: np.ndarray
    reachable: bool


@dataclass
class Site:
    name: str
    arm: str
    generation: str
    chain: Chain
    home_q: np.ndarray
    scene: Scene
    profile: SafetyProfile
    poses: dict[str, NamedPose] = field(default_factory=dict)
    jaw_angles: dict[str, float] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)

    def twin(self) -> KinematicTwin:
        return KinematicTwin(self.chain, self.home_q, jaw=self.jaw_angles.get("open", 0.0))


def _profile(spec: dict[str, Any]) -> SafetyProfile:
    return SafetyProfile(
        name=spec["name"], version=str(spec["version"]),
        free_step_m=spec["free_step_mm"] / 1000, free_step_rad=math.radians(spec["free_step_deg"]),
        near_step_m=spec["near_step_mm"] / 1000, near_step_rad=math.radians(spec["near_step_deg"]),
        near_distance_m=spec["near_distance_mm"] / 1000, workspace_half_m=spec["workspace_half_mm"] / 1000,
        track_pos_m=spec["tracking_stop_mm"] / 1000, track_rot_rad=math.radians(spec["tracking_stop_deg"]),
        joint_margin_rad=math.radians(spec["joint_margin_deg"]), joint_margin_m=spec["joint_margin_mm"] / 1000,
        clearance_m=spec["clearance_mm"] / 1000)


def build_site(spec: dict[str, Any], chain: Chain) -> Site:
    home_q = np.array(spec["start_joints_si"], dtype=float)
    T_start = chain.fk(home_q)
    p0 = T_start[:3, 3]
    up = np.array(spec["up_axis"], dtype=float)
    up = up / np.linalg.norm(up)
    mm = lambda v: np.array(v, dtype=float) / 1000.0           # noqa: E731
    scene = Scene(workspace_center=p0.copy(),
                  surface_point=p0 - up * spec["pad_below_start_mm"] / 1000.0,
                  surface_normal=up,
                  near_points=tuple(p0 + mm(n["offset_mm"]) for n in spec.get("near_objects", [])))
    site = Site(name=spec["name"], arm=spec["arm"], generation=spec["generation"], chain=chain, home_q=home_q,
                scene=scene, profile=_profile(spec["safety_profile"]),
                jaw_angles={k: math.radians(v) for k, v in spec["jaw_deg"].items()},
                source={"site_file": spec.get("_file", ""), **chain.source})
    for name, ps in spec["named_poses"].items():
        if "joints_si" in ps:
            q = np.array(ps["joints_si"], dtype=float)
            T, ok = chain.fk(q), True
        else:
            T = T_start.copy()
            T[:3, 3] = p0 + mm(ps["offset_mm"])
            rot = ps.get("rotate_deg")
            if rot:
                T[:3, :3] = from_rotvec(np.radians(np.array(rot, dtype=float))) @ T_start[:3, :3]
            q, ok = chain.ik(T, home_q)
        site.poses[name] = NamedPose(name=name, description=ps.get("description", ""), T=T, q=q, reachable=ok)
    return site


def load_site(name: str = "site_a", dvrk_dir: Path | None = None) -> Site:
    s = get_settings()
    path = s.sites_dir / f"{name}.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["_file"] = path.name
    base = dvrk_dir or s.dvrk_dir
    arm_file, tool_file = base / spec["kinematic_file"], base / spec["instrument_file"]
    for f in (arm_file, tool_file):
        if not f.exists():
            raise FileNotFoundError(f"{f} is missing. Run: python scripts/fetch_dvrk_config.py")
    return build_site(spec, load_chain(arm_file, tool_file))
