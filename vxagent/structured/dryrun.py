"""Dry application: what an accepted patch would do to the poses the site has named.

The patch is applied to the kinematic twin's inputs, never to the robot: the site is built again
from the patched copy, every named pose is recomputed, and the differences are reported in
millimetres and degrees for the person who has to approve. This shows the geometric effect on
named poses. It says nothing about contact.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ..robot.kinematics import load_chain, pose_distance
from ..robot.motion import check_goal
from ..robot.site import Site, build_site
from .rules import InstrumentSet


def _compare(before: Site, after: Site, Bb: np.ndarray | None = None, Ba: np.ndarray | None = None) -> dict[str, Any]:
    rows, newly_rejected, newly_accepted = [], [], []
    for name in sorted(set(before.poses) | set(after.poses)):
        pb, pa = before.poses.get(name), after.poses.get(name)
        if pb is None or pa is None:
            rows.append({"pose": name, "note": "added" if pb is None else "removed"})
            continue
        Tb = pb.T if Bb is None else Bb @ pb.T
        Ta = pa.T if Ba is None else Ba @ pa.T
        d_m, d_rad = pose_distance(Tb, Ta)
        ok_b = not check_goal(before.profile, before.scene, before.chain, pb.T, before.home_q)
        ok_a = not check_goal(after.profile, after.scene, after.chain, pa.T, after.home_q)
        rules_a = [v.rule for v in check_goal(after.profile, after.scene, after.chain, pa.T, after.home_q)]
        rows.append({"pose": name, "moved_mm": round(d_m * 1000, 3), "turned_deg": round(math.degrees(d_rad), 3),
                     "height_over_pad_mm_before": round(before.scene.height(pb.T[:3, 3]) * 1000, 2),
                     "height_over_pad_mm_after": round(after.scene.height(pa.T[:3, 3]) * 1000, 2),
                     "accepted_before": ok_b, "accepted_after": ok_a, "rules_failing_after": rules_a})
        if ok_b and not ok_a:
            newly_rejected.append(name)
        if ok_a and not ok_b:
            newly_accepted.append(name)
    moved = [r for r in rows if r.get("moved_mm", 0) > 1e-6 or r.get("turned_deg", 0) > 1e-6]
    lifted = [abs(r["height_over_pad_mm_after"] - r["height_over_pad_mm_before"]) for r in rows if "moved_mm" in r]
    return {"poses": rows, "poses_that_move": [r["pose"] for r in moved],
            "largest_change_of_height_over_pad_mm": round(max(lifted, default=0.0), 2),
            "largest_move_mm": max((r["moved_mm"] for r in moved), default=0.0),
            "largest_turn_deg": max((r["turned_deg"] for r in moved), default=0.0),
            "newly_rejected_by_the_safety_filter": newly_rejected,
            "newly_accepted_by_the_safety_filter": newly_accepted}


def dry_apply(file_type: str, before: Any, after: Any, site_spec: dict[str, Any], dvrk_dir: Path | str,
              changes: list[dict[str, Any]]) -> dict[str, Any]:
    base = Path(dvrk_dir)
    arm_file, tool_file = base / site_spec["kinematic_file"], base / site_spec["instrument_file"]
    chain = load_chain(arm_file, tool_file)

    if file_type == "vx-site":
        out = _compare(build_site(dict(before), chain), build_site(dict(after), chain))
        out.update({"applied_to": "the site's kinematic twin", "poses_expressed_in": "the arm's base frame"})
        return out

    if file_type == "dvrk-psm" and any(c["path"] == "tool" for c in changes) and after.get("tool"):
        new_file = InstrumentSet(base / "share" / "tool").file_for(after["tool"])
        if new_file is None:
            return {"geometric_effect": "not computed: the instrument has no file"}
        chain_after = load_chain(arm_file, base / "share" / "tool" / new_file)
        out = _compare(build_site(dict(site_spec), chain), build_site(dict(site_spec), chain_after))
        out.update({"applied_to": "the site's kinematic twin with the new instrument file",
                    "instrument_before": tool_file.name, "instrument_after": new_file,
                    "poses_expressed_in": "the arm's base frame, at the site's start joint values"})
        return out

    if file_type == "dvrk-system":
        arm = site_spec["arm"]
        hit = [c for c in changes if c["path"].startswith(f"arms/{arm}/base_frame/transform")]
        if hit:
            def frame(doc: Any) -> np.ndarray:
                a = next(x for x in doc["arms"] if x.get("name") == arm)
                return np.array(a["base_frame"]["transform"], dtype=float)
            site = build_site(dict(site_spec), chain)
            out = _compare(site, site, frame(before), frame(after))
            ref = next(x for x in after["arms"] if x.get("name") == arm)["base_frame"].get("reference_frame")
            out.update({"applied_to": f"the base frame of {arm} in front of the site's kinematic twin",
                        "poses_expressed_in": f"the reference frame {ref}"})
            return out
        other = sorted({c["path"].split("/")[1] for c in changes if c["path"].startswith("arms/")})
        if other:
            return {"poses": [], "poses_that_move": [], "largest_move_mm": 0.0, "largest_turn_deg": 0.0,
                    "note": f"the change is to {', '.join(other)}; this site's arm is {arm}, so none of its named "
                            f"poses moves"}
    return {"poses": [], "poses_that_move": [], "largest_move_mm": 0.0, "largest_turn_deg": 0.0,
            "note": "this change has no geometric effect on the named poses"}
