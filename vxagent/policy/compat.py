"""Is this policy's contract compatible with this site? Decided in code.

The answer is one of: compatible, compatible within a stated sub-region, not compatible. A
start pose or target outside the range the policy was trained on is never "run and see": it
fails here, and the pre-flight of a trial fails with it.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..robot.motion import check_goal
from ..structured.units import MEASURE, convert
from .adapter import Adapter
from .contract import Contract

RANGE_TOLERANCE_M = 0.005          # how far outside the trained start-to-goal range a trial may be


def training_range(contract: Contract) -> tuple[np.ndarray, np.ndarray]:
    k = convert(1.0, contract.translation_unit, "m")
    return np.array(contract.start_to_goal_min) * k, np.array(contract.start_to_goal_max) * k


def in_training_range(contract: Contract, start_T: np.ndarray, goal_T: np.ndarray) -> tuple[bool, float]:
    """(inside, distance outside in metres) of the displacement start -> goal."""
    lo, hi = training_range(contract)
    d = goal_T[:3, 3] - start_T[:3, 3]
    out = np.maximum(d - hi, 0) + np.maximum(lo - d, 0)
    far = float(np.linalg.norm(out))
    return far <= RANGE_TOLERANCE_M, far


def check(contract: Contract, site: Any, start_pose: str, goal_pose: str, control_rate_hz: float,
          policy_start: Any = None, policy_goal: Any = None, seen_positions: Any = None) -> dict[str, Any]:
    """policy_start / policy_goal: a start and a goal the policy was trained on, in policy space.
    seen_positions: translations as the policy's data holds them, for the unit check."""
    items: list[dict[str, Any]] = []

    def item(name: str, ok: bool | None, detail: str, blocks: bool = True) -> None:
        items.append({"check": name, "ok": ok, "detail": detail, "blocks": blocks})

    # 1. every observation field can be produced at this site
    item("the arm's pose can be read at the site", True, "measured pose of the arm, from the controller")
    item("the goal pose can be produced at the site", None,
         "the goal is the needle's grasp pose; this site has it as a declared pose, not as a measurement", blocks=False)

    # 2. units
    known = MEASURE.get(contract.translation_unit) == "length"
    item("the translation unit is a length the site can convert", known, f"contract says {contract.translation_unit!r}")
    if seen_positions is not None and known:
        med = float(np.median(np.linalg.norm(np.asarray(seen_positions, dtype=float), axis=1))) * convert(1.0, contract.translation_unit, "m")
        plausible = 0.02 <= med <= 0.5                 # a PSM tool tip is centimetres to decimetres from its base
        item("the magnitude of the data agrees with the declared unit", plausible,
             f"median distance of the tool tip from the base would be {med * 1000:.0f} mm"
             + ("" if plausible else "; a dVRK PSM works between 20 and 500 mm: the unit is probably wrong"))

    # 3. rate
    if contract.control_rate_hz is None:
        item("the policy's control rate is stated", None, "the sources do not state it; the site runs at "
             f"{control_rate_hz:g} Hz and the trial steps the policy at that rate", blocks=False)
    else:
        item("the site can run at the policy's control rate", contract.control_rate_hz <= control_rate_hz + 1e-9,
             f"policy {contract.control_rate_hz:g} Hz, site {control_rate_hz:g} Hz")

    # 4. the largest step an action can ask for against the safety profile
    ad = Adapter(contract)
    a = max(abs(contract.action_low), abs(contract.action_high))
    per_axis = a * contract.step_translation * convert(1.0, contract.translation_unit, "m")
    worst_m, _ = ad.largest_step()
    rot = a * contract.step_rotation_rad
    p = site.profile
    item("one action on one axis stays inside the step limit near the needle", per_axis <= p.near_step_m + 1e-7
         and rot <= p.near_step_rad + 1e-7,
         f"{per_axis * 1000:.2f} mm and {math.degrees(rot):.2f} deg per action; limit {p.near_step_m * 1000:g} mm and "
         f"{math.degrees(p.near_step_rad):g} deg")
    item("an action at its bound on all three axes stays inside the step limit near the needle", worst_m <= p.near_step_m + 1e-12,
         f"{worst_m * 1000:.2f} mm against {p.near_step_m * 1000:g} mm: such an action would be rejected by the safety "
         f"filter, not clipped", blocks=False)

    # 5. coverage: does the site's own start -> goal lie in the trained range, and if not, where would it
    start_T, goal_T = site.poses[start_pose].T, site.poses[goal_pose].T
    inside, far = in_training_range(contract, start_T, goal_T)
    lo, hi = training_range(contract)
    item(f"the site's {start_pose} -> {goal_pose} lies in the trained start-to-goal range", inside,
         f"site displacement {np.round((goal_T[:3, 3] - start_T[:3, 3]) * 1000, 1).tolist()} mm; trained "
         f"{np.round(lo * 1000, 1).tolist()} to {np.round(hi * 1000, 1).tolist()} mm; {far * 1000:.1f} mm outside", blocks=False)
    region = None
    if policy_start is not None and policy_goal is not None:
        anchored = Adapter.anchored(contract, policy_goal, goal_T)
        s_T, g_T = anchored.pose_to_site(policy_start), anchored.pose_to_site(policy_goal)
        bad_s = check_goal(site.profile, site.scene, site.chain, s_T, site.home_q)
        bad_g = check_goal(site.profile, site.scene, site.chain, g_T, site.home_q)
        ok = not bad_s and not bad_g
        item("a trained start and goal, anchored on the site's grasp pose, are accepted by the safety filter", ok,
             "accepted" if ok else "start: " + (", ".join(v.rule for v in bad_s) or "ok") + "; goal: "
             + (", ".join(v.rule for v in bad_g) or "ok"))
        if ok:
            region = {"start_offset_from_site_start_mm": np.round((s_T[:3, 3] - start_T[:3, 3]) * 1000, 1).tolist(),
                      "goal": goal_pose, "tolerance_mm": RANGE_TOLERANCE_M * 1000}
    blocking = [i for i in items if i["ok"] is False and i["blocks"]]
    if blocking:
        verdict = "not compatible"
    elif inside:
        verdict = "compatible"
    elif region is not None:
        verdict = "compatible within a stated sub-region"
    else:
        verdict = "not compatible"
    return {"verdict": verdict, "sub_region": region if verdict.startswith("compatible within") else None, "items": items,
            "failing": [i["check"] for i in items if i["ok"] is False], "open": [i["check"] for i in items if i["ok"] is None]}
