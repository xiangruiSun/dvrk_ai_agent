"""A deployment run on the twin, with one fault put in on purpose.

This is the smallest pipeline in which each family of sim-to-real failure has somewhere to
live: the camera sees the needle, a calibration turns that into a goal for the arm, a policy
(here: the shape of a recorded approach, emitted as small steps) moves towards it, a mapping
turns the policy's actions into set points, the safety filter checks every one and the arm
follows. Each step is recorded, and the diagnostic tools read that record.

Every fault here is INJECTED. Each one is a clean single fault, which real failures are not.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ..pipelines.paths import fit_path
from ..robot.kinematics import Chain, from_rotvec, interpolate
from ..robot.safety import check_step, check_tracking
from ..robot.twin import KinematicTwin

CAUSES: dict[str, dict[str, str]] = {
    "calibration_offset": {"family": "robot_calibration",
                           "what": "the camera-to-robot transform has a constant translation error, so the arm "
                                   "goes to a point shifted from where the camera saw the needle"},
    "calibration_axis_flip": {"family": "robot_calibration",
                              "what": "one axis of the camera-to-robot transform is inverted, so positions "
                                      "seen by the camera are mirrored in the robot's frame"},
    "action_unit_scale": {"family": "policy_transfer",
                          "what": "the mapping from policy actions to robot set points uses the wrong unit, so "
                                  "every commanded step is far too large or far too small"},
    "start_outside_training_range": {"family": "policy_transfer",
                                     "what": "the run starts from a pose the policy never saw in training, so "
                                             "its observations are out of range from the first step"},
    "joint_limit_tightened": {"family": "control_and_safety",
                              "what": "a joint limit in the arm's configuration is tighter than the task needs, "
                                      "so the path runs into the margin kept to that limit"},
    "tracking_error": {"family": "control_and_safety",
                       "what": "the arm does not follow its set point closely enough: the measured pose "
                               "differs from the set point by more than the stop threshold"},
    "perception_dropout": {"family": "visual_perception",
                           "what": "the needle detection drops out for long stretches, so the pipeline has no "
                                   "target and holds still"},
}
PATIENCE = 5                 # steps without a valid detection before the pipeline holds still
STEP_M = 0.001               # spacing of the policy's steps along the recorded shape
T_BC = np.eye(4)             # robot base from camera: the truth of this scenario
T_BC[:3, :3] = from_rotvec([0.3, -0.2, 0.5])
T_BC[:3, 3] = [0.05, -0.02, 0.10]


def _inv(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def _apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return T[:3, :3] @ p + T[:3, 3]


def _by_cause(fault: Any) -> dict[str, dict[str, Any]]:
    """One fault, several, or none, as a table by cause."""
    if fault is None:
        return {}
    return {f["cause"]: f for f in (fault if isinstance(fault, list) else [fault])}


def calibration(fault: Any, centre: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    """The calibration record the run uses, with held-out point pairs: where the camera saw a
    point and where the arm touched it."""
    T_est = T_BC.copy()
    fs = _by_cause(fault)
    if "calibration_offset" in fs:
        d = np.array(fs["calibration_offset"]["direction"], dtype=float)
        T_est[:3, 3] += d / np.linalg.norm(d) * fs["calibration_offset"]["mm"] / 1000.0
    if "calibration_axis_flip" in fs:
        flip = np.eye(3)
        flip[fs["calibration_axis_flip"]["axis"], fs["calibration_axis_flip"]["axis"]] = -1.0
        T_est[:3, :3] = T_BC[:3, :3] @ flip
        T_est[:3, 3] = centre - T_est[:3, :3] @ _apply(_inv(T_BC), centre)     # still right at the centre
    touched = centre + rng.uniform(-0.03, 0.03, size=(12, 3))
    seen = np.array([_apply(_inv(T_BC), p) for p in touched]) + rng.normal(0, 0.0002, size=(12, 3))
    return {"id": "cal-" + ("%04x" % int(rng.integers(0, 65535))), "T_est": T_est, "touched_m": touched, "seen_m": seen}


def policy_path(points: np.ndarray, T_from: np.ndarray, T_to: np.ndarray) -> list[np.ndarray]:
    """The recorded shape fitted between two poses and walked in equal steps of STEP_M: what the
    policy emits, one target per control tick."""
    way = [T_from] + fit_path(points, T_from, T_to, STEP_M / 2)[0]
    ps = np.array([w[:3, 3] for w in way])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ps, axis=0), axis=1))])
    n = max(1, int(np.ceil(arc[-1] / STEP_M)))
    out = []
    for i in range(1, n + 1):
        a = min(arc[-1], i * STEP_M)
        T = interpolate(T_from, T_to, a / arc[-1] if arc[-1] > 0 else 1.0)
        T[:3, 3] = [np.interp(a, arc, ps[:, k]) for k in range(3)]
        out.append(T)
    out[-1] = T_to.copy()
    return out


def nominal_waypoints(site: Any, points: np.ndarray) -> list[np.ndarray]:
    return policy_path(points, site.poses["start"].T, site.poses["pregrasp"].T)


def make_contract(site: Any, stretches: list[np.ndarray]) -> dict[str, Any]:
    """What the policy was trained on, read from fault-free replays of the stretches: the range
    of its observation (vector from tip to goal) and the size of its steps."""
    goal = site.poses["pregrasp"].T[:3, 3]
    obs, steps = [], []
    for pts in stretches:
        way = [site.poses["start"].T] + nominal_waypoints(site, pts)
        ps = np.array([w[:3, 3] for w in way])
        obs.append((goal - ps) * 1000.0)
        steps.append(np.linalg.norm(np.diff(ps, axis=0), axis=1) * 1000.0)
    o, s = np.concatenate(obs), np.concatenate(steps)
    return {"action_unit": "m", "observation": "vector from tip to goal, mm",
            "observation_min_mm": [round(float(x) - 2.0, 2) for x in o.min(axis=0)],
            "observation_max_mm": [round(float(x) + 2.0, 2) for x in o.max(axis=0)],
            "action_step_max_mm": round(float(s.max()) * 1.1, 3), "episodes": len(stretches)}


def run_episode(site: Any, points: np.ndarray, contract: dict[str, Any], fault: Any, seed: int = 0) -> dict[str, Any]:
    """fault: one fault, a list of faults that are all present at once, or None."""
    rng = np.random.default_rng(seed)
    fs = _by_cause(fault)
    cause = "+".join(fs) or None
    chain: Chain = site.chain
    start_T = site.poses["start"].T.copy()
    needle_B = site.scene.near_points[0]
    approach = site.poses["pregrasp"].T[:3, 3] - needle_B
    cal = calibration(fault, site.scene.workspace_center, rng)
    nominal = nominal_waypoints(site, points)
    budget = len(nominal) + 4

    if "joint_limit_tightened" in fs:
        q0, q1 = site.home_q, site.poses["pregrasp"].q
        j = int(fs["joint_limit_tightened"]["joint"])
        tight = q0[j] + fs["joint_limit_tightened"]["fraction"] * (q1[j] - q0[j])
        joints = list(chain.joints)
        joints[j] = dataclasses.replace(joints[j], **({"qmax": float(tight)} if q1[j] > q0[j] else {"qmin": float(tight)}))
        chain = Chain(joints, chain.tooltip, chain.jaw_limits, chain.source)
    twin = KinematicTwin(chain, site.home_q, jaw=site.jaw_angles.get("open", 0.0))
    twin.enable()
    twin.home()
    if "start_outside_training_range" in fs:
        T0 = start_T.copy()
        T0[:3, 3] += np.array(fs["start_outside_training_range"]["offset_mm"], dtype=float) / 1000.0
        q, _ = chain.ik(T0, site.home_q)
        twin.set_joints(q)
        start_T = twin.state().setpoint_cp.copy()

    scale = fs["action_unit_scale"]["scale"] if "action_unit_scale" in fs else 1.0
    drop = fs.get("perception_dropout")
    dropout_from = drop.get("from_step", 2) if drop else None
    track = fs.get("tracking_error")
    needle_C = _apply(_inv(T_BC), needle_B)

    steps: list[dict[str, Any]] = []
    stops: list[dict[str, Any]] = []
    way: list[np.ndarray] | None = None
    goal_est: np.ndarray | None = None
    since_valid, k, command_id = 0, 0, f"episode-{seed}"
    twin.begin_motion(command_id)
    outcome = "completed"
    for t in range(budget):
        valid = True
        conf = float(np.clip(rng.normal(0.9, 0.03), 0, 1))
        if dropout_from is not None and t >= dropout_from and rng.random() < drop["rate"]:
            valid, conf = False, float(np.clip(rng.normal(0.12, 0.05), 0, 1))
        if valid:
            seen = needle_C + rng.normal(0, 0.0002, 3)
            goal_est = _apply(cal["T_est"], seen) + approach
            since_valid = 0
        else:
            since_valid += 1
        s = twin.state()
        row: dict[str, Any] = {"step": t, "detection_valid": valid, "detection_confidence": round(conf, 3)}
        if goal_est is not None:
            row["observation_mm"] = [round(float(x) * 1000, 2) for x in (goal_est - s.measured_cp[:3, 3])]
        if goal_est is None or since_valid > PATIENCE:
            row["held"] = "no target"
            steps.append(row)
            continue
        if way is None:
            goal_T = site.poses["pregrasp"].T.copy()
            goal_T[:3, 3] = goal_est
            way = policy_path(points, start_T, goal_T)
        if k >= len(way):
            steps.append(row)
            break
        prev = start_T[:3, 3] if k == 0 else way[k - 1][:3, 3]
        action = way[k][:3, 3] - prev                              # the policy's action, in its own unit (m)
        target = way[k].copy()
        target[:3, 3] = s.setpoint_cp[:3, 3] + scale * action       # the mapping to a set point
        row.update({"action_norm_policy_unit": round(float(np.linalg.norm(action)), 6),
                    "commanded_step_mm": round(float(np.linalg.norm(scale * action)) * 1000, 4),
                    "commanded_mm": [round(float(x) * 1000, 3) for x in target[:3, 3]]})
        q_next, _ = chain.ik(target, s.q)
        bad = check_step(site.profile, site.scene, chain, s.setpoint_cp, target, q_next)
        if bad:
            stops.append({"step": t, "kind": "rejected_by_safety_filter", "rule": bad[0].rule, **bad[0].as_dict()})
            steps.append(row)
            outcome = "stopped"
            break
        if track is not None and k == track["from_step"]:
            twin.inject_tracking_offset(np.array(track["offset_mm"], dtype=float) / 1000.0)
        after = twin.servo_cp(target, command_id)
        row.update({"setpoint_mm": [round(float(x) * 1000, 3) for x in after.setpoint_cp[:3, 3]],
                    "measured_mm": [round(float(x) * 1000, 3) for x in after.measured_cp[:3, 3]]})
        steps.append(row)
        tracked = check_tracking(site.profile, target, after.setpoint_cp, after.measured_cp)
        if tracked:
            stops.append({"step": t, "kind": "tracking_stop", "rule": tracked[0].rule, **tracked[0].as_dict()})
            outcome = "stopped"
            break
        k += 1
    else:
        outcome = "completed" if way is not None and k >= len(way) else "out_of_steps"
    if outcome == "completed" and (way is None or k < len(way)):
        outcome = "out_of_steps"
    end = twin.end_motion(command_id)
    d_mm = float(np.linalg.norm(end.measured_cp[:3, 3] - (needle_B + approach))) * 1000.0
    last = steps[-1]["step"] + 1 if steps else 0
    if outcome == "stopped":
        symptom = (f"The run stopped at step {last} of {budget}. The console says the motion was stopped by the safety "
                   f"system. The tip was {d_mm:.0f} mm from the grasp point.")
    elif outcome == "out_of_steps":
        symptom = (f"The run used all {budget} steps it is allowed and never reached the needle. The tip ended "
                   f"{d_mm:.0f} mm from the grasp point.")
    elif d_mm > 2.0:
        symptom = (f"The run finished its {last} steps and closed the jaw, but the tip was {d_mm:.0f} mm from the grasp "
                   f"point and the grasp missed.")
    else:
        symptom = f"The run finished its {last} steps with the tip {d_mm:.1f} mm from the grasp point."
    return {"injected_cause": cause, "fault": fault, "outcome": outcome, "symptom": symptom, "budget_steps": budget,
            "steps_recorded": len(steps), "tip_to_grasp_point_mm": round(d_mm, 2), "steps": steps, "stops": stops,
            "calibration": {"id": cal["id"], "T_est": cal["T_est"].tolist(), "touched_m": cal["touched_m"].tolist(),
                            "seen_m": cal["seen_m"].tolist()},
            "contract": contract, "tracking_stop_mm": round(site.profile.track_pos_m * 1000, 3)}


def make_fault(cause: str, rng: np.random.Generator) -> dict[str, Any]:
    """A fault of this kind with a magnitude drawn at random inside a plausible band."""
    if cause == "calibration_offset":
        return {"cause": cause, "mm": round(float(rng.uniform(5, 10)), 1), "direction": [1.0, float(rng.uniform(-1, 1)), 0.0]}
    if cause == "calibration_axis_flip":
        return {"cause": cause, "axis": int(rng.integers(0, 2))}
    if cause == "action_unit_scale":
        return {"cause": cause, "scale": 1000.0 if rng.random() < 0.5 else 0.001}
    if cause == "start_outside_training_range":
        return {"cause": cause, "offset_mm": [-float(rng.uniform(18, 26)), -float(rng.uniform(4, 10)), 0.0]}
    if cause == "joint_limit_tightened":
        return {"cause": cause, "joint": 2, "fraction": round(float(rng.uniform(0.45, 0.75)), 2)}
    if cause == "tracking_error":
        return {"cause": cause, "from_step": int(rng.integers(3, 8)), "offset_mm": [round(float(rng.uniform(2.6, 4.0)), 1), 0.0, 0.0]}
    if cause == "perception_dropout":
        return {"cause": cause, "from_step": int(rng.integers(1, 4)), "rate": round(float(rng.uniform(0.85, 0.95)), 2)}
    raise ValueError(cause)

