"""Kinematics, the twin and the safety filter. Every limit is tested on both sides."""
from __future__ import annotations

import math

import numpy as np
import pytest

from vxagent.robot import jsonc
from vxagent.robot.backend import BackendError
from vxagent.robot.kinematics import from_rotvec, interpolate, pose_distance, rotvec
from vxagent.robot.motion import MotionStopped, check_goal, execute_path
from vxagent.robot.safety import check_pose, check_step, check_tracking, plan_steps


def shifted(T, dx=0.0, dy=0.0, dz=0.0, rot_deg=0.0):
    out = T.copy()
    out[:3, 3] += np.array([dx, dy, dz])
    if rot_deg:
        out[:3, :3] = from_rotvec([0.0, 0.0, math.radians(rot_deg)]) @ out[:3, :3]
    return out


def rules(violations):
    return {v.rule for v in violations}


# ------------------------------------------------------------------------------ files and kinematics

def test_jsonc_strips_comments_but_not_inside_strings():
    text = '{ /* block */ "a": 1, // line\n "b": "http://x/*y*/" }'
    assert jsonc.loads(text) == {"a": 1, "b": "http://x/*y*/"}


def test_forward_kinematics_of_the_start_pose(chain):
    T = chain.fk([0, 0, 0.12, 0, 0, 0])
    # insertion 0.12 - 0.40 + tool 0.38 = 0.10 along the shaft, plus the 0.01 wrist link
    assert np.allclose(T[:3, 3], [0.0, 0.0, -0.11], atol=1e-5)
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)


def test_jacobian_matches_finite_differences(chain):
    q = np.array([0.2, -0.1, 0.15, 0.3, 0.2, -0.4])
    J, eps = chain.jacobian(q), 1e-7
    for i in range(chain.n):
        dq = np.zeros(chain.n)
        dq[i] = eps
        num = (chain.fk(q + dq)[:3, 3] - chain.fk(q - dq)[:3, 3]) / (2 * eps)
        assert np.allclose(J[:3, i], num, atol=1e-6)


def test_inverse_kinematics_round_trip(chain):
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = chain.qmin + (chain.qmax - chain.qmin) * rng.uniform(0.2, 0.8, chain.n)
        found, ok = chain.ik(chain.fk(q), q + rng.normal(0, 0.03, chain.n) * [1, 1, 0.1, 1, 1, 1])
        assert ok
        d_m, d_rad = pose_distance(chain.fk(q), chain.fk(found))
        assert d_m < 1e-6 and d_rad < 1e-5


def test_inverse_kinematics_reports_a_target_beyond_the_joint_limits(chain):
    T = shifted(chain.fk([0, 0, 0.24, 0, 0, 0]), dz=-0.05)       # needs insertion 0.29, limit 0.25
    q, ok = chain.ik(T, [0, 0, 0.24, 0, 0, 0])
    assert not ok and chain.within_limits(q)


def test_rotation_vector_round_trip():
    for v in ([0.3, -0.2, 0.5], [0.0, 0.0, 3.1], [1e-4, 0, 0]):
        assert np.allclose(rotvec(from_rotvec(v)), v, atol=1e-6)


# ------------------------------------------------------------------------------ safety filter

@pytest.mark.parametrize("mm,ok", [(5.0, True), (5.1, False)])
def test_free_space_position_step(site, mm, ok):
    T0 = site.poses["start"].T
    T1 = shifted(T0, dx=mm / 1000)
    q1, _ = site.chain.ik(T1, site.home_q)
    assert ("step_position" not in rules(check_step(site.profile, site.scene, site.chain, T0, T1, q1))) is ok


@pytest.mark.parametrize("deg,ok", [(5.0, True), (5.2, False)])
def test_free_space_rotation_step(site, deg, ok):
    T0 = site.poses["start"].T
    T1 = shifted(T0, rot_deg=deg)
    q1, _ = site.chain.ik(T1, site.home_q)
    assert ("step_rotation" not in rules(check_step(site.profile, site.scene, site.chain, T0, T1, q1))) is ok


@pytest.mark.parametrize("mm,ok", [(2.0, True), (2.1, False)])
def test_near_the_needle_the_step_limit_is_two_millimetres(site, mm, ok):
    T0 = site.poses["pregrasp"].T                     # 8 mm over the pad: inside the 20 mm zone
    T1 = shifted(T0, dx=mm / 1000)
    q1, _ = site.chain.ik(T1, site.poses["pregrasp"].q)
    assert ("step_position" not in rules(check_step(site.profile, site.scene, site.chain, T0, T1, q1))) is ok


@pytest.mark.parametrize("height_mm,near", [(20.5, False), (19.5, True)])
def test_the_near_zone_starts_twenty_millimetres_from_the_pad(site, height_mm, near):
    p = site.scene.surface_point + site.scene.surface_normal * height_mm / 1000 + np.array([-0.03, -0.03, 0])
    assert site.scene.is_near(p, site.profile.near_distance_m) is near


@pytest.mark.parametrize("mm,ok", [(40.0, True), (40.5, False)])
def test_local_workspace(site, mm, ok):
    T = shifted(site.poses["start"].T, dy=mm / 1000)
    q, _ = site.chain.ik(T, site.home_q)
    assert ("workspace" not in rules(check_pose(site.profile, site.scene, site.chain, T, q))) is ok


@pytest.mark.parametrize("height_mm,ok", [(3.0, True), (2.5, False)])
def test_clearance_over_the_pad(site, height_mm, ok):
    T = shifted(site.poses["start"].T, dz=-(30.0 - height_mm) / 1000)
    q, _ = site.chain.ik(T, site.home_q)
    assert ("clearance" not in rules(check_pose(site.profile, site.scene, site.chain, T, q))) is ok


def test_joint_limit_margin(site):
    lim = site.chain.joints[4].qmax
    inside = np.array([0, 0, 0.12, 0, lim - math.radians(5.5), 0])
    outside = np.array([0, 0, 0.12, 0, lim - math.radians(4.5), 0])
    assert "joint_margin" not in rules(check_pose(site.profile, site.scene, site.chain, site.chain.fk(inside), inside))
    assert "joint_margin" in rules(check_pose(site.profile, site.scene, site.chain, site.chain.fk(outside), outside))


@pytest.mark.parametrize("mm,ok", [(1.9, True), (2.1, False)])
def test_tracking_stop(site, mm, ok):
    T = site.poses["start"].T
    assert (not check_tracking(site.profile, T, T, shifted(T, dx=mm / 1000))) is ok


def test_tracking_detects_a_clipped_set_point(site):
    T = site.poses["start"].T
    assert "clipped" in rules(check_tracking(site.profile, T, shifted(T, dz=0.001), shifted(T, dz=0.001)))


def test_planned_path_respects_the_limit_that_applies_at_every_step(site):
    T0, T1 = site.poses["start"].T, site.poses["pregrasp"].T      # free space into the near zone
    steps = plan_steps(site.profile, site.scene, T0, T1)
    current, q = T0, site.home_q
    for T in steps:
        q, ok = site.chain.ik(T, q)
        assert ok and not check_step(site.profile, site.scene, site.chain, current, T, q)
        current = T
    assert pose_distance(current, T1)[0] < 1e-6
    assert len(steps) > len(plan_steps(site.profile, site.scene, T0, site.poses["hover_needle"].T))


def test_declared_poses_are_accepted_or_rejected_by_the_expected_rule(site):
    expect = {"start": set(), "hover_needle": set(), "pregrasp": set(), "left_edge": set(), "right_edge": set(),
              "retract": {"workspace"}, "pad_touch": {"clearance"}, "wrist_limit": {"joint_margin"}}
    for name, want in expect.items():
        got = rules(check_goal(site.profile, site.scene, site.chain, site.poses[name].T, site.home_q))
        assert got == want, name


# ------------------------------------------------------------------------------ twin

def test_twin_refuses_motion_until_enabled_and_homed(site):
    twin = site.twin()
    with pytest.raises(BackendError) as e:
        twin.begin_motion("c1")
    assert e.value.kind == "robot_not_ready"
    twin.enable()
    with pytest.raises(BackendError):
        twin.begin_motion("c1")
    twin.home()
    twin.begin_motion("c1")


def test_a_command_id_is_accepted_once(site):
    twin = site.twin()
    twin.enable(), twin.home()
    twin.begin_motion("c1")
    twin.end_motion("c1")
    with pytest.raises(BackendError) as e:
        twin.begin_motion("c1")
    assert e.value.kind == "duplicate_command"


def test_every_change_of_state_advances_the_sequence_number(site):
    twin = site.twin()
    seqs = [twin.state().seq]
    twin.enable(); seqs.append(twin.state().seq)
    twin.home(); seqs.append(twin.state().seq)
    twin.perturb([0.01, 0, 0, 0, 0, 0]); seqs.append(twin.state().seq)
    assert seqs == sorted(set(seqs))


def run_path(site, twin, name, cid="c1"):
    twin.begin_motion(cid)
    try:
        return execute_path(twin, site.chain, site.profile, site.scene, site.poses[name].T, cid, tick=lambda dt: None)
    finally:
        twin.end_motion(cid)


def test_path_execution_reaches_the_named_pose(site):
    twin = site.twin()
    twin.enable(), twin.home()
    info = run_path(site, twin, "hover_needle")
    assert info["steps"] == info["planned_steps"] > 0
    assert pose_distance(twin.state().measured_cp, site.poses["hover_needle"].T)[0] < 1e-6


def test_path_stops_when_the_arm_does_not_follow(site):
    twin = site.twin()
    twin.enable(), twin.home()
    twin.inject_dropped_commands()                    # INJECTED fault
    with pytest.raises(MotionStopped) as e:
        run_path(site, twin, "left_edge")
    assert e.value.kind == "tracking_error" and e.value.steps_done == 1


def test_path_stops_on_an_injected_tracking_offset(site):
    twin = site.twin()
    twin.enable(), twin.home()
    twin.inject_tracking_offset([0.003, 0, 0])        # INJECTED: 3 mm, over the 2 mm stop
    with pytest.raises(MotionStopped) as e:
        run_path(site, twin, "left_edge")
    assert e.value.kind == "tracking_error"


def test_rehearsal_copy_does_not_move_the_original(site):
    twin = site.twin()
    twin.enable(), twin.home()
    before = twin.state()
    copy = twin.clone()
    run_path(site, copy, "left_edge")
    assert twin.state().seq == before.seq
    assert np.allclose(twin.state().measured_cp, before.measured_cp)


def test_interpolate_endpoints(site):
    a, b = site.poses["start"].T, site.poses["pregrasp"].T
    assert np.allclose(interpolate(a, b, 0.0), a) and np.allclose(interpolate(a, b, 1.0), b, atol=1e-9)
