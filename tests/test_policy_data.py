"""Policy contracts and episode data. What must not regress: a demonstration file is never
unpickled with code execution, a contract field without a quotation that is really in the
source stays empty, a start outside the trained range is not compatible, the adapter returns
the pose it was given, and the episode checks catch what they are named for.
"""
from __future__ import annotations

import io
import json
import os
import pickle

import numpy as np
import pytest

from vxagent.data import quality
from vxagent.data.surgicai import UnsafePickle, safe_load
from vxagent.llm.scripted import ScriptedLLM, say
from vxagent.policy import compat
from vxagent.policy.adapter import Adapter, R_to_rpy, angle_between_rpy, rpy_to_R
from vxagent.policy.contract import FIELDS, Contract, draft_contract
from vxagent.robot.site import build_site
from vxagent.structured.units import UnitError, convert


def contract(**kw) -> Contract:
    base = dict(name="p", version="1", source_simulator="sim", observation_size=21, translation_unit="cm", rotation_unit="rad",
                rotation_convention="RPY", action_size=7, action_low=-1.0, action_high=1.0, step_translation=0.15,
                step_rotation_rad=float(np.radians(3.0)), step_jaw=0.05, max_episode_steps=200, success_translation=0.5,
                success_rotation_deg=30.0, control_rate_hz=None, start_to_goal_min=[-2.3, -0.3, -2.1],
                start_to_goal_max=[-2.1, -0.2, -1.9], provenance={})
    return Contract(**{**base, **kw})


# ------------------------------------------------------------------------------ the restricted loader

class Evil:
    def __reduce__(self):
        return (os.system, ("echo should-never-run",))


def test_arrays_and_plain_containers_load(tmp_path):
    p = tmp_path / "ok.pkl"
    p.write_bytes(pickle.dumps([{"obs": {"achieved_goal": np.arange(7.0)}, "action": np.zeros(7), "done": np.float32(1.0)}]))
    back = safe_load(p)
    assert np.allclose(back[0]["obs"]["achieved_goal"], np.arange(7.0)) and float(back[0]["done"]) == 1.0


def test_a_pickle_that_asks_for_anything_else_is_not_loaded(tmp_path):
    p = tmp_path / "evil.pkl"
    p.write_bytes(pickle.dumps([Evil()]))
    with pytest.raises(UnsafePickle, match="not data"):
        safe_load(p)
    q = tmp_path / "evil2.pkl"
    q.write_bytes(pickle.dumps({"f": io.BytesIO}))
    with pytest.raises(UnsafePickle):
        safe_load(q)


# ------------------------------------------------------------------------------ contract drafting

SOURCE = {"env.py": "class Env:\n    def __init__(self, max_episode_step=200, step_size=None):\n        self.shape = (21,)\n"
                    "        # limits = [-0.02, 0.02]\n"}


def reply(**fields) -> str:
    body = {k: {"value": None, "file": None, "quote": None} for k in FIELDS}
    body.update(fields)
    return json.dumps(body)


def test_a_field_is_kept_only_with_a_quotation_that_is_in_the_file():
    llm = ScriptedLLM([say(reply(
        max_episode_steps={"value": "200", "file": "env.py", "quote": "max_episode_step=200"},
        observation_size={"value": "21", "file": "env.py", "quote": "self.shape   =  (21,)"},          # spacing differs: fine
        control_rate_hz={"value": "50", "file": "env.py", "quote": "the policy runs at 50 Hz"},        # not in the file
        action_size={"value": "7", "file": "other.py", "quote": "shape = (7,)"}))])                    # not a file it was given
    d = draft_contract(llm, SOURCE)
    assert d.fields["max_episode_steps"]["status"] == "sourced" and d.fields["observation_size"]["value"] == "21"
    assert d.fields["control_rate_hz"]["value"] is None and d.fields["control_rate_hz"]["status"].startswith("quotation not found")
    assert d.fields["control_rate_hz"]["offered"] == "50"               # what the model offered is kept for the record
    assert d.fields["action_size"]["value"] is None
    assert d.fields["action_step_size"]["status"] == "not stated in the sources"
    assert sorted(d.filled()) == ["max_episode_steps", "observation_size"] and len(d.for_a_person()) == len(FIELDS) - 2


def test_a_model_that_cannot_be_read_leaves_every_field_for_a_person():
    d = draft_contract(ScriptedLLM([say("not json")]), SOURCE)
    assert d.error and d.filled() == [] and len(d.for_a_person()) == len(FIELDS)


# ------------------------------------------------------------------------------ adapter

def test_a_pose_taken_to_the_site_and_back_is_the_same_pose():
    ad = Adapter(contract(), anchor_m=(0.01, -0.02, 0.03))
    rng = np.random.default_rng(1)
    for _ in range(200):
        pose = np.concatenate([rng.uniform(-15, 15, 3), rng.uniform(-3.1, 3.1, 1), rng.uniform(-1.5, 1.5, 1), rng.uniform(-3.1, 3.1, 1), [0.4]])
        back = ad.pose_to_policy(ad.pose_to_site(pose), jaw=0.4)
        assert np.allclose(back[:3], pose[:3], atol=1e-9) and angle_between_rpy(back, pose) < 1e-6 and back[6] == 0.4
    R = rpy_to_R(0.3, -0.2, 1.1)
    assert np.allclose(rpy_to_R(*R_to_rpy(R)), R) and np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)


def test_an_action_moves_the_pose_by_the_step_size_and_is_clipped_to_its_bounds():
    ad = Adapter(contract())
    out = ad.step(np.zeros(7), [1.0, -0.5, 0.0, 2.0, 0.0, 0.0, -1.0])
    assert np.allclose(out, [0.15, -0.075, 0.0, np.radians(3.0), 0.0, 0.0, -0.05])
    assert ad.pose_to_site([10.0, 0, 0, 0, 0, 0, 0])[0, 3] == pytest.approx(0.10)           # centimetres to metres
    worst_m, worst_rad = ad.largest_step()
    assert worst_m == pytest.approx(np.sqrt(3) * 0.0015)


def test_centimetres_can_be_read_but_not_written_into_a_patch():
    from vxagent.structured.schemas import Operation
    assert convert(2.5, "cm", "mm") == pytest.approx(25.0)
    with pytest.raises(UnitError):
        convert(1.0, "inch", "m")
    with pytest.raises(Exception):
        Operation(op="replace", path="a/b", value=1.0, unit="cm")


# ------------------------------------------------------------------------------ compatibility

def test_a_start_outside_the_trained_range_is_only_compatible_where_the_policy_was_trained(site):
    c = contract(start_to_goal_min=[-2.3, -0.3, -2.1], start_to_goal_max=[-2.1, -0.2, -1.9])
    goal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    start = np.array([2.2, 0.25, 2.0, 0.0, 0.0, 0.0, 0.8])
    ad = Adapter(c)
    T = ad.pose_to_policy(site.poses["pregrasp"].T)                       # keep the site's own orientation
    goal[3:6], start[3:6] = T[3:6], T[3:6]
    res = compat.check(c, site, "start", "pregrasp", 10.0, policy_start=start, policy_goal=goal)
    assert res["verdict"] == "compatible within a stated sub-region"
    assert res["sub_region"]["start_offset_from_site_start_mm"] == [37.0, 12.5, -2.0]
    inside = contract(start_to_goal_min=[1.4, 0.9, -2.3], start_to_goal_max=[1.6, 1.1, -2.1])     # the site's own displacement
    assert compat.check(inside, site, "start", "pregrasp", 10.0)["verdict"] == "compatible"
    assert compat.check(c, site, "start", "pregrasp", 10.0)["verdict"] == "not compatible"        # no trained start offered
    ok, far = compat.in_training_range(c, site.poses["start"].T, site.poses["pregrasp"].T)
    assert not ok and far > 0.03


def test_a_step_larger_than_the_sites_limit_or_a_wrong_unit_is_not_compatible(site, chain):
    spec = json.loads((__import__("pathlib").Path(__file__).resolve().parents[1] / "sites" / "site_a.json").read_text(encoding="utf-8"))
    spec["named_poses"]["wrist_limit"]["joints_si"] = [0.0, 0.0, 0.12, 0.0, 1.36, 0.0]
    spec["safety_profile"]["near_step_mm"] = 1.0
    tight = build_site(spec, chain)
    inside = contract(start_to_goal_min=[1.4, 0.9, -2.3], start_to_goal_max=[1.6, 1.1, -2.1])
    res = compat.check(inside, tight, "start", "pregrasp", 10.0)
    assert res["verdict"] == "not compatible" and res["failing"][0].startswith("one action on one axis")
    seen = np.array([[-5.0, 1.0, -11.0]] * 10)                            # centimetres
    wrong = compat.check(inside.model_copy(update={"translation_unit": "m"}), site, "start", "pregrasp", 10.0, seen_positions=seen)
    assert wrong["verdict"] == "not compatible" and "the unit is probably wrong" in next(
        i["detail"] for i in wrong["items"] if i["check"].startswith("the magnitude"))
    slow = compat.check(inside.model_copy(update={"control_rate_hz": 50.0}), site, "start", "pregrasp", 10.0)
    assert "the site can run at the policy's control rate" in slow["failing"]


# ------------------------------------------------------------------------------ episode checks

def episode(n=600, **kw) -> quality.Episode:
    t = np.arange(n) * 0.02
    pos = np.stack([0.05 + 0.02 * np.sin(t), 0.01 * np.cos(t), -0.12 + 0.01 * np.sin(2 * t)], axis=1)
    base = dict(source="ROSMA", episode_id="e1", task="Peg", operator="X01", frame="f", unit="m", rate_hz=50.0, t=t, position=pos)
    return quality.Episode(**{**base, **kw})


def flags(ep, limits=None):
    return [f["check"] for f in quality.check(ep, limits)["flags"]]


def test_a_clean_episode_is_accepted():
    row = quality.check(episode())
    assert row["verdict"] == "accepted" and row["flags"] == []


def test_each_check_catches_what_it_is_named_for():
    e = episode()
    assert "unit_magnitude" in flags(episode(position=e.position * 1000))
    jumped = e.position.copy()
    jumped[300:] += [0.03, 0, 0]
    assert flags(episode(position=jumped)) == ["jump"]
    assert "time_runs_backwards" in flags(episode(t=e.t[np.random.default_rng(0).permutation(len(e.t))]))
    late = e.t.copy()
    late[100] -= 0.98
    assert flags(episode(t=late)) == ["seconds_field_late"]
    gap = e.t.copy()
    gap[400:] += 0.5
    assert flags(episode(t=gap)) == ["sample_gaps"]
    frozen = e.position.copy()
    frozen[200:350] = frozen[200]
    assert "frozen" in flags(episode(position=frozen))
    joints = np.zeros((len(e.t), 6))
    joints[50:60, 2] = 0.5
    assert flags(episode(joints=joints), (np.full(6, -0.1), np.full(6, 0.24))) == ["joint_limits"]
    assert flags(episode(n=1)) == ["empty"]


def test_excluding_flags_exclude_and_a_copy_is_a_duplicate():
    e = episode()
    rows = quality.check_all([e, episode(episode_id="e2"), episode(episode_id="e3", position=e.position * 1000),
                              episode(episode_id="e4", position=e.position + 1e-4)])
    assert [r["verdict"] for r in rows] == ["accepted", "excluded", "excluded", "accepted"]
    assert rows[1]["excluded_because"] == ["duplicate"] and rows[2]["excluded_because"] == ["unit_magnitude"]


def test_the_split_is_by_operator_and_never_puts_one_on_both_sides():
    rows = quality.check_all([episode(episode_id=f"{op}_{i}", operator=op, position=episode().position + k * 1e-3 + i * 1e-4)
                              for k, op in enumerate(["X01", "X02", "X03", "X04"]) for i in range(3)]
                             + [episode(episode_id="sim_0", operator=None, source="SurgicAI", position=episode().position + 0.02)])
    sp = quality.split_by_operator(rows)
    assert sp["evaluation_operators"] == ["X04"] and sp["operators_on_both_sides"] == [] and len(sp["evaluation"]) == 3
    assert "sim_0" in sp["train"] and sp["without_an_operator"] == ["SurgicAI"]
    card = quality.card(rows, sp, {"ROSMA": {"source": "s", "licence": "CC BY 4.0", "cannot_show": "x"},
                                   "SurgicAI": {"source": "s", "licence": "MIT", "cannot_show": "y"}})
    assert "Operators on both sides: 0" in card and "Licence: MIT" in card and "Episodes: 12" in card
