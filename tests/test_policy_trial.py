"""The policy trial pipeline, with a made-up arm and a made-up rollout. What must not regress:
no motion before every pre-flight check has passed and the operator has said go, each failed
check names the runbook that repairs it, the needle counts as held only if it came up with the
jaws and stayed, and an operator's override never replaces the machine verdict.
"""
from __future__ import annotations

import numpy as np
import pytest

from runbook_bench import make_site
from test_policy_data import contract
from vxagent.diagnostics.episode import calibration
from vxagent.memory import site as site_memory
from vxagent.pipelines import policy_trial as PT
from vxagent.pipelines import trial
from vxagent.policy import compat
from vxagent.policy.adapter import Adapter
from vxagent.profiles import principal
from vxagent.robot.kinematics import pose_distance
from vxagent.robot.needle import DeclaredNeedle
from vxagent.robot.preflight import run_preflight
from vxagent.runtime.events import EventLog
from vxagent.runtime.graph import Runner
from vxagent.store import Database
from vxagent.structured import proposals
from vxagent.tools.executor import decide_approval
from vxagent.tools.service import store_calibration_record

OP = principal("op.test", "operator", "site-a")
DISPLACEMENT_CM = np.array([-2.2, -0.3, -2.0])


def rollout(site, c):
    """A straight rollout in policy space that ends on the goal, as steps the policy could emit."""
    free = Adapter(c)
    goal = free.pose_to_policy(site.poses["pregrasp"].T)
    start = goal.copy()
    start[:3] -= DISPLACEMENT_CM
    n = int(np.ceil(np.abs(DISPLACEMENT_CM).max() / (0.9 * c.step_translation)))
    action = np.concatenate([DISPLACEMENT_CM / n / c.step_translation, [0, 0, 0, 0]])
    poses = [start]
    for _ in range(n):
        poses.append(free.step(poses[-1], action))
    return {"id": "made_up_00", "poses": np.array(poses), "actions": np.array([action] * n), "goal": goal}


class Bench:
    def __init__(self, needle=None, ready=True, perception="ok", session="s1", scene="s1", cal_offset_mm=0.0, shift_cm=None):
        self.site = make_site()
        self.db = Database(":memory:")
        self.c = contract(start_to_goal_min=list(DISPLACEMENT_CM - 0.05), start_to_goal_max=list(DISPLACEMENT_CM + 0.05))
        self.demo = rollout(self.site, self.c)
        if shift_cm is not None:
            self.demo["poses"] = self.demo["poses"].copy()
            self.demo["poses"][:, :3] += np.array(shift_cm)
        self.twin = self.site.twin()
        if ready:
            self.twin.enable()
            self.twin.home()
        fault = {"cause": "calibration_offset", "mm": cal_offset_mm, "direction": [1.0, 0.0, 0.0]} if cal_offset_mm else None
        cal = calibration(fault, self.site.scene.workspace_center, np.random.default_rng(2))
        store_calibration_record(self.db, self.site.name, {"id": cal["id"], "T_est": cal["T_est"].tolist(), "seen_m": cal["seen_m"].tolist(),
                                                           "touched_m": cal["touched_m"].tolist(), "method": "pp", "session": session})
        if scene:
            self.db.execute("INSERT INTO kv(k,v) VALUES (?,?)", (f"scene_session:{self.site.name}", f'"{scene}"'))
        proposals.set_calibration(self.db, self.site.name, True, "test")
        self.needle = DeclaredNeedle(self.site, **(needle or {})).attach(self.twin)
        ad = Adapter.anchored(self.c, self.demo["goal"], self.site.poses["pregrasp"].T)
        cp = compat.check(self.c, self.site, "start", "pregrasp", 10.0, policy_start=self.demo["poses"][0], policy_goal=self.demo["goal"])
        look = {"ok": lambda: {"visible": True, "latency_s": 0.01}, "slow": lambda: {"visible": True, "latency_s": 0.9}, "none": None}[perception]
        self.env = trial.make_env(self.site, self.twin, self.db, tick=lambda dt: None,
                                  extras={"contract": self.c, "adapter": ad, "demo": self.demo, "needle": self.needle, "compat": cp,
                                          "perception": look})
        self.sheet = trial.make_sheet(self.site, OP.user_id, None)
        self.runner = Runner(PT.build_graph(), self.env)

    def start(self, run_id="r1"):
        return PT.start(self.env, self.sheet, OP, run_id)

    def go(self, run_id="r1"):
        res = self.start(run_id)
        decide_approval(self.db, res.waiting_for["approval_id"], True, OP.user_id)
        return self.runner.resume(run_id)

    def commands(self, run_id="r1"):
        return len(self.db.query("SELECT 1 FROM commands WHERE run_id=?", (run_id,)))


def test_nothing_moves_before_preflight_rehearsal_and_the_operators_go():
    b = Bench()
    seq = b.twin.state().seq
    res = b.start()
    assert res.status == "waiting" and [h["node"] for h in res.state["history"]] == ["preflight", "rehearse"]
    assert res.state["data"]["preflight"] == {"checks": 9, "failed": []} and b.commands() == 0 and b.twin.state().seq == seq


def test_a_held_needle_is_placed_and_the_run_waits_for_the_operator_to_confirm():
    b = Bench()
    res = b.go()
    assert res.status == "waiting" and res.waiting_for["kind"] == "operator_confirmation" and res.waiting_for["machine_verdict"] == "success"
    v = res.state["data"]["verdict"]
    assert v["reach_tolerance_met"] and v["needle_held"] and v["lateral_distance_from_the_entry_mm"] < 0.01
    assert v["needle_height_over_pad_mm_at_release"] == pytest.approx(3.5, abs=0.01) and v["outcome_source"] == "declared"
    PT.decide_verdict(b.db, "r1", OP.user_id, "confirm")
    res = b.runner.resume("r1")
    assert res.outcome == "success" and res.state["data"]["operator_verdict"]["verdict"] == "success"
    assert pose_distance(b.twin.state().measured_cp, b.site.poses["start"].T)[0] < 1e-6
    assert res.state["data"]["record"]["machine_verdict"]["machine"] == "success" and res.state["data"]["summary_source"] == "template"


def test_the_first_step_of_every_stage_is_sent_alone_and_checked():
    b = Bench()
    b.go()
    events = EventLog(b.db).events("r1")
    firsts = [e["data"]["node"] for e in events if e["type"] == "check_passed" and e["data"]["check"] == "first step of the stage is tracked"]
    assert firsts == ["stage", "reach", "grasp:descent", "grasp:lift", "place"]
    labels = [e["data"]["label"] for e in events if e["type"] == "action_started"]
    assert labels[:4] == ["stage:first_step", "stage", "reach:first_step", "reach"]


@pytest.mark.parametrize("needle, said", [({"slips_on_lift": True}, "the needle slipped out during the lift"),
                                          ({"slips_while_holding": True}, "the needle slipped out while it was held still"),
                                          ({"grasp_succeeds": False}, "the jaw closed and missed the needle")])
def test_a_needle_that_does_not_stay_routes_to_recovery_and_never_to_place(needle, said):
    b = Bench(needle=needle)
    res = b.go()
    assert said in b.needle.log and "place" not in [h["node"] for h in res.state["history"]]
    assert res.state["data"]["verdict"]["machine"] == "failed_recovered" and res.state["data"]["verdict"]["arm_at_start_pose"]
    assert res.state["data"]["recovery"]["reason"] == "retention:not_held"


def test_an_override_needs_a_reason_and_is_kept_beside_the_machine_verdict():
    b = Bench()
    b.go()
    PT.decide_verdict(b.db, "r1", OP.user_id, "override", to="failed", reason="no")
    res = b.runner.resume("r1")
    assert res.status == "waiting" and res.waiting_for["note"] == "an override needs a reason"
    PT.decide_verdict(b.db, "r1", OP.user_id, "override", to="failed", reason="The needle sat crooked in the jaw.")
    res = b.runner.resume("r1")
    d = res.state["data"]
    assert d["operator_verdict"] == {"decision": "override", "by": "op.test", "verdict": "failed", "reason": "The needle sat crooked in the jaw."}
    assert d["verdict"]["machine"] == "success" and d["record"]["machine_verdict"]["machine"] == "success"
    assert "override to failed" in d["summary"]


@pytest.mark.parametrize("kw, check, repair", [
    ({"ready": False}, "arm is powered, enabled and homed, with no fault", "bring_up"),
    ({"cal_offset_mm": 6.0}, "the calibration residual is under the site's limit", "calibration_validity"),
    ({"scene": "s0"}, "pad-plane and site records are of the same session as the calibration", "rig_check"),
    ({"scene": None}, "pad-plane and site records are of the same session as the calibration", "rig_check"),
    ({"perception": "slow"}, "the perception source answered within its time limit", "rig_check"),
    ({"perception": "none"}, "the perception source answered within its time limit", "rig_check"),
    ({"shift_cm": [3.0, 0.0, 0.0]}, "start and target lie in the policy's training range and inside the workspace", "escalate_to_engineer"),
])
def test_each_failed_preflight_check_stops_the_run_and_names_its_repair(kw, check, repair):
    b = Bench(**kw)
    res = b.start()
    failed = {f["check"]: f["repair"] for f in res.state["data"]["preflight"]["failed"]}
    assert res.outcome == "preflight_failed" and failed.get(check) == repair and b.commands() == 0
    assert repair in res.state["data"]["repairs"]


def test_an_event_that_ends_the_calibration_fails_preflight():
    b = Bench()
    site_memory.record_event(b.db, b.site.name, "ecm_moved", "eng")
    res = b.start()
    assert "the calibration has not been ended by an event" in [f["check"] for f in res.state["data"]["preflight"]["failed"]]


def test_preflight_reports_every_check_with_its_result():
    b = Bench()
    poses = PT.trial_poses(type("R", (), {"env": b.env, "state": {"sheet": b.sheet.model_dump()}})())
    rows = run_preflight(b.twin, b.site, b.sheet.model_dump(), b.db, start_T=poses["start"], goal_T=poses["goal"],
                         compat=None, in_range=None, perception=lambda: (_ for _ in ()).throw(TimeoutError()))
    by = {r["check"]: r for r in rows}
    assert by["the policy contract is compatible with the site"]["repair"] == "escalate_to_engineer"
    assert "TimeoutError" in by["the perception source answered within its time limit"]["detail"]
    assert all(r["repair"] is None for r in rows if r["ok"])


def test_a_rollout_the_envelope_does_not_allow_is_stopped_in_rehearsal():
    b = Bench()
    object.__setattr__(b.site, "profile", type(b.site.profile)(**{**b.site.profile.__dict__, "clearance_m": 0.006}))
    b.env.extras.pop("poses", None)
    res = b.start()
    assert res.outcome == "rehearsal_rejected" and res.state["data"]["dry_run"]["rejections"][0]["stage"] == "descent"
    assert b.commands() == 0 and "approval_requested" not in [e["type"] for e in EventLog(b.db).events("r1")]
