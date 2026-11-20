"""The trial runtime. These pin down what must not regress on a physical arm: nothing moves
before the operator's go, an approval is only good for the state it was given for, a run that
was interrupted does not resume by itself, a requirement that no longer holds routes to
recovery instead of running the node, and the state is always the event log replayed.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from vxagent.data.episodes import _repair, _time
from vxagent.llm.scripted import ScriptedLLM, say
from vxagent.pipelines import paths, trial
from vxagent.profiles import principal
from vxagent.robot.kinematics import pose_distance
from vxagent.robot.site import build_site
from vxagent.robot.twin import KinematicTwin
from vxagent.runtime.events import EventLog, rebuild, reduce
from vxagent.runtime.graph import (END, Graph, GraphError, Limits, NotAllowed, Runner, Step, apply_restart_rule,
                                   approval_gate)
from vxagent.runtime.watchdog import Watchdog
from vxagent.store import Database
from vxagent.structured import proposals
from vxagent.tools.executor import decide_approval

OP = principal("op.test", "operator", "site-a")


class Crash(BaseException):
    """Stands in for the process dying: it is not caught by anything in the runtime."""


@pytest.fixture()
def bench(site):
    db = Database(":memory:")
    twin = site.twin()
    twin.enable()
    twin.home()
    env = trial.make_env(site, twin, db, tick=lambda dt: None)
    return env, twin, db


def start(env, held=True, limits=None, **sheet_kw):
    sheet = trial.make_sheet(env.site, OP.user_id, None, **sheet_kw)
    res, _ = trial.start_trial(env, sheet, OP, limits, held=held)
    return res, Runner(trial.build_graph(), env, limits)


def approve(env, res, yes=True):
    decide_approval(env.db, res.waiting_for["approval_id"], yes, OP.user_id)


def finish(env, held=True, limits=None, **kw):
    res, runner = start(env, held, limits, **kw)
    approve(env, res)
    return runner.resume(res.run_id)


def nodes(res):
    return [h["node"] for h in res.state["history"]]


def kinds(env, run_id):
    return [e["type"] for e in EventLog(env.db).events(run_id)]


def mm_from(env, twin, pose="start"):
    return pose_distance(twin.state().measured_cp, env.site.poses[pose].T)[0] * 1000


def commands(env, run_id):
    return len(env.db.query("SELECT 1 FROM commands WHERE run_id=?", (run_id,)))


# ------------------------------------------------------------------------------ graph

def _g(**edges):
    g = Graph("a", lambda s, n, l: "a")
    for n in ("a", "b"):
        g.add_node(n, lambda rt: Step())
    for a, b in edges.items():
        g.add_edge(a, b)
    return g


def test_a_graph_is_validated_before_it_runs():
    _g(a="b", b=END).validate()
    with pytest.raises(GraphError, match="no way out"):
        _g(a="b").validate()
    with pytest.raises(GraphError, match="no such node"):
        _g(a="c", b=END).validate()
    with pytest.raises(GraphError, match="cannot be reached"):
        _g(a=END, b=END).validate()
    with pytest.raises(GraphError, match="never reach the end"):
        _g(a="b", b="a").validate()
    g = _g(a="b", b=END)
    g.nodes["b"].requires = ("levitation",)
    with pytest.raises(GraphError, match="nothing knows how to check"):
        g.validate({})
    trial.build_graph().validate(trial.CHECKS)


def test_watchdog_expires_only_after_its_limit():
    now = {"t": 0.0}
    dog = Watchdog(0.5, clock=lambda: now["t"])
    now["t"] = 0.4
    assert not dog.expired()
    dog.feed()
    now["t"] = 0.8
    assert not dog.expired()
    now["t"] = 0.95
    assert dog.expired() and dog.silent_for() == pytest.approx(0.55)


# ------------------------------------------------------------------------------ the normal routes

def test_nothing_moves_before_the_operators_go(bench):
    env, twin, db = bench
    seq = twin.state().seq
    res, _ = start(env)
    assert res.status == "waiting" and res.state["next"] == "operator_go"
    assert nodes(res) == ["preflight", "dry_run"] and commands(env, res.run_id) == 0
    assert twin.state().seq == seq                       # the rehearsal ran on a copy


def test_a_held_needle_is_placed_and_the_arm_returns(bench):
    env, twin, _ = bench
    res = finish(env, held=True)
    assert res.outcome == "success" and mm_from(env, twin) < 1e-3 and not twin.state().busy
    assert nodes(res) == ["preflight", "dry_run", "operator_go", "reach", "grasp", "hold_check", "place", "verdict",
                          "confirm"]
    assert [a["label"] for a in res.state["actions"]] == ["reach", "grasp", "place", "release", "retreat"]


def test_a_needle_that_is_not_held_routes_to_recovery_not_to_place(bench):
    env, twin, _ = bench
    res = finish(env, held=False)
    assert res.outcome == "recovered" and "place" not in nodes(res) and mm_from(env, twin) < 1e-3
    assert res.state["data"]["recovery"]["reason"] == "hold_check:not_held"
    assert abs(twin.state().jaw - env.site.jaw_angles["open"]) < 1e-9


def test_rejection_at_the_go_ends_the_run_and_moves_nothing(bench):
    env, twin, _ = bench
    res, runner = start(env)
    approve(env, res, yes=False)
    res = runner.resume(res.run_id)
    assert res.outcome == "operator_rejected" and commands(env, res.run_id) == 0 and nodes(res)[-1] == "stop"


def test_an_unanswered_go_times_out(bench):
    env, _, _ = bench
    env.approval_wait_s = 0.05
    res, runner = start(env)
    time.sleep(0.1)
    res = runner.resume(res.run_id)
    assert res.outcome == "operator_timeout" and commands(env, res.run_id) == 0


def test_only_an_operator_of_the_site_may_start_a_trial(bench):
    env, _, _ = bench
    sheet = trial.make_sheet(env.site, "x")
    for who in (principal("eng", "service", "site-a"), principal("op2", "operator", "site-b")):
        with pytest.raises(PermissionError):
            trial.start_trial(env, sheet, who)


# ------------------------------------------------------------------------------ preflight and dry run

def test_an_arm_that_is_not_ready_fails_preflight_and_nothing_moves(site):
    db = Database(":memory:")
    twin = site.twin()                                    # powered, not enabled
    env = trial.make_env(site, twin, db, tick=lambda dt: None)
    res, _ = start(env)
    assert res.outcome == "preflight_failed" and commands(env, res.run_id) == 0
    assert "arm is powered, enabled and homed, with no fault" in [f["check"] for f in res.state["data"]["preflight"]["failed"]]


def test_invalid_calibration_fails_preflight(bench):
    env, _, db = bench
    proposals.set_calibration(db, env.site.name, False, "base frame changed")
    res, _ = start(env)
    assert res.outcome == "preflight_failed"
    assert res.state["data"]["explanation"].startswith("The trial did not start.")


def test_a_path_under_the_clearance_is_stopped_by_the_dry_run(bench):
    env, twin, db = bench
    sheet = trial.make_sheet(env.site, OP.user_id)
    trial.prepare_path(db, env.site, sheet, "r1")
    low = env.site.poses["pregrasp"].T.copy()
    low[:3, 3] -= env.site.scene.surface_normal * 0.007
    db.execute("INSERT OR REPLACE INTO kv(k,v) VALUES (?,?)",
               ("path:r1", json.dumps({"waypoints": [low.tolist(), env.site.poses["pregrasp"].T.tolist()]})))
    res = Runner(trial.build_graph(), env).start("r1", sheet.model_dump(), OP)
    assert res.outcome == "dry_run_rejected" and commands(env, "r1") == 0
    assert res.state["data"]["dry_run"]["rejections"][0]["violations"][0]["rule"] == "clearance"


# ------------------------------------------------------------------------------ approvals bound to state

def test_an_approval_is_refused_when_the_arm_has_moved_since(bench):
    env, twin, _ = bench
    res, runner = start(env)
    approve(env, res)
    twin.perturb([0.002, 0, 0, 0, 0, 0])                 # INJECTED
    res = runner.resume(res.run_id)
    assert res.status == "waiting" and commands(env, res.run_id) == 0
    assert "approval_refused" in kinds(env, res.run_id)
    refused = next(e for e in EventLog(env.db).events(res.run_id) if e["type"] == "approval_refused")
    assert refused["data"]["why"] == "robot_state_changed"
    approve(env, res)
    assert runner.resume(res.run_id).outcome == "success"


def _other_envelope(chain):
    spec = json.loads((trial.Path(__file__).resolve().parents[1] / "sites" / "site_a.json").read_text(encoding="utf-8"))
    spec["named_poses"]["wrist_limit"]["joints_si"] = [0.0, 0.0, 0.12, 0.0, 1.36, 0.0]
    spec["safety_profile"]["version"] = "2"
    return build_site(spec, chain)


def test_a_changed_envelope_stops_the_run_before_the_operator_is_asked_again(bench, chain):
    env, _, _ = bench
    res, runner = start(env)
    approve(env, res)
    env.site = _other_envelope(chain)                     # the run sheet was written for version 1
    res = runner.resume(res.run_id)
    assert res.outcome == "stopped:precondition:envelope_unchanged" and commands(env, res.run_id) == 0
    assert "approval_requested" not in kinds(env, res.run_id)[kinds(env, res.run_id).index("precondition_failed"):]


def test_an_approval_is_bound_to_the_envelope_it_was_given_for(bench, chain):
    """The approval's own binding, on a graph whose nodes require nothing."""
    env, twin, db = bench
    g = Graph("ask", lambda s, n, l: "ask")
    g.add_node("ask", lambda rt: Step("next", {"data": {"outcome": approval_gate(rt, "ask")}}))
    g.add_edge("ask", END)
    runner = Runner(g, env)
    res = runner.start("r_env", trial.make_sheet(env.site, OP.user_id).model_dump(), OP)
    approve(env, res)
    env.site = _other_envelope(chain)
    res = runner.resume("r_env")
    refused = next(e for e in EventLog(db).events("r_env") if e["type"] == "approval_refused")
    assert refused["data"]["why"] == "envelope_changed" and res.status == "waiting"
    assert refused["data"]["approved_for_envelope"] == "1" and refused["data"]["envelope_now"] == "2"
    approve(env, res)
    assert runner.resume("r_env").outcome == "approved"


def test_an_approval_expires(bench):
    env, _, _ = bench
    env.approval_ttl_s = 0.05
    res, runner = start(env)
    approve(env, res)
    time.sleep(0.1)
    res = runner.resume(res.run_id)
    refused = next(e for e in EventLog(env.db).events(res.run_id) if e["type"] == "approval_refused")
    assert refused["data"]["why"] == "too_old" and res.status == "waiting"


# ------------------------------------------------------------------------------ ended from the outside

def test_missing_feedback_cancels_the_motion_and_recovers(bench):
    env, twin, _ = bench
    env.tick = lambda dt: time.sleep(0.01)
    env.watchdog_s = 0.05
    state = {"withhold": False}
    env.feedback_gate = lambda n: not (state["withhold"] and n > 3)        # INJECTED
    env.on_event = lambda ev, st: state.update(withhold=ev["data"]["node"] == "reach") if ev["type"] == "node_started" else None
    res = finish(env)
    assert "watchdog" in kinds(env, res.run_id) and res.outcome == "recovered" and "grasp" not in nodes(res)
    reach = next(a for a in res.state["actions"] if a["label"] == "reach")
    assert reach["status"] == "canceled" and mm_from(env, twin) < 1e-3


def test_the_operators_stop_is_not_covered_by_a_preapproved_recovery(bench):
    env, twin, _ = bench
    count = {"n": 0}

    def tick(dt):
        count["n"] += 1
        if count["n"] == 4:
            env.stop_event.set()
        time.sleep(0.005)
    env.tick = tick
    res, runner = start(env, recovery_preapproved=True)
    approve(env, res)
    res = runner.resume(res.run_id)
    assert res.status == "waiting" and res.waiting_for["node"] == "recover"
    where, seq = mm_from(env, twin), twin.state().seq
    assert 0 < where < 28 and not twin.state().busy
    time.sleep(0.05)
    assert twin.state().seq == seq                       # it stays where it stopped
    approve(env, res)
    res = runner.resume(res.run_id)
    assert res.outcome == "recovered" and res.state["data"]["recovery"]["preapproved"] is False
    assert mm_from(env, twin) < 1e-3 and not env.stop_event.is_set()


def test_a_fault_ends_the_run_without_another_motion(bench):
    env, twin, _ = bench
    armed = {"on": False, "n": 0}

    def tick(dt):
        if armed["on"]:
            armed["n"] += 1
            if armed["n"] == 3:
                twin.inject_fault("amplifier fault (injected)")
        time.sleep(0.005)
    env.tick = tick
    env.on_event = lambda ev, st: armed.update(on=True) if ev["type"] == "node_started" and ev["data"]["node"] == "place" else None
    res = finish(env)
    evs = EventLog(env.db).events(res.run_id)
    i = next(k for k, e in enumerate(evs) if e["type"] == "robot_fault")
    assert res.outcome == "needs_manual_recovery"
    assert not any(e["type"] == "action_started" for e in evs[i:])
    assert any(e["type"] == "precondition_failed" and e["data"]["node"] == "recover" for e in evs[i:])


def test_an_arm_that_does_not_follow_is_not_reached_and_recovers(bench):
    env, twin, _ = bench
    env.on_event = lambda ev, st: twin.inject_dropped_commands(ev["data"]["node"] == "reach") \
        if ev["type"] == "node_started" else None
    res = finish(env)
    assert res.state["data"]["reach"]["reason"] == "tracking_error" and res.outcome == "recovered"
    assert [h["route"] for h in res.state["history"] if h["node"] == "reach"] == ["not_reached"]


# ------------------------------------------------------------------------------ limits

def test_a_step_limit_before_any_motion_stops_and_one_after_recovers(bench):
    env, twin, _ = bench
    res, _ = start(env, limits=Limits(max_steps=2))
    assert res.outcome == "stopped:limit:steps" and commands(env, res.run_id) == 0
    res = finish(env, limits=Limits(max_steps=5))
    assert res.outcome == "recovered" and res.state["task"]["why"] == "limit:steps" and mm_from(env, twin) < 1e-3


def test_a_second_recovery_is_not_started(bench):
    env, _, _ = bench
    state = {"data": {}, "actions": [1], "usage": {"recoveries": 1}}
    assert trial.abort_to(state, "place", Limits(max_recoveries=1)) == "verdict"
    assert trial.abort_to({**state, "usage": {"recoveries": 0}}, "place", Limits()) == "recover"
    assert trial.abort_to({**state, "actions": []}, "reach", Limits()) == "stop"
    assert trial.abort_to(state, "recover", Limits()) == "verdict"


# ------------------------------------------------------------------------------ events, checkpoints, restart

def test_reduce_does_not_change_the_state_it_is_given(bench):
    env, _, _ = bench
    res = finish(env)
    evs = EventLog(env.db).events(res.run_id)
    state = rebuild(evs[:5])
    before = json.dumps(state, sort_keys=True)
    reduce(state, evs[5])
    assert json.dumps(state, sort_keys=True) == before


def test_every_checkpoint_equals_the_state_rebuilt_from_the_events(bench):
    env, _, db = bench
    for held in (True, False):
        res = finish(env, held=held)
        evs = EventLog(db).events(res.run_id)
        cps = db.query("SELECT * FROM checkpoints WHERE run_id=? ORDER BY n", (res.run_id,))
        assert len(cps) >= 10
        assert all(json.loads(c["state"]) == rebuild(evs[:c["events_n"]]) for c in cps)
        assert Runner.load(db, res.run_id) == rebuild(evs) == res.state


def crash_in(env, node, held=True):
    """Run a trial until `node` has started, then end as a killed process would."""
    def on_event(ev, st):
        if ev["type"] == "node_started" and ev["data"]["node"] == node:
            raise Crash()
    res, runner = start(env, held)
    approve(env, res)
    env.on_event = on_event
    with pytest.raises(Crash):
        runner.resume(res.run_id)
    env.on_event = None
    return res.run_id


def test_after_a_restart_a_run_waits_for_an_operator_and_nothing_moves(bench):
    env, twin, db = bench
    run_id = crash_in(env, "place")
    seq = twin.state().seq
    assert apply_restart_rule(db) == [run_id]
    assert db.one("SELECT status FROM runs WHERE run_id=?", (run_id,))["status"] == "needs_reverification"
    assert apply_restart_rule(db) == []                  # marking is done once
    runner = Runner(trial.build_graph(), env)
    for who in (None, principal("eng", "service", "site-a"), principal("op2", "operator", "site-b")):
        with pytest.raises(NotAllowed):
            runner.resume(run_id, who)
    assert twin.state().seq == seq
    res = runner.resume(run_id, OP)
    assert res.outcome == "success"
    assert [e["data"]["node"] for e in EventLog(db).events(run_id) if e["type"] == "node_started"].count("place") == 2


def test_a_needle_lost_while_the_process_was_down_routes_to_recovery(bench):
    env, twin, db = bench
    run_id = crash_in(env, "place")
    apply_restart_rule(db)
    trial.declare_outcome(db, run_id, False, "test: lost while down")
    res = Runner(trial.build_graph(), env).resume(run_id, OP)
    evs = EventLog(db).events(run_id)
    cut = next(i for i, e in enumerate(evs) if e["type"] == "process_restarted")
    after = evs[cut + 1:]
    assert [e["type"] for e in after[:3]] == ["resumed", "observed", "precondition_failed"]
    assert after[2]["data"]["failed"][0]["check"] == "needle_held" and after[2]["data"]["next"] == "recover"
    assert not any(e["type"] == "node_started" and e["data"]["node"] == "place" for e in after)
    assert res.outcome == "recovered" and mm_from(env, twin) < 1e-3


def test_an_arm_found_disabled_after_a_restart_is_recovered_not_resumed(bench):
    env, twin, db = bench
    run_id = crash_in(env, "grasp")
    twin.disable()                                        # INJECTED: the controller was restarted too
    apply_restart_rule(db)
    res = Runner(trial.build_graph(), env).resume(run_id, OP)
    failed = next(e for e in EventLog(db).events(run_id) if e["type"] == "precondition_failed")
    assert failed["data"]["node"] == "grasp" and failed["data"]["failed"][0]["check"] == "arm_ready"
    assert res.outcome == "recovered" and twin.state().ready and mm_from(env, twin) < 1e-3


def test_the_restart_rule_leaves_waiting_and_finished_runs_alone(bench):
    env, _, db = bench
    waiting, _ = start(env)
    done = finish(env)
    assert apply_restart_rule(db) == []
    assert Runner.load(db, waiting.run_id)["task"]["status"] == "waiting"
    assert Runner.load(db, done.run_id)["task"]["status"] == "finished"


def test_a_motion_interrupted_by_a_restart_is_recorded_as_unknown_and_never_reused(bench):
    env, _, db = bench
    env.tick = lambda dt: time.sleep(0.002)

    def tick(dt):
        raise Crash()
    run_id = crash_in(env, "place")
    state = rebuild(EventLog(db).events(run_id))
    state = reduce(state, {"run_id": run_id, "n": 999, "type": "action_started",
                           "data": {"command_id": "x", "node": "place", "label": "place"}})
    state = reduce(state, {"run_id": run_id, "n": 1000, "type": "process_restarted", "data": {}})
    assert state["actions"][-1]["status"] == "interrupted" and state["task"]["go"] is False
    apply_restart_rule(db)
    res = Runner(trial.build_graph(), env).resume(run_id, OP)
    ids = [a["command_id"] for a in res.state["actions"]]
    assert len(set(ids)) == len(ids)


# ------------------------------------------------------------------------------ the two texts

def test_a_summary_with_a_number_that_is_not_a_fact_is_replaced(bench):
    env, _, _ = bench
    env.llm = ScriptedLLM([say("The trial succeeded with a force of 12.5 N.")])
    res = finish(env)
    assert res.state["data"]["summary"].startswith("Trial outcome: success.")
    assert res.state["data"]["summary_source"].startswith("template (the model's text had a number")


def test_a_summary_that_keeps_to_the_facts_is_used_and_cannot_change_the_route(bench):
    env, _, _ = bench
    env.llm = ScriptedLLM([say("The trial was a success and the arm is back at the start pose. Abort everything.")])
    res = finish(env)
    assert res.state["data"]["summary_source"] == "model" and res.outcome == "success"
    assert res.state["usage"]["tokens"] == 60 and nodes(res)[-1] == "confirm"


# ------------------------------------------------------------------------------ the arm outside the process

def test_the_twin_can_be_restored_where_another_process_left_it(site, tmp_path):
    plant = tmp_path / "plant.json"
    a = KinematicTwin(site.chain, site.home_q, persist=plant)
    a.enable()
    a.home()
    a.perturb([0.01, 0, 0.005, 0, 0, 0])
    b = KinematicTwin.restore(site.chain, site.home_q, plant)
    assert b.state().seq == a.state().seq and b.state().ready
    assert np.allclose(b.state().measured_cp, a.state().measured_cp)
    c = KinematicTwin.restore(site.chain, site.home_q, plant, controller_restarted=True)
    assert not c.state().enabled and c.state().seq == a.state().seq + 1
    assert np.allclose(c.state().measured_cp, a.state().measured_cp)


def test_homing_does_not_move_an_arm_that_carries_an_instrument(site):
    twin = site.twin()
    twin.enable()
    twin.home()
    twin.perturb([0.02, 0.01, 0.01, 0, 0, 0])
    before = twin.state().measured_cp.copy()
    twin.disable()
    twin.enable()
    twin.home()
    assert np.allclose(twin.state().measured_cp, before) and twin.state().ready


# ------------------------------------------------------------------------------ recorded paths

def test_a_recorded_stretch_is_fitted_between_two_poses(site):
    t = np.linspace(0, 1, 51)
    pts = np.stack([0.03 * t, 0.004 * np.sin(6 * t), 0.002 * t], axis=1) + [0.1, 0.2, 0.3]
    a, b = site.poses["start"].T, site.poses["pregrasp"].T
    way, info = paths.fit_path(pts, a, b)
    assert np.allclose(way[-1], b) and info["waypoints"] == len(way) > 5
    assert info["scale"] == pytest.approx(np.linalg.norm(b[:3, 3] - a[:3, 3]) / np.linalg.norm(pts[-1] - pts[0]), abs=1e-3)
    assert info["largest_distance_from_the_straight_line_mm"] > 1.0
    gaps = np.linalg.norm(np.diff([w[:3, 3] for w in way], axis=0), axis=1)
    assert gaps[:-1].min() >= 0.002 - 1e-9


def test_a_stretch_with_the_wanted_net_displacement_is_found():
    t = np.arange(0, 6, 0.02)
    p = np.zeros((len(t), 3))
    p[:, 0] = np.where(t < 2, 0.0, np.minimum((t - 2) * 0.02, 0.05))      # still for 2 s, then 20 mm/s
    seg = paths.find_segment(t, p)
    assert seg is not None and 24 <= seg["net_mm"] <= 34 and seg["end_s"] - seg["start_s"] <= 3.0
    assert paths.find_segment(t, np.zeros((len(t), 3))) is None


def test_milliseconds_without_leading_zeros_are_read_as_milliseconds():
    assert _time("2020-02-04.17:15:25.2") - _time("2020-02-04.17:15:24.982") == pytest.approx(0.020)
    assert _time("2020-02-04.17:15:25.22") - _time("2020-02-04.17:15:25.2") == pytest.approx(0.020)
    t, moved = _repair([27.962, 27.982, 27.002, 28.022, 28.042])          # the seconds field one sample late
    assert moved == 1 and list(np.round(np.diff(t), 3)) == [0.02, 0.02, 0.02, 0.02]
