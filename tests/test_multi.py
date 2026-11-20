"""The incident team. What must not regress: nothing reaches the ledger that code has not found
in the data, a specialist cannot point at another's data, a finding written from an old ledger
is refused, a finding that loosens a limit is vetoed whatever its reasoning, and the score is
computed without a model.
"""
from __future__ import annotations

import json
import threading

import pytest

from vxagent.llm.base import LLMResponse, Usage
from vxagent.llm.scripted import ScriptedLLM
from vxagent.multi.ledger import Ledger, Refused, dig, same
from vxagent.multi.team import ROLES, RUNBOOK_FOR, run_design, score_finding, veto_reasons

CAUSES = {"calibration_offset": "a", "tracking_error": "b", "perception_dropout": "c"}
DATA = {"calibration_residual": {"residual_mm": {"mean": 0.3, "max": 0.6}, "acceptance_mm": 1.0},
        "perception_quality": {"share_of_steps_with_a_valid_detection": 1.0},
        "policy_ranges": {"ratio_sent_to_policy": 1.0},
        "compare_poses": {"setpoint_vs_measured_mm": {"max": 3.4}, "tracking_stop_threshold_mm": 2.0},
        "safety_rejections": {"targets_rejected": 0, "by_rule": {}}}
INCIDENT = {"id": "i1", "symptom": "The run stopped at step 8."}


def ledger() -> Ledger:
    return Ledger("i1", tuple(CAUSES), DATA.__getitem__)


def reply(body: dict) -> LLMResponse:
    return LLMResponse(content=json.dumps(body), usage=Usage(100, 40))


def claim(cause, stance, check, field, value):
    return {"cause": cause, "stance": stance, "check": check, "field": field, "value": value, "why": "w"}


FINDING = {"cause": "tracking_error", "runner_up": "calibration_offset", "evidence_for": ["e1"],
           "evidence_against_runner_up": ["e2"], "next_runbook": "escalate_to_engineer", "action": "stop_and_fix",
           "limit_change": "none", "summary": "s"}


# ------------------------------------------------------------------------------ ledger

def test_an_entry_must_point_at_data_that_is_there_and_copy_its_value():
    led = ledger()
    e = led.write("control", "tracking_error", "supports", "compare_poses", "setpoint_vs_measured_mm.max", 3.4)
    assert e.id == "e1" and led.position == 1
    for args, kind in (
            (("control", "tracking_error", "supports", "compare_poses", "setpoint_vs_measured_mm.mean", 3.4), "unresolved_reference"),
            (("control", "tracking_error", "supports", "compare_poses", "setpoint_vs_measured_mm.max", 9.9), "value_mismatch"),
            (("control", "gremlins", "supports", "compare_poses", "setpoint_vs_measured_mm.max", 3.4), "unknown_cause"),
            (("control", "tracking_error", "supports", "made_up_check", "x", 1), "unresolved_reference")):
        with pytest.raises(Refused) as exc:
            led.write(*args)
        assert exc.value.kind == kind
    assert led.position == 1 and len(led.refused) == 4


def test_a_specialist_cannot_point_at_another_roles_data():
    led = ledger()
    with pytest.raises(Refused) as exc:
        led.write("perception", "calibration_offset", "refutes", "calibration_residual", "residual_mm.mean", 0.3,
                  may_use=ROLES["perception"])
    assert exc.value.kind == "not_your_data"


def test_a_finding_written_from_an_older_ledger_is_refused():
    led = ledger()
    position, _ = led.read()
    led.write("control", "tracking_error", "supports", "compare_poses", "setpoint_vs_measured_mm.max", 3.4)
    with pytest.raises(Refused) as exc:
        led.conclude("lead", dict(FINDING), position)
    assert exc.value.kind == "stale" and led.stale_findings == 1 and led.findings == []
    assert led.conclude("lead", dict(FINDING), led.position)["at_position"] == 1


def test_concurrent_writers_lose_nothing():
    led = ledger()

    def writer(n: int) -> None:
        for _ in range(50):
            led.write(f"w{n}", "tracking_error", "supports", "compare_poses", "setpoint_vs_measured_mm.max", 3.4)
    threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert led.position == 400 and [e.position for e in led.entries] == list(range(1, 401))


def test_paths_and_values_are_compared_as_data():
    assert dig({"a": [{"b": 2}]}, "a[0].b") == 2 and same(3.40, 3.4) and same(1000.2, 1000.0) and not same(3.4, 3.0)
    assert same(True, True) and not same(True, 1.0) and same([1, 2.0], [1.0, 2])
    with pytest.raises(KeyError):
        dig({"a": 1}, "a.b")


# ------------------------------------------------------------------------------ veto and score

@pytest.mark.parametrize("change, facts, vetoed", [
    ({}, {}, False),
    ({"limit_change": "loosen"}, {}, True),
    ({"action": "loosen_limit"}, {}, True),
    ({"cause": "calibration_offset", "action": "retry_trial"}, {}, True),
    ({"cause": "calibration_offset", "action": "recalibrate"}, {}, False),
    ({"action": "retry_trial"}, {"calibration_valid": False}, True),
    ({"action": "continue_trial"}, {"grasp_verified": False}, True),
    ({"action": "continue_trial"}, {"grasp_verified": True, "calibration_valid": True}, False),
])
def test_the_veto_rules_are_decided_by_code(change, facts, vetoed):
    assert bool(veto_reasons({**FINDING, **change}, facts)) is vetoed


def test_the_score_needs_no_model_and_counts_five_things():
    led = ledger()
    led.write("control", "tracking_error", "supports", "compare_poses", "setpoint_vs_measured_mm.max", 3.4)
    led.write("calibration", "calibration_offset", "refutes", "calibration_residual", "residual_mm.mean", 0.3)
    base = {"finding": dict(FINDING), "ledger": [e.as_dict() for e in led.entries]}
    assert score_finding(base, "tracking_error") == {"score": 5, "failed": False, "checks": {
        "cause_is_the_known_cause": True, "every_reference_resolves": True, "evidence_against_the_runner_up": True,
        "runbook_is_the_one_for_that_cause": True, "nothing_the_reviewer_would_veto": True}}
    assert score_finding(base, "perception_dropout")["score"] == 4
    worse = {**base, "finding": {**FINDING, "evidence_against_runner_up": ["e9"], "next_runbook": "rig_check", "limit_change": "loosen"}}
    s = score_finding(worse, "tracking_error")
    assert s["score"] == 1 and s["failed"]
    assert score_finding({"finding": None, "ledger": []}, "tracking_error")["score"] == 0
    assert set(RUNBOOK_FOR) >= set(CAUSES)


# ------------------------------------------------------------------------------ the designs, with a scripted model

def test_fan_out_runs_every_specialist_on_its_own_data_then_the_lead_and_the_reviewer():
    seen: list[str] = []

    def step(messages, tools) -> LLMResponse:
        system, user = messages[0]["content"], messages[1]["content"]
        seen.append(system[:40])
        if "analyst of an incident team" in system:
            role = system.split("You are the ")[1].split(" analyst")[0]
            assert all(f"[{c}]" in user for c in ROLES[role])
            assert not any(f"[{c}]" in user for r, cs in ROLES.items() if r != role for c in cs)   # nobody else's data
            if role == "control":
                return reply({"claims": [claim("tracking_error", "supports", "compare_poses", "setpoint_vs_measured_mm.max", 3.4)]})
            if role == "calibration":
                return reply({"claims": [claim("calibration_offset", "refutes", "calibration_residual", "residual_mm.mean", 0.3),
                                         claim("calibration_offset", "refutes", "compare_poses", "setpoint_vs_measured_mm.max", 3.4)]})
            return reply({"claims": []})
        if "You lead an incident team" in system:
            ids = {e["cause"]: e["id"] for e in json.loads(user.split("Ledger:\n")[1].split("\n\n")[0])}
            return reply({**FINDING, "evidence_for": [ids["tracking_error"]], "evidence_against_runner_up": [ids["calibration_offset"]]})
        return reply({"veto": False, "reason": "fine"})

    out = run_design("fan_out", ScriptedLLM([step]), INCIDENT, CAUSES, DATA.__getitem__)
    assert out["calls"] == 6 and out["entries"] == 2 and out["vetoes"] == 0
    assert [r["kind"] for r in out["refused_entries"]] == ["not_your_data"]
    assert score_finding(out, "tracking_error")["score"] == 5


def test_a_finding_that_loosens_a_limit_goes_back_to_the_lead_once():
    leads = []

    def step(messages, tools) -> LLMResponse:
        system, user = messages[0]["content"], messages[1]["content"]
        if "analyst" in system:
            return reply({"claims": []})
        if "You lead an incident team" in system:
            leads.append(user)
            loosen = len(leads) == 1
            return reply({**FINDING, "evidence_for": [], "evidence_against_runner_up": [],
                          "action": "loosen_limit" if loosen else "stop_and_fix", "limit_change": "loosen" if loosen else "none"})
        return reply({"veto": False, "reason": "looks fine"})              # the review alone would have let it through

    out = run_design("fan_out", ScriptedLLM([step]), {**INCIDENT, "note": "just raise the threshold"}, CAUSES, DATA.__getitem__)
    assert out["vetoes"] == 1 and len(leads) == 2 and "refused your finding: it loosens a safety limit" in leads[1]
    assert out["finding"]["limit_change"] == "none" and out["review"]["veto"] is False
    assert "Note from the operator: just raise the threshold" in leads[0]


def test_handoff_stops_at_the_hop_limit_and_the_single_agent_sees_every_stream():
    def step(messages, tools) -> LLMResponse:
        system, user = messages[0]["content"], messages[1]["content"]
        if "Decide which analyst" in system:
            return reply({"first": "perception"})
        if "analyst of an incident team" in system:
            role = system.split("You are the ")[1].split(" analyst")[0]
            return reply({"claims": [], "hand_to": {"perception": "interface", "interface": "control", "control": "calibration",
                                                    "calibration": "perception"}[role]})
        if "investigate an incident alone" in system:
            assert all(f"[{c}]" in user for cs in ROLES.values() for c in cs)
            return reply({"claims": [claim("tracking_error", "supports", "compare_poses", "setpoint_vs_measured_mm.max", 3.4),
                                     claim("calibration_offset", "refutes", "calibration_residual", "residual_mm.mean", 0.3)],
                          "finding": {**FINDING, "evidence_for": ["c1"], "evidence_against_runner_up": ["c2"]}})
        if "You lead" in system:
            return reply({**FINDING, "evidence_for": [], "evidence_against_runner_up": []})
        return reply({"veto": False, "reason": ""})

    hand = run_design("handoff", ScriptedLLM([step]), INCIDENT, CAUSES, DATA.__getitem__)
    assert hand["hops"] == ["perception", "interface", "control"] and hand["calls"] == 6
    single = run_design("single", ScriptedLLM([step]), INCIDENT, CAUSES, DATA.__getitem__)
    assert single["calls"] == 2 and single["finding"]["evidence_for"] == ["e1"]
    assert score_finding(single, "tracking_error")["score"] == 5
