"""Structured change proposals. These pin down what must not regress: the model never converts
a unit, a patch never edits the file it was checked against, a loosened limit cannot hide in a
patch that says it tightens, a failed generation proposes nothing, and nobody approves their
own change.

The schemas and files under tests/fixtures/config are made up for these tests.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from vxagent.llm.base import LLMError
from vxagent.llm.scripted import ScriptedLLM, say
from vxagent.profiles import principal
from vxagent.robot import config as cfg
from vxagent.robot.frames import FrameGraph
from vxagent.store import Database
from vxagent.structured import proposals
from vxagent.structured.dryrun import dry_apply
from vxagent.structured.generate import extract_json, generate_patch
from vxagent.structured.rules import InstrumentSet, RuleContext, describe_paths, validate_patch
from vxagent.structured.schemas import ConfigPatch, RunSheet, SiteFile
from vxagent.structured.units import UnitError, convert
from vxagent.tools.registry import Principal

FIX = Path(__file__).parent / "fixtures"
CONF = FIX / "config"
ROOT = Path(__file__).resolve().parents[1]
SITE = json.loads((ROOT / "sites" / "site_a.json").read_text(encoding="utf-8"))


def ctx_for(target: str, evidence=("rec:1",)) -> RuleContext:
    doc = {"site": copy.deepcopy(SITE), "system": json.loads((CONF / "system.json").read_text()),
           "arm": json.loads((CONF / "arm.json").read_text())}[target]
    return RuleContext(target=target, file_type=cfg.file_type(doc), doc=doc, schemas=cfg.SchemaSet(CONF / "schemas"),
                       instruments=InstrumentSet(CONF / "tool"), frames_declared=frozenset({"world", "ECM"}),
                       evidence=frozenset(evidence))


def patch(target: str, ops: list[dict], limit: str = "none", evidence=("rec:1",)) -> ConfigPatch:
    return ConfigPatch(target=target, operations=ops, limit_change=limit, reason="measured again on site",
                       evidence=list(evidence))


def op(path: str, value, unit: str = "none", kind: str = "replace") -> dict:
    return {"op": kind, "path": path, "value": value, "unit": unit}


def check(target: str, ops: list[dict], limit: str = "none"):
    c = ctx_for(target)
    return c, validate_patch(patch(target, ops, limit), c)


# ------------------------------------------------------------------------------ units and schemas

def test_units_are_converted_by_code_and_only_within_a_quantity():
    assert convert(210, "mm", "m") == pytest.approx(0.21)
    assert convert(-45, "deg", "rad") == pytest.approx(-math.pi / 4)
    assert convert(2, "ms", "s") == pytest.approx(0.002)
    with pytest.raises(UnitError):
        convert(4, "deg", "mm")


def test_patch_schema_forbids_unknown_fields_and_requires_every_field():
    good = {"target": "site", "operations": [op("a/b", 1, "mm")], "limit_change": "none",
            "reason": "measured again on site", "evidence": ["rec:1"]}
    ConfigPatch.model_validate(good)
    with pytest.raises(ValidationError):
        ConfigPatch.model_validate({**good, "comment": "extra"})
    for missing in ("limit_change", "evidence", "reason"):
        with pytest.raises(ValidationError):
            ConfigPatch.model_validate({k: v for k, v in good.items() if k != missing})
    with pytest.raises(ValidationError):                      # every value carries a unit
        ConfigPatch.model_validate({**good, "operations": [{"op": "replace", "path": "a/b", "value": 1}]})
    with pytest.raises(ValidationError):                      # from a closed list
        ConfigPatch.model_validate({**good, "operations": [op("a/b", 1, "cm")]})


def test_a_flag_stays_a_flag_and_a_number_a_number():
    p = patch("x", [op("a", True), op("b", 1, "mm")])
    assert p.operations[0].value is True and p.operations[1].value == 1 and p.operations[1].value is not True


def test_the_site_file_satisfies_its_own_schema():
    SiteFile.model_validate(SITE)
    with pytest.raises(ValidationError):
        SiteFile.model_validate({**SITE, "unknown_key": 1})


def test_run_sheet_requires_the_path_source_to_be_stated_even_if_null():
    base = dict(site="s", arm="PSM1", instrument="i", stages=["reach"], envelope_profile="trial", envelope_version="1",
                start_pose="start", grasp_pose="pregrasp", place_pose="place",
                acceptance={"reach_tolerance_mm": 1.0, "hold_required": True, "max_recoveries": 1}, operator="op",
                outcome_source="declared", recovery_preapproved=True)
    with pytest.raises(ValidationError):
        RunSheet.model_validate(base)
    RunSheet.model_validate({**base, "path_source": None})


# ------------------------------------------------------------------------------ files and paths

def test_list_elements_are_addressed_by_name():
    doc = json.loads((CONF / "system.json").read_text())
    assert cfg.get(doc, "arms/PSM2/base_frame/reference_frame") == "ECM"
    assert cfg.get(doc, "consoles/console/teleop_PSMs/MTMR-PSM1/configure_parameter/scale") == 0.4
    assert cfg.get(doc, "settings/audio_volume") is cfg.MISSING
    with pytest.raises(cfg.PathError):
        cfg.get(doc, "arms/PSM9/period")


def test_schema_references_are_resolved_locally():
    s = cfg.SchemaSet(CONF / "schemas")
    doc = json.loads((CONF / "system.json").read_text())
    assert s.errors(doc) == ["consoles/0/teleop_PSMs/0/configure_parameter: Additional properties are not allowed "
                             "('old_key' was unexpected)"]
    doc["arms"][0]["period"] = -1                              # the arm entry is behind a $ref
    assert any(e.startswith("arms/0/period") for e in s.errors(doc))
    assert s.errors({"no": "schema"}) is None


def test_allowed_paths_are_listed_with_their_value_now():
    c = ctx_for("system")
    rows = {r["path"]: r for r in describe_paths(c.file_type, c.doc)}
    assert rows["arms/PSM1/base_frame/transform/translation"]["now"] == [-0.2, 0.0, 0.0]
    assert rows["settings/audio_volume"]["now"] == "(not set)" and rows["settings/audio_volume"]["operation"] == "add"
    assert rows["arms/PSM1/base_frame/transform/translation"]["accepted_units"] == ["m", "mm"]


# ------------------------------------------------------------------------------ the rules

def test_a_valid_patch_is_applied_to_a_copy_and_converted_by_code():
    c, v = check("system", [op("arms/PSM1/base_frame/transform/translation", [-210, 0, 0], "mm")])
    assert v.ok and v.patched["arms"][0]["base_frame"]["transform"][0][3] == pytest.approx(-0.21)
    assert c.doc["arms"][0]["base_frame"]["transform"][0][3] == -0.2          # the original is untouched
    assert v.invalidates_calibration
    assert v.schema_errors_already_in_file                                   # reported, not blamed on the patch


def test_a_path_outside_the_allowlist_is_refused():
    _, v = check("system", [op("arms/PSM1/arm_file", "other.json")])
    assert v.stage == "allowlist"
    _, v = check("arm", [op("homing_zero_position", True, kind="add")])
    assert v.stage == "allowlist"


def test_replace_needs_a_value_and_add_needs_none():
    _, v = check("system", [op("settings/audio_volume", 0.3, "ratio")])
    assert v.stage == "allowlist" and "use op 'add'" in v.errors[0]
    _, v = check("system", [op("consoles/console/teleop_PSMs/MTMR-PSM1/configure_parameter/scale", 0.3, "ratio", "add")])
    assert v.stage == "allowlist" and "use op 'replace'" in v.errors[0]
    _, v = check("system", [op("settings/audio_volume", 0.3, "ratio", "add")])
    assert v.ok and v.patched["settings"] == {"audio_volume": 0.3}


def test_a_new_schema_error_stops_the_patch_but_an_old_one_does_not():
    _, v = check("system", [op("arms/PSM1/period", -2, "ms", "add")])
    assert v.stage == "file_schema" and "arms/0/period" in v.errors[0]


def test_millimetres_labelled_as_metres_are_refused_with_a_hint():
    _, v = check("system", [op("arms/PSM1/base_frame/transform/translation", [-210, 0, 0], "m")])
    assert v.stage == "physical" and "plausible range" in v.errors[0] and "'mm'" in v.errors[0]


def test_a_unit_of_another_quantity_is_refused():
    _, v = check("site", [op("safety_profile/clearance_mm", 4, "deg")], "tighten")
    assert v.stage == "physical" and "cannot be written" in v.errors[0]


def test_a_change_larger_than_the_step_is_refused():
    _, v = check("site", [op("safety_profile/clearance_mm", 12, "mm")], "tighten")
    assert v.stage == "physical" and "one patch may change" in v.errors[0]


@pytest.mark.parametrize("rotation, says", [
    ([[1.1, 0, 0], [0, 1, 0], [0, 0, 1]], "not orthonormal"),
    ([[1, 0, 0], [0, 1, 0], [0, 0, -1]], "reflection"),
    ([[1, 0, 0], [0, 0, -1], [0, 1, 0]], "rotation changes by 90.0 deg"),
])
def test_a_rotation_must_be_a_rotation_and_a_small_one(rotation, says):
    _, v = check("system", [op("arms/PSM1/base_frame/transform/rotation", rotation)])
    assert v.stage == "physical" and says in v.errors[0]


def test_a_transform_needs_its_last_row():
    T = [[1, 0, 0, -0.2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
    _, v = check("system", [op("arms/PSM1/base_frame/transform", T, "m")])
    assert v.stage == "physical" and "last row" in v.errors[0]


def test_frames_must_exist_and_must_not_form_a_cycle():
    _, v = check("system", [op("arms/PSM1/base_frame/reference_frame", "Cart")])
    assert v.stage == "physical" and "is not a frame of this site" in v.errors[0]
    _, v = check("system", [op("arms/PSM1/base_frame/reference_frame", "PSM2"),
                            op("arms/PSM2/base_frame/reference_frame", "PSM1")])
    assert v.stage == "physical" and "its own ancestor" in v.errors[0]
    _, v = check("system", [op("arms/PSM1/base_frame/reference_frame", "world")])
    assert v.ok


def test_frame_graph_finds_a_cycle():
    assert FrameGraph(parents={"a": "b", "b": "c"}).cycle() is None
    assert FrameGraph(parents={"a": "b", "b": "a"}).cycle() == ["a", "b", "a"]


@pytest.mark.parametrize("value, declared, ok", [
    (4, "tighten", True), (4, "loosen", False), (4, "none", False),
    (2, "loosen", True), (2, "tighten", False), (2, "none", False)])
def test_the_direction_of_a_limit_change_is_computed_not_believed(value, declared, ok):
    _, v = check("site", [op("safety_profile/clearance_mm", value, "mm")], declared)
    assert v.ok is ok
    assert v.limit_effect == ("tighten" if value > 3 else "loosen")


def test_a_patch_that_touches_no_limit_must_say_none():
    _, v = check("site", [op("pad_below_start_mm", 28, "mm")], "tighten")
    assert v.stage == "physical" and "the values say 'none'" in v.errors[0]


def test_the_near_step_cannot_exceed_the_free_step():
    _, v = check("site", [op("safety_profile/free_step_mm", 1.5, "mm")], "tighten")
    assert v.stage == "physical" and any("cannot exceed the free-space step" in e for e in v.errors)


def test_an_instrument_must_have_a_file():
    _, v = check("arm", [op("tool", "TEST_TOOL:000001", kind="add")])
    assert v.ok
    _, v = check("arm", [op("tool", "GHOST_TOOL:000002", kind="add")])      # listed, file missing
    assert v.stage == "physical" and "no instrument" in v.errors[0]
    _, v = check("arm", [op("tool_detection", "FIXED")])
    assert v.stage == "physical" and "must name the instrument" in v.errors[0]


def test_evidence_must_be_a_record_of_the_run():
    c = ctx_for("site")
    v = validate_patch(patch("site", [op("safety_profile/clearance_mm", 4, "mm")], "tighten", evidence=["rec:9"]), c)
    assert v.stage == "physical" and "is not a record of this run" in v.errors[0]


def test_the_target_must_be_the_file_the_request_is_about():
    c = ctx_for("site")
    v = validate_patch(patch("system", [op("safety_profile/clearance_mm", 4, "mm")], "tighten"), c)
    assert v.stage == "target"


# ------------------------------------------------------------------------------ generation

GOOD = {"target": "site", "operations": [op("safety_profile/clearance_mm", 4, "mm")], "limit_change": "tighten",
        "reason": "The jaw touches the pad at 3 mm.", "evidence": ["rec:1"]}
BAD = {**GOOD, "limit_change": "loosen"}


def test_a_failed_attempt_is_sent_back_with_numbered_errors():
    llm = ScriptedLLM([say(json.dumps(BAD)), say(json.dumps(GOOD))])
    g = generate_patch(llm, "Raise the clearance to 4 mm.", ctx_for("site"), {"rec:1": "note"})
    assert g.ok and [a.stage for a in g.attempts] == ["physical", "ok"]
    sent_back = llm.calls[1][-1]["content"]
    assert sent_back.startswith("That patch was not accepted. Problems:\n1. ") and "limit_change" in sent_back
    assert llm.calls[1][-2]["content"] == json.dumps(BAD)          # with the model's own reply


def test_no_valid_patch_means_nothing_is_proposed():
    llm = ScriptedLLM([say(json.dumps(BAD))])
    g = generate_patch(llm, "Raise the clearance to 4 mm.", ctx_for("site"), {"rec:1": "note"})
    assert not g.ok and g.patch is None and len(g.attempts) == 3 and len(llm.calls) == 3


def test_free_text_mode_extracts_the_object_and_reports_what_it_cannot_parse():
    fenced = "Here is the patch:\n```json\n" + json.dumps(GOOD) + "\n```\nLet me know."
    g = generate_patch(ScriptedLLM([say("I think 4 mm is right."), say(fenced)]), "r", ctx_for("site"), {"rec:1": "n"},
                       mode="free")
    assert [a.stage for a in g.attempts] == ["json", "ok"]
    assert extract_json('noise {"a": "}"} more') == {"a": "}"}


def test_schema_errors_are_their_own_stage():
    wrong = {k: v for k, v in GOOD.items() if k != "evidence"}
    g = generate_patch(ScriptedLLM([say(json.dumps(wrong)), say(json.dumps(GOOD))]), "r", ctx_for("site"), {"rec:1": "n"})
    assert [a.stage for a in g.attempts] == ["schema", "ok"]


def test_an_unreachable_model_is_not_retried_and_proposes_nothing():
    llm = ScriptedLLM([LLMError("unavailable", "down")])
    g = generate_patch(llm, "r", ctx_for("site"), {"rec:1": "n"})
    assert not g.ok and [a.stage for a in g.attempts] == ["llm_error"] and len(llm.calls) == 1


# ------------------------------------------------------------------------------ who may do what

@pytest.fixture()
def people():
    return {"operator": principal("op", "operator", "site-a"), "engineer": principal("eng", "service", "site-a"),
            "elsewhere": principal("eng2", "service", "site-b"), "support": principal("sup", "support", ""),
            "safety": principal("saf", "safety", "site-a"), "safety_elsewhere": principal("saf2", "safety", "site-b")}


def valid(target="site", ops=None, limit="tighten"):
    c = ctx_for(target)
    p = patch(target, ops or [op("safety_profile/clearance_mm", 4, "mm")], limit)
    return c, p, validate_patch(p, c)


def test_only_the_sites_own_engineer_may_propose(people):
    db = Database(":memory:")
    c, p, v = valid()
    for who, kind in (("operator", "role"), ("support", "role"), ("safety", "role"), ("elsewhere", "site")):
        with pytest.raises(proposals.NotPermitted) as e:
            proposals.propose(db, people[who], "site-a", p, v, c.doc, c.file_type)
        assert e.value.kind == kind
    assert proposals.propose(db, people["engineer"], "site-a", p, v, c.doc, c.file_type) == 1
    assert len(db.query("SELECT 1 FROM audit WHERE allowed=0")) == 4


def test_an_invalid_patch_cannot_be_proposed(people):
    db = Database(":memory:")
    c, p, v = valid(limit="loosen")
    with pytest.raises(proposals.NotPermitted) as e:
        proposals.propose(db, people["engineer"], "site-a", p, v, c.doc, c.file_type)
    assert e.value.kind == "invalid"


def test_only_the_safety_lead_approves_and_never_their_own_change(people):
    db = Database(":memory:")
    c, p, v = valid()
    pid = proposals.propose(db, people["engineer"], "site-a", p, v, c.doc, c.file_type)
    for who, kind in (("engineer", "role"), ("operator", "role"), ("safety_elsewhere", "site")):
        with pytest.raises(proposals.NotPermitted) as e:
            proposals.decide(db, people[who], pid, True)
        assert e.value.kind == kind
    both = Principal("both", "safety", people["engineer"].scopes | people["safety"].scopes, site="site-a")
    mine = proposals.propose(db, both, "site-a", p, v, c.doc, c.file_type)
    with pytest.raises(proposals.NotPermitted) as e:
        proposals.decide(db, both, mine, True)
    assert e.value.kind == "self_approval"
    assert proposals.decide(db, people["safety"], pid, True)["status"] == "approved"


def test_a_loosened_limit_needs_a_written_reason(people):
    db = Database(":memory:")
    c, p, v = valid(ops=[op("safety_profile/workspace_half_mm", 45, "mm")], limit="loosen")
    pid = proposals.propose(db, people["engineer"], "site-a", p, v, c.doc, c.file_type)
    with pytest.raises(proposals.NotPermitted) as e:
        proposals.decide(db, people["safety"], pid, True, "ok")
    assert e.value.kind == "reason_required"
    proposals.decide(db, people["safety"], pid, True, "Phantom B is 86 mm wide; reviewed on site.")


def test_apply_writes_a_new_version_and_leaves_the_source(tmp_path, people):
    db = Database(":memory:")
    source = tmp_path / "site_a.json"
    source.write_text(json.dumps(SITE), encoding="utf-8")
    store = cfg.ConfigStore(tmp_path / "work", {"site": source})
    before = source.read_bytes()
    c, p, v = valid()
    pid = proposals.propose(db, people["engineer"], "site-a", p, v, c.doc, c.file_type)
    with pytest.raises(proposals.NotPermitted):                 # not approved yet
        proposals.apply(db, store, pid)
    other = proposals.propose(db, people["engineer"], "site-a", p, v, c.doc, c.file_type)
    proposals.decide(db, people["safety"], pid, True)
    proposals.decide(db, people["safety"], other, True)
    res = proposals.apply(db, store, pid)
    doc, _, version = store.current("site")
    assert version == 1 and doc["safety_profile"]["clearance_mm"] == 4 and source.read_bytes() == before
    assert res["envelope_version"] == "2" == doc["safety_profile"]["version"]
    with pytest.raises(proposals.NotPermitted) as e:            # checked against version 0, which is gone
        proposals.apply(db, store, other)
    assert e.value.kind == "stale"


def test_a_base_frame_change_invalidates_the_calibration(tmp_path, people):
    db = Database(":memory:")
    store = cfg.ConfigStore(tmp_path / "work", {"system": CONF / "system.json"})
    c, p, v = valid("system", [op("arms/PSM1/base_frame/transform/translation", [-210, 0, 0], "mm")], "none")
    assert proposals.calibration(db, "site-a")["valid"]
    pid = proposals.propose(db, people["engineer"], "site-a", p, v, c.doc, c.file_type)
    proposals.decide(db, people["safety"], pid, True)
    assert proposals.apply(db, store, pid)["calibration_invalidated"]
    assert not proposals.calibration(db, "site-a")["valid"]


# ------------------------------------------------------------------------------ dry application

def test_dry_application_shows_which_named_pose_the_filter_would_now_reject():
    spec = {**copy.deepcopy(SITE), "kinematic_file": "test_arm.json", "instrument_file": "test_tool.json"}
    spec["named_poses"]["wrist_limit"]["joints_si"] = [0.0, 0.0, 0.12, 0.0, 1.36, 0.0]
    c = ctx_for("site")
    v = validate_patch(patch("site", [op("pad_below_start_mm", 24, "mm")]), c)
    after = {**v.patched, "kinematic_file": "test_arm.json", "instrument_file": "test_tool.json",
             "named_poses": spec["named_poses"]}
    effect = dry_apply("vx-site", spec, after, spec, FIX, v.changes)
    assert effect["newly_rejected_by_the_safety_filter"] == ["place", "pregrasp"]
    assert effect["poses_that_move"] == [] and effect["largest_change_of_height_over_pad_mm"] == 6.0
