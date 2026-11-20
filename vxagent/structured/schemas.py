"""The typed objects a model may propose. All of them forbid unknown fields, and every field a
rule checks is required: a field the model would rather leave out has to be written as null
where null is allowed, so "not stated" is something the model says, not something it omits.

These models say whether an object is well formed. Whether it is physically sensible is decided
in rules.py, and whether the caller may propose or approve it in proposals.py.
"""
from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr

Unit = Literal["m", "mm", "rad", "deg", "s", "ms", "ratio", "none"]
Number = Union[StrictInt, StrictFloat]
Value = Union[StrictBool, StrictInt, StrictFloat, StrictStr, list[Number], list[list[Number]]]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------------------ ConfigPatch

class Operation(Strict):
    op: Literal["replace", "add"] = Field(description="replace: the path has a value now. add: it does not.")
    path: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_$.:\-]+(/[A-Za-z0-9_$.:\-]+)*$",
                      description="Path inside the file, segments separated by '/'. Use the paths listed as allowed.")
    value: Value = Field(description="The new value, as the request states it. Do not convert units.")
    unit: Unit = Field(description="Unit the value is given in. 'ratio' for a pure number, 'none' for text, "
                                   "true/false and rotation matrices.")


class ConfigPatch(Strict):
    """A small change to one existing configuration file. The only way configuration changes."""
    target: str = Field(description="The file to change, exactly as given in the request.")
    operations: list[Operation] = Field(min_length=1, max_length=8)
    limit_change: Literal["none", "tighten", "loosen"] = Field(
        description="'tighten' if the patch makes a safety limit stricter, 'loosen' if it relaxes one, "
                    "'none' if it touches no safety limit.")
    reason: str = Field(min_length=10, max_length=400, description="Why the change is needed, in one or two sentences.")
    evidence: list[str] = Field(min_length=1, max_length=8,
                                description="Ids of the records of this run that support the change.")


# ------------------------------------------------------------------------------ RunSheet

Stage = Literal["reach", "grasp", "hold_check", "place"]


class PathSource(Strict):
    dataset: str
    episode_id: str
    arm: Literal["PSM1", "PSM2"]
    first_sample: int = Field(ge=0)
    last_sample: int = Field(gt=0)
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)


class Acceptance(Strict):
    reach_tolerance_mm: float = Field(gt=0, le=5)
    hold_required: bool
    max_recoveries: int = Field(ge=0, le=3)


class RunSheet(Strict):
    """What a trial is allowed to be, written down before anything moves."""
    site: str
    arm: str
    instrument: str = Field(description="Instrument file name without the extension.")
    stages: list[Stage] = Field(min_length=1)
    envelope_profile: str
    envelope_version: str
    start_pose: str
    grasp_pose: str
    place_pose: str
    acceptance: Acceptance
    operator: str
    path_source: PathSource | None = Field(description="Recorded path the approach follows, or null for a straight approach.")
    outcome_source: Literal["declared", "sensor"] = Field(
        description="Where 'needle held' comes from. The twin has no needle, so on the twin it is declared.")
    recovery_preapproved: bool


# ------------------------------------------------------------------------------ FaultReport

class EvidenceRef(Strict):
    source_id: str
    locator: str | None = Field(description="Where in the source: a line, a time or a field. Null if the whole record.")


class FaultReport(Strict):
    symptom: str = Field(min_length=5)
    cause_class: Literal["configuration", "calibration", "communication", "mechanical", "instrument", "software",
                         "operator", "unknown"]
    component: str | None
    evidence: list[EvidenceRef] = Field(min_length=1)
    next_check: str = Field(description="Name of the runbook to run next.")
    confidence: Literal["low", "medium", "high"]


# ------------------------------------------------------------------------------ CalibrationRecord

class Frames(Strict):
    parent: str
    child: str


class Residuals(Strict):
    translation_rms_mm: float = Field(ge=0)
    rotation_rms_deg: float | None = Field(ge=0)


class CalibrationRecord(Strict):
    method: str
    frames: Frames
    transform: list[list[float]] = Field(min_length=4, max_length=4, description="Parent from child, 4 x 4, metres.")
    residuals: Residuals
    samples: int = Field(ge=1)
    valid_until: list[Literal["instrument_change", "base_frame_change", "camera_moved", "arm_remounted",
                              "software_update"]] = Field(min_length=1)


# ------------------------------------------------------------------------------ PolicyContract

class PolicyContract(Strict):
    """The interface a learned policy declares before it may be onboarded. Extended in task 10."""
    name: str
    version: str
    observation_keys: list[str] = Field(min_length=1)
    action_frame: str
    action_kind: Literal["delta_pose", "absolute_pose", "joint_delta"]
    action_unit_translation: Literal["m", "mm"]
    action_unit_rotation: Literal["rad", "deg"]
    rate_hz: float = Field(gt=0, le=100)
    max_step_mm: float = Field(gt=0)


# ------------------------------------------------------------------------------ the site file

class SafetyProfileSpec(Strict):
    name: str
    version: str
    free_step_mm: float
    free_step_deg: float
    near_step_mm: float
    near_step_deg: float
    near_distance_mm: float
    workspace_half_mm: float
    tracking_stop_mm: float
    tracking_stop_deg: float
    joint_margin_deg: float
    joint_margin_mm: float
    clearance_mm: float


class NearObject(Strict):
    name: str
    offset_mm: list[float] = Field(min_length=3, max_length=3)


class NamedPoseSpec(Strict):
    description: str = ""
    offset_mm: list[float] | None = Field(default=None, min_length=3, max_length=3)
    rotate_deg: list[float] | None = Field(default=None, min_length=3, max_length=3)
    joints_si: list[float] | None = None


class JawSpec(Strict):
    open: float
    closed: float


class SiteFile(Strict):
    """sites/<name>.json. This project's own file, so this project's own schema."""
    name: str
    description: str = ""
    arm: str
    generation: Literal["Classic", "Si"]
    software_release: str = ""
    kinematic_file: str
    instrument_file: str
    arm_file: str = ""
    system_file: str = ""
    frames: list[str] = Field(default_factory=list)
    start_joints_si: list[float]
    up_axis: list[float] = Field(min_length=3, max_length=3)
    pad_below_start_mm: float
    near_objects: list[NearObject] = Field(default_factory=list)
    jaw_deg: JawSpec
    safety_profile: SafetyProfileSpec
    named_poses: dict[str, NamedPoseSpec]
