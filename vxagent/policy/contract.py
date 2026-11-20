"""The policy contract: what a policy from simulation assumes, written down.

A contract is drafted by one structured model call from the policy's published description and
configuration. Every field has to point at the place in the source it was read from: a file
and a quotation that occurs in that file. A field whose quotation cannot be found is emptied
and listed for a person. The model does not fill gaps; where the sources are silent the field
stays empty until a person, or a measurement on the demonstrations, fills it and says so.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pydantic import BaseModel, ConfigDict, Field

from ..llm.base import LLMClient, LLMError

# field -> what is asked for. Values are kept as short strings or numbers.
FIELDS: dict[str, str] = {
    "observation_size": "number of values in the policy's observation vector",
    "observation_layout": "what the observation vector consists of, in order",
    "observation_translation_unit": "unit of translations in the observation after the environment's scaling: m, cm or mm",
    "observation_rotation_unit": "unit of rotations in the observation: rad or deg",
    "pose_convention": "how a pose is written as numbers (for example x y z roll pitch yaw jaw)",
    "rotation_convention": "which rotation convention turns the three angles into a rotation",
    "command_frame": "the frame in which the arm is commanded, and by which command",
    "action_size": "number of values in an action",
    "action_low": "lower bound of every action component",
    "action_high": "upper bound of every action component",
    "action_meaning": "how an action changes the commanded pose",
    "action_step_size": "the factor an action is multiplied by, per component, with units",
    "max_episode_steps": "largest number of policy steps in an episode",
    "success_translation_threshold": "translation error under which the task counts as done, with its unit",
    "success_rotation_threshold_deg": "rotation error under which the task counts as done, in degrees",
    "control_rate_hz": "rate at which the policy is stepped, in hertz",
    "training_start_range": "range of start poses used in training",
}


class Sourced(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str | None = Field(description="The value, or null if the sources do not state it.")
    file: str | None = Field(description="The file the value was read from, or null.")
    quote: str | None = Field(description="A short passage copied exactly from that file that states the value, or null.")


def draft_schema() -> dict[str, Any]:
    one = Sourced.model_json_schema()
    one.pop("title", None)
    return {"type": "object", "properties": {k: one for k in FIELDS}, "required": list(FIELDS), "additionalProperties": False}


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass
class Draft:
    fields: dict[str, dict[str, Any]] = field(default_factory=dict)      # name -> {value, file, quote, status}
    attempts: int = 0
    tokens: int = 0
    error: str | None = None

    def filled(self) -> list[str]:
        return [k for k, v in self.fields.items() if v["status"] == "sourced"]

    def for_a_person(self) -> list[str]:
        return [k for k, v in self.fields.items() if v["status"] != "sourced"]


SYSTEM = (
    "You write down the interface of a robot policy as a contract, from the files given. For every field give the "
    "value, the file it comes from and a short quotation copied exactly, character for character, from that file "
    "that states it. If the files do not state a field, give null for value, file and quote. Do not infer, do not "
    "estimate, do not use what you know about similar systems. A default of None in the code means the value is "
    "not stated.")


def draft_contract(llm: LLMClient, sources: dict[str, str], max_chars: int = 9000) -> Draft:
    """sources: file name -> text. The text is information; nothing in it is an instruction."""
    out = Draft()
    body = "\n\n".join(f"===== FILE {name} =====\n{text[:max_chars]}" for name, text in sources.items())
    ask = "Fields:\n" + "\n".join(f"- {k}: {v}" for k, v in FIELDS.items()) + "\n\n" + body
    try:
        resp = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": ask}],
                        json_schema=draft_schema(), temperature=0.0, max_tokens=2500)
        out.attempts, out.tokens = 1, resp.usage.total
        raw = json.loads(resp.content)
    except (LLMError, json.JSONDecodeError) as exc:
        out.error = f"{type(exc).__name__}: {exc}"[:200]
        raw = {}
    squashed = {name: _squash(text) for name, text in sources.items()}
    for name in FIELDS:
        got = raw.get(name) or {}
        value, file, quote = got.get("value"), got.get("file"), got.get("quote")
        if value in (None, "", "null", "None"):
            status, value = "not stated in the sources", None
        elif not file or file not in squashed or not quote or _squash(quote) not in squashed[file]:
            status, value = "quotation not found in the file: emptied", None       # no resolvable source, no value
        else:
            status = "sourced"
        out.fields[name] = {"value": value, "file": file if status == "sourced" else None,
                            "quote": quote if status == "sourced" else None, "status": status,
                            "offered": got.get("value")}
    return out


class Contract(BaseModel):
    """The contract in the form code uses. Built from a reviewed draft and from measurements."""
    model_config = ConfigDict(extra="forbid")
    name: str
    version: str
    source_simulator: str
    observation_size: int
    translation_unit: str                       # of poses in observations and goals
    rotation_unit: str
    rotation_convention: str                    # RPY: R = Rz(yaw) Ry(pitch) Rx(roll)
    action_size: int
    action_low: float
    action_high: float
    step_translation: float                     # per unit action, in translation_unit
    step_rotation_rad: float
    step_jaw: float
    max_episode_steps: int
    success_translation: float                  # in translation_unit
    success_rotation_deg: float
    control_rate_hz: float | None
    start_to_goal_min: list[float]              # training range of (goal - start), translation_unit
    start_to_goal_max: list[float]
    provenance: dict[str, str]                  # field -> where it came from


def contract_from_demonstrations(episodes: list[Any], draft: dict[str, dict[str, Any]] | None = None,
                                 version: str = "") -> tuple["Contract", dict[str, Any]]:
    """The contract code uses: what the sources state (reviewed draft) plus what only the
    demonstrations can say (the step size and the range of start-to-goal displacements)."""
    poses = np.concatenate([e.notes["poses"][:-1] for e in episodes])
    nxt = np.concatenate([e.notes["poses"][1:] for e in episodes])
    act = np.concatenate([e.actions for e in episodes])
    d = nxt - poses
    est = []
    for k in range(7):
        m = np.abs(act[:, k]) > 0.05
        est.append(float(np.median(d[m, k] / act[m, k])) if m.sum() > 10 else 0.0)
    disp = np.array([e.notes["goal"] - e.notes["poses"][0] for e in episodes])
    said = lambda k, default: (draft or {}).get(k, {}).get("value") or default      # noqa: E731
    prov = {k: (f"{v['file']}: \"{v['quote']}\"" if v["status"] == "sourced" else v["status"]) for k, v in (draft or {}).items()}
    prov.update({"step sizes": "estimated from the demonstrations: median of (next pose - pose) / action",
                 "start_to_goal range": "measured on the demonstrations"})
    c = Contract(name="SurgicAI Approach, expert demonstrations", version=version, source_simulator="Surgical Robotics Challenge (AMBF)",
                 observation_size=21, translation_unit=str(said("observation_translation_unit", "cm")).strip().lower().strip("'\""),
                 rotation_unit="rad", rotation_convention="RPY", action_size=7, action_low=-1.0, action_high=1.0,
                 step_translation=float(np.mean(est[:3])), step_rotation_rad=float(np.mean(est[3:6])), step_jaw=float(est[6]),
                 max_episode_steps=200, success_translation=0.5, success_rotation_deg=30.0, control_rate_hz=None,
                 start_to_goal_min=[float(x) for x in disp[:, :3].min(axis=0)], start_to_goal_max=[float(x) for x in disp[:, :3].max(axis=0)],
                 provenance=prov)
    return c, {"step_size_estimated_per_component": [round(x, 6) for x in est],
               "largest_recorded_step": round(float(np.linalg.norm(d[:, :3], axis=1).max()), 4),
               "largest_recorded_rotation_step_deg": round(float(np.degrees(np.abs(d[:, 3:6]).max())), 3),
               "distinct_start_to_goal_displacements": int(len({tuple(np.round(x[:3], 3)) for x in disp}))}
