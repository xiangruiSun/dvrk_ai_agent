"""Is a proposed change sensible? The checks a JSON Schema cannot make.

A patch goes through four stages, each of which can stop it:

    allowlist     every path is one this kind of file may have changed, with the right operation
                  and the right kind of value
    file_schema   the patched copy still satisfies the schema of its file type, and has no schema
                  error the file did not already have
    physical      units, plausible range, step size, rotations, frames, instruments, the
                  direction of a limit change, evidence
    ok

The ranges and step sizes below are declared by hand, per path. They are deliberately narrow:
a change outside them is not impossible, it is just not something one patch should do unseen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from pydantic import ValidationError

from ..robot import config as cfg
from ..robot import jsonc
from ..robot.frames import FrameGraph
from ..robot.kinematics import rotvec
from .schemas import ConfigPatch, SiteFile
from .units import QUANTITY, UnitError, convert


@dataclass(frozen=True)
class PathRule:
    pattern: str                       # segments separated by "/", "*" matches one segment
    kind: str                          # number | vector3 | matrix3 | transform | flag | choice | frame | instrument
    unit: str = "none"                 # the unit the file stores the value in
    lo: float | None = None            # plausible range, in the file's unit
    hi: float | None = None
    step: float | None = None          # largest change one patch may make, in the file's unit
    step_deg: float | None = None      # the same for a rotation
    limit: str = ""                    # lower_is_tighter | higher_is_tighter, for a safety limit
    choices: tuple[str, ...] = ()
    ops: tuple[str, ...] = ("replace",)
    invalidates_calibration: bool = False
    what: str = ""


_BOTH = ("replace", "add")
_TELEOP = "consoles/*/teleop_PSMs/*/configure_parameter/"

ALLOW: dict[str, list[PathRule]] = {
    "vx-site": [
        PathRule("safety_profile/free_step_mm", "number", "mm", 0.5, 5.0, 2.0, limit="lower_is_tighter",
                 what="largest step in free space"),
        PathRule("safety_profile/free_step_deg", "number", "deg", 0.5, 5.0, 2.0, limit="lower_is_tighter",
                 what="largest rotation step in free space"),
        PathRule("safety_profile/near_step_mm", "number", "mm", 0.2, 2.0, 1.0, limit="lower_is_tighter",
                 what="largest step near the needle or the pad"),
        PathRule("safety_profile/near_step_deg", "number", "deg", 0.5, 3.0, 1.5, limit="lower_is_tighter",
                 what="largest rotation step near the needle or the pad"),
        PathRule("safety_profile/near_distance_mm", "number", "mm", 10.0, 60.0, 10.0, limit="higher_is_tighter",
                 what="distance from the needle or pad inside which the smaller step applies"),
        PathRule("safety_profile/workspace_half_mm", "number", "mm", 10.0, 60.0, 10.0, limit="lower_is_tighter",
                 what="half width of the local workspace around the start pose"),
        PathRule("safety_profile/tracking_stop_mm", "number", "mm", 0.5, 2.0, 1.0, limit="lower_is_tighter",
                 what="position error at which a motion is stopped"),
        PathRule("safety_profile/tracking_stop_deg", "number", "deg", 0.5, 2.0, 1.0, limit="lower_is_tighter",
                 what="rotation error at which a motion is stopped"),
        PathRule("safety_profile/joint_margin_deg", "number", "deg", 1.0, 20.0, 5.0, limit="higher_is_tighter",
                 what="room kept to a revolute joint limit"),
        PathRule("safety_profile/joint_margin_mm", "number", "mm", 1.0, 20.0, 5.0, limit="higher_is_tighter",
                 what="room kept to the insertion limit"),
        PathRule("safety_profile/clearance_mm", "number", "mm", 1.0, 15.0, 5.0, limit="higher_is_tighter",
                 what="height the tip must keep over the pad"),
        PathRule("pad_below_start_mm", "number", "mm", 5.0, 100.0, 10.0, what="distance of the pad below the start pose"),
        PathRule("named_poses/*/offset_mm", "vector3", "mm", -60.0, 60.0, 20.0,
                 what="offset of a named pose from the start pose"),
        PathRule("near_objects/*/offset_mm", "vector3", "mm", -60.0, 60.0, 20.0,
                 what="offset of the needle or a phantom feature from the start pose"),
    ],
    "dvrk-system": [
        PathRule("arms/*/base_frame/transform", "transform", "m", -2.0, 2.0, 0.05, step_deg=10.0,
                 invalidates_calibration=True, what="4 x 4 transform of the arm's base in its reference frame"),
        PathRule("arms/*/base_frame/transform/translation", "vector3", "m", -2.0, 2.0, 0.05,
                 invalidates_calibration=True, what="translation part of the base frame transform"),
        PathRule("arms/*/base_frame/transform/rotation", "matrix3", "none", step_deg=10.0,
                 invalidates_calibration=True, what="3 x 3 rotation part of the base frame transform"),
        PathRule("arms/*/base_frame/reference_frame", "frame", invalidates_calibration=True,
                 what="name of the frame the base transform is expressed in"),
        PathRule("arms/*/period", "number", "s", 0.0002, 0.01, 0.005, ops=_BOTH, what="period of the arm component"),
        PathRule(_TELEOP + "scale", "number", "ratio", 0.05, 1.0, 0.3, ops=_BOTH,
                 what="teleoperation scale, instrument motion per hand motion"),
        PathRule(_TELEOP + "start_orientation_tolerance", "number", "rad", 0.0, 0.5, 0.2, ops=_BOTH,
                 what="orientation error allowed between hand and instrument when teleoperation starts"),
        PathRule(_TELEOP + "rotation_locked", "flag", ops=_BOTH, what="lock the instrument's orientation"),
        PathRule(_TELEOP + "translation_locked", "flag", ops=_BOTH, what="lock the instrument's position"),
        PathRule(_TELEOP + "ignore_jaw", "flag", ops=_BOTH, what="do not drive the jaw from the gripper"),
        PathRule(_TELEOP + "align_MTM", "flag", ops=_BOTH, what="move the hand controller to match the instrument"),
        PathRule("settings/audio_volume", "number", "ratio", 0.05, 1.0, None, ops=_BOTH, what="volume of the audio cues"),
        PathRule("settings/chatty", "flag", ops=_BOTH, what="speak state changes aloud"),
    ],
    "dvrk-psm": [
        PathRule("tool_detection", "choice", choices=("AUTOMATIC", "MANUAL", "FIXED"), ops=_BOTH,
                 what="how the instrument type is determined"),
        PathRule("tool", "instrument", ops=_BOTH, invalidates_calibration=True,
                 what="instrument as NAME:MODEL, used when tool_detection is FIXED"),
        PathRule("mounting_pitch", "number", "rad", -1.5708, 1.5708, 1.3, ops=_BOTH,
                 what="pitch of the arm's mounting, for the direction of gravity"),
        PathRule("re_home", "flag", ops=_BOTH, what="force homing again at start-up"),
    ],
}
_VIRTUAL = {"translation": "vector3", "rotation": "matrix3"}


def find_rule(file_type: str, path: str) -> PathRule | None:
    segs = path.split("/")
    for rule in ALLOW.get(file_type, []):
        pat = rule.pattern.split("/")
        if len(pat) == len(segs) and all(p == "*" or p == s for p, s in zip(pat, segs)):
            return rule
    return None


def _virtual(path: str, rule: PathRule) -> tuple[str, str]:
    """(path of the stored value, part) for the two views of a transform."""
    last = path.rsplit("/", 1)[-1]
    if rule.kind in ("vector3", "matrix3") and last in _VIRTUAL and "/transform/" in path + "/":
        base = path.rsplit("/", 1)[0]
        if base.endswith("transform"):
            return base, last
    return path, ""


def read(doc: Any, path: str, rule: PathRule) -> Any:
    base, part = _virtual(path, rule)
    value = cfg.get(doc, base)
    if value is cfg.MISSING or not part:
        return value
    m = np.array(value, dtype=float)
    return [float(x) for x in m[:3, 3]] if part == "translation" else [[float(x) for x in row] for row in m[:3, :3]]


def _children(node: Any) -> list[tuple[str, Any]]:
    if isinstance(node, dict):
        return list(node.items())
    if isinstance(node, list):
        out = []
        for i, it in enumerate(node):
            name = str(i)
            if isinstance(it, dict):
                name = it.get("name") or (f"{it.get('MTM')}-{it.get('PSM')}" if "MTM" in it or "PSM" in it else str(i))
            out.append((name, it))
        return out
    return []


def expand(rule: PathRule, doc: Any) -> Iterator[tuple[str, Any]]:
    """Concrete paths of this file the rule covers, with the value now (cfg.MISSING if unset)."""
    pat = rule.pattern.split("/")

    def walk(node: Any, i: int, done: list[str]) -> Iterator[tuple[str, Any]]:
        if i == len(pat):
            yield "/".join(done), node
            return
        seg = pat[i]
        if seg == "*":
            for name, child in _children(node):
                yield from walk(child, i + 1, done + [name])
        elif isinstance(node, dict) and seg in node:
            yield from walk(node[seg], i + 1, done + [seg])
        elif i == len(pat) - 1 and seg in _VIRTUAL and isinstance(node, list):
            path = "/".join(done + [seg])
            yield path, read(doc, path, rule)
        elif isinstance(node, dict) and "add" in rule.ops and "*" not in pat[i:]:
            yield "/".join(done + pat[i:]), cfg.MISSING

    yield from walk(doc, 0, [])


def describe_paths(file_type: str, doc: Any) -> list[dict[str, Any]]:
    """The paths of this file a patch may touch, for the person or the model writing one."""
    out = []
    for rule in ALLOW.get(file_type, []):
        for path, now in expand(rule, doc):
            row: dict[str, Any] = {"path": path, "what": rule.what, "takes": rule.kind,
                                   "now": "(not set)" if now is cfg.MISSING else now,
                                   "operation": "add" if now is cfg.MISSING else "replace"}
            if rule.kind in ("number", "vector3", "transform"):
                row["file_stores"] = rule.unit
                row["accepted_units"] = [u for u, q in QUANTITY.items() if q == QUANTITY[rule.unit]]
            if rule.choices:
                row["choices"] = list(rule.choices)
            if rule.limit:
                row["safety_limit"] = rule.limit.replace("_", " ")
            out.append(row)
    return out


# ------------------------------------------------------------------------------ instruments

class InstrumentSet:
    """The instruments the dVRK software knows: share/tool/index.json and the files it names."""

    def __init__(self, tool_dir: Path | str):
        self.folder = Path(tool_dir)
        index = jsonc.load(self.folder / "index.json")
        self.entries = index.get("instruments", [])

    def file_for(self, tool: str) -> str | None:
        """File of "NAME:MODEL" (a version in square brackets is ignored), or None."""
        name, _, model = tool.partition(":")
        model = model.split("[")[0]
        for e in self.entries:
            if e.get("model") == model and name in e.get("names", []):
                return e["file"] if (self.folder / e["file"]).exists() else None
        return None


# ------------------------------------------------------------------------------ validation

@dataclass
class RuleContext:
    target: str                                  # the file the request is about
    file_type: str
    doc: Any                                     # that file as it is now
    schemas: cfg.SchemaSet | None = None
    instruments: InstrumentSet | None = None
    frames_declared: frozenset[str] = frozenset()
    evidence: frozenset[str] = frozenset()       # ids of records retrieved or measured in this run


@dataclass
class Validation:
    stage: str                                   # target | allowlist | file_schema | physical | ok
    errors: list[str] = field(default_factory=list)
    patched: Any = None
    changes: list[dict[str, Any]] = field(default_factory=list)
    limit_effect: str = "none"                   # none | tighten | loosen, computed from the values
    invalidates_calibration: bool = False
    schema_errors_already_in_file: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.stage == "ok"


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _shape_problem(kind: str, value: Any, rule: PathRule) -> str | None:
    def rows(n: int) -> bool:
        return (isinstance(value, list) and len(value) == n
                and all(isinstance(r, list) and len(r) == n and all(_is_number(x) for x in r) for r in value))
    if kind == "number" and not _is_number(value):
        return "takes one number"
    if kind == "vector3" and not (isinstance(value, list) and len(value) == 3 and all(_is_number(x) for x in value)):
        return "takes a list of 3 numbers"
    if kind == "matrix3" and not rows(3):
        return "takes a 3 x 3 matrix (3 rows of 3 numbers)"
    if kind == "transform" and not rows(4):
        return "takes a 4 x 4 matrix (4 rows of 4 numbers)"
    if kind == "flag" and not isinstance(value, bool):
        return "takes true or false"
    if kind == "choice" and value not in rule.choices:
        return f"takes one of {', '.join(rule.choices)}"
    if kind in ("frame", "instrument") and not isinstance(value, str):
        return "takes text"
    return None


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.6g}"
    if isinstance(x, list):
        return "[" + ", ".join(_fmt(v) for v in x) + "]"
    return str(x)


def rotation_problem(R: np.ndarray, tol: float = 1e-4) -> str | None:
    """None if R is a rotation: orthonormal with determinant +1."""
    if not np.all(np.isfinite(R)):
        return "contains a value that is not a finite number"
    off = float(np.abs(R @ R.T - np.eye(3)).max())
    if off > tol:
        return f"is not orthonormal (R times its transpose differs from the identity by {off:.3g})"
    det = float(np.linalg.det(R))
    if det < 0:
        return f"has determinant {det:.3f}: it is a reflection, not a rotation"
    return None


def _angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return math.degrees(float(np.linalg.norm(rotvec(Rb @ Ra.T))))


def validate_patch(patch: ConfigPatch, ctx: RuleContext) -> Validation:
    v = Validation(stage="target")
    if patch.target != ctx.target:
        v.errors.append(f"target is {patch.target!r} but the request is about {ctx.target!r}")
        return v

    # ---- allowlist: path, operation, kind of value
    v.stage = "allowlist"
    plan: list[tuple[int, Any, PathRule, Any]] = []            # (number, operation, rule, value now)
    for i, op in enumerate(patch.operations, 1):
        where = f"operation {i} ({op.path})"
        rule = find_rule(ctx.file_type, op.path)
        if rule is None:
            v.errors.append(f"{where}: this path may not be changed by a patch to a {ctx.file_type or 'file of this'} "
                            f"file. Use only the paths listed as allowed.")
            continue
        try:
            old = read(ctx.doc, op.path, rule)
        except (cfg.PathError, IndexError, ValueError) as exc:
            v.errors.append(f"{where}: {exc}")
            continue
        if op.op not in rule.ops:
            v.errors.append(f"{where}: this path can only be changed with {' or '.join(rule.ops)}")
            continue
        if op.op == "replace" and old is cfg.MISSING:
            v.errors.append(f"{where}: there is no value to replace at this path"
                            + ("; use op 'add'" if "add" in rule.ops else ""))
            continue
        if op.op == "add" and old is not cfg.MISSING:
            v.errors.append(f"{where}: this path already has the value {_fmt(old)}; use op 'replace'")
            continue
        problem = _shape_problem(rule.kind, op.value, rule)
        if problem:
            v.errors.append(f"{where}: this path {problem}")
            continue
        plan.append((i, op, rule, old))
    paths = [op.path for op in patch.operations]
    if len(set(paths)) != len(paths):
        v.errors.append("the same path appears in more than one operation")
    if v.errors:
        return v

    # ---- units: the dimension must match; the conversion is done here, never by the model
    unit_errors: list[str] = []
    writes: list[tuple[str, Any, bool]] = []
    work = cfg.applied(ctx.doc, [])
    for i, op, rule, old in plan:
        where = f"operation {i} ({op.path})"
        new: Any = op.value
        try:
            if rule.kind == "number":
                new = convert(op.value, op.unit, rule.unit)
            elif rule.kind == "vector3":
                new = [convert(x, op.unit, rule.unit) for x in op.value]
            elif rule.kind == "transform":
                new = [[float(x) for x in row] for row in op.value]
                for r in range(3):
                    new[r][3] = convert(op.value[r][3], op.unit, rule.unit)
            elif rule.kind == "matrix3":
                new = [[float(x) for x in row] for row in op.value]
        except UnitError as exc:
            unit_errors.append(f"{where}: {exc}")
            continue
        base, part = _virtual(op.path, rule)
        stored = new
        if part:
            m = np.array(cfg.get(work, base), dtype=float)
            if part == "translation":
                m[:3, 3] = new
            else:
                m[:3, :3] = np.array(new, dtype=float)
            stored = [[float(x) for x in row] for row in m]
        cfg.put(work, base, stored, create_parents=op.op == "add")
        writes.append((base, stored, op.op == "add"))
        v.changes.append({"operation": i, "path": op.path, "op": op.op, "kind": rule.kind, "unit": rule.unit,
                          "before": None if old is cfg.MISSING else old, "after": new,
                          "given": op.value, "given_unit": op.unit})
        v.invalidates_calibration |= rule.invalidates_calibration
    if unit_errors:
        v.stage, v.errors = "physical", unit_errors
        return v
    v.patched = work

    # ---- the file's own schema
    v.stage = "file_schema"
    if ctx.file_type == "vx-site":
        try:
            SiteFile.model_validate(work)
        except ValidationError as exc:
            v.errors += [f"{'/'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
    elif ctx.schemas is not None:
        before = ctx.schemas.errors(ctx.doc) or []
        after = ctx.schemas.errors(work)
        if after is None:
            v.errors.append("the patched file no longer names a schema of the dVRK project")
        else:
            v.schema_errors_already_in_file = [e for e in after if e in before]
            v.errors += [f"schema: {e}" for e in after if e not in before]
    if v.errors:
        return v

    # ---- physical
    v.stage = "physical"
    effects: list[str] = []
    for (i, op, rule, old), change in zip(plan, v.changes):
        where = f"operation {i} ({op.path})"
        new = change["after"]
        u = "" if rule.unit in ("none", "ratio") else f" {rule.unit}"
        if rule.kind == "number":
            if (rule.lo is not None and new < rule.lo - 1e-12) or (rule.hi is not None and new > rule.hi + 1e-12):
                v.errors.append(f"{where}: {_fmt(new)}{u} is outside the plausible range {_fmt(rule.lo)} to "
                                f"{_fmt(rule.hi)}{u} for this path" + _unit_hint(op, rule, [new]))
            elif old is not cfg.MISSING and rule.step is not None and abs(new - old) > rule.step + 1e-9:
                v.errors.append(f"{where}: a change from {_fmt(float(old))} to {_fmt(new)}{u} is larger than the "
                                f"{_fmt(rule.step)}{u} one patch may change this value by")
            if rule.limit and old is not cfg.MISSING and abs(new - old) > 1e-12:
                tighter = (new < old) == (rule.limit == "lower_is_tighter")
                effects.append("tighten" if tighter else "loosen")
                change["limit"] = "tightened" if tighter else "loosened"
        elif rule.kind == "vector3":
            vec = np.array(new, dtype=float)
            if (rule.lo is not None and float(vec.min()) < rule.lo - 1e-12) or \
               (rule.hi is not None and float(vec.max()) > rule.hi + 1e-12):
                v.errors.append(f"{where}: {_fmt(new)}{u} has a component outside the plausible range {_fmt(rule.lo)} "
                                f"to {_fmt(rule.hi)}{u}" + _unit_hint(op, rule, list(vec)))
            elif old is not cfg.MISSING and rule.step is not None:
                moved = float(np.linalg.norm(vec - np.array(old, dtype=float)))
                if moved > rule.step + 1e-9:
                    v.errors.append(f"{where}: the value moves by {_fmt(moved)}{u}, more than the {_fmt(rule.step)}{u} "
                                    f"one patch may move it by")
        elif rule.kind in ("transform", "matrix3"):
            m = np.array(new, dtype=float)
            R = m[:3, :3]
            if rule.kind == "transform" and not np.allclose(m[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
                v.errors.append(f"{where}: the last row of a transform must be 0 0 0 1, it is {_fmt(list(m[3]))}")
            bad = rotation_problem(R)
            if bad:
                v.errors.append(f"{where}: the rotation part {bad}")
            elif old is not cfg.MISSING:
                Rold = np.array(old, dtype=float)[:3, :3]
                turned = _angle_deg(Rold, R)
                if rule.step_deg is not None and turned > rule.step_deg + 1e-6:
                    v.errors.append(f"{where}: the rotation changes by {turned:.1f} deg, more than the "
                                    f"{_fmt(rule.step_deg)} deg one patch may change it by")
            if rule.kind == "transform":
                t = m[:3, 3]
                if float(np.abs(t).max()) > (rule.hi or math.inf) + 1e-12:
                    v.errors.append(f"{where}: translation {_fmt(list(t))} m is outside the plausible range "
                                    f"{_fmt(rule.lo)} to {_fmt(rule.hi)} m" + _unit_hint(op, rule, list(t)))
                elif old is not cfg.MISSING and rule.step is not None:
                    moved = float(np.linalg.norm(t - np.array(old, dtype=float)[:3, 3]))
                    if moved > rule.step + 1e-9:
                        v.errors.append(f"{where}: the translation moves by {_fmt(moved)} m, more than the "
                                        f"{_fmt(rule.step)} m one patch may move it by")
        elif rule.kind == "instrument":
            if ctx.instruments is not None and ctx.instruments.file_for(new) is None:
                v.errors.append(f"{where}: no instrument {new!r} in the instrument set (share/tool/index.json)")

    if ctx.file_type == "dvrk-system":
        base = FrameGraph.from_system(ctx.doc)
        known = set(ctx.frames_declared) | base.frames | set(base.parents.values())
        g = FrameGraph.from_system(work, known)
        for name in g.undeclared():
            v.errors.append(f"frame {name!r} is not a frame of this site (frames: {', '.join(sorted(known))})")
        loop = g.cycle()
        if loop:
            v.errors.append(f"the patch makes a frame its own ancestor: {' -> '.join(loop)}")
    if ctx.file_type == "dvrk-psm":
        if work.get("tool_detection") == "FIXED":
            tool = work.get("tool")
            if not tool:
                v.errors.append("tool_detection is FIXED, so the file must name the instrument in 'tool'")
            elif ctx.instruments is not None and ctx.instruments.file_for(tool) is None and \
                    not any(c["path"] == "tool" for c in v.changes):
                v.errors.append(f"tool_detection is FIXED but the instrument {tool!r} is not in the instrument set")
    if ctx.file_type == "vx-site":
        sp = work["safety_profile"]
        for a, b in (("near_step_mm", "free_step_mm"), ("near_step_deg", "free_step_deg")):
            if sp[a] > sp[b] + 1e-12:
                v.errors.append(f"safety_profile/{a} ({_fmt(float(sp[a]))}) would be larger than safety_profile/{b} "
                                f"({_fmt(float(sp[b]))}): the step near the needle cannot exceed the free-space step")

    v.limit_effect = "loosen" if "loosen" in effects else ("tighten" if "tighten" in effects else "none")
    if v.limit_effect != patch.limit_change:
        detail = "; ".join(f"{c['path']} is {c['limit']} ({_fmt(float(c['before']))} to {_fmt(float(c['after']))})"
                           for c in v.changes if "limit" in c) or "no safety limit is changed"
        v.errors.append(f"limit_change is {patch.limit_change!r} but the values say {v.limit_effect!r}: {detail}."
                        + (" A loosened limit must be declared as 'loosen'." if v.limit_effect == "loosen" else ""))
    unknown = [e for e in patch.evidence if e not in ctx.evidence]
    if unknown:
        v.errors.append(f"evidence {', '.join(repr(e) for e in unknown)} is not a record of this run "
                        f"(records: {', '.join(sorted(ctx.evidence)) or 'none'})")
    if not v.errors:
        v.stage = "ok"
    return v


def _unit_hint(op: Any, rule: PathRule, values: list[float]) -> str:
    """If the number would be plausible in the other unit of the same quantity, say so. The
    model still has to state the unit; nothing is converted for it on a guess."""
    for other in (u for u, q in QUANTITY.items() if q == QUANTITY[rule.unit] and u != op.unit):
        try:
            given = op.value if isinstance(op.value, list) else [op.value]
            flat = [x[3] for x in given[:3]] if given and isinstance(given[0], list) else given
            alt = [convert(x, other, rule.unit) for x in flat]
        except (UnitError, TypeError, IndexError):
            continue
        if all((rule.lo is None or a >= rule.lo) and (rule.hi is None or a <= rule.hi) for a in alt):
            return (f". The number would be plausible in {other!r}: if the request gave it in {other}, state unit "
                    f"{other!r} and leave the number as it is.")
    return ""
