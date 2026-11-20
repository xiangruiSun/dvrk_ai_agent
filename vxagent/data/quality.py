"""One schema for recorded episodes from different sources, and the checks an episode has to
pass before it is used to train or to judge a policy.

A stream a source does not contain is absent. It is never filled in: the ROSMA recordings have
no jaw signal and no commands; the SurgicAI demonstrations have no time stamps and no operator.

Checks: time runs forward; gaps in the sample rate; joint values inside the arm's limits;
the magnitude of positions agrees with the declared unit; jumps from one sample to the next;
frozen signals; duplicates by content hash. An episode is accepted, accepted with flags, or
excluded, each with the reason.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..structured.units import convert
from . import episodes as rosma
from . import surgicai

JUMP_M = 0.010               # more than this between two samples is not motion, it is a fault in the record
GAP_FACTOR = 3.0             # a step in time this many times the usual one is a gap
FROZEN_SAMPLES = 100         # this many identical positions in a row is a frozen signal
EXCLUDING = ("time_runs_backwards", "unit_magnitude", "duplicate", "empty")


@dataclass
class Episode:
    source: str
    episode_id: str
    task: str
    operator: str | None             # None: the source does not say who or which session
    frame: str
    unit: str                        # of positions
    rate_hz: float | None            # None: the source has no time stamps
    t: np.ndarray | None             # seconds, as recorded (not repaired)
    position: np.ndarray             # N x 3, in `unit`
    joints: np.ndarray | None = None
    actions: np.ndarray | None = None
    absent: tuple[str, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(np.round(self.position, 9)).tobytes()).hexdigest()[:16]


def from_rosma(ep: rosma.Episode, arm: str = "PSM1") -> Episode:
    t, cols = rosma.load(ep)
    pos = np.stack([cols[f"{arm}_position_{a}"] for a in "xyz"], axis=1)
    joints = np.stack([cols[f"{arm}_joint_position_{i}"] for i in range(1, 7)], axis=1)
    raw = rosma.raw_times(ep)
    return Episode("ROSMA", ep.episode_id, ep.task, ep.operator, f"{arm} tool tip, frame of the recording", "m",
                   round(float(1.0 / np.median(np.diff(t)[np.diff(t) > 0])), 1), raw, pos, joints,
                   absent=("jaw", "commands", "camera"),
                   notes={"time_stamps_moved_by_one_second_on_reading": int(cols["_timestamps_repaired"][0])})


def from_surgicai(task: str, base: Any = None) -> list[Episode]:
    transitions = surgicai.load_task(task, base)
    out, start = [], 0
    for i, tr in enumerate(transitions):
        if float(np.asarray(tr["done"]).ravel()[0]) > 0.5:
            block = transitions[start:i + 1]
            poses = np.array([b["obs"]["achieved_goal"] for b in block] + [block[-1]["next_obs"]["achieved_goal"]], dtype=float)
            out.append(Episode("SurgicAI", f"{task}_{len(out):02d}", task, None, "PSM base frame of the simulator", "cm",
                               None, None, poses[:, :3], None, np.array([b["action"] for b in block], dtype=float),
                               absent=("time stamps", "operator", "joint values"),
                               notes={"poses": poses, "goal": np.asarray(block[0]["obs"]["desired_goal"], dtype=float),
                                      "success": bool(np.asarray(block[-1]["info"]["is_success"]))}))
            start = i + 1
    return out


def check(ep: Episode, joint_limits: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
    flags: list[dict[str, Any]] = []

    def flag(name: str, detail: str) -> None:
        flags.append({"check": name, "detail": detail})

    n = len(ep.position)
    if n < 2:
        flag("empty", f"{n} samples")
    if ep.t is not None and n >= 2:
        dt = np.diff(ep.t)
        back = int((dt < 0).sum())
        if back > max(1, 0.05 * n) or (back and float(dt.min()) < -1.5):
            flag("time_runs_backwards", f"{back} of {n - 1} steps go back in time, by up to {abs(float(dt.min())):.2f} s")
        elif back:
            flag("seconds_field_late", f"{back} samples carry a seconds field that is one sample late")
        usual = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0
        gaps = int((dt > GAP_FACTOR * usual).sum()) - back if usual > 0 else 0
        if gaps > 0:
            flag("sample_gaps", f"{gaps} gaps longer than {GAP_FACTOR:g} times the usual {usual * 1000:.0f} ms")
    if n >= 2:
        k = convert(1.0, ep.unit, "m")
        reach = float(np.median(np.linalg.norm(ep.position, axis=1))) * k
        if not 0.02 <= reach <= 0.5:
            flag("unit_magnitude", f"with unit {ep.unit!r} the tool tip would be {reach * 1000:.0f} mm from the origin "
                                   f"at the median; a dVRK PSM works between 20 and 500 mm")
        step = np.linalg.norm(np.diff(ep.position, axis=0), axis=1) * k
        if float(step.max()) > JUMP_M:
            i = int(step.argmax())
            flag("jump", f"{float(step.max()) * 1000:.1f} mm between samples {i} and {i + 1}")
        still, longest = 0, 0
        for s in step:
            still = still + 1 if s == 0.0 else 0
            longest = max(longest, still)
        if longest >= FROZEN_SAMPLES:
            flag("frozen", f"{longest} consecutive samples with exactly the same position")
    if ep.joints is not None and joint_limits is not None:
        lo, hi = joint_limits
        q = ep.joints[:, :len(lo)]
        out = int(((q < lo - 1e-6) | (q > hi + 1e-6)).any(axis=1).sum())
        if out:
            worst = int(np.argmax(np.maximum(q - hi, lo - q).max(axis=0)))
            flag("joint_limits", f"{out} of {n} samples have a joint outside the limits of the kinematic file "
                                 f"(most often joint {worst + 1})")
    excluded = [f["check"] for f in flags if f["check"] in EXCLUDING]
    return {"episode_id": ep.episode_id, "source": ep.source, "task": ep.task, "operator": ep.operator, "samples": n,
            "hash": ep.content_hash(), "absent": list(ep.absent), "flags": flags,
            "verdict": "excluded" if excluded else ("accepted with flags" if flags else "accepted"),
            "excluded_because": excluded}


def check_all(episodes: list[Episode], joint_limits: tuple[np.ndarray, np.ndarray] | None = None) -> list[dict[str, Any]]:
    """Every episode checked, and a later copy of an earlier one excluded as a duplicate."""
    out, seen = [], {}
    for ep in episodes:
        row = check(ep, joint_limits if ep.joints is not None else None)
        if row["hash"] in seen:
            row["flags"].append({"check": "duplicate", "detail": f"same content as {seen[row['hash']]}"})
            row["verdict"], row["excluded_because"] = "excluded", row["excluded_because"] + ["duplicate"]
        else:
            seen[row["hash"]] = ep.episode_id
        out.append(row)
    return out


def split_by_operator(rows: list[dict[str, Any]], eval_share: float = 0.25) -> dict[str, Any]:
    """Train and evaluation sets split by operator, never by episode. Episodes with no operator
    (one source, one unnamed session) all go to the training side, and that is said."""
    usable = [r for r in rows if r["verdict"] != "excluded"]
    operators = sorted({r["operator"] for r in usable if r["operator"]})
    n_eval = max(1, round(len(operators) * eval_share)) if len(operators) > 1 else 0
    eval_ops = set(operators[-n_eval:]) if n_eval else set()
    train = [r["episode_id"] for r in usable if r["operator"] not in eval_ops]
    evaluation = [r["episode_id"] for r in usable if r["operator"] in eval_ops]
    train_ops = {r["operator"] for r in usable if r["operator"] and r["operator"] not in eval_ops}
    return {"by": "operator", "train_operators": sorted(train_ops), "evaluation_operators": sorted(eval_ops),
            "operators_on_both_sides": sorted(train_ops & eval_ops), "train": train, "evaluation": evaluation,
            "without_an_operator": sorted({r["source"] for r in usable if not r["operator"]}),
            "note": "episodes whose source names no operator or session are kept on the training side as one session"}


def card(rows: list[dict[str, Any]], split: dict[str, Any], sources: dict[str, dict[str, str]]) -> str:
    """The dataset card, generated from the checks. Markdown."""
    lines = ["# Dataset card", "", "Generated by code from the episode checks. Nothing here is typed by hand.", ""]
    for name, meta in sources.items():
        mine = [r for r in rows if r["source"] == name]
        tasks: dict[str, int] = {}
        for r in mine:
            tasks[r["task"]] = tasks.get(r["task"], 0) + 1
        lines += [f"## {name}", "", f"- Source: {meta['source']}", f"- Licence: {meta['licence']}",
                  f"- Episodes: {len(mine)} (" + ", ".join(f"{t}: {c}" for t, c in sorted(tasks.items())) + ")",
                  f"- Accepted: {sum(r['verdict'] == 'accepted' for r in mine)}, accepted with flags: "
                  f"{sum(r['verdict'] == 'accepted with flags' for r in mine)}, excluded: {sum(r['verdict'] == 'excluded' for r in mine)}",
                  f"- Streams the source does not contain: {', '.join(mine[0]['absent']) if mine else 'n/a'}",
                  f"- What it cannot show: {meta['cannot_show']}", ""]
        reasons: dict[str, int] = {}
        for r in mine:
            for f in r["flags"]:
                reasons[f["check"]] = reasons.get(f["check"], 0) + 1
        if reasons:
            lines += ["Flags: " + ", ".join(f"{k}: {v}" for k, v in sorted(reasons.items())), ""]
    lines += ["## Split", "", f"- By {split['by']}, never by episode.",
              f"- Training operators: {', '.join(split['train_operators']) or 'none'} ({len(split['train'])} episodes)",
              f"- Evaluation operators: {', '.join(split['evaluation_operators']) or 'none'} ({len(split['evaluation'])} episodes)",
              f"- Operators on both sides: {len(split['operators_on_both_sides'])}",
              f"- Sources without an operator: {', '.join(split['without_an_operator']) or 'none'}. {split['note'].capitalize()}.", ""]
    return "\n".join(lines)
