"""Recorded dVRK episodes from the ROSMA dataset (Zenodo 3932964, CC BY 4.0).

Each recording is a CSV sampled at about 50 Hz with, for MTML, MTMR, PSM1 and PSM2: Cartesian
position and orientation, linear and angular velocity, wrench, and joint position, velocity and
effort. There is no jaw signal. File names are <operator>_<task>_<trial>.csv, and scores.csv
gives the completion time and error count of each trial.

Recordings are large, so tools never return them. They return statistics computed here, and a
bounded window of one signal on request.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_settings

NAME = re.compile(r"^(X\d{2})_(.+)_(\d{2})$")
ARMS = ("PSM1", "PSM2", "MTML", "MTMR")
SIGNALS = {"position": ("position_x", "position_y", "position_z"),
           "velocity_linear": ("velocity_linear_x", "velocity_linear_y", "velocity_linear_z"),
           "joint_position": tuple(f"joint_position_{i}" for i in range(1, 7))}


@dataclass
class Episode:
    episode_id: str
    operator: str
    task: str
    trial: int
    path: Path


def _dir(base: Path | None = None) -> Path:
    return base or get_settings().rosma_dir


def list_episodes(base: Path | None = None) -> list[Episode]:
    out = []
    for p in sorted(_dir(base).glob("X*.csv")):
        m = NAME.match(p.stem)
        if m:
            out.append(Episode(p.stem, m.group(1), m.group(2).replace("_", " "), int(m.group(3)), p))
    return out


def get_episode(episode_id: str, base: Path | None = None) -> Episode | None:
    return next((e for e in list_episodes(base) if e.episode_id == episode_id), None)


def _time(stamp: str) -> float:
    """The dataset writes the milliseconds without leading zeros: "17:15:25.2" is 25.002 s, not
    25.2 s. (A first version of this function read the fraction as a decimal and put about one
    sample in ten at the wrong time.)"""
    base, _, ms = stamp.rpartition(".")
    return datetime.strptime(base, "%Y-%m-%d.%H:%M:%S").timestamp() + int(ms) / 1000.0


def _repair(times: list[float]) -> tuple[np.ndarray, int]:
    """In some samples the seconds field is one sample late at a roll-over ("27.982" followed by
    "27.2" where "28.2" is meant), which makes time run back by 0.98 s for one sample. Such a
    sample is moved forward by one second. Returns the times and how many were moved."""
    t = np.array(times, dtype=float)
    moved = 0
    for i in range(1, len(t)):
        if t[i] < t[i - 1] - 0.5:
            t[i] += 1.0
            moved += 1
    return t, moved


def load(ep: Episode) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(time in seconds from the first sample, columns by name). The dataset's trailing summary
    row and any row that does not parse are dropped."""
    with ep.path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        times: list[float] = []
        rows: list[list[float]] = []
        width = len(header)
        for row in reader:
            if len(row) < width or row[0].startswith("Date"):
                continue
            try:
                t = _time(row[0])
                vals = [float(x) for x in row[1:width]]
            except ValueError:
                continue
            times.append(t)
            rows.append(vals)
    data = np.array(rows, dtype=float)
    t, moved = _repair(times)
    cols = {name: data[:, i] for i, name in enumerate(header[1:width])}
    cols["_timestamps_repaired"] = np.array([moved])
    return t - t[0], cols


def _xyz(cols: dict[str, np.ndarray], arm: str, signal: str) -> np.ndarray:
    return np.stack([cols[f"{arm}_{s}"] for s in SIGNALS[signal]], axis=1)


def compute_stats(ep: Episode) -> dict[str, Any]:
    t, cols = load(ep)
    dt = np.diff(t)
    out: dict[str, Any] = {
        "episode_id": ep.episode_id, "operator": ep.operator, "task": ep.task, "trial": ep.trial,
        "samples": int(len(t)), "duration_s": round(float(t[-1] - t[0]), 3),
        "sample_rate_hz": round(float(1.0 / np.median(dt[dt > 0])), 1) if np.any(dt > 0) else None,
        "signals_recorded": "position, orientation, velocity, wrench and joints of MTML, MTMR, PSM1, PSM2",
        "signals_absent": "jaw angle, camera calibration",
        "timestamps_repaired": int(cols["_timestamps_repaired"][0]),
        "time_runs_forward": bool(np.all(dt >= 0)),
    }
    for arm in ("PSM1", "PSM2"):
        p = _xyz(cols, arm, "position")
        v = _xyz(cols, arm, "velocity_linear")
        speed = np.linalg.norm(v, axis=1)
        k = arm.lower()
        out[f"{k}_path_length_mm"] = round(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) * 1000, 1)
        out[f"{k}_peak_speed_mm_per_s"] = round(float(speed.max()) * 1000, 1)
        out[f"{k}_mean_speed_mm_per_s"] = round(float(speed.mean()) * 1000, 1)
        extent = p.max(axis=0) - p.min(axis=0)
        out[f"{k}_workspace_extent_mm"] = [round(float(x) * 1000, 1) for x in extent]
    return out


def stats(ep: Episode) -> dict[str, Any]:
    """compute_stats with a cache keyed on the file's size and modification time."""
    cache_file = get_settings().runtime_dir / "episode_stats.json"
    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    st = ep.path.stat()
    key = f"v2:{ep.path.name}:{st.st_size}:{int(st.st_mtime)}"      # v2: milliseconds read correctly
    if key not in cache:
        cache[key] = compute_stats(ep)
        cache_file.write_text(json.dumps(cache), encoding="utf-8")
    return cache[key]


def scores(base: Path | None = None) -> dict[str, dict[str, float]]:
    """The dataset's own completion time, error count and score per trial."""
    path = _dir(base) / "scores.csv"
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                out[row["File_name"]] = {"dataset_time_s": float(row["Time"]), "dataset_errors": int(float(row["Errors"])),
                                         "dataset_score": float(row["Score"])}
            except (KeyError, ValueError):
                continue
    return out


def window(ep: Episode, arm: str, signal: str, start_s: float, count: int) -> dict[str, Any]:
    """A bounded slice of one signal: at most `count` samples from start_s."""
    t, cols = load(ep)
    i0 = int(np.searchsorted(t, start_s))
    sl = slice(i0, min(len(t), i0 + count))
    values = _xyz(cols, arm, signal)[sl]
    scale, unit = (1000.0, "mm") if signal == "position" else ((1000.0, "mm_per_s") if signal == "velocity_linear" else (1.0, "si"))
    return {"episode_id": ep.episode_id, "arm": arm, "signal": signal, "unit": unit,
            "t_s": [round(float(x), 3) for x in t[sl]],
            "values": [[round(float(x) * scale, 3) for x in row] for row in values],
            "samples_returned": int(values.shape[0]), "samples_in_episode": int(len(t))}
