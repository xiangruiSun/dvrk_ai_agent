"""Similar recorded episodes.

A recording is far too large to compare by reading it. Each one is reduced by code to a short
vector of statistics, the vectors are put on a common scale, and "similar" means near in that
space. A neighbour is returned with the features that made it near, so the answer can be
checked by a person. The features are chosen by hand; nothing here is learned.

The recordings have no jaw signal, so there are no jaw events among the features.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import episodes as ep_store

FEATURES = ("duration_s", "psm1_path_mm", "psm2_path_mm", "psm1_mean_speed", "psm1_peak_speed", "psm2_mean_speed",
            "psm1_extent_mm", "psm2_extent_mm", "psm1_pauses", "psm2_share_of_travel")
PAUSE_SPEED = 0.002          # m/s: slower than this is standing still
PAUSE_S = 0.5


def features(ep: ep_store.Episode) -> dict[str, float]:
    t, cols = ep_store.load(ep)
    out: dict[str, float] = {"duration_s": float(t[-1] - t[0])}
    paths = {}
    for arm in ("PSM1", "PSM2"):
        p = np.stack([cols[f"{arm}_position_{a}"] for a in "xyz"], axis=1)
        v = np.linalg.norm(np.stack([cols[f"{arm}_velocity_linear_{a}"] for a in "xyz"], axis=1), axis=1)
        k = arm.lower()
        paths[arm] = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) * 1000
        out[f"{k}_path_mm"] = paths[arm]
        out[f"{k}_mean_speed"] = float(v.mean()) * 1000
        out[f"{k}_extent_mm"] = float(np.linalg.norm(p.max(axis=0) - p.min(axis=0))) * 1000
        if arm == "PSM1":
            out["psm1_peak_speed"] = float(v.max()) * 1000
            still = v < PAUSE_SPEED
            pauses, start = 0, None
            for i, s in enumerate(still):
                if s and start is None:
                    start = t[i]
                elif not s and start is not None:
                    pauses += int(t[i] - start >= PAUSE_S)
                    start = None
            out["psm1_pauses"] = float(pauses)
    out["psm2_share_of_travel"] = paths["PSM2"] / max(1e-9, paths["PSM1"] + paths["PSM2"])
    return {k: round(float(out[k]), 4) for k in FEATURES}


class EpisodeIndex:
    def __init__(self, base: Path | None = None, table: dict[str, dict[str, float]] | None = None):
        self.episodes = {e.episode_id: e for e in ep_store.list_episodes(base)}
        self.table = table or {eid: features(e) for eid, e in self.episodes.items()}
        self.ids = sorted(self.table)
        raw = np.array([[self.table[i][f] for f in FEATURES] for i in self.ids], dtype=float)
        self.mean, self.std = raw.mean(axis=0), np.where(raw.std(axis=0) > 1e-12, raw.std(axis=0), 1.0)
        self.z = (raw - self.mean) / self.std

    def task(self, episode_id: str) -> str:
        e = self.episodes.get(episode_id)
        return e.task if e else episode_id.split("_", 1)[1].rsplit("_", 1)[0].replace("_", " ")

    def nearest(self, episode_id: str, k: int = 3) -> list[dict[str, Any]]:
        i = self.ids.index(episode_id)
        d = np.linalg.norm(self.z - self.z[i], axis=1)
        order = [j for j in np.argsort(d) if j != i][:k]
        out = []
        for j in order:
            gap = np.abs(self.z[j] - self.z[i])
            close = [FEATURES[m] for m in np.argsort(gap)[:3]]
            out.append({"episode_id": self.ids[j], "task": self.task(self.ids[j]), "distance": round(float(d[j]), 3),
                        "closest_in": {f: {"this": self.table[episode_id][f], "that": self.table[self.ids[j]][f]} for f in close}})
        return out
