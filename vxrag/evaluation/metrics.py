"""Retrieval metrics over ranked units, and the paired bootstrap used to compare two runs."""
from __future__ import annotations

import math

import numpy as np


def dcg(grades: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    """Gain 2^grade - 1, discount 1/log2(rank + 1), ideal ordering from the labelled units."""
    ideal = dcg(sorted(relevant.values(), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([relevant.get(u, 0) for u in ranked[:k]]) / ideal


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int = 10, grade: int = 2) -> float:
    """Share of the units labelled with at least `grade` that appear in the first k results."""
    primary = {u for u, g in relevant.items() if g >= grade}
    if not primary:
        return 0.0
    return len(primary & set(ranked[:k])) / len(primary)


def mrr_at_k(ranked: list[str], relevant: dict[str, int], k: int = 10, grade: int = 2) -> float:
    for i, u in enumerate(ranked[:k], 1):
        if relevant.get(u, 0) >= grade:
            return 1.0 / i
    return 0.0


def unlabelled_share(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    """Share of the first k results that carry no label. Weak labels are incomplete; this says
    how much of what was returned could not be scored either way."""
    top = ranked[:k]
    return sum(u not in relevant for u in top) / len(top) if top else 0.0


def paired_bootstrap(a: list[float], b: list[float], resamples: int = 2000, seed: int = 7) -> dict[str, float]:
    """95% interval of the mean per-query difference b - a, resampling queries with replacement."""
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if diff.size == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "wins": 0, "losses": 0, "ties": 0}
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(resamples, diff.size))].mean(axis=1)
    return {"mean": round(float(diff.mean()), 4), "low": round(float(np.percentile(means, 2.5)), 4),
            "high": round(float(np.percentile(means, 97.5)), 4),
            "wins": int((diff > 1e-12).sum()), "losses": int((diff < -1e-12).sum()),
            "ties": int((np.abs(diff) <= 1e-12).sum())}
