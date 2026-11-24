"""Ranking metrics and the statistics that say whether a difference is real.

Conventions, fixed here so that every number in the reports means the same thing:

  grade        0 not relevant, 1 related but not usable for the need, 2 usable, 3 exactly what was asked
  relevant     grade >= 2, unless a metric says otherwise
  unjudged     a returned table with no grade. It counts as grade 0 in every metric (the
               conservative reading) and is reported separately as unjudged@k, because a system
               that was not part of the pool is punished for finding things nobody looked at.
  nDCG@k       gain 2^grade - 1, discount log2(rank + 1), ideal ranking from all judged tables of the query
  recall@k     relevant tables in the top k / relevant tables known for the query (judged ones)
  success@k    1 if the top k holds at least one relevant table
  MRR@k        1 / rank of the first relevant table in the top k, else 0

A query with no relevant table at all has no recall and no nDCG; it is left out of those means
and counted, never scored as zero or one.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Sequence

RELEVANT = 2


def dcg(grades: Sequence[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at(ranked: Sequence[str], qrels: dict[str, int], k: int = 10) -> float | None:
    ideal = dcg(sorted(qrels.values(), reverse=True)[:k])
    if ideal == 0:
        return None
    return dcg([qrels.get(t, 0) for t in ranked[:k]]) / ideal


def recall_at(ranked: Sequence[str], qrels: dict[str, int], k: int = 10, threshold: int = RELEVANT) -> float | None:
    rel = {t for t, g in qrels.items() if g >= threshold}
    if not rel:
        return None
    return len(rel & set(ranked[:k])) / len(rel)


def precision_at(ranked: Sequence[str], qrels: dict[str, int], k: int = 10, threshold: int = RELEVANT) -> float:
    top = list(ranked[:k])
    return sum(qrels.get(t, 0) >= threshold for t in top) / k if k else 0.0


def success_at(ranked: Sequence[str], qrels: dict[str, int], k: int = 10, threshold: int = RELEVANT) -> float:
    return float(any(qrels.get(t, 0) >= threshold for t in ranked[:k]))


def mrr_at(ranked: Sequence[str], qrels: dict[str, int], k: int = 10, threshold: int = RELEVANT) -> float:
    for i, t in enumerate(ranked[:k], 1):
        if qrels.get(t, 0) >= threshold:
            return 1.0 / i
    return 0.0


def unjudged_at(ranked: Sequence[str], qrels: dict[str, int], k: int = 10) -> float:
    top = list(ranked[:k])
    return sum(t not in qrels for t in top) / len(top) if top else 0.0


def hit_at(ranked: Sequence[str], target: str, k: int = 10) -> float:
    """Known-item: is the one table the need was written from in the top k?"""
    return float(target in ranked[:k])


def rank_of(ranked: Sequence[str], target: str) -> int | None:
    for i, t in enumerate(ranked, 1):
        if t == target:
            return i
    return None


METRICS: dict[str, Callable[..., float | None]] = {
    "ndcg@10": lambda r, q: ndcg_at(r, q, 10), "recall@10": lambda r, q: recall_at(r, q, 10),
    "precision@10": lambda r, q: precision_at(r, q, 10), "success@10": lambda r, q: success_at(r, q, 10),
    "mrr@10": lambda r, q: mrr_at(r, q, 10), "unjudged@10": lambda r, q: unjudged_at(r, q, 10),
}


def per_query(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], metric: str) -> dict[str, float]:
    """metric value per query id, for the queries where the metric is defined."""
    fn = METRICS[metric]
    out = {}
    for qid, ranked in run.items():
        v = fn(ranked, qrels.get(qid, {}))
        if v is not None:
            out[qid] = float(v)
    return out


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(values: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 7) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean over queries."""
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    m = len(values)
    means = sorted(mean([values[rng.randrange(m)] for _ in range(m)]) for _ in range(n))
    return means[int(alpha / 2 * n)], means[min(n - 1, int((1 - alpha / 2) * n))]


def summarise(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], metrics: Sequence[str] = tuple(METRICS)) -> dict[str, Any]:
    out: dict[str, Any] = {"queries": len(run)}
    for m in metrics:
        vals = list(per_query(run, qrels, m).values())
        lo, hi = bootstrap_ci(vals)
        out[m] = {"mean": round(mean(vals), 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(vals)}
    return out


def paired_bootstrap(a: dict[str, float], b: dict[str, float], n: int = 5000, seed: int = 11) -> dict[str, Any]:
    """b minus a over the queries both have. The interval is over resampled queries; `p_two_sided`
    is the share of resamples on the other side of zero, doubled."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"n": 0}
    diffs = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    m = len(diffs)
    means = sorted(mean([diffs[rng.randrange(m)] for _ in range(m)]) for _ in range(n))
    d = mean(diffs)
    below = sum(x <= 0 for x in means) / n
    p = 2 * min(below, 1 - below)
    return {"n": m, "mean_difference": round(d, 4), "ci95": [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)],
            "p_two_sided": round(min(1.0, p), 4), "wins": sum(x > 0 for x in diffs), "losses": sum(x < 0 for x in diffs),
            "ties": sum(x == 0 for x in diffs)}


def randomization_test(a: dict[str, float], b: dict[str, float], n: int = 10000, seed: int = 13) -> float:
    """Two-sided paired randomization (sign-flip) test of the mean difference."""
    ids = sorted(set(a) & set(b))
    diffs = [b[i] - a[i] for i in ids]
    if not diffs:
        return float("nan")
    observed = abs(mean(diffs))
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs) / len(diffs)
        hits += abs(s) >= observed - 1e-12
    return hits / n


def minimum_detectable_effect(values: Sequence[float], other: Sequence[float] | None = None, power: float = 0.8,
                              alpha: float = 0.05) -> float:
    """The smallest true mean difference a paired comparison on this many queries would find with
    the given power. Uses the standard deviation of per-query differences when `other` is given,
    else of the values (a pessimistic stand-in). Normal approximation."""
    xs = [o - v for v, o in zip(values, other)] if other is not None else list(values)
    m = len(xs)
    if m < 2:
        return float("nan")
    mu = mean(xs)
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (m - 1))
    z_a, z_b = _z(1 - alpha / 2), _z(power)
    return (z_a + z_b) * sd / math.sqrt(m)


def _z(p: float) -> float:
    """Inverse of the standard normal distribution function (Acklam's approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - 0.02425:
        return -_z(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def cohen_kappa(x: Sequence[Any], y: Sequence[Any]) -> float:
    """Agreement of two labellers beyond chance, on the same items."""
    assert len(x) == len(y) and x
    n = len(x)
    labels = sorted(set(x) | set(y), key=str)
    po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((sum(a == l for a in x) / n) * (sum(b == l for b in y) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def weighted_kappa(x: Sequence[int], y: Sequence[int], levels: int = 4) -> float:
    """Quadratic-weighted kappa for ordered grades: a 3 against a 0 costs more than a 3 against a 2."""
    n = len(x)
    w = [[((i - j) ** 2) / ((levels - 1) ** 2) for j in range(levels)] for i in range(levels)]
    obs = [[0.0] * levels for _ in range(levels)]
    for a, b in zip(x, y):
        obs[a][b] += 1 / n
    px = [sum(a == i for a in x) / n for i in range(levels)]
    py = [sum(b == i for b in y) / n for i in range(levels)]
    num = sum(w[i][j] * obs[i][j] for i in range(levels) for j in range(levels))
    den = sum(w[i][j] * px[i] * py[j] for i in range(levels) for j in range(levels))
    return 1 - num / den if den else 1.0


def kendall_tau(order_a: Sequence[str], order_b: Sequence[str]) -> float:
    """Rank agreement of two orderings of the same systems (tau-a)."""
    items = [x for x in order_a if x in set(order_b)]
    pa = {x: i for i, x in enumerate(order_a)}
    pb = {x: i for i, x in enumerate(order_b)}
    conc = disc = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            s = (pa[items[i]] - pa[items[j]]) * (pb[items[i]] - pb[items[j]])
            conc += s > 0
            disc += s < 0
    pairs = len(items) * (len(items) - 1) / 2
    return (conc - disc) / pairs if pairs else float("nan")
