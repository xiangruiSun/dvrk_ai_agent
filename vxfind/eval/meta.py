"""Evaluating the evaluation: can these numbers be trusted, and how far?

Four questions, each answered by a measurement and not by an argument:

  1. Does the judge agree with truth that no model produced?      calibrate()
  2. Does the evaluation notice a system that was made worse?      degraded() + detected()
  3. How small a difference can this many queries show?            power()
  4. Do the query sets rank systems the same way?                  set_agreement()

If the answer to 2 is no, or the answer to 3 is "only very large ones", a result table built on
these query sets says less than it appears to, and the report says so next to the table.
"""
from __future__ import annotations

import math
import random
from typing import Any

from . import metrics as M
from .harness import HEADLINE


# ---------------------------------------------------------------------------- 1. the judge

def calibrate(truth: list[int], judged: list[int], relevant_truth: int = 3, relevant_judged: int = 2) -> dict[str, Any]:
    """Judge grades against code-computed grades on the same (need, table) pairs.

    The code grades are 3 (robot and all constraints match), 1 (robot matches, a constraint does
    not) and 0 (other robot). The question asked of the judge is binary: does it call relevant
    (grade >= 2) exactly the pairs where everything matches?"""
    n = len(truth)
    t = [x >= relevant_truth for x in truth]
    j = [x >= relevant_judged for x in judged]
    tp = sum(a and b for a, b in zip(t, j))
    fp = sum((not a) and b for a, b in zip(t, j))
    fn = sum(a and (not b) for a, b in zip(t, j))
    tn = n - tp - fp - fn
    by_truth: dict[str, dict[str, int]] = {}
    for a, b in zip(truth, judged):
        by_truth.setdefault(str(a), {}).setdefault(str(b), 0)
        by_truth[str(a)][str(b)] += 1
    return {"pairs": n, "agreement": round((tp + tn) / n, 4), "kappa": round(M.cohen_kappa(t, j), 4),
            "judge_precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "judge_recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else None,
            "confusion": {"true_pos": tp, "false_pos": fp, "false_neg": fn, "true_neg": tn},
            "judge_grade_by_true_grade": {k: dict(sorted(v.items())) for k, v in sorted(by_truth.items())}}


def agreement(x: list[int], y: list[int]) -> dict[str, Any]:
    """Two judges on the same pairs."""
    n = len(x)
    return {"pairs": n, "exact": round(sum(a == b for a, b in zip(x, y)) / n, 4),
            "within_one": round(sum(abs(a - b) <= 1 for a, b in zip(x, y)) / n, 4),
            "weighted_kappa": round(M.weighted_kappa(x, y), 4),
            "kappa_relevant": round(M.cohen_kappa([a >= 2 for a in x], [b >= 2 for b in y]), 4),
            "mean_first": round(sum(x) / n, 3), "mean_second": round(sum(y) / n, 3)}


# ---------------------------------------------------------------------------- 2. planted faults

def degraded(run: dict[str, dict[str, Any]], how: str, all_ids: list[str], seed: int = 5) -> dict[str, dict[str, Any]]:
    """A copy of a run, made worse in a known way. Nothing is searched again."""
    rng = random.Random(seed)
    out = {}
    for qid, v in run.items():
        ids, fams = list(v["ids"]), list(v["families"])
        if how == "shuffled_top10":                    # same tables, order destroyed: rank metrics must fall, set metrics must not move
            order = list(range(min(10, len(ids))))
            rng.shuffle(order)
            ids[:10], fams[:10] = [ids[i] for i in order], [fams[i] for i in order]
        elif how == "reversed_top10":
            ids[:10], fams[:10] = ids[:10][::-1], fams[:10][::-1]
        elif how == "second_page":                     # ranks 11-20 shown as if they were 1-10
            ids, fams = ids[10:], fams[10:]
        elif how == "half_random":                     # five of the ten replaced by arbitrary tables
            for i in rng.sample(range(min(10, len(ids))), min(5, len(ids))):
                ids[i], fams[i] = rng.choice(all_ids), None
        elif how == "one_random":                      # one of the ten replaced: a small fault
            if ids:
                i = rng.randrange(min(10, len(ids)))
                ids[i], fams[i] = rng.choice(all_ids), None
        else:
            raise ValueError(how)
        out[qid] = {**v, "ids": ids, "families": fams, "deep": []}
    return out


def detected(cmp: dict[str, Any], alpha: float = 0.05) -> dict[str, Any]:
    """Was a planted fault seen as a loss on each set's headline metric?"""
    return {k: {"metric": v["metric"], "difference": v["mean_difference"], "ci95": v["ci95"], "p": v["p_randomization"],
                "seen_as_a_loss": bool(v["mean_difference"] < 0 and v["p_randomization"] < alpha)} for k, v in cmp.items()}


# ---------------------------------------------------------------------------- 3. power

def power(base: dict[str, dict[str, dict[str, float]]], other: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    """From the spread of per-query differences between two real configurations: the smallest
    true difference this many queries would show (80 % power, 5 % two-sided), and how many
    queries a difference of 0.05 or 0.02 would need."""
    out = {}
    for kind, metric in HEADLINE.items():
        a, b = base.get(kind, {}).get(metric, {}), other.get(kind, {}).get(metric, {})
        ids = sorted(set(a) & set(b))
        if len(ids) < 3:
            continue
        d = [b[i] - a[i] for i in ids]
        mu = sum(d) / len(d)
        sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1))
        z = M._z(0.975) + M._z(0.8)
        out[kind] = {"metric": metric, "queries": len(ids), "sd_of_paired_differences": round(sd, 4),
                     "smallest_detectable_difference": round(z * sd / math.sqrt(len(ids)), 4),
                     "queries_needed_for_0.05": math.ceil((z * sd / 0.05) ** 2) if sd else 0,
                     "queries_needed_for_0.02": math.ceil((z * sd / 0.02) ** 2) if sd else 0}
    return out


# ---------------------------------------------------------------------------- 4. do the sets agree

def set_agreement(table: dict[str, dict[str, float]]) -> dict[str, Any]:
    """table: configuration -> {A: mean, B: mean, C: mean}. Kendall's tau between the orders the
    sets put the configurations in. Near 1: the cheap sets can stand in for the costly one.
    Near 0: a gain on one set says nothing about another."""
    kinds = sorted({k for v in table.values() for k in v})
    order = {k: sorted((c for c in table if k in table[c]), key=lambda c: (-table[c][k], c)) for k in kinds}
    out: dict[str, Any] = {"configurations": len(table), "best_by_set": {k: order[k][:3] for k in kinds}}
    for i, a in enumerate(kinds):
        for b in kinds[i + 1:]:
            out[f"tau_{a}_{b}"] = round(M.kendall_tau(order[a], order[b]), 3)
    return out
