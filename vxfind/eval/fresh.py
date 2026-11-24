"""Freshness: how often would the assistant hand a person an address that no longer works?

The index is built from a snapshot of the catalog. The site moves on: tables are deleted, made
private, renamed to another owner, changed. The measurement:

  1. Take the tables the serving configuration shows for every query (top 10).
  2. Ask the site about each one, now (the same check the vx-site server makes).
  3. Before verification: the share of shown addresses that are dead, and the share of answers
     with at least one dead address. After verification: dead addresses are removed and the list
     is refilled from the next candidates; moved tables get their new address.

The site's answers are recorded (data/vxfind/eval/site_answers.json) so that later runs, and the
agent cases, can replay them without the network. The record is a measurement at one moment and
carries its date.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ANSWERS = ROOT / "data" / "vxfind" / "eval" / "site_answers.json"


def load_answers() -> dict[str, dict[str, Any]]:
    return json.loads(ANSWERS.read_text(encoding="utf-8"))["answers"] if ANSWERS.exists() else {}


def check_live(site: Any, ids: list[str], workers: int = 4, log: Any = None) -> dict[str, dict[str, Any]]:
    """Ask the site about each id not asked before; keep every answer that is a fact."""
    known = load_answers()
    todo = [t for t in ids if t not in known]
    t0 = time.time()
    done = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for ans in ex.map(site.check, todo):
            done += 1
            if ans["state"] != "unknown":
                known[ans["table_id"]] = {k: ans[k] for k in ("table_id", "state", "url", "moved_to", "last_modified", "revision",
                                                              "gated", "http_status", "reason", "checked_at")}
            if log and done % 100 == 0:
                log(f"  {done}/{len(todo)} live checks, {time.time() - t0:.0f} s")
            if done % 200 == 0:
                _save(known)
    _save(known)
    return known


def _save(known: dict[str, dict[str, Any]]) -> None:
    ANSWERS.parent.mkdir(parents=True, exist_ok=True)
    ANSWERS.write_text(json.dumps({"recorded": time.strftime("%Y-%m-%dT%H:%M:%S"), "answers": known}), encoding="utf-8")


def staleness(run: dict[str, dict[str, Any]], answers: dict[str, dict[str, Any]], indexed_modified: dict[str, str | None],
              k: int = 10) -> dict[str, Any]:
    """Dead, moved and changed addresses in what was shown, before and after the live check."""
    shown = dead = moved = changed = unknown = 0
    answers_with_dead = answers_with_any_problem = full_after = 0
    removed_per_answer = []
    for v in run.values():
        top = v["ids"][:k]
        d = m = c = 0
        for t in top:
            a = answers.get(t)
            shown += 1
            if a is None:
                unknown += 1
                continue
            d += a["state"] == "gone"
            m += a["state"] == "moved"
            c += bool(a["state"] in ("live", "moved") and a.get("last_modified") and indexed_modified.get(t)
                      and a["last_modified"][:19] > str(indexed_modified[t])[:19])
        dead += d
        moved += m
        changed += c
        answers_with_dead += d > 0
        answers_with_any_problem += (d + m + c) > 0
        removed_per_answer.append(d)
        # after verification: dead ones dropped, list refilled from ranks k+1.. that are not dead
        alive = [t for t in v["ids"] if answers.get(t, {}).get("state") != "gone"]
        full_after += len(alive[:k]) >= min(k, len(v["ids"][:k]))
    n = len(run)
    judged = shown - unknown
    return {"answers": n, "addresses_shown": shown, "addresses_checked": judged, "not_checked": unknown,
            "dead": dead, "moved": moved, "changed_since_indexed": changed,
            "share_dead": round(dead / judged, 4) if judged else None, "share_moved": round(moved / judged, 4) if judged else None,
            "share_changed": round(changed / judged, 4) if judged else None,
            "answers_with_a_dead_address": answers_with_dead, "share_answers_with_a_dead_address": round(answers_with_dead / n, 4) if n else None,
            "answers_with_dead_moved_or_changed": answers_with_any_problem,
            "after_live_check": {"dead_addresses_shown": 0, "mean_removed_per_answer": round(sum(removed_per_answer) / n, 3) if n else None,
                                 "answers_still_full_after_refill": full_after}}


def by_age(ids: list[str], answers: dict[str, dict[str, Any]], created: dict[str, str | None]) -> dict[str, Any]:
    """Dead share by the year the table was created: is staleness a property of old tables or new ones?"""
    tot: Counter[str] = Counter()
    dead: Counter[str] = Counter()
    for t in ids:
        a = answers.get(t)
        if a is None or not created.get(t):
            continue
        y = str(created[t])[:4]
        tot[y] += 1
        dead[y] += a["state"] == "gone"
    return {y: {"checked": tot[y], "dead": dead[y], "share_dead": round(dead[y] / tot[y], 4)} for y in sorted(tot)}


def check_latency(site_factory: Any, ids: list[str], per_answer: int = 10, answers: int = 12) -> dict[str, Any]:
    """Seconds added to one answer by checking its ten tables, one after another and four at a time.
    A new site object each time, so nothing is answered from memory."""
    seq, par = [], []
    for i in range(answers):
        chunk = ids[i * per_answer:(i + 1) * per_answer]
        if len(chunk) < per_answer:
            break
        site = site_factory()
        t0 = time.perf_counter()
        if i % 2 == 0:
            for t in chunk:
                site.check(t)
            seq.append(time.perf_counter() - t0)
        else:
            with cf.ThreadPoolExecutor(4) as ex:
                list(ex.map(site.check, chunk))
            par.append(time.perf_counter() - t0)
    med = lambda xs: round(sorted(xs)[len(xs) // 2], 3) if xs else None
    return {"tables_per_answer": per_answer, "one_after_another_s": {"median": med(seq), "max": round(max(seq), 3) if seq else None, "answers": len(seq)},
            "four_at_a_time_s": {"median": med(par), "max": round(max(par), 3) if par else None, "answers": len(par)}}
