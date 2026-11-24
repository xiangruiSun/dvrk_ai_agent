"""From roll-out to evaluation: what people did with an answer becomes the next test case.

An offline query set is somebody's guess at what people will ask. Once people use the assistant,
their questions are the better set, and their reactions are free (noisy) labels. This module is
the path between the two:

  log       one line per event: a question, the tables shown, and what the person did
            (opened an address, marked a table useful or not, said that nothing fits, asked again)
  measure   online signals per day or per release: answers with no click, first click position,
            share marked not useful, share of questions asked again in other words
  triage    events that say the system failed become CANDIDATE cases: a question where nothing
            was opened, a table marked not useful, a "nothing fits". A candidate is not truth: a
            person looks at it, labels it, and only then does it join the query set (kind F).

What is logged is the question, table ids and the action. No name, no free-text comment and no
identity beyond an opaque session id is kept, and questions enter the query set only after a
person has read them (a question can hold something that should not be stored).
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

ACTIONS = ("shown", "opened", "useful", "not_useful", "nothing_fits", "asked_again", "requested_access")


class FeedbackLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, session: str, need: str, action: str, table_id: str | None = None, shown: list[str] | None = None,
            rank: int | None = None, release: str = "", at: float | None = None) -> None:
        if action not in ACTIONS:
            raise ValueError(action)
        ev = {"at": at or time.time(), "session": session, "release": release, "need": need[:600], "action": action,
              "table_id": table_id, "rank": rank, "shown": shown}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({k: v for k, v in ev.items() if v is not None}, ensure_ascii=False) + "\n")

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]


def questions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group events into questions: one per (session, need)."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for e in events:
        q = out.setdefault((e["session"], e["need"]), {"session": e["session"], "need": e["need"], "release": e.get("release", ""),
                                                        "shown": [], "opened": [], "useful": [], "not_useful": [],
                                                        "nothing_fits": False, "asked_again": False, "requested": []})
        a = e["action"]
        if a == "shown":
            q["shown"] = e.get("shown") or []
        elif a in ("opened", "useful", "not_useful"):
            q[a].append({"table_id": e.get("table_id"), "rank": e.get("rank")})
        elif a == "requested_access":
            q["requested"].append(e.get("table_id"))
        else:
            q[a] = True
    return list(out.values())


def online(events: list[dict[str, Any]]) -> dict[str, Any]:
    qs = questions(events)
    n = len(qs)
    if not n:
        return {"questions": 0}
    first = [min(o["rank"] for o in q["opened"] if o["rank"]) for q in qs if any(o["rank"] for o in q["opened"])]
    shown = sum(len(q["shown"]) for q in qs)
    return {"questions": n, "sessions": len({q["session"] for q in qs}),
            "answers_with_nothing_opened": sum(not q["opened"] for q in qs),
            "share_nothing_opened": round(sum(not q["opened"] for q in qs) / n, 4),
            "mean_rank_of_first_opened": round(sum(first) / len(first), 2) if first else None,
            "first_opened_was_rank_1": round(sum(r == 1 for r in first) / len(first), 4) if first else None,
            "tables_marked_not_useful": sum(len(q["not_useful"]) for q in qs),
            "share_of_shown_marked_not_useful": round(sum(len(q["not_useful"]) for q in qs) / shown, 4) if shown else None,
            "said_nothing_fits": sum(q["nothing_fits"] for q in qs), "asked_again": sum(q["asked_again"] for q in qs),
            "access_requests": sum(len(q["requested"]) for q in qs)}


def triage(events: list[dict[str, Any]], known_needs: set[str]) -> list[dict[str, Any]]:
    """Candidate evaluation cases from failures. One per distinct need; a need already in the
    query sets is not added twice."""
    out, seen = [], set()
    for q in questions(events):
        key = " ".join(q["need"].lower().split())
        if key in seen or key in known_needs:
            continue
        why = []
        if q["nothing_fits"]:
            why.append("the person said nothing fits")
        if q["shown"] and not q["opened"]:
            why.append("nothing shown was opened")
        if q["not_useful"]:
            why.append(f"{len(q['not_useful'])} shown table(s) marked not useful")
        if q["asked_again"]:
            why.append("the person asked again in other words")
        if not why:
            continue
        seen.add(key)
        qid = "F" + hashlib.sha256(key.encode()).hexdigest()[:8]
        out.append({"qid": qid, "kind": "F", "need": q["need"], "status": "candidate: needs a person to read and label it",
                    "why": why, "shown": q["shown"][:10],
                    "weak_labels": {"not_useful": [x["table_id"] for x in q["not_useful"]], "useful": [x["table_id"] for x in q["useful"]],
                                    "opened": [x["table_id"] for x in q["opened"]]}})
    return out


def label_mix(cands: list[dict[str, Any]]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for x in cands:
        for w in x["why"]:
            c["a shown table was marked not useful" if "marked" in w else w] += 1
    return dict(c)
