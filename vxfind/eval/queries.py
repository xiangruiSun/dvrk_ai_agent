"""The query sets: what the evaluation asks, and where each set's truth comes from.

Four sets, each with a different source of truth and therefore a different weakness. No single
one is trusted alone; a change counts as an improvement when the sets agree.

  A  attribute needs    "Franka data, at least 200 episodes, with a wrist camera". Written from
                        templates. Truth is COMPUTED by code from the catalog for every table, so
                        there are no unjudged results and no labeller. Weakness: template wording,
                        and only what the metadata can express.
  B  known-item needs   A language model reads one described table and writes the question a
                        newcomer would ask to find it, without its name. Truth is that table (and
                        its near-identical siblings). Weakness: the question is written from the
                        text that is indexed, which favours word overlap; other tables may fit
                        equally well and are not credited.
  C  written needs      Questions written by hand before any result was seen, in a newcomer's
                        words. Truth is graded relevance of pooled results (a language-model judge,
                        calibrated in meta.py). Weakness: one author, who is not a newcomer; judge
                        error; unjudged results.
  D  no-answer needs    Things the catalog does not hold, or sizes no table has. Truth is that
                        nothing fits, confirmed by judging the pooled results. Used for abstention.

Each query belongs to `dev` or `test` by a hash of its id. Settings are chosen on dev. Test is
scored for the final table and for the regression gate, and for nothing else.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from .families import FAMILIES, family_of, is_bimanual, label

ROOT = Path(__file__).resolve().parents[2]
QDIR = ROOT / "evals" / "vxfind" / "queries"
DATA = ROOT / "data" / "vxfind"


def split_of(qid: str) -> str:
    return "dev" if int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 2 == 0 else "test"


def load(kind: str, split: str | None = None) -> list[dict[str, Any]]:
    p = QDIR / f"{kind}.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if split is None or r["split"] == split]


def save(kind: str, rows: Iterable[dict[str, Any]]) -> Path:
    QDIR.mkdir(parents=True, exist_ok=True)
    p = QDIR / f"{kind}.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return p


def records(snapshot: str = "old") -> list[dict[str, Any]]:
    """The catalog rows the labels are computed from: the snapshot the index was built from."""
    import pyarrow.parquet as pq
    cols = ["table_id", "owner", "name", "robot_type", "fps", "total_episodes", "n_cameras", "cameras_json", "codebase_version",
            "is_head", "family", "has_description", "is_scratch", "description", "summary", "content_hash"]
    rows = pq.read_table(DATA / snapshot / "tables.parquet", columns=cols).to_pylist()
    for r in rows:
        r["cameras"] = [c.rsplit(".", 1)[-1].lower() for c in json.loads(r.pop("cameras_json") or "[]")]
        r["robot_family"] = family_of(r["robot_type"])
    return rows


# ---------------------------------------------------------------------------- A: attribute needs

CHECKS = {
    "min_episodes": lambda r, v: (r["total_episodes"] or 0) >= v,
    "fps": lambda r, v: r["fps"] is not None and abs(r["fps"] - v) < 0.5,
    "min_cameras": lambda r, v: (r["n_cameras"] or 0) >= v,
    "camera": lambda r, v: any(v in c for c in r["cameras"]),
    "bimanual": lambda r, v: is_bimanual(r["robot_type"]) == v,
    "format": lambda r, v: r["codebase_version"] == v,
}
PHRASES = {
    "min_episodes": ["with at least {v} episodes", "{v} episodes or more", "no fewer than {v} episodes"],
    "fps": ["recorded at {v} fps", "at {v} frames per second", "{v} Hz recordings"],
    "min_cameras": ["with at least {v} camera views", "{v} or more cameras"],
    "camera": ["with a {v} camera", "that includes a {v} camera view"],
    "bimanual": ["two-arm (bimanual) set-up", "bimanual"],
    "format": ["in the {v} dataset format", "stored as format {v}"],
}
OPENERS = ["I need {robot} data {rest}.", "Looking for datasets of the {robot} {rest}.", "Which tables have {robot} recordings {rest}?",
           "{robot} demonstrations, {rest}", "For training I want {robot} data {rest}."]
VALUES = {"min_episodes": [50, 100, 200, 500], "fps": [10, 15, 20, 50, 60], "min_cameras": [2, 3],
          "camera": ["wrist", "top", "front", "side"], "bimanual": [True], "format": ["v2.1", "v3.0"]}


def grade_attribute(spec: dict[str, Any], rec: dict[str, Any]) -> int:
    """3: the robot and every stated constraint match. 1: the robot matches, a constraint does
    not. 0: another robot, or the table does not say which robot.

    A table with no episodes is never a 3. FOUND WHILE CALIBRATING THE JUDGE: the judge refused
    tables that met every stated constraint and held 0 episodes, and it was right. Truth computed
    by code is only as good as the code; this rule was missing."""
    if rec["robot_family"] != spec["family"]:
        return 0
    if not (rec["total_episodes"] or 0) > 0:
        return 1
    return 3 if all(CHECKS[k](rec, v) for k, v in spec["constraints"].items()) else 1


def attribute_text(spec: dict[str, Any], rng: random.Random) -> str:
    parts = [rng.choice(PHRASES[k]).format(v=v) for k, v in spec["constraints"].items()]
    rest = " and ".join(parts) if len(parts) < 3 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return rng.choice(OPENERS).format(robot=label(spec["family"]), rest=rest).replace("  ", " ")


def build_attribute(recs: list[dict[str, Any]], n: int = 60, seed: int = 20261123, per_family: int = 6) -> list[dict[str, Any]]:
    """Specs whose constraints matter: at least 5 tables satisfy all of them, and at most half of
    the robot's tables do, so returning any table of that robot is not enough."""
    rng = random.Random(seed)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        if r["robot_family"]:
            by_family.setdefault(r["robot_family"], []).append(r)
    out, seen, used = [], set(), {f: 0 for f in FAMILIES}
    attempts = 0
    while len(out) < n and attempts < 20000:
        attempts += 1
        fam = rng.choice(sorted(by_family))
        if used[fam] >= per_family:
            continue
        keys = rng.sample(sorted(VALUES), rng.choice([1, 2, 2, 3]))
        cons = {k: rng.choice(VALUES[k]) for k in sorted(keys)}
        sig = fam + json.dumps(cons, sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        spec = {"family": fam, "constraints": cons}
        pool = by_family[fam]
        full = sum(grade_attribute(spec, r) == 3 for r in pool)
        if full < 5 or full > len(pool) * 0.5:
            continue
        qid = "A" + hashlib.sha256(sig.encode()).hexdigest()[:8]
        heads = len({r["family"] for r in pool if grade_attribute(spec, r) == 3})
        out.append({"qid": qid, "kind": "A", "split": split_of(qid), "need": attribute_text(spec, rng), "spec": spec,
                    "matching_tables": full, "matching_groups": heads, "tables_of_robot": len(pool)})
        used[fam] += 1
    return out


# ---------------------------------------------------------------------------- B: known-item needs

WRITE_PROMPT = ("You write the question a new team member would ask when they need a dataset like the one described, "
                "without knowing that it exists. Write one or two plain sentences that say what they need: the robot, the "
                "task, and anything distinctive such as cameras, size or setting. Do not use the dataset's name, its "
                "owner's name or any identifier from it. Do not say that a description was given.")
WRITE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["need"], "properties": {"need": {"type": "string"}}}
COMMON = {"pick", "place", "cube", "block", "robot", "data", "dataset", "test", "task", "demo", "eval", "record", "lerobot",
          "franka", "panda", "aloha", "koch", "so100", "so101", "follower", "leader", "stack", "push", "grasp", "table",
          "left", "right", "bimanual", "simulation", "real", "libero", "droid", "object", "train", "policy"}


def leaks(need: str, rec: dict[str, Any]) -> str | None:
    """Why a written need gives the target away, or None."""
    low = need.lower()
    if len(rec["owner"]) >= 4 and rec["owner"].lower() in low:
        return "names the owner"
    flat = re.sub(r"[^a-z0-9]+", " ", rec["name"].lower()).strip()
    if len(flat) >= 8 and (flat in re.sub(r"[^a-z0-9]+", " ", low) or rec["name"].lower() in low):
        return "contains the table's name"
    for tok in re.split(r"[^a-z0-9]+", rec["name"].lower()):
        if len(tok) >= 6 and tok not in COMMON and any(ch.isdigit() for ch in tok) and tok in low:
            return f"contains the identifier {tok}"
    return None


def candidates_known_item(recs: list[dict[str, Any]], n: int, seed: int = 20261123, per_family: int = 14) -> list[dict[str, Any]]:
    """Tables a need can be written from: described by a person, not scratch, one per group of
    near-identical tables, and spread over robots so that the commonest arm is not most of the set."""
    rng = random.Random(seed)
    pool = [r for r in recs if r["has_description"] and not r["is_scratch"] and r["is_head"] and len(r["description"] or "") >= 200]
    rng.shuffle(pool)
    count: dict[str, int] = {}
    out = []
    for r in pool:
        f = r["robot_family"] or "other"
        if count.get(f, 0) >= (per_family * 3 if f == "other" else per_family):
            continue
        count[f] = count.get(f, 0) + 1
        out.append(r)
        if len(out) >= n:
            break
    return out


def write_known_item(llm: Any, rec: dict[str, Any]) -> tuple[str | None, str]:
    """(need, note). Two attempts; a need that names the table is discarded."""
    text = f"{rec['summary'][:700]}\n\nDescription:\n{(rec['description'] or '')[:1300]}"
    note = ""
    for attempt in range(2):
        extra = "" if attempt == 0 else " Your last answer gave the dataset away; describe the need without any name from it."
        try:
            r = llm.chat([{"role": "system", "content": WRITE_PROMPT + extra}, {"role": "user", "content": text}],
                         json_schema=WRITE_SCHEMA, temperature=0.3 if attempt else 0.0, max_tokens=160)
            need = str(json.loads(r.content).get("need", "")).strip()
        except Exception as exc:
            note = f"model error: {type(exc).__name__}"
            continue
        if len(need) < 25:
            note = "too short"
            continue
        why = leaks(need, rec)
        if why is None:
            return need, ""
        note = why
    return None, note


def known_item_query(rec: dict[str, Any], need: str) -> dict[str, Any]:
    qid = "B" + hashlib.sha256(rec["table_id"].encode()).hexdigest()[:8]
    return {"qid": qid, "kind": "B", "split": split_of(qid), "need": need, "target": rec["table_id"], "target_family": rec["family"],
            "robot_family": rec["robot_family"]}


# ---------------------------------------------------------------------------- C and D: written by hand

def stamp(kind: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, r in enumerate(rows, 1):
        qid = f"{kind}{i:03d}"
        out.append({"qid": qid, "kind": kind, "split": split_of(qid), **r})
    return out


def all_queries(split: str | None = None, kinds: str = "ABCD") -> list[dict[str, Any]]:
    return [q for k in kinds for q in load(k, split)]
