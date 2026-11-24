"""From a dataset card to a catalog record.

A "table" in this system is one training dataset in the catalog: it has a stable id, an owner,
a URL, a schema (the features recorded per frame), size figures, and sometimes a description a
person wrote. This module reads one row of the card dump and produces that record. It is pure
Python with no I/O, so the same functions run inside the Spark job, in the incremental sync and
in the tests.

What the cards are really like (measured on the 2026-11-23 snapshot, see verification/REPORT.md):
almost all carry the machine-written schema block, and only a few in a hundred carry a sentence
written by a person. So the record keeps apart what a person wrote (`description`) from what
the tooling wrote (the schema), and says which of the two it has.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import yaml

SITE = "https://huggingface.co/datasets/"
TAG = "LeRobot"
MAX_DESCRIPTION = 12000

_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.S)
_INFO = re.compile(r"\[meta/info\.json\]\(meta/info\.json\):\s*```json\s*(\{.*?\})\s*```", re.S)
_CODE = re.compile(r"```.*?```", re.S)
_HTML = re.compile(r"<[^>]+>")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BOILER = [re.compile(p, re.I) for p in (
    r"this dataset was created using \[?lerobot\]?(\([^)]*\))?\.?",
    r"\[?more information needed\]?",
    r"^\s*-\s*\*\*(homepage|paper|license|repository|curated by|funded by|shared by|language\(s\)[^*]*)\:?\*\*\:?.*$",
    r"^\s*\*\*bibtex:?\*\*\:?\s*$",
    r"^\s*#+\s*(dataset description|dataset structure|citation|dataset card for .*|dataset details|uses|"
    r"dataset creation|bias, risks, and limitations|more information|dataset card contact|dataset card authors)\s*$",
)]
_SCRATCH = re.compile(r"(^|[_\-.\d])(test|tests|testing|tmp|temp|debug|dummy|demo|sample|example|try|trial|foo|bar|"
                      r"delete|del|todo|untitled|asdf|hello)([_\-.\d]|$)", re.I)
_STAMP = re.compile(r"\d{6,}")
_TOKEN = re.compile(r"[A-Za-z]+|\d+")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")
INDEX_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index", "next.done", "next.reward",
                  "next.success"}


def sha(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:n]


def name_tokens(name: str) -> list[str]:
    """`so101_PickPlace-v2` -> so, 101, so101, pick, place, v, 2. Compound tokens are kept too."""
    out: list[str] = []
    for part in re.split(r"[_\-./\s]+", name):
        if not part:
            continue
        out.append(part.lower())
        pieces = [p.lower() for p in _TOKEN.findall(_CAMEL.sub(" ", part))]
        if len(pieces) > 1:
            out += pieces
    seen: set[str] = set()
    return [t for t in out if not (t in seen or seen.add(t))]


def front_matter(card: str) -> tuple[dict[str, Any], str]:
    m = _FRONT.match(card)
    if not m:
        return {}, card
    try:
        meta = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        meta = None
    return (meta if isinstance(meta, dict) else {}), card[m.end():]


def schema_block(card: str) -> tuple[dict[str, Any] | None, str]:
    """The machine-written schema, if the card carries it, and how reading it went."""
    m = _INFO.search(card)
    if not m:
        return None, "absent"
    try:
        info = json.loads(m.group(1))
    except ValueError:
        return None, "unparsable"
    return (info, "ok") if isinstance(info, dict) else (None, "unparsable")


def person_text(body: str) -> str:
    """What is left of a card after the template is taken away: the part a person wrote."""
    text = _INFO.sub("", body)
    text = _CODE.sub(" ", text)
    text = _HTML.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    lines = []
    for line in text.splitlines():
        for pat in _BOILER:
            line = pat.sub("", line)
        if line.strip(" -*#:|>`"):
            lines.append(line.strip())
    return re.sub(r"\s+\n", "\n", "\n".join(lines)).strip()[:MAX_DESCRIPTION]


def _shape(v: Any) -> list[int]:
    return [int(x) for x in v if isinstance(x, (int, float))] if isinstance(v, list) else []


def columns(info: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    feats = info.get("features")
    if not isinstance(feats, dict):
        return out
    for name, f in feats.items():
        if not isinstance(f, dict):
            continue
        dtype = str(f.get("dtype", ""))
        names = f.get("names")
        if isinstance(names, dict):                        # older cards: {"motors": [...]} or {"axes": [...]}
            names = next((v for v in names.values() if isinstance(v, list)), None)
        names = [str(x) for x in names][:64] if isinstance(names, list) else []
        if dtype in ("video", "image"):
            kind = "camera"
        elif name in INDEX_FEATURES:
            kind = "index"
        elif name == "action" or name.startswith("action."):
            kind = "action"
        elif name.startswith("observation"):
            kind = "state"
        else:
            kind = "other"
        out.append({"name": str(name)[:120], "dtype": dtype[:24], "shape": _shape(f.get("shape")), "names": names, "kind": kind})
    return out


def _int(v: Any) -> int | None:
    try:
        return int(v) if v is not None and not isinstance(v, bool) else None
    except (TypeError, ValueError):
        return None


def _num(v: Any) -> float | None:
    try:
        return float(v) if v is not None and not isinstance(v, bool) else None
    except (TypeError, ValueError):
        return None


def parse_row(table_id: str, owner: str | None, last_modified: str | None, created: str | None, downloads: Any, likes: Any,
              tags: list[str] | None, card: str | None) -> dict[str, Any]:
    """One catalog record. Never raises on a malformed card: what could not be read is said in
    `schema_status` and `problems`, because a record the pipeline drops silently is a table nobody
    can find."""
    card = card or ""
    tags = [str(t) for t in (tags or [])]
    meta, body = front_matter(card)
    info, status = schema_block(card)
    cols = columns(info) if info else []
    name = table_id.split("/", 1)[-1]
    cams = [c["name"] for c in cols if c["kind"] == "camera"]
    action = next((c for c in cols if c["name"] == "action"), None)
    state = next((c for c in cols if c["name"] == "observation.state"), None)
    text = person_text(body)
    licence = meta.get("license")
    problems = []
    if status != "ok":
        problems.append(f"schema {status}")
    if info and not cols:
        problems.append("schema without features")
    rec: dict[str, Any] = {
        "table_id": table_id, "owner": owner or table_id.split("/", 1)[0], "name": name, "url": SITE + table_id,
        "created": created, "last_modified": last_modified, "downloads": _int(downloads) or 0, "likes": _int(likes) or 0,
        "tags": tags, "licence": str(licence)[:60] if licence else None,
        "arxiv": [t.split(":", 1)[1] for t in tags if t.startswith("arxiv:")],
        "schema_status": status,
        "codebase_version": str(info.get("codebase_version"))[:12] if info and info.get("codebase_version") else None,
        "robot_type": str(info.get("robot_type"))[:80] if info and info.get("robot_type") else None,
        "fps": _num(info.get("fps")) if info else None,
        "total_episodes": _int(info.get("total_episodes")) if info else None,
        "total_frames": _int(info.get("total_frames")) if info else None,
        "total_tasks": _int(info.get("total_tasks")) if info else None,
        "data_mb": _num(info.get("data_files_size_in_mb")) if info else None,
        "video_mb": _num(info.get("video_files_size_in_mb")) if info else None,
        "columns": cols, "n_columns": len(cols), "cameras": cams, "n_cameras": len(cams),
        "action_dim": (action["shape"][0] if action and action["shape"] else None),
        "state_dim": (state["shape"][0] if state and state["shape"] else None),
        "action_names": action["names"] if action else [],
        "description": text, "has_description": len(text) >= 80,
        "name_tokens": name_tokens(name), "is_scratch": bool(_SCRATCH.search(name)),
        "has_stamp": bool(_STAMP.search(name)),
        "card_chars": len(card), "card_hash": sha(card),
        "schema_hash": sha(json.dumps({"f": cols, "fps": info.get("fps"), "r": info.get("robot_type"),
                                       "v": info.get("codebase_version")}, sort_keys=True, default=str)) if info else None,
        "problems": problems,
    }
    rec["summary"] = summary(rec)
    rec["content_hash"] = sha(rec["summary"] + "\n" + rec["description"] + "\n" + str(rec["last_modified"]))
    return rec


def summary(rec: dict[str, Any]) -> str:
    """The table's summary, written by code from its record: what a search over tables reads and
    what is embedded. A person's description, when there is one, comes first."""
    parts = [f"Dataset {rec['name']} by {rec['owner']}."]
    if rec["description"]:
        parts.append(rec["description"][:1200])
    facts = []
    if rec["robot_type"]:
        facts.append(f"robot {rec['robot_type']}")
    if rec["fps"]:
        facts.append(f"{rec['fps']:g} fps")
    if rec["total_episodes"] is not None:
        facts.append(f"{rec['total_episodes']} episodes")
    if rec["total_frames"] is not None:
        facts.append(f"{rec['total_frames']} frames")
    if rec["total_tasks"]:
        facts.append(f"{rec['total_tasks']} tasks")
    if rec["codebase_version"]:
        facts.append(f"format {rec['codebase_version']}")
    if facts:
        parts.append("Recorded: " + ", ".join(facts) + ".")
    if rec["cameras"]:
        parts.append("Cameras: " + ", ".join(c.rsplit(".", 1)[-1] for c in rec["cameras"][:8]) + ".")
    if rec["action_names"]:
        parts.append("Action: " + ", ".join(rec["action_names"][:16]) + ".")
    elif rec["action_dim"]:
        parts.append(f"Action of {rec['action_dim']} values.")
    other = [c["name"] for c in rec["columns"] if c["kind"] in ("state", "other") and c["name"] != "observation.state"]
    if other:
        parts.append("Also: " + ", ".join(other[:10]) + ".")
    return " ".join(parts)
