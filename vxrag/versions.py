"""Index versions: built, validated, activated, rolled back.

A version is a folder holding a snapshot of the units it was built from and a manifest. The
snapshot makes a version independent of the store: when the store moves on, an older version
still serves exactly what it served before, which is what makes a rollback real. Everything
else (BM25 statistics, vectors) is derived from the snapshot when the version is loaded, and
vectors come from the embedding cache.

    build      store -> snapshot + manifest                    status "built"
    validate   release gates (vxrag/gates.py)                  status "validated" or "rejected"
    activate   only a validated version; the one serving becomes "previous"
    rollback   the previous version serves again

The pointer to the active version is one small file, replaced atomically.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .gates import run_gates
from .retrieval import Index
from .store import SOURCES, CorpusStore, Unit, units_hash

POINTER = "ACTIVE.json"


class VersionError(RuntimeError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _dir(settings: Settings | None) -> Path:
    return (settings or get_settings()).indexes_dir


def _write(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def pointer(settings: Settings | None = None) -> dict[str, Any]:
    path = _dir(settings) / POINTER
    if not path.exists():
        return {"active": None, "previous": None, "log": []}
    return json.loads(path.read_text(encoding="utf-8"))


def manifest(version: str, settings: Settings | None = None) -> dict[str, Any]:
    path = _dir(settings) / version / "manifest.json"
    if not path.exists():
        raise VersionError(f"there is no index version {version!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_versions(settings: Settings | None = None) -> list[dict[str, Any]]:
    out = []
    for p in sorted(_dir(settings).glob("*/manifest.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        out.append({k: m.get(k) for k in ("version", "built", "status", "config", "units", "note")})
    return out


def build(store: CorpusStore, strategy: str = "bm25:ident", chunking: str = "section_ctx", trust: str = "off",
          sources: tuple[str, ...] = SOURCES, note: str = "", settings: Settings | None = None) -> dict[str, Any]:
    units = store.units(sources)
    if not units:
        raise VersionError("the store has no units to build from")
    snapshot_hash = units_hash(sorted((u.unit_id, u.content_hash) for u in units))
    root = _dir(settings)
    n = 1 + max([int(p.name[1:4]) for p in root.glob("v[0-9][0-9][0-9]-*") if p.is_dir()], default=0)
    version = f"v{n:03d}-{snapshot_hash[:8]}"
    folder = root / version
    folder.mkdir(parents=True)
    with open(folder / "units.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for u in units:
            f.write(json.dumps(asdict(u), ensure_ascii=False) + "\n")
    by_source: dict[str, int] = {}
    for u in units:
        by_source[u.source] = by_source.get(u.source, 0) + 1
    m = {"version": version, "built": _now(), "status": "built", "note": note,
         "config": {"strategy": strategy, "chunking": chunking, "trust": trust},
         "snapshot_hash": snapshot_hash, "public_corpus_hash": store.corpus_hash(), "units": by_source,
         "collections": sorted({u.collection for u in units}), "store_change_id": store.last_change_id()}
    _write(folder / "manifest.json", m)
    return m


def load(version: str, settings: Settings | None = None) -> tuple[Index, dict[str, Any]]:
    m = manifest(version, settings)
    units = []
    with open(_dir(settings) / version / "units.jsonl", encoding="utf-8") as f:
        for line in f:
            units.append(Unit(**json.loads(line)))
    index = Index(units, m["config"]["chunking"])
    m["chunks"] = len(index.chunks)
    return index, m


def validate(version: str, settings: Settings | None = None, policy: dict[str, float] | None = None) -> dict[str, Any]:
    """Run the release gates against the version, with the active version as the reference."""
    index, m = load(version, settings)
    active = pointer(settings)["active"]
    reference, reference_index = None, None
    if active and active != version:
        reference_index, am = load(active, settings)
        reference = {"version": active, "units": am["units"], "config": am["config"], **am.get("measured", {})}
    report = run_gates(index, m, reference, policy, reference_index)
    m.update(status="validated" if report["passed"] else "rejected", validated=_now(), gates=report["gates"],
             measured=report["measured"], reference=reference["version"] if reference else None,
             suspect_units=report["suspect_units"], chunks=len(index.chunks))
    _write(_dir(settings) / version / "manifest.json", m)
    return m


def activate(version: str, settings: Settings | None = None) -> dict[str, Any]:
    m = manifest(version, settings)
    if m["status"] != "validated":
        failed = [g["gate"] for g in m.get("gates", []) if not g["passed"]]
        raise VersionError(f"{version} is {m['status']}, not validated"
                           + (f" (failed: {', '.join(failed)})" if failed else "") + "; it cannot be activated")
    p = pointer(settings)
    if p["active"] == version:
        return p
    p = {"active": version, "previous": p["active"],
         "log": [*p["log"], {"at": _now(), "action": "activate", "version": version, "replaced": p["active"]}]}
    _write(_dir(settings) / POINTER, p)
    return p


def rollback(settings: Settings | None = None) -> dict[str, Any]:
    p = pointer(settings)
    if not p["previous"]:
        raise VersionError("there is no previous version to go back to")
    manifest(p["previous"], settings)                     # it must still exist
    p = {"active": p["previous"], "previous": p["active"],
         "log": [*p["log"], {"at": _now(), "action": "rollback", "version": p["previous"], "replaced": p["active"]}]}
    _write(_dir(settings) / POINTER, p)
    return p


def sync(store: CorpusStore, settings: Settings | None = None, sources: tuple[str, ...] = SOURCES) -> dict[str, Any]:
    """Bring the store up to date with the downloaded sources and say exactly what changed.
    Unchanged units are not written, so the change events are the work an index has to redo."""
    from .ingest import ingest
    before = store.last_change_id()
    report = ingest(store, settings, sources)
    changes = store.changes_since(before)
    summary: dict[str, dict[str, int]] = {}
    for c in changes:
        summary.setdefault(c["source"], {"added": 0, "changed": 0, "removed": 0})[c["change"]] += 1
    return {"changes": summary, "change_events": len(changes), "rebuild_needed": bool(changes),
            "units": report["units"], "problems": report["problems"], "validation": report["validation"],
            "changed_unit_ids": [c["unit_id"] for c in changes]}
