"""Configuration files as data: reading them, addressing a value inside one, changing a copy,
validating against the dVRK project's own JSON Schemas, and keeping versions.

A published or deployed file is never edited. A change is applied to a copy in memory, and an
approved change is written as a new numbered version beside the earlier ones.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from . import jsonc

MISSING = object()


class PathError(ValueError):
    pass


# ------------------------------------------------------------------------------ addressing

def _pick(items: list[Any], seg: str) -> int:
    """An element of a list is addressed by its index, by its "name", or, for a teleoperation
    pair, by "<MTM>-<PSM>". Names survive a reordering of the file; indices do not."""
    if re.fullmatch(r"\d+", seg):
        i = int(seg)
        if i < len(items):
            return i
        raise PathError(f"index {i} is outside the list ({len(items)} elements)")
    hits = [i for i, it in enumerate(items) if isinstance(it, dict)
            and (it.get("name") == seg or (("MTM" in it or "PSM" in it) and f"{it.get('MTM')}-{it.get('PSM')}" == seg))]
    if len(hits) == 1:
        return hits[0]
    names = [it.get("name") or f"{it.get('MTM')}-{it.get('PSM')}" for it in items if isinstance(it, dict)]
    raise PathError(f"no element named {seg!r} (elements: {', '.join(map(str, names)) or 'none'})"
                    if not hits else f"{seg!r} names more than one element")


def get(doc: Any, path: str) -> Any:
    """The value at path, or MISSING. Raises PathError if the path cannot exist in this file."""
    node = doc
    for seg in path.split("/"):
        if isinstance(node, list):
            node = node[_pick(node, seg)]
        elif isinstance(node, dict):
            if seg not in node:
                return MISSING
            node = node[seg]
        else:
            raise PathError(f"{path}: {seg!r} is below a value that is not an object or a list")
    return node


def put(doc: Any, path: str, value: Any, create_parents: bool = False) -> None:
    """Write value at path, in place (the caller passes a copy)."""
    segs = path.split("/")
    node = doc
    for i, seg in enumerate(segs[:-1]):
        if isinstance(node, list):
            node = node[_pick(node, seg)]
        elif isinstance(node, dict):
            if seg not in node:
                if not create_parents:
                    raise PathError(f"{'/'.join(segs[:i + 1])} does not exist in the file")
                node[seg] = {}
            node = node[seg]
        else:
            raise PathError(f"{path}: {seg!r} is below a value that is not an object or a list")
    last = segs[-1]
    if isinstance(node, list):
        node[_pick(node, last)] = value
    elif isinstance(node, dict):
        node[last] = value
    else:
        raise PathError(f"{path}: the parent is not an object or a list")


def canonical(doc: Any) -> str:
    def norm(x: Any) -> Any:
        if isinstance(x, float):
            return round(x, 9) + 0.0
        if isinstance(x, dict):
            return {k: norm(v) for k, v in sorted(x.items())}
        if isinstance(x, list):
            return [norm(v) for v in x]
        return x
    return json.dumps(norm(doc), sort_keys=True, ensure_ascii=False)


def doc_hash(doc: Any) -> str:
    return hashlib.sha256(canonical(doc).encode()).hexdigest()[:16]


def file_type(doc: Any) -> str:
    """dvrk-system | dvrk-psm | dvrk-ecm | dvrk-mtm from the schema a dVRK file names; vx-site
    for this project's site files; "" for a file that names no schema."""
    if isinstance(doc, dict):
        sid = doc.get("$id")
        if isinstance(sid, str) and sid.endswith(".schema.json"):
            return sid[: -len(".schema.json")]
        if "safety_profile" in doc and "named_poses" in doc:
            return "vx-site"
    return ""


# ------------------------------------------------------------------------------ dVRK JSON Schemas

class SchemaSet:
    """The JSON Schemas the dVRK project publishes in share/schemas, with the references between
    them resolved locally. Nothing is fetched."""

    def __init__(self, folder: Path | str):
        import jsonschema
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT7

        self._jsonschema = jsonschema
        self.schemas = {p.name: jsonc.load(p) for p in sorted(Path(folder).glob("*.schema.json"))}
        if not self.schemas:
            raise FileNotFoundError(f"no schemas in {folder}. Run: python scripts/fetch_dvrk_config.py")
        self._registry = Registry().with_resources(
            [(name, Resource.from_contents(s, default_specification=DRAFT7)) for name, s in self.schemas.items()])

    def names(self) -> list[str]:
        return sorted(self.schemas)

    def errors(self, doc: Any, schema_name: str | None = None) -> list[str] | None:
        """Schema errors as "path: message", or None if the file names no schema of this set."""
        name = schema_name or (doc.get("$id") if isinstance(doc, dict) else None)
        if name not in self.schemas:
            return None
        v = self._jsonschema.Draft7Validator(self.schemas[name], registry=self._registry)
        out = []
        for e in sorted(v.iter_errors(doc), key=lambda e: [str(p) for p in e.absolute_path]):
            where = "/".join(str(p) for p in e.absolute_path) or "(top level)"
            out.append(f"{where}: {e.message[:200]}")
        return out


# ------------------------------------------------------------------------------ versions

class ConfigStore:
    """A site's working set of configuration files. `sources` maps a target name to the file it
    starts from, which is only ever read. Approved changes become numbered versions under root."""

    def __init__(self, root: Path | str, sources: dict[str, Path]):
        self.root = Path(root)
        self.sources = {k: Path(v) for k, v in sources.items()}

    def _folder(self, target: str) -> Path:
        return self.root / re.sub(r"[^A-Za-z0-9_.-]", "_", target)

    def versions(self, target: str) -> list[int]:
        return sorted(int(p.stem[1:]) for p in self._folder(target).glob("v*.json"))

    def current(self, target: str) -> tuple[Any, str, int]:
        """(document, hash, version). Version 0 is the source file itself."""
        if target not in self.sources:
            raise KeyError(f"no file named {target!r} in this site's configuration "
                           f"(files: {', '.join(sorted(self.sources))})")
        vs = self.versions(target)
        if vs:
            doc = json.loads((self._folder(target) / f"v{vs[-1]:04d}.json").read_text(encoding="utf-8"))
            return doc, doc_hash(doc), vs[-1]
        doc = jsonc.load(self.sources[target])
        return doc, doc_hash(doc), 0

    def commit(self, target: str, doc: Any, note: dict[str, Any]) -> int:
        """Write doc as the next version. The source file and earlier versions are left as they are."""
        _, _, last = self.current(target)
        folder = self._folder(target)
        folder.mkdir(parents=True, exist_ok=True)
        n = last + 1
        (folder / f"v{n:04d}.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        log = folder / "log.jsonl"
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"version": n, "hash": doc_hash(doc), **note}, ensure_ascii=False) + "\n")
        return n


def site_sources(site_file: Path | str, dvrk_dir: Path | str) -> dict[str, Path]:
    """The files a site's configuration consists of: its own site file, and the arm and system
    files of the dVRK project that name a schema. Target names are "site/...", "arm/...",
    "system/..."."""
    out = {f"site/{Path(site_file).name}": Path(site_file)}
    for folder in ("arm", "system"):
        for p in sorted((Path(dvrk_dir) / "share" / folder).glob("*.json")):
            try:
                doc = jsonc.load(p)
            except ValueError:
                continue
            if file_type(doc):
                out[f"{folder}/{p.name}"] = p
    return out


def applied(doc: Any, writes: list[tuple[str, Any, bool]]) -> Any:
    """A copy of doc with (path, value, create_parents) written into it."""
    out = copy.deepcopy(doc)
    for path, value, create in writes:
        put(out, path, value, create)
    return out
