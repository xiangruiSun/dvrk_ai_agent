"""Source adapters: documentation, issue threads, the change log and configuration schemas, each
read into units of the authoritative store, with a validation report.

Ingestion is idempotent. Every unit carries a hash of its content; running it again on unchanged
sources writes nothing, an edited page updates only its units, and a page that has disappeared
upstream disappears here.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from . import rst
from .config import Settings, get_settings
from .store import PUBLIC_SOURCES, CorpusStore, Unit

MAINTAINER = {"OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR"}
MIN_SECTION_CHARS = 40


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]


def _generation(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    if "Classic" in parts:
        return "Classic"
    if "Si" in parts:
        return "Si"
    return "both"


# ------------------------------------------------------------------------------ documentation

def read_docs(s: Settings) -> tuple[list[Unit], dict[str, Any]]:
    root = s.sources_dir / s.docs_folder
    units: list[Unit] = []
    report: dict[str, Any] = {"files": 0, "sections": 0, "stub_sections": 0, "code_blocks": 0, "references": 0,
                              "labels": 0}
    for p in sorted(root.rglob("*.rst")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith("includes/"):
            continue                                   # diagram snippets pulled into other pages
        report["files"] += 1
        key = re.sub(r"^pages/", "", rel[:-4])
        doc_id = f"doc:{key}"
        sections = rst.parse(p.read_text(encoding="utf-8", errors="replace"))
        seen: dict[str, int] = {}
        for i, sec in enumerate(sections):
            anchor = rst.slug(sec.title) or "top"
            seen[anchor] = seen.get(anchor, 0) + 1
            if seen[anchor] > 1:
                anchor = f"{anchor}-{seen[anchor]}"
            stub = len(sec.text) < MIN_SECTION_CHARS
            report["sections"] += 1
            report["stub_sections"] += stub
            report["code_blocks"] += len(sec.code)
            report["references"] += len(sec.refs)
            report["labels"] += len(sec.labels)
            url = f"{s.docs_base_url}{rel[:-4]}.html" + ("" if i == 0 else f"#{anchor}")
            units.append(Unit(
                unit_id=f"{doc_id}#{anchor}", doc_id=doc_id, source="doc", kind="section", ord=i,
                title=sec.title, path=sec.path, text=sec.text, url=url, generation=_generation(rel),
                labels=sec.labels, refs=[{"target": r.target, "text": r.text, "sentence": r.sentence} for r in sec.refs],
                meta={"file": rel, "level": sec.level, "stub": stub, "code_blocks": len(sec.code)},
                content_hash=_hash(sec.title, sec.text, ",".join(sec.labels))))
    return units, report


# ------------------------------------------------------------------------------ issue threads

def read_threads(s: Settings) -> tuple[list[Unit], dict[str, Any]]:
    units: list[Unit] = []
    report: dict[str, Any] = {"threads": 0, "closed": 0, "with_replies": 0, "replies": 0, "maintainer_replies": 0}
    for f in sorted((s.sources_dir / "issues").glob("*.json")):
        for t in json.loads(f.read_text(encoding="utf-8")):
            name = t["repo"].split("/")[1]
            doc_id = f"thread:{name}/{t['number']}"
            url = f"https://github.com/{t['repo']}/issues/{t['number']}"
            meta = {"state": t["state"], "created_at": t["created_at"], "closed_at": t.get("closed_at"),
                    "labels": t.get("labels", []), "repo": t["repo"], "number": t["number"],
                    "opened_by": t.get("author_association")}
            report["threads"] += 1
            report["closed"] += t["state"] == "closed"
            body = (t.get("body") or "").strip()
            opener = "maintainer" if t.get("author_association") in MAINTAINER else "community"
            units.append(Unit(unit_id=f"{doc_id}#q", doc_id=doc_id, source="thread", kind="question", ord=0,
                              title=t["title"], path=[f"{name} #{t['number']}", t["title"]],
                              text=(t["title"] + "\n\n" + body).strip(), url=url, authority=opener, meta=meta,
                              content_hash=_hash(t["title"], body)))
            replies = [c for c in t.get("comments", []) if (c.get("body") or "").strip()]
            if not replies:
                continue
            report["with_replies"] += 1
            report["replies"] += len(replies)
            maint = [c for c in replies if c.get("author_association") in MAINTAINER]
            report["maintainer_replies"] += len(maint)
            parts = [{"by": "maintainer" if c.get("author_association") in MAINTAINER else "community",
                      "by_opener": bool(c.get("by_opener")), "created_at": c["created_at"], "text": c["body"].strip()}
                     for c in replies]
            units.append(Unit(unit_id=f"{doc_id}#a", doc_id=doc_id, source="thread", kind="replies", ord=1,
                              title=t["title"], path=[f"{name} #{t['number']}", t["title"], "replies"],
                              text="\n\n".join(p["text"] for p in parts), url=url,
                              authority="maintainer" if maint else "community",
                              meta={**meta, "replies": parts},
                              content_hash=_hash(*[p["text"] for p in parts])))
    return units, report


# ------------------------------------------------------------------------------ change log

RELEASE = re.compile(r"^(\d+\.\d+(?:\.\d+)?)\s*\(([^)]+)\)\s*$")


def read_changelog(s: Settings) -> tuple[list[Unit], dict[str, Any]]:
    path = s.sources_dir / "changelog" / "CHANGELOG.md"
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    units: list[Unit] = []
    report: dict[str, Any] = {"releases": 0, "units": 0}
    version, date, category, intro, items = "", "", "", [], []
    order = 0

    def flush() -> None:
        nonlocal items, order
        if version and items:
            text = "\n".join(items).strip()
            units.append(Unit(unit_id=f"changelog:{version}#{rst.slug(category) or 'notes'}", doc_id=f"changelog:{version}",
                              source="changelog", kind="release", ord=order, title=f"dVRK {version}: {category or 'Notes'}",
                              path=["Change log", f"{version} ({date})", category or "Notes"], text=text,
                              url="https://github.com/jhu-dvrk/sawIntuitiveResearchKit/blob/main/CHANGELOG.md",
                              version=version, meta={"released": date, "category": category or "Notes"},
                              content_hash=_hash(version, category, text)))
            order += 1
        items = []

    for i, line in enumerate(lines):
        m = RELEASE.match(line.strip())
        if m and i + 1 < len(lines) and set(lines[i + 1].strip()) <= {"="} and lines[i + 1].strip():
            flush()
            version, date, category = m.group(1), m.group(2), ""
            report["releases"] += 1
            continue
        if not version or (line.strip() and set(line.strip()) <= {"="}):
            continue
        top = re.match(r"^\* (.+?):\s*$", line)
        if top:
            flush()
            category = top.group(1).strip()
        elif line.strip():
            if not category and not line.startswith(("*", " ")):
                flush()
                category = "Notes"
            items.append(line.rstrip())
    flush()
    seen: dict[str, int] = {}
    for u in units:                                        # a category can repeat inside one release
        seen[u.unit_id] = seen.get(u.unit_id, 0) + 1
        if seen[u.unit_id] > 1:
            u.unit_id = f"{u.unit_id}-{seen[u.unit_id]}"
    report["units"] = len(units)
    return units, report


# ------------------------------------------------------------------------------ configuration schemas

def read_schemas(s: Settings) -> tuple[list[Unit], dict[str, Any]]:
    units: list[Unit] = []
    report: dict[str, Any] = {"schemas": 0, "properties": 0, "without_description": 0}
    for f in sorted(s.schemas_dir.glob("*.schema.json")):
        schema = json.loads(f.read_text(encoding="utf-8"))
        name = f.name[:-len(".schema.json")]
        title = schema.get("title", name)
        report["schemas"] += 1
        doc_id = f"schema:{name}"
        order = 0
        if schema.get("description"):
            units.append(Unit(unit_id=f"{doc_id}#root", doc_id=doc_id, source="schema", kind="property", ord=0,
                              title=title, path=["Configuration schemas", title], text=schema["description"],
                              meta={"file": f.name}, content_hash=_hash(schema["description"])))
            order = 1

        def walk(node: dict[str, Any], prefix: str, required: set[str]) -> None:
            nonlocal order
            for key, prop in (node.get("properties") or {}).items():
                if not isinstance(prop, dict):
                    continue
                path = f"{prefix}.{key}" if prefix else key
                kind = prop.get("type") or ("reference" if "$ref" in prop else "any")
                desc = (prop.get("description") or "").strip()
                report["properties"] += 1
                if not desc:
                    report["without_description"] += 1
                facts = [f"`{path}` in `{f.name}`", f"type: {kind}"]
                if key in required:
                    facts.append("required")
                if prop.get("enum"):
                    facts.append("allowed values: " + ", ".join(f"`{v}`" for v in prop["enum"]))
                if "default" in prop:
                    facts.append(f"default: `{json.dumps(prop['default'])}`")
                if "$ref" in prop:
                    facts.append(f"defined by `{prop['$ref']}`")
                text = "; ".join(facts) + ".\n\n" + desc if desc else "; ".join(facts) + "."
                units.append(Unit(unit_id=f"{doc_id}#{path}", doc_id=doc_id, source="schema", kind="property",
                                  ord=order, title=f"{title}: {path}", path=["Configuration schemas", title, path],
                                  text=text, meta={"file": f.name, "key_path": path, "type": kind,
                                                   "has_description": bool(desc)},
                                  content_hash=_hash(path, text)))
                order += 1
                walk(prop, path, set(prop.get("required") or []))
                items = prop.get("items")
                if isinstance(items, dict):
                    walk(items, path + "[]", set(items.get("required") or []))

        walk(schema, "", set(schema.get("required") or []))
    return units, report


# ------------------------------------------------------------------------------ a site's own notes

ATX = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")


def split_markdown(text: str) -> list[tuple[list[str], str]]:
    """Sections of a Markdown or plain-text note: (heading path, body). Fenced code is never cut."""
    lines = text.replace("\r\n", "\n").split("\n")
    out: list[tuple[list[str], str]] = []
    path: list[tuple[int, str]] = []
    body: list[str] = []
    fenced = False

    def flush() -> None:
        content = "\n".join(body).strip()
        if content or path:
            out.append(([t for _, t in path], content))
        body.clear()

    def open_section(level: int, title: str) -> None:
        flush()
        while path and path[-1][0] >= level:
            path.pop()
        path.append((level, title))

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            fenced = not fenced
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        m = None if fenced else ATX.match(line)
        if m:
            open_section(len(m.group(1)), m.group(2))
        elif not fenced and line.strip() and len(nxt) >= 3 and set(nxt) <= {"="}:
            open_section(1, line.strip())
            i += 1
        elif not fenced and line.strip() and len(nxt) >= 3 and set(nxt) <= {"-"} and not line.lstrip().startswith(("*", "-")):
            open_section(2, line.strip())
            i += 1
        else:
            body.append(line)
        i += 1
    flush()
    return [(p, b) for p, b in out if b]


def read_site_notes(s: Settings) -> tuple[list[Unit], dict[str, Any]]:
    """Notes a site keeps beside its configuration files. Each site's notes are a collection of
    their own, `site:<name>`, which retrieval shows to that site only (vxrag/access.py)."""
    root = s.sources_dir / "sites"
    manifest_path = s.sources_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")).get("sites", {}) if manifest_path.exists() else {}
    units: list[Unit] = []
    report: dict[str, Any] = {"sites": 0, "files": 0, "sections": 0, "by_site": {}}
    for site_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        site = site_dir.name
        info = manifest.get(site, {})
        report["sites"] += 1
        n = 0
        for f in sorted(site_dir.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in (".md", ".txt"):
                continue
            rel = f.relative_to(site_dir).as_posix()
            report["files"] += 1
            doc_id = f"note:{site}/{rel.rsplit('.', 1)[0]}"
            system = rel.split("/")[0] if "/" in rel else ""
            seen: dict[str, int] = {}
            for i, (path, body) in enumerate(split_markdown(f.read_text(encoding="utf-8", errors="replace"))):
                title = path[-1] if path else rel
                anchor = rst.slug(title) or "top"
                seen[anchor] = seen.get(anchor, 0) + 1
                if seen[anchor] > 1:
                    anchor = f"{anchor}-{seen[anchor]}"
                units.append(Unit(
                    unit_id=f"{doc_id}#{anchor}", doc_id=doc_id, source="note", kind="section", ord=i, title=title,
                    path=[f"{site} site notes", rel, *path], text=body, generation=_generation_of_system(system),
                    authority="site", collection=f"site:{site}",
                    meta={"site": site, "file": rel, "system": system, "release": info.get("release", ""),
                          "repository": info.get("repository", ""), "commit": info.get("commit", "")},
                    content_hash=_hash(title, body)))
                n += 1
        report["sections"] += n
        report["by_site"][site] = {"sections": n, "release": info.get("release", "")}
    return units, report


def _generation_of_system(system: str) -> str:
    """A site names its systems, and a system built from Si arms carries "Si" in its name."""
    if not system:
        return "both"
    return "Si" if re.search(r"(?:^|-)Si(?:$|-)", system) else "Classic"


# ------------------------------------------------------------------------------ run

READERS = {"doc": read_docs, "thread": read_threads, "changelog": read_changelog, "schema": read_schemas,
           "note": read_site_notes}


def ingest(store: CorpusStore | None = None, settings: Settings | None = None,
           sources: tuple[str, ...] = PUBLIC_SOURCES) -> dict[str, Any]:
    s = settings or get_settings()
    store = store or CorpusStore(s.db_path)
    report: dict[str, Any] = {"sources": {}, "problems": []}
    all_units: list[Unit] = []
    for src in sources:
        units, rep = READERS[src](s)
        ids = [u.unit_id for u in units]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            report["problems"].append({"source": src, "duplicate_unit_ids": dupes[:10]})
            units = list({u.unit_id: u for u in units}.values())
        rep["units"] = len(units)
        rep["written"] = store.replace_source(src, units)
        report["sources"][src] = rep
        all_units.extend(units)
    # Validation across sources: every cross-reference should point at a label that exists.
    labels = {lb for u in all_units for lb in u.labels}
    refs = [(u.unit_id, r["target"]) for u in all_units for r in u.refs]
    dangling = sorted({t for _, t in refs if t not in labels})
    report["validation"] = {"labels": len(labels), "references": len(refs),
                            "references_to_unknown_labels": len([1 for _, t in refs if t not in labels]),
                            "unknown_labels": dangling[:25]}
    report["corpus_hash"] = store.corpus_hash()
    report["units"] = store.counts(sources)
    store.log_run(report)
    return report
