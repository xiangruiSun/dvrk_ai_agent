"""Command line for the retrieval system.

    python -m vxrag.cli ingest                      public sources into the store
    python -m vxrag.cli sync                        all sources; prints exactly what changed
    python -m vxrag.cli search "firewire nodes not found after power on"
    python -m vxrag.cli search "console.json" --release 2.3.0 --generation Classic
    python -m vxrag.cli answer "How do I calibrate the potentiometers of a PSM?" --generation Si
    python -m vxrag.cli index build | validate <v> | activate <v> | rollback | list
    python -m vxrag.cli keys add --role engineer --site ucl --generation Classic --release 2.3.0
    python -m vxrag.cli serve

`search` and `answer` on the command line act as a caller who may read the public collection
only, unless --site is given. The command line runs on the machine that holds the store, so it
is trusted with that choice; the HTTP service is not, and takes the caller from the key.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import versions
from .access import KeyRing, Principal
from .config import get_settings
from .engine import Engine
from .ingest import ingest
from .retrieval import Index
from .store import SOURCES, CorpusStore

DEFAULT_STRATEGY = "bm25:ident"
DEFAULT_CHUNKING = "section_ctx"


def _engine(with_model: bool) -> Engine | None:
    """The active version if there is one, otherwise an index built from the store on the spot."""
    generator = None
    if with_model:
        from .llm import OllamaGenerator
        generator = OllamaGenerator()
    active = versions.pointer()["active"]
    if active:
        index, m = versions.load(active)
        return Engine(index, m["config"], version=active, generator=generator)
    units = CorpusStore().units()
    if not units:
        return None
    return Engine(Index(units, DEFAULT_CHUNKING), version="unversioned", generator=generator)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vxrag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="read the downloaded public sources into the store and print the report")
    sub.add_parser("sync", help="bring the store up to date with all downloaded sources and print what changed")
    for name in ("search", "answer"):
        s = sub.add_parser(name)
        s.add_argument("query")
        s.add_argument("--sources", nargs="*", choices=list(SOURCES))
        s.add_argument("--generation", choices=["Classic", "Si"])
        s.add_argument("--release")
        s.add_argument("--site", help="act as an engineer of this site (reads its notes)")
        s.add_argument("-k", type=int, default=5)
    ix = sub.add_parser("index", help="build, validate, activate, roll back and list index versions")
    ix.add_argument("action", choices=["build", "validate", "activate", "rollback", "list"])
    ix.add_argument("version", nargs="?")
    ix.add_argument("--strategy", default=DEFAULT_STRATEGY)
    ix.add_argument("--chunking", default=DEFAULT_CHUNKING)
    ix.add_argument("--note", default="")
    keys = sub.add_parser("keys", help="create an API key; the token is printed once")
    keys.add_argument("action", choices=["add", "list"])
    keys.add_argument("--role", default="engineer")
    keys.add_argument("--site", default="")
    keys.add_argument("--generation", choices=["Classic", "Si"])
    keys.add_argument("--release")
    keys.add_argument("--id", default="")
    sub.add_parser("serve", help="run the HTTP service on the loopback address")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = get_settings()

    if a.cmd == "ingest":
        print(json.dumps(ingest(CorpusStore()), indent=1))
        return 0
    if a.cmd == "sync":
        report = versions.sync(CorpusStore())
        report.pop("changed_unit_ids")
        print(json.dumps(report, indent=1))
        return 0
    if a.cmd == "index":
        try:
            if a.action == "build":
                out = versions.build(CorpusStore(), a.strategy, a.chunking, note=a.note)
            elif a.action == "list":
                out = {"pointer": versions.pointer(), "versions": versions.list_versions()}
            elif a.action == "rollback":
                out = versions.rollback()
            elif not a.version:
                print("name the version", file=sys.stderr)
                return 2
            elif a.action == "validate":
                m = versions.validate(a.version)
                out = {"version": m["version"], "status": m["status"], "gates": m["gates"]}
            else:
                out = versions.activate(a.version)
        except versions.VersionError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(out, indent=1))
        return 0
    if a.cmd == "keys":
        ring = KeyRing(settings.keys_path)
        if a.action == "list":
            for p in ring.principals():
                print(f"{p.key_id}: role {p.role}, site {p.site or '-'}, {p.generation or '?'} / {p.release or '?'}")
            return 0
        key_id = a.id or f"{a.role}-{a.site or 'any'}-{int(time.time())}"
        token = ring.issue(Principal(key_id, a.role, a.site, a.generation, a.release))
        print(f"key {key_id} created. The token is shown once:\n{token}")
        return 0
    if a.cmd == "serve":
        from .llm import OllamaGenerator
        from .service import EngineHolder, Server
        holder = EngineHolder(settings=settings, generator=OllamaGenerator())
        if holder.reload() is None:
            print("no index version is active: run `index build`, `index validate` and `index activate` first",
                  file=sys.stderr)
            return 1
        server = Server(holder, KeyRing(settings.keys_path), settings.host, settings.port).start()
        print(f"serving {holder.engine.version} on {server.url}")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            server.stop()
        return 0

    engine = _engine(with_model=a.cmd == "answer")
    if engine is None:
        print("the store is empty: run scripts/fetch_sources.py, then python -m vxrag.cli sync", file=sys.stderr)
        return 1
    who = Principal("cli", "engineer" if a.site else "support", a.site, a.generation, a.release)
    if a.cmd == "search":
        out = engine.search(who, a.query, k=a.k, sources=a.sources, chars=240)
        if out["expanded_query"]:
            print(f"searched as: {out['expanded_query']}")
        if not out["results"]:
            print("nothing found in the indexed sources")
        for d in out["results"]:
            print(f"{d['rank']}. [{d['id']}]  ({d['source']}, {d['generation']}, {d['authority']}, "
                  f"release {d['release'] or '?'}, applies: {d['applies']})")
            print(f"   {d['title']}")
            print(f"   {d['text']!r}")
            for n in d["notes"]:
                print(f"   note: {n}")
        return 0
    out = engine.answer(who, a.query, k=a.k)
    if not out["answered"]:
        print(f"no answer ({out['abstained_by']}): {out['reason']}")
        return 0
    for i, s in enumerate(out["steps"], 1):
        flag = "  [unconfirmed: rests on a community reply]" if s["unconfirmed"] else ""
        print(f"{i}. {s['text']}  {' '.join('[' + c + ']' for c in s['citations'])}{flag}")
    print(f"relies on: {out['relies_on']} text; applies to: {out['applies_to']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
