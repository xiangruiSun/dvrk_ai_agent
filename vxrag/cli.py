"""Command line for the retrieval system.

    python -m vxrag.cli ingest
    python -m vxrag.cli search "firewire nodes not found after power on"
    python -m vxrag.cli search "tool_detection" --strategy bm25:ident --sources doc schema
    python -m vxrag.cli search "potentiometer calibration" --generation Si -k 5
"""
from __future__ import annotations

import argparse
import json
import sys

from .ingest import ingest
from .retrieval import Index
from .store import CorpusStore

DEFAULT_STRATEGY = "bm25:ident"
DEFAULT_CHUNKING = "section_ctx"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vxrag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="read the downloaded sources into the store and print the report")
    s = sub.add_parser("search", help="search the corpus")
    s.add_argument("query")
    s.add_argument("--strategy", default=DEFAULT_STRATEGY)
    s.add_argument("--chunking", default=DEFAULT_CHUNKING)
    s.add_argument("--sources", nargs="*", choices=["doc", "thread", "changelog", "schema"])
    s.add_argument("--generation", choices=["Classic", "Si"])
    s.add_argument("-k", type=int, default=5)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    store = CorpusStore()
    if a.cmd == "ingest":
        print(json.dumps(ingest(store), indent=1))
        return 0
    units = store.units()
    if not units:
        print("the store is empty: run scripts/fetch_sources.py, then python -m vxrag.cli ingest", file=sys.stderr)
        return 1
    index = Index(units, a.chunking)
    hits = index.search(a.query, a.strategy, k=a.k, sources=set(a.sources) if a.sources else None,
                        generation=a.generation)
    if not hits:
        print("nothing found in the indexed sources")
        return 0
    for h in hits:
        d = index.describe(h, chars=240)
        print(f"{d['rank']}. [{d['id']}]  ({d['source']}, {d['generation']}, {d['authority']})")
        print(f"   {d['title']}")
        print(f"   {d['text']!r}")
        if d["url"]:
            print(f"   {d['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
