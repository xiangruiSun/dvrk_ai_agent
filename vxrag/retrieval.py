"""An index over the corpus and the retrieval strategies that run on it.

Every strategy is a configuration, not a code path chosen in advance:

    bm25:<tokenizer>                     lexical
    dense:<model>                        embeddings
    rrf:<strategy>+<strategy>[+...]      reciprocal-rank fusion of any of the above

Chunks are scored; units are returned. A unit's score is that of its best chunk, so a long
section does not win by having more chunks.

Filters are applied inside the search, before ranking, so a filtered search still returns k
results and a unit the caller may not read is never a candidate:

    collections   which collections the caller may read (default: the public one only)
    generation    drop pages written for the other hardware generation
    max_release   drop release notes of releases newer than the site runs
    sources       documentation, threads, change log, schemas, notes
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .applicability import version_key
from .chunking import Chunk, chunk_units
from .dense import Embedder
from .lexical import BM25
from .store import PUBLIC, Unit
from .untrusted import scan

RRF_K = 60
PUBLIC_ONLY = frozenset({PUBLIC})
# How much a source's standing counts when trust is switched on. "tier" is the strict version:
# official text first, then maintainers' replies, then everyone else's.
TRUST_WEIGHT = {"official": 1.0, "site": 1.0, "maintainer": 0.9, "community": 0.8}
TRUST_TIER = {"official": 0, "site": 0, "maintainer": 1, "community": 2}
TRUST_MODES = ("off", "prior", "tier")


@dataclass
class Hit:
    unit_id: str
    score: float
    chunk_id: str
    rank: int = 0
    parts: dict[str, int] = field(default_factory=dict)       # rank given by each fused strategy


class Index:
    def __init__(self, units: list[Unit], chunking: str = "section_ctx"):
        self.units = {u.unit_id: u for u in units}
        self.chunking = chunking
        self.chunks: list[Chunk] = chunk_units(units, chunking)
        self.chunk_by_id = {c.chunk_id: c for c in self.chunks}
        self.chunk_unit = np.array([c.unit_id for c in self.chunks])
        owners = [self.units[c.unit_id] for c in self.chunks]
        self._collection = np.array([u.collection for u in owners])
        self._source = np.array([u.source for u in owners])
        self._generation = np.array([u.generation for u in owners])
        self._release = [version_key(u.version) if u.source == "changelog" and u.version else None for u in owners]
        self._trust = np.array([TRUST_WEIGHT.get(u.authority, 0.8) for u in owners])
        # Units that contain text addressed to an automated reader (vxrag/untrusted.py).
        self.suspect: dict[str, list[dict[str, Any]]] = {}
        for u in units:
            if u.source in ("thread", "note"):
                found = scan(u.text)
                if found:
                    self.suspect[u.unit_id] = found
        self._bm25: dict[str, BM25] = {}
        self._dense: dict[str, np.ndarray] = {}
        self._embedders: dict[str, Embedder] = {}
        self.build_seconds: dict[str, float] = {}

    # ---------------------------------------------------------------- building
    def bm25(self, tokenizer: str) -> BM25:
        if tokenizer not in self._bm25:
            t0 = time.perf_counter()
            self._bm25[tokenizer] = BM25([c.index_text for c in self.chunks], tokenizer)
            self.build_seconds[f"bm25:{tokenizer}"] = round(time.perf_counter() - t0, 2)
        return self._bm25[tokenizer]

    def dense(self, model: str) -> np.ndarray:
        if model not in self._dense:
            t0 = time.perf_counter()
            self._embedders[model] = Embedder(model)
            self._dense[model] = self._embedders[model].embed([c.index_text for c in self.chunks])
            self.build_seconds[f"dense:{model}"] = round(time.perf_counter() - t0, 2)
        return self._dense[model]

    def prepare(self, strategy: str) -> None:
        for part in _parts(strategy):
            kind, _, arg = part.partition(":")
            self.bm25(arg) if kind == "bm25" else self.dense(arg)

    # ---------------------------------------------------------------- searching
    def _chunk_scores(self, part: str, query: str) -> np.ndarray:
        kind, _, arg = part.partition(":")
        if kind == "bm25":
            return self.bm25(arg).scores(query)
        if kind == "dense":
            matrix = self.dense(arg)
            return matrix @ self._embedders[arg].embed([query], query=True)[0]
        raise ValueError(f"unknown retrieval strategy {part!r}")

    def _rank_units(self, scores: np.ndarray, allowed: np.ndarray, depth: int) -> list[Hit]:
        """Best chunk per unit, best units first. Chunks that scored nothing are not results."""
        order = np.argsort(-scores, kind="stable")
        seen: set[str] = set()
        hits: list[Hit] = []
        for i in order:
            if scores[i] <= 0 or len(hits) >= depth:
                break
            if not allowed[i]:
                continue
            uid = self.chunk_unit[i]
            if uid in seen:
                continue
            seen.add(uid)
            hits.append(Hit(unit_id=str(uid), score=float(scores[i]), chunk_id=self.chunks[i].chunk_id))
        return hits

    def allowed(self, collections: Iterable[str] = PUBLIC_ONLY, exclude_units: set[str] | None = None,
                sources: set[str] | None = None, generation: str | None = None,
                max_release: str | None = None) -> np.ndarray:
        mask = np.isin(self._collection, list(collections))
        if sources:
            mask &= np.isin(self._source, list(sources))
        if generation:
            mask &= np.isin(self._generation, ["both", generation])     # not the other generation's pages
        if exclude_units:
            mask &= ~np.isin(self.chunk_unit, list(exclude_units))
        if max_release:
            limit = version_key(max_release)
            mask &= np.array([r is None or r <= limit for r in self._release])
        return mask

    def search(self, query: str, strategy: str = "bm25:ident", k: int = 10, exclude_units: set[str] | None = None,
               sources: set[str] | None = None, generation: str | None = None, depth: int = 50,
               collections: Iterable[str] = PUBLIC_ONLY, max_release: str | None = None,
               trust: str = "off") -> list[Hit]:
        if trust not in TRUST_MODES:
            raise ValueError(f"unknown trust mode {trust!r}; choose from {TRUST_MODES}")
        allowed = self.allowed(collections, exclude_units, sources, generation, max_release)
        parts = _parts(strategy)
        lists = {}
        for p in parts:
            scores = self._chunk_scores(p, query)
            if trust == "prior":
                scores = scores * self._trust
            lists[p] = self._rank_units(scores, allowed, depth)
        if len(parts) == 1:
            hits = lists[parts[0]]
        else:
            fused: dict[str, Hit] = {}
            for p, ranked in lists.items():
                for r, h in enumerate(ranked, 1):
                    f = fused.setdefault(h.unit_id, Hit(unit_id=h.unit_id, score=0.0, chunk_id=h.chunk_id))
                    f.score += 1.0 / (RRF_K + r)
                    f.parts[p] = r
            hits = sorted(fused.values(), key=lambda h: (-h.score, h.unit_id))
        if trust == "tier":
            hits = sorted(hits, key=lambda h: TRUST_TIER.get(self.units[h.unit_id].authority, 2))     # stable
        hits = hits[:k]
        for r, h in enumerate(hits, 1):
            h.rank = r
        return hits

    def describe(self, hit: Hit, chars: int = 400) -> dict[str, Any]:
        u = self.units[hit.unit_id]
        chunk = self.chunk_by_id[hit.chunk_id]
        return {"id": u.unit_id, "rank": hit.rank, "score": round(hit.score, 4), "source": u.source,
                "title": " > ".join(u.path), "url": u.url, "generation": u.generation, "authority": u.authority,
                "version": u.version, "text": chunk.text[:chars]}


def _parts(strategy: str) -> list[str]:
    if strategy.startswith("rrf:"):
        return strategy[4:].split("+")
    return [strategy]
