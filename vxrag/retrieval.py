"""An index over the corpus and the retrieval strategies that run on it.

Every strategy is a configuration, not a code path chosen in advance:

    bm25:<tokenizer>                     lexical
    dense:<model>                        embeddings
    rrf:<strategy>+<strategy>[+...]      reciprocal-rank fusion of any of the above

Chunks are scored; units are returned. A unit's score is that of its best chunk, so a long
section does not win by having more chunks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .chunking import Chunk, chunk_units
from .dense import Embedder
from .lexical import BM25
from .store import Unit

RRF_K = 60


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
        self.chunk_unit = np.array([c.unit_id for c in self.chunks])
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

    def search(self, query: str, strategy: str = "bm25:ident", k: int = 10, exclude_units: set[str] | None = None,
               sources: set[str] | None = None, generation: str | None = None, depth: int = 50) -> list[Hit]:
        allowed = np.ones(len(self.chunks), dtype=bool)
        if exclude_units or sources or generation:
            for i, c in enumerate(self.chunks):
                u = self.units[c.unit_id]
                if exclude_units and c.unit_id in exclude_units:
                    allowed[i] = False
                elif sources and u.source not in sources:
                    allowed[i] = False
                elif generation and u.generation not in ("both", generation):
                    allowed[i] = False           # a page written for the other hardware generation
        parts = _parts(strategy)
        lists = {p: self._rank_units(self._chunk_scores(p, query), allowed, depth) for p in parts}
        if len(parts) == 1:
            hits = lists[parts[0]][:k]
        else:
            fused: dict[str, Hit] = {}
            for p, ranked in lists.items():
                for r, h in enumerate(ranked, 1):
                    f = fused.setdefault(h.unit_id, Hit(unit_id=h.unit_id, score=0.0, chunk_id=h.chunk_id))
                    f.score += 1.0 / (RRF_K + r)
                    f.parts[p] = r
            hits = sorted(fused.values(), key=lambda h: (-h.score, h.unit_id))[:k]
        for r, h in enumerate(hits, 1):
            h.rank = r
        return hits

    def describe(self, hit: Hit, chars: int = 400) -> dict[str, Any]:
        u = self.units[hit.unit_id]
        chunk = next(c for c in self.chunks if c.chunk_id == hit.chunk_id)
        return {"id": u.unit_id, "rank": hit.rank, "score": round(hit.score, 4), "source": u.source,
                "title": " > ".join(u.path), "url": u.url, "generation": u.generation, "authority": u.authority,
                "version": u.version, "text": chunk.text[:chars]}


def _parts(strategy: str) -> list[str]:
    if strategy.startswith("rrf:"):
        return strategy[4:].split("+")
    return [strategy]
