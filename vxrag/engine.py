"""The engine: one index version, the rules that decide what a caller may read and what
applies to their site, a cache, and counters. The command line and the HTTP service both go
through it, so there is one place where access is decided.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from typing import Any

import numpy as np

from . import answer as answering
from .access import Principal, visible_collections
from .applicability import Applicability, SiteProfile
from .llm import Generator
from .logline import normalise
from .retrieval import Index
from .store import PUBLIC_SOURCES, SOURCES

DEFAULT_CONFIG = {"strategy": "bm25:ident", "chunking": "section_ctx", "trust": "off"}


class NotAuthorized(Exception):
    pass


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counts: dict[str, int] = {}
        self.latency: dict[str, deque[float]] = {}

    def count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counts[name] = self.counts.get(name, 0) + n

    def observe(self, op: str, ms: float) -> None:
        with self._lock:
            self.latency.setdefault(op, deque(maxlen=2000)).append(ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            lat = {op: {"n": len(v), "p50_ms": round(float(np.percentile(v, 50)), 2),
                        "p95_ms": round(float(np.percentile(v, 95)), 2)} for op, v in self.latency.items() if v}
            hits, misses = self.counts.get("cache_hit", 0), self.counts.get("cache_miss", 0)
            return {"counts": dict(self.counts), "latency": lat,
                    "cache_hit_rate": round(hits / (hits + misses), 3) if hits + misses else None}


class Engine:
    def __init__(self, index: Index, config: dict[str, Any] | None = None, version: str = "adhoc",
                 generator: Generator | None = None, cache_size: int = 512):
        self.index = index
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.version = version
        self.generator = generator
        self.applicability = Applicability(u for u in index.units.values() if u.source in PUBLIC_SOURCES)
        self.metrics = Metrics()
        self._cache: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        index.prepare(self.config["strategy"])

    # ------------------------------------------------------------------ helpers
    def _profile(self, principal: Principal, generation: str | None, release: str | None) -> SiteProfile:
        return SiteProfile(generation or principal.generation, release or principal.release)

    def _retrieve(self, principal: Principal, query: str, k: int, sources: list[str] | None,
                  profile: SiteProfile, expand: bool, exclude_units: set[str] | None = None):
        widened, terms = self.applicability.expand(query, profile) if expand else (query, [])
        hits = self.index.search(
            widened, self.config["strategy"], k=k, exclude_units=exclude_units,
            sources=set(sources) if sources else None, generation=profile.generation,
            collections=visible_collections(principal), max_release=profile.release, trust=self.config["trust"])
        return widened, terms, hits

    def _result(self, hit, profile: SiteProfile, chars: int) -> dict[str, Any]:
        u = self.index.units[hit.unit_id]
        chunk = self.index.chunk_by_id[hit.chunk_id]
        note = self.applicability.annotate(u, profile, chunk.text)
        suspect = hit.unit_id in self.index.suspect
        # Text that addresses an automated reader is not handed to one. A person can open the url.
        text = "[withheld: this text addresses an automated reader]" if suspect else chunk.text[:chars]
        return {"id": u.unit_id, "rank": hit.rank, "score": round(hit.score, 4), "source": u.source,
                "authority": u.authority, "title": " > ".join(u.path), "url": u.url, "generation": u.generation,
                "release": note["era"], "applies": note["applies"], "notes": note["notes"],
                "addressed_to_automated_reader": suspect, "text": text}

    # ------------------------------------------------------------------ operations
    def search(self, principal: Principal | None, query: str, k: int = 5, sources: list[str] | None = None,
               generation: str | None = None, release: str | None = None, expand: bool = True,
               chars: int = 500, log_line: bool = False) -> dict[str, Any]:
        if principal is None:
            raise NotAuthorized("a search needs a key")
        t0 = time.perf_counter()
        for s in sources or []:
            if s not in SOURCES:
                raise ValueError(f"unknown source {s!r}; choose from {SOURCES}")
        profile = self._profile(principal, generation, release)
        removed: list[str] = []
        asked = query
        if log_line:
            query, removed = normalise(query)
        # The key holds what the caller may read, so one site's cached results are never another's.
        key = (self.version, tuple(sorted(visible_collections(principal))), query, k, tuple(sources or ()),
               profile.generation, profile.release, expand, chars)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
        if cached is not None:
            self.metrics.count("cache_hit")
            self.metrics.observe("search", (time.perf_counter() - t0) * 1000)
            return {**cached, "cached": True}
        self.metrics.count("cache_miss")
        widened, terms, hits = self._retrieve(principal, query, k, sources, profile, expand)
        out = {"index_version": self.version, "query": query, "expanded_query": widened if widened != query else None,
               "terms": terms, "profile": {"generation": profile.generation, "release": profile.release},
               "results": [self._result(h, profile, chars) for h in hits], "cached": False}
        if log_line:
            out["log_line"] = {"removed": removed, "characters_in": len(asked), "characters_searched": len(query)}
        with self._lock:
            self._cache[key] = out
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        self.metrics.count("search")
        self.metrics.observe("search", (time.perf_counter() - t0) * 1000)
        return out

    def answer(self, principal: Principal | None, question: str, k: int = 6, generation: str | None = None,
               release: str | None = None, budget_tokens: int = 2400, exclude_units: set[str] | None = None,
               **options: Any) -> dict[str, Any]:
        """A grounded answer or an abstention. `exclude_units` exists for evaluation in process;
        the HTTP service does not accept it."""
        if principal is None:
            raise NotAuthorized("an answer needs a key")
        if self.generator is None:
            raise RuntimeError("this engine was started without a model")
        t0 = time.perf_counter()
        profile = self._profile(principal, generation, release)
        _, terms, hits = self._retrieve(principal, question, k, None, profile, True, exclude_units)
        bundle = answering.build_bundle(self.index, hits, budget_tokens, profile, self.applicability,
                                        screen_untrusted=options.pop("screen_untrusted", True))
        out = answering.answer(question, bundle, self.generator, profile, terms, **options)
        out["index_version"] = self.version
        out["bundle_tokens"] = bundle.tokens
        self.metrics.count("answer" if out["answered"] else "abstain")
        self.metrics.observe("answer", (time.perf_counter() - t0) * 1000)
        return out

    def unit(self, principal: Principal | None, unit_id: str) -> dict[str, Any] | None:
        """One unit in full. A unit the caller may not read is reported as not existing."""
        if principal is None:
            raise NotAuthorized("reading a unit needs a key")
        u = self.index.units.get(unit_id)
        if u is None or u.collection not in visible_collections(principal):
            self.metrics.count("unit_not_found")
            return None
        note = self.applicability.annotate(u, principal.profile)
        return {"id": u.unit_id, "source": u.source, "authority": u.authority, "title": " > ".join(u.path),
                "url": u.url, "generation": u.generation, "release": note["era"], "applies": note["applies"],
                "notes": note["notes"], "text": u.text[:8000]}

    def describe(self) -> dict[str, Any]:
        by_source: dict[str, int] = {}
        for u in self.index.units.values():
            by_source[u.source] = by_source.get(u.source, 0) + 1
        return {"index_version": self.version, "config": self.config, "units": by_source,
                "chunks": len(self.index.chunks), "latest_release": self.applicability.latest,
                "renames_known": len(self.applicability.renames), "model": getattr(self.generator, "model", None)}
