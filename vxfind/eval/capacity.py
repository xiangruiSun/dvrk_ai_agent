"""Capacity: how many questions per second the read path answers, and what the index costs.

A closed-loop load test: N workers, each sends a search, waits for the reply, sends the next,
for a fixed time. Reported: completed searches per second and the latency a caller saw (median,
95th and 99th percentile), per retriever and per N. Query vectors are computed beforehand unless
the test says otherwise, so the search engine is measured and not the embedding model.

What this does and does not show: the engine, the load generator and the embedding model share
one machine, and the index is one shard on one node. The numbers are this machine's, at this
catalog size. They show the shape (which stage costs what, where throughput stops growing), not
what a cluster would do.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from ..search.engine import Engine, SearchConfig


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def load_test(engine: Engine, cfg: SearchConfig, needs: list[str], workers: int, seconds: float = 15.0,
              warm_vectors: bool = True) -> dict[str, Any]:
    if warm_vectors and cfg.retriever in ("knn", "rrf"):
        for n in needs:
            engine.embed(cfg.embedding, n)
    elif not warm_vectors:
        engine._qcache.clear()
    lat: list[float] = []
    errors = [0]
    lock = threading.Lock()
    stop = time.perf_counter() + seconds

    def work(offset: int) -> None:
        i = offset
        mine = []
        while time.perf_counter() < stop:
            need = needs[i % len(needs)]
            i += 1
            t0 = time.perf_counter()
            try:
                if not warm_vectors:
                    engine._qcache.pop((cfg.embedding, need), None)
                engine.search(need, cfg)
                mine.append((time.perf_counter() - t0) * 1000)
            except Exception:
                with lock:
                    errors[0] += 1
        with lock:
            lat.extend(mine)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=work, args=(k * 7,)) for k in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    took = time.perf_counter() - t0
    return {"retriever": cfg.name, "workers": workers, "seconds": round(took, 1), "searches": len(lat), "errors": errors[0],
            "per_second": round(len(lat) / took, 1), "p50_ms": round(percentile(lat, 0.5), 1), "p95_ms": round(percentile(lat, 0.95), 1),
            "p99_ms": round(percentile(lat, 0.99), 1), "query_vectors": "computed beforehand" if warm_vectors else "computed per search"}


def growth(created_month: dict[str, int], last_n: int = 6) -> dict[str, Any]:
    """Tables created per month (real, from the catalog) and a straight-line projection."""
    months = sorted(m for m in created_month if m)
    recent = months[-(last_n + 1):-1]                 # the last month is incomplete in a snapshot
    per_month = sum(created_month[m] for m in recent) / len(recent) if recent else 0.0
    before = months[-(2 * last_n + 1):-(last_n + 1)]
    per_month_before = sum(created_month[m] for m in before) / len(before) if before else 0.0
    total = sum(created_month.values())
    return {"first_month": months[0] if months else None, "last_full_month": recent[-1] if recent else None,
            "created_per_month_recent": {m: created_month[m] for m in recent}, "mean_created_per_month_recent": round(per_month),
            "tables_now": total, "monthly_growth_of_catalog": round(per_month / total, 4) if total else None,
            "mean_created_per_month_in_the_six_months_before": round(per_month_before),
            "projected_tables_in_12_months": round(total + 12 * per_month),
            "note": "a straight line from the last six full months. The counts are of tables still carried by the dump, which never "
                    "drops one, so they are creations, not net growth; the newest months are also the ones whose tables have had the "
                    "least time to get a card"}
