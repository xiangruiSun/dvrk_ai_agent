"""The node's end of the perception worker: every request has a time limit, and a late or
missing answer is "perception unavailable", never a guess."""
from __future__ import annotations

import time
from typing import Any

import httpx

from ..config import EgressBlocked, assert_egress_allowed
from .. import net

TIME_LIMIT_S = 0.5


class PerceptionUnavailable(Exception):
    pass


class PerceptionClient:
    def __init__(self, url: str, time_limit_s: float = TIME_LIMIT_S, scene: int = 0):
        self.url, self.time_limit_s, self.scene = url.rstrip("/"), time_limit_s, scene
        self.latencies_ms: list[float] = []
        self.failures = 0

    def estimate(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            assert_egress_allowed(self.url)
            r = net.post(f"{self.url}/estimate", json={"scene": self.scene}, timeout=self.time_limit_s)
            r.raise_for_status()
            out = r.json()
        except (httpx.HTTPError, EgressBlocked, ValueError) as exc:
            self.failures += 1
            raise PerceptionUnavailable(f"{type(exc).__name__} after {time.perf_counter() - t0:.2f} s "
                                        f"(limit {self.time_limit_s:g} s)") from exc
        took = time.perf_counter() - t0
        self.latencies_ms.append(took * 1000.0)
        out["latency_s"] = took
        return out

    __call__ = estimate

    def health(self) -> dict[str, Any]:
        try:
            assert_egress_allowed(self.url)
            r = net.get(f"{self.url}/health", timeout=self.time_limit_s)
            return {"reachable": r.status_code == 200, **(r.json() if r.status_code == 200 else {})}
        except (httpx.HTTPError, EgressBlocked, ValueError) as exc:
            return {"reachable": False, "error": type(exc).__name__}
