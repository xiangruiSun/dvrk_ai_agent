"""The live check: what the source site says about a table right now.

The index is a copy and the catalog database is a copy. Between the moment a table was indexed
and the moment a person is shown its address, the table may have been deleted, made private,
moved to another owner, or changed. The only way to know is to ask the site.

One request per table, to the site's public metadata endpoint. What leaves this machine is the
table's public id and nothing else: no user text, no user identity, no token.

    200              the table exists; the reply carries its last change and revision
    301/302/307/308  the table has moved; the Location header names its new address
    401, 404         gone or private (the site answers 401 for both to an anonymous caller)
    429              the site is limiting us: a transient failure, not a fact about the table

The site allows 500 requests per 300 seconds per caller. `RateLimiter` keeps this process under
a budget below that, and `HubSite` remembers answers for a while, so a table shown to three
people in ten minutes is asked about once.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable

import httpx

API = "https://huggingface.co/api/datasets/"
PAGE = "https://huggingface.co/datasets/"
AGENT = "generic-client/1.0"


class RateLimited(Exception):
    """The budget for live checks is used up for now."""


class RateLimiter:
    """At most `calls` in any window of `per_s` seconds (sliding window)."""

    def __init__(self, calls: int = 400, per_s: float = 300.0, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.calls, self.per_s, self.clock, self.sleep = calls, per_s, clock, sleep
        self._times: deque[float] = deque()
        self._lock = threading.Lock()
        self.waited_s = 0.0

    def wait_needed(self) -> float:
        with self._lock:
            now = self.clock()
            while self._times and now - self._times[0] >= self.per_s:
                self._times.popleft()
            return 0.0 if len(self._times) < self.calls else self.per_s - (now - self._times[0])

    def acquire(self, max_wait_s: float = 0.0) -> None:
        """Take one call from the budget, waiting up to `max_wait_s`; raise RateLimited otherwise."""
        while True:
            wait = self.wait_needed()
            if wait <= 0:
                with self._lock:
                    self._times.append(self.clock())
                return
            if wait > max_wait_s:
                raise RateLimited(f"live-check budget used up; free again in {wait:.0f} s")
            self.sleep(min(wait, 1.0))
            self.waited_s += min(wait, 1.0)
            max_wait_s -= min(wait, 1.0)

    def remaining(self) -> int:
        self.wait_needed()
        return self.calls - len(self._times)


def http_get(url: str, client: httpx.Client | None = None) -> tuple[int, dict[str, str], Any]:
    c = client or _client()
    r = c.get(url)
    body = None
    if r.status_code == 200:
        try:
            body = r.json()
        except ValueError:
            body = None
    return r.status_code, {k.lower(): v for k, v in r.headers.items()}, body


_shared: httpx.Client | None = None


def _client() -> httpx.Client:
    global _shared
    if _shared is None:
        _shared = httpx.Client(follow_redirects=False, timeout=15.0, headers={"User-Agent": AGENT})
    return _shared


class HubSite:
    def __init__(self, get: Callable[[str], tuple[int, dict[str, str], Any]] | None = None, limiter: RateLimiter | None = None,
                 cache_ttl_s: float = 900.0, clock: Callable[[], float] = time.time, max_wait_s: float = 0.0):
        self.get = get or http_get
        self.limiter = limiter or RateLimiter()
        self.cache_ttl_s, self.clock, self.max_wait_s = cache_ttl_s, clock, max_wait_s
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.requests = 0
        self.cache_hits = 0

    def check(self, table_id: str) -> dict[str, Any]:
        """What the site says about one table now. Never raises for a fact about the table; a
        failure to ask (limit, network) comes back as state 'unknown' with the reason."""
        with self._lock:
            hit = self._cache.get(table_id)
            if hit and self.clock() - hit["checked_at"] < self.cache_ttl_s:
                self.cache_hits += 1
                return {**hit, "from_cache": True}
        out = self._ask(table_id)
        if out["state"] != "unknown":
            with self._lock:
                self._cache[table_id] = out
        return {**out, "from_cache": False}

    def _fetch(self, table_id: str) -> tuple[int, dict[str, str], Any]:
        self.limiter.acquire(self.max_wait_s)
        self.requests += 1
        return self.get(API + table_id)

    def _ask(self, table_id: str) -> dict[str, Any]:
        base: dict[str, Any] = {"table_id": table_id, "state": "unknown", "url": PAGE + table_id, "moved_to": None,
                                "last_modified": None, "revision": None, "gated": None, "http_status": None,
                                "reason": "", "checked_at": self.clock()}
        try:
            status, headers, body = self._fetch(table_id)
        except RateLimited as exc:
            return {**base, "reason": str(exc)}
        except httpx.HTTPError as exc:
            return {**base, "reason": f"the site could not be reached ({type(exc).__name__})"}
        base["http_status"] = status
        if status in (301, 302, 307, 308):
            new = _id_from_location(headers.get("location", ""))
            if not new or new == table_id:
                return {**base, "reason": "the site redirected to an address that is not a table"}
            out = {**base, "state": "moved", "moved_to": new, "url": PAGE + new}
            try:                                               # one hop: what the new address says
                s2, _, b2 = self._fetch(new)
            except (RateLimited, httpx.HTTPError):
                return out
            if s2 == 200 and isinstance(b2, dict):
                out.update(last_modified=_iso(b2.get("lastModified")), revision=b2.get("sha"), gated=_gated(b2.get("gated")))
            elif s2 in (401, 404):
                out.update(state="gone", url=PAGE + table_id, reason="it moved, and the new address is gone or private")
            return out
        if status == 200 and isinstance(body, dict):
            if body.get("disabled"):
                return {**base, "state": "gone", "reason": "the site has disabled this table"}
            return {**base, "state": "live", "last_modified": _iso(body.get("lastModified")), "revision": body.get("sha"),
                    "gated": _gated(body.get("gated"))}
        if status in (401, 403, 404):
            return {**base, "state": "gone", "reason": "the site says it does not exist or is private"}
        if status == 429:
            return {**base, "reason": "the site is limiting requests; try again later"}
        return {**base, "reason": f"unexpected reply from the site ({status})"}


def _id_from_location(loc: str) -> str | None:
    for mark in ("/api/datasets/", "/datasets/"):
        if mark in loc:
            rest = loc.split(mark, 1)[1].split("?", 1)[0].strip("/")
            parts = rest.split("/")
            if len(parts) >= 2:
                return "/".join(parts[:2])
    return None


def _iso(v: Any) -> str | None:
    return str(v).replace(".000Z", "Z") if v else None


def _gated(v: Any) -> str:
    return v if v in ("auto", "manual") else "no"


def verdict(indexed_last_modified: str | None, live: dict[str, Any]) -> dict[str, Any]:
    """Compare what the index knew with what the site says: is the indexed description still good?"""
    changed = bool(live.get("state") in ("live", "moved") and live.get("last_modified") and indexed_last_modified
                   and str(live["last_modified"])[:19] > str(indexed_last_modified)[:19])
    usable = live.get("state") in ("live", "moved")
    return {"table_id": live["table_id"], "state": live["state"], "usable": usable, "url": live["url"] if usable else None,
            "moved_to": live.get("moved_to"), "changed_since_indexed": changed, "live_last_modified": live.get("last_modified"),
            "indexed_last_modified": indexed_last_modified, "gated": live.get("gated"), "reason": live.get("reason") or "",
            "from_cache": bool(live.get("from_cache"))}


class InjectedSite:
    """INJECTED faults for tests and evaluation: the named tables answer as told, every other
    table is passed to the real site (or to `base`). The answers are marked."""

    def __init__(self, base: Any, faults: dict[str, dict[str, Any]]):
        self.base, self.faults = base, faults

    def check(self, table_id: str) -> dict[str, Any]:
        f = self.faults.get(table_id)
        if f is None:
            return self.base.check(table_id)
        state = f["state"]
        new = f.get("moved_to")
        return {"table_id": table_id, "state": state, "url": PAGE + (new or table_id), "moved_to": new,
                "last_modified": f.get("last_modified"), "revision": None, "gated": f.get("gated", "no"), "http_status": None,
                "reason": f.get("reason", "INJECTED"), "checked_at": time.time(), "from_cache": False, "injected": True}


class RecordedSite:
    """Answers from a file of earlier real answers, for runs that must not depend on the network."""

    def __init__(self, answers: dict[str, dict[str, Any]]):
        self.answers = answers
        self.requests = 0

    def check(self, table_id: str) -> dict[str, Any]:
        self.requests += 1
        a = self.answers.get(table_id)
        if a is None:
            return {"table_id": table_id, "state": "unknown", "url": PAGE + table_id, "moved_to": None, "last_modified": None,
                    "revision": None, "gated": None, "http_status": None, "reason": "no recorded answer for this table",
                    "checked_at": time.time(), "from_cache": False}
        return {**a, "from_cache": False}
