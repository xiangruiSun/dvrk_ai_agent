"""Store and forward between an edge node's outbox and the fleet store.

The node sends what is pending, oldest first, and marks as sent only what the store has
acknowledged. If the link is down, nothing is marked and nothing is lost; the next attempt
sends the same records again and the store ignores the ones it already has.

The fleet store is reached over HTTP with a token per site. `serve` runs it; `HttpLink` is the
node's end. `DirectLink` calls a store in the same process, for tests.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from ..config import EgressBlocked, assert_egress_allowed
from .outbox import Outbox
from .store import FleetStore


class LinkDown(Exception):
    pass


class DirectLink:
    def __init__(self, store: FleetStore):
        self.store, self.up = store, True
        self.lose_next_ack = False           # INJECTED in tests: the store got the batch, the node never heard

    def send(self, site: str, batch: list[dict[str, Any]]) -> dict[str, int]:
        if not self.up:
            raise LinkDown("the fleet link is down")
        ack = self.store.ingest(site, batch)
        if self.lose_next_ack:
            self.lose_next_ack = False
            raise LinkDown("the acknowledgement was lost")
        return ack


class HttpLink:
    def __init__(self, url: str, token: str, timeout_s: float = 3.0):
        self.url, self._token, self.timeout_s = url.rstrip("/"), token, timeout_s

    def send(self, site: str, batch: list[dict[str, Any]]) -> dict[str, int]:
        import httpx
        try:
            assert_egress_allowed(self.url)            # blocked before a byte leaves the lab
        except EgressBlocked as exc:
            raise LinkDown(f"blocked by the egress guard: {exc}") from exc
        try:
            r = httpx.post(f"{self.url}/v1/ingest", json={"site": site, "batch": batch},
                           headers={"Authorization": f"Bearer {self._token}"}, timeout=self.timeout_s)
        except httpx.HTTPError as exc:
            raise LinkDown(type(exc).__name__) from exc
        if r.status_code != 200:
            raise LinkDown(f"the fleet store answered {r.status_code}")
        return r.json()


def sync(outbox: Outbox, link: Any, batch_size: int = 20, max_batches: int = 50) -> dict[str, Any]:
    """Send what is pending. Returns what happened; never raises for a link that is down."""
    sent = dup = 0
    for _ in range(max_batches):
        batch = outbox.pending(batch_size)
        if not batch:
            break
        try:
            ack = link.send(outbox.site, batch)
        except LinkDown as exc:
            return {"delivered": sent, "duplicates_ignored_by_the_store": dup, "link": "down", "why": str(exc),
                    "backlog": outbox.backlog()}
        outbox.mark_sent(ack["ack"])
        sent += ack["accepted"]
        dup += ack["duplicates"]
        if ack["accepted"] == 0 and ack["duplicates"] == 0:
            break                                          # the store wants an earlier record first; try again later
    return {"delivered": sent, "duplicates_ignored_by_the_store": dup, "link": "up", "backlog": outbox.backlog()}


# ------------------------------------------------------------------ the fleet store as a service

class FleetServer:
    """The fleet store behind HTTP on this machine. One token per site; a site may only write
    records under its own name."""

    def __init__(self, store: FleetStore, tokens: dict[str, str], host: str = "127.0.0.1", port: int = 0):
        self.store, self.tokens, self.host, self.port = store, dict(tokens), host, port
        self._server: Any = None
        self._thread: threading.Thread | None = None

    def app(self) -> Any:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        async def ingest(request: Request) -> JSONResponse:
            auth = request.headers.get("authorization", "")
            body = await request.json()
            site = str(body.get("site", ""))
            if not auth.startswith("Bearer ") or self.tokens.get(site) != auth[7:]:
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return JSONResponse(self.store.ingest(site, body.get("batch") or []))

        async def health(request: Request) -> JSONResponse:
            return JSONResponse({"ok": True, "sites": len(self.store.sites())})

        return Starlette(routes=[Route("/v1/ingest", ingest, methods=["POST"]), Route("/health", health)])

    def start(self) -> "FleetServer":
        import socket

        import uvicorn
        if not self.port:
            with socket.socket() as s:
                s.bind((self.host, 0))
                self.port = s.getsockname()[1]
        self._server = uvicorn.Server(uvicorn.Config(self.app(), host=self.host, port=self.port, log_level="error"))
        self._thread = threading.Thread(target=self._server.run, daemon=True, name="fleet-store")
        self._thread.start()
        deadline = time.time() + 10
        while not self._server.started and time.time() < deadline:
            time.sleep(0.02)
        return self

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            if self._thread is not None:
                self._thread.join(timeout=5)
            self._server = None
