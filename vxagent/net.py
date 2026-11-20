"""One HTTP client for the whole process, kept open.

Creating an HTTP client per request costs about 0.2 s on the development PC (it builds a TLS
context each time). That was first seen as a perception request taking 0.2 s for 0.04 s of work
(task 14), and it was in every model call, retrieval call and gateway command before that.

The client also counts the requests it sends, so a verification can show that a call blocked
by the egress guard never reached the network layer.
"""
from __future__ import annotations

import threading

import httpx

_lock = threading.Lock()
_client: httpx.Client | None = None
_sent = 0


def _count(request: httpx.Request) -> None:
    global _sent
    with _lock:
        _sent += 1


def client() -> httpx.Client:
    global _client
    with _lock:
        if _client is None:
            _client = httpx.Client(event_hooks={"request": [_count]}, timeout=30.0)
        return _client


def sent() -> int:
    """Requests handed to the network layer by this process so far."""
    with _lock:
        return _sent


def request(method: str, url: str, **kw: object) -> httpx.Response:
    """Send through the kept client. A kept connection whose other end has gone away (a worker
    that was restarted) shows as a protocol error before any reply: that one case is sent once
    more, on a new connection."""
    try:
        return client().request(method, url, **kw)                  # type: ignore[arg-type]
    except httpx.RemoteProtocolError:
        return client().request(method, url, **kw)                  # type: ignore[arg-type]


def post(url: str, **kw: object) -> httpx.Response:
    return request("POST", url, **kw)


def get(url: str, **kw: object) -> httpx.Response:
    return request("GET", url, **kw)
