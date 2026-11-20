"""The HTTP service the agents call.

    GET  /health                 no key needed; says whether an index version is serving
    GET  /v1/version             the serving version, its configuration and unit counts
    POST /v1/search              {"query", "k", "sources", "generation", "release"}
    POST /v1/search/log-line     {"line", "k"}: a pasted log line, cleaned before it is searched
    POST /v1/answer              {"question", "k", "generation", "release"}
    GET  /v1/units/<unit id>     one unit in full
    GET  /v1/metrics             counters, latency percentiles, cache hit rate

Every /v1 request carries `Authorization: Bearer <key>`. The key decides who the caller is and
therefore what they may read; no field of a request can widen that. A unit the caller may not
read answers 404, the same as a unit that does not exist.

The server binds to the loopback address unless told otherwise. Operations that change what
is served (build, validate, activate, rollback) are not on the network at all: they are
commands run on the machine (python -m vxrag.cli index ...), after which `reload()` is called.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Literal
from urllib.parse import unquote

from pydantic import BaseModel, Field, ValidationError

from . import versions
from .access import KeyRing
from .config import Settings, get_settings
from .engine import Engine, NotAuthorized
from .llm import Generator

MAX_BODY = 64 * 1024
Source = Literal["doc", "thread", "changelog", "schema", "note"]
Generation = Literal["Classic", "Si"]
RELEASE = r"^\d+\.\d+(\.\d+)?$"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    sources: list[Source] | None = None
    generation: Generation | None = None
    release: str | None = Field(default=None, pattern=RELEASE)


class LogLineRequest(BaseModel):
    line: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=6, ge=1, le=10)
    generation: Generation | None = None
    release: str | None = Field(default=None, pattern=RELEASE)


class EngineHolder:
    """The engine now serving. `reload` swaps it for the active version in one assignment, so a
    request sees either the old version or the new one, never a mixture."""

    def __init__(self, engine: Engine | None = None, settings: Settings | None = None,
                 generator: Generator | None = None):
        self.settings = settings or get_settings()
        self.generator = generator
        self.engine = engine
        self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S") if engine else None

    def reload(self) -> str | None:
        active = versions.pointer(self.settings)["active"]
        if active is None:
            self.engine = None
            return None
        if self.engine is None or self.engine.version != active:
            index, m = versions.load(active, self.settings)
            self.engine = Engine(index, m["config"], version=active, generator=self.generator)
            self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")
        return active


class _Problem(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code = status, code


def make_handler(holder: EngineHolder, keys: KeyRing, fault: dict[str, float] | None = None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "vxrag"

        def log_message(self, *args: Any) -> None:            # request lines are not written to stderr
            return

        def _send(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                if status >= 400:
                    # An error may be decided before the request body was read (a missing key, a body
                    # that is too large). The body is read and dropped first: left unread, it would be
                    # taken for the next request on a kept connection, and closing a connection with
                    # unread data can destroy the error reply on its way. Then the connection ends.
                    self._drain()
                    self.send_header("Connection", "close")
                    self.close_connection = True
                self.end_headers()
                self.wfile.write(data)
            except ConnectionError:              # the caller gave up waiting; there is nobody to tell
                self.close_connection = True

        def _drain(self) -> None:
            if getattr(self, "_body_read", True):
                return
            self._body_read = True
            try:
                left = min(int(self.headers.get("Content-Length") or 0), 4 * MAX_BODY)
                while left > 0:
                    chunk = self.rfile.read(min(left, 65536))
                    if not chunk:
                        break
                    left -= len(chunk)
            except (ValueError, OSError):
                pass

        def _principal(self):
            header = self.headers.get("Authorization", "")
            token = header[7:].strip() if header.lower().startswith("bearer ") else None
            principal = keys.resolve(token)
            if principal is None:
                raise _Problem(401, "not_authorized", "a valid key is required")
            return principal

        def _engine(self) -> Engine:
            if holder.engine is None:
                raise _Problem(503, "no_active_index", "no index version is active")
            return holder.engine

        def _body(self, model: type[BaseModel]) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise _Problem(413, "too_large", f"a request body may be at most {MAX_BODY} bytes")
            self._body_read = True
            try:
                return model.model_validate_json(self.rfile.read(length) or b"{}")
            except ValidationError as exc:
                first = exc.errors()[0]
                raise _Problem(422, "invalid_request",
                               f"{'.'.join(str(x) for x in first['loc']) or 'body'}: {first['msg']}") from exc

        def _handle(self, method: str) -> None:
            self._body_read = method != "POST"         # a POST has a body that someone has to read
            if fault and fault["requests"] > 0:                # used by tests and fault injection only
                fault["requests"] -= 1
                time.sleep(fault["delay_s"])
            path = self.path.split("?", 1)[0]
            try:
                if method == "GET" and path == "/health":
                    e = holder.engine
                    return self._send(200, {"status": "ok" if e else "no_active_index",
                                            "index_version": e.version if e else None})
                principal = self._principal()
                engine = self._engine()
                if method == "GET" and path == "/v1/version":
                    return self._send(200, {**engine.describe(), "loaded": holder.loaded_at})
                if method == "GET" and path == "/v1/metrics":
                    return self._send(200, engine.metrics.snapshot())
                if method == "GET" and path.startswith("/v1/units/"):
                    unit = engine.unit(principal, unquote(path[len("/v1/units/"):]))
                    if unit is None:
                        raise _Problem(404, "not_found", "there is no such unit")
                    return self._send(200, unit)
                if method == "POST" and path == "/v1/search":
                    r = self._body(SearchRequest)
                    return self._send(200, engine.search(principal, r.query, r.k, r.sources, r.generation, r.release))
                if method == "POST" and path == "/v1/search/log-line":
                    r = self._body(LogLineRequest)
                    return self._send(200, engine.search(principal, r.line, r.k, log_line=True))
                if method == "POST" and path == "/v1/answer":
                    r = self._body(AnswerRequest)
                    if engine.generator is None:
                        raise _Problem(503, "no_model", "this service was started without a model")
                    return self._send(200, engine.answer(principal, r.question, r.k, r.generation, r.release))
                raise _Problem(404, "not_found", "there is no such endpoint")
            except _Problem as p:
                if holder.engine is not None:
                    holder.engine.metrics.count(f"http_{p.status}")
                self._send(p.status, {"error": {"code": p.code, "message": str(p)}})
            except NotAuthorized as exc:
                self._send(401, {"error": {"code": "not_authorized", "message": str(exc)}})
            except ValueError as exc:
                self._send(422, {"error": {"code": "invalid_request", "message": str(exc)}})
            except Exception as exc:                            # noqa: BLE001 - the caller gets a typed error
                self._send(500, {"error": {"code": "internal", "message": type(exc).__name__}})

        def do_GET(self) -> None:                               # noqa: N802
            self._handle("GET")

        def do_POST(self) -> None:                              # noqa: N802
            self._handle("POST")

    return Handler


class Server:
    """The service in a background thread. Used by the command line, the verification scripts
    and the tests. `slow(n, seconds)` makes the next n responses late, to exercise a caller's
    timeout and retry."""

    def __init__(self, holder: EngineHolder, keys: KeyRing, host: str = "127.0.0.1", port: int = 0):
        self.fault = {"requests": 0, "delay_s": 0.0}
        base = make_handler(holder, keys, self.fault)
        open_connections: set[Any] = set()

        class Tracked(base):                          # so that stop() can end connections a caller keeps open
            def setup(self) -> None:
                super().setup()
                open_connections.add(self.connection)

            def finish(self) -> None:
                open_connections.discard(self.connection)
                super().finish()

        self._connections = open_connections
        self.httpd = ThreadingHTTPServer((host, port), Tracked)
        self.httpd.daemon_threads = True
        self.host, self.port = self.httpd.server_address[0], self.httpd.server_address[1]
        self.url = f"http://{self.host}:{self.port}"
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self) -> "Server":
        self._thread.start()
        return self

    def slow(self, requests: int, delay_s: float) -> None:
        self.fault.update(requests=requests, delay_s=delay_s)

    def stop(self) -> None:
        """Stop listening and end the connections that are still open. Without the second part a
        caller that keeps its connection would go on being served by a "stopped" service."""
        import socket
        self.httpd.shutdown()
        self.httpd.server_close()
        for conn in list(getattr(self, "_connections", ())):
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
