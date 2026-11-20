"""One connection to one MCP server, for a host whose own code is not asynchronous.

It initialises, lists, calls, reads, listens. It holds the connection and nothing else: what
may be called, and what a result may be used for, is the host's business (host.py).
The SDK's client lives on an event loop in a background thread; the methods here block.
"""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import json
import threading
from typing import Any, Callable

import httpx2
from mcp import Client, types
from mcp.client.streamable_http import streamable_http_client


class McpUnavailable(Exception):
    pass


class Connection:
    def __init__(self, name: str, url: str, token: str | None,
                 on_confirm: Callable[[str, str], bool | None] | None = None, timeout_s: float = 30.0):
        self.name, self.url, self._token, self.timeout_s = name, url, token, timeout_s
        self.on_confirm = on_confirm            # (server, message) -> True / False / None for "cancel"
        self.confirmations: list[dict[str, Any]] = []
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name=f"mcp-client-{name}", daemon=True)
        self._client: Client | None = None
        self._closed: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._session: cf.Future[Any] | None = None

    # ------------------------------------------------------------------ connection
    def connect(self) -> "Connection":
        self._thread.start()
        self._session = asyncio.run_coroutine_threadsafe(self._run(), self._loop)
        if not self._ready.wait(self.timeout_s) or self._error is not None:
            reason = _flat(self._error) if self._error else "timed out"
            self.close()
            raise McpUnavailable(f"cannot connect to {self.name}: {reason}")
        return self

    async def _run(self) -> None:
        self._closed = asyncio.Event()
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        try:
            async with httpx2.AsyncClient(headers=headers, timeout=self.timeout_s) as http:
                async with Client(streamable_http_client(self.url, http_client=http),
                                  elicitation_callback=self._elicit) as client:
                    self._client = client
                    self._ready.set()
                    await self._closed.wait()
        except BaseException as exc:                    # includes the refusal of a missing token
            self._error = exc
            self._ready.set()

    async def _elicit(self, context: Any, params: Any) -> types.ElicitResult:
        """The server asks the connected person. The model is not on this path."""
        message = getattr(params, "message", "")
        answer = self.on_confirm(self.name, message) if self.on_confirm else None
        self.confirmations.append({"server": self.name, "message": message, "answer": answer})
        if answer is None:
            return types.ElicitResult(action="cancel")
        return types.ElicitResult(action="accept" if answer else "decline", content={"confirm": bool(answer)})

    def close(self) -> None:
        if self._closed is not None:
            self._loop.call_soon_threadsafe(self._closed.set)
        if self._session is not None:
            try:
                self._session.result(timeout=5)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    def _do(self, coro: Any, timeout: float | None = None) -> Any:
        if self._client is None:
            raise McpUnavailable(f"{self.name} is not connected")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout or self.timeout_s)

    # ------------------------------------------------------------------ what a client does
    def info(self) -> dict[str, Any]:
        c = self._client
        assert c is not None
        caps = c.server_capabilities.model_dump(exclude_none=True) if c.server_capabilities else {}
        return {"server": c.server_info.name, "version": c.server_info.version, "protocol": c.protocol_version,
                "capabilities": caps}

    def list_tools(self) -> list[dict[str, Any]]:
        res = self._do(self._client.list_tools(cache_mode="refresh") if _has_cache() else self._client.list_tools())   # type: ignore[union-attr]
        return [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema, "meta": t.meta or {}}
                for t in res.tools]

    def list_resources(self) -> list[str]:
        return [str(r.uri) for r in self._do(self._client.list_resources()).resources]        # type: ignore[union-attr]

    def list_prompts(self) -> list[str]:
        return [p.name for p in self._do(self._client.list_prompts()).prompts]                # type: ignore[union-attr]

    def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> str:
        res = self._do(self._client.get_prompt(name, arguments))                              # type: ignore[union-attr]
        return "\n".join(getattr(m.content, "text", "") for m in res.messages)

    def read_resource(self, uri: str) -> Any:
        res = self._do(self._client.read_resource(uri, cache_mode="refresh") if _has_cache()  # type: ignore[union-attr]
                       else self._client.read_resource(uri))                                  # type: ignore[union-attr]
        text = res.contents[0].text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: float | None = None,
                  on_progress: Callable[[float, str], None] | None = None,
                  cancel: threading.Event | None = None) -> dict[str, Any]:
        """The server's reply as data. Setting `cancel` cancels the request, which stops a motion."""
        async def progress(done: float, total: float | None = None, message: str | None = None) -> None:
            if on_progress:
                on_progress(done / total if total else done, message or "")

        fut = asyncio.run_coroutine_threadsafe(
            self._client.call_tool(name, arguments, progress_callback=progress), self._loop)   # type: ignore[union-attr]
        waited, limit = 0.0, timeout or self.timeout_s
        while True:
            try:
                res = fut.result(timeout=0.05)
                break
            except cf.TimeoutError:
                waited += 0.05
                if cancel is not None and cancel.is_set():
                    fut.cancel()
                    return {"ok": False, "error_kind": "canceled", "error": "the request was cancelled"}
                if waited >= limit:
                    fut.cancel()
                    return {"ok": False, "error_kind": "timeout", "error": f"{name} did not answer within {limit:g} s"}
            except BaseException as exc:
                return {"ok": False, "error_kind": "failed", "error": _flat(exc)[:300]}
        text = "".join(getattr(b, "text", "") for b in res.content)
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            return {"ok": False, "error_kind": "failed" if res.is_error else "bad_response", "error": text[:300]}
        if not isinstance(body, dict) or "ok" not in body:
            return {"ok": False, "error_kind": "bad_response", "error": text[:300]}
        return body

    def listen(self, uri: str, on_event: Callable[[str], None]) -> Callable[[], None]:
        """Be told when a resource changes. Returns a function that ends the subscription."""
        started = threading.Event()
        stop = asyncio.Event()

        async def run() -> None:
            async with self._client.listen(resource_subscriptions=[uri]) as sub:              # type: ignore[union-attr]
                started.set()

                async def pump() -> None:
                    async for event in sub:
                        on_event(str(getattr(event, "uri", "")))
                task = asyncio.ensure_future(pump())
                await stop.wait()
                task.cancel()
        fut = asyncio.run_coroutine_threadsafe(run(), self._loop)
        if not started.wait(self.timeout_s):
            raise McpUnavailable(f"{self.name} did not acknowledge the subscription: {_flat(fut.exception(0.1)) if fut.done() else 'timeout'}")
        return lambda: self._loop.call_soon_threadsafe(stop.set)


def _has_cache() -> bool:
    return True


def _flat(exc: BaseException | None) -> str:
    """The innermost message of a (possibly grouped) exception."""
    seen = exc
    while isinstance(seen, BaseExceptionGroup) and seen.exceptions:
        seen = seen.exceptions[0]
    return f"{type(seen).__name__}: {seen}" if seen is not None else ""
