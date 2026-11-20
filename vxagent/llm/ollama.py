"""Ollama adapter (native /api/chat). Supports tool calling and schema-constrained JSON output."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

import httpx

from ..config import EgressBlocked, assert_egress_allowed, get_settings
from .. import net
from .base import LLMError, LLMResponse, ToolCall, Usage


class OllamaClient:
    def __init__(self, model: str | None = None, base_url: str | None = None, num_ctx: int | None = None):
        s = get_settings()
        self.model = model or s.model
        self.base_url = (base_url or s.ollama_url).rstrip("/")
        self.num_ctx = num_ctx or s.num_ctx
        self.default_timeout = s.llm_timeout_s
        self.empty_replies_repeated = 0        # turns asked a second time because the first reply was empty

    @staticmethod
    def _wire_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                out.append({"role": "assistant", "content": m.get("content") or "",
                            "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}}
                                           for c in m["tool_calls"]]})
            elif m["role"] == "tool":
                out.append({"role": "tool", "content": m["content"], "tool_name": m.get("name", "")})
            else:
                out.append({"role": m["role"], "content": m.get("content") or ""})
        return out

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             json_schema: dict[str, Any] | None = None, temperature: float = 0.0,
             timeout_s: float | None = None, max_tokens: int | None = None) -> LLMResponse:
        options: dict[str, Any] = {"temperature": temperature, "num_ctx": self.num_ctx}
        if max_tokens:
            options["num_predict"] = max_tokens
        body: dict[str, Any] = {"model": self.model, "stream": False, "think": False,
                                "messages": self._wire_messages(messages), "options": options}
        if tools:
            body["tools"] = tools
        if json_schema is not None:
            body["format"] = json_schema
        try:
            assert_egress_allowed(self.base_url)       # checked before a byte of the prompt leaves this machine
        except EgressBlocked as exc:
            raise LLMError("egress_blocked", str(exc)) from exc
        t0 = time.perf_counter()

        def once() -> dict[str, Any]:
            try:
                r = net.post(f"{self.base_url}/api/chat", json=body, timeout=timeout_s or self.default_timeout)
            except httpx.TimeoutException as exc:
                raise LLMError("timeout", f"model call exceeded {timeout_s or self.default_timeout}s") from exc
            except httpx.HTTPError as exc:
                raise LLMError("unavailable", f"cannot reach Ollama at {self.base_url}: {exc}") from exc
            if r.status_code != 200:
                raise LLMError("unavailable", f"Ollama returned {r.status_code}: {r.text[:200]}")
            return r.json()

        data = once()
        msg = data.get("message") or {}
        spent = (data.get("prompt_eval_count") or 0, data.get("eval_count") or 0)
        if tools and not (msg.get("content") or "").strip() and not msg.get("tool_calls"):
            # The model produced tokens but the server could parse neither text nor a tool call from
            # them. Seen with qwen3:14b on "Run the rig check": with its reasoning switched on, the
            # same request gives a well-formed tool call. The reasoning text is not used.
            body["think"] = True
            self.empty_replies_repeated += 1
            data = once()
            msg = data.get("message") or {}
            data["prompt_eval_count"] = (data.get("prompt_eval_count") or 0) + spent[0]
            data["eval_count"] = (data.get("eval_count") or 0) + spent[1]
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):  # some models return the arguments as a JSON string
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append(ToolCall(id=c.get("id") or f"call_{uuid.uuid4().hex[:8]}", name=fn.get("name", ""),
                                  arguments=args if isinstance(args, dict) else {"_raw": args}))
        return LLMResponse(content=msg.get("content") or "", tool_calls=calls,
                           usage=Usage(data.get("prompt_eval_count") or 0, data.get("eval_count") or 0),
                           latency_ms=(time.perf_counter() - t0) * 1000, model=self.model)


    def health(self, timeout_s: float = 2.0) -> dict[str, Any]:
        """Whether the model server answers and whether this model is installed. No prompt is sent."""
        try:
            assert_egress_allowed(self.base_url)
            r = net.get(f"{self.base_url}/api/tags", timeout=timeout_s)
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return {"reachable": r.status_code == 200, "models_installed": len(names), "model": self.model,
                    "model_installed": self.model in names}
        except Exception as exc:                       # unreachable, blocked or not a model server
            return {"reachable": False, "models_installed": 0, "model": self.model, "error": type(exc).__name__}


def get_llm(model: str | None = None) -> OllamaClient:
    return OllamaClient(model=model)
