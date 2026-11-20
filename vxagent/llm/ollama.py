"""Ollama adapter (native /api/chat). Supports tool calling and schema-constrained JSON output."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

import httpx

from ..config import assert_egress_allowed, get_settings
from .base import LLMError, LLMResponse, ToolCall, Usage


class OllamaClient:
    def __init__(self, model: str | None = None, base_url: str | None = None, num_ctx: int | None = None):
        s = get_settings()
        self.model = model or s.model
        self.base_url = (base_url or s.ollama_url).rstrip("/")
        self.num_ctx = num_ctx or s.num_ctx
        self.default_timeout = s.llm_timeout_s

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
        assert_egress_allowed(self.base_url)
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=body, timeout=timeout_s or self.default_timeout)
        except httpx.TimeoutException as exc:
            raise LLMError("timeout", f"model call exceeded {timeout_s or self.default_timeout}s") from exc
        except httpx.HTTPError as exc:
            raise LLMError("unavailable", f"cannot reach Ollama at {self.base_url}: {exc}") from exc
        if r.status_code != 200:
            raise LLMError("unavailable", f"Ollama returned {r.status_code}: {r.text[:200]}")
        data = r.json()
        msg = data.get("message") or {}
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


def get_llm(model: str | None = None) -> OllamaClient:
    return OllamaClient(model=model)
