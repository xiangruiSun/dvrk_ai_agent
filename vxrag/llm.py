"""The one model call the retrieval system makes: a structured answer from a local model.

Text never leaves the machine. The client refuses any endpoint that is not local, as the
embedding client does. Tests pass a scripted object with the same `generate` method.
"""
from __future__ import annotations

import json
import time
from typing import Any, Protocol

import httpx

from .config import get_settings


class GenerationError(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind                      # unavailable | timeout | bad_response


class Generator(Protocol):
    model: str

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]: ...


class OllamaGenerator:
    def __init__(self, model: str | None = None, url: str | None = None, timeout_s: float = 180.0):
        s = get_settings()
        self.model = model or s.model
        self.url = (url or s.ollama_url).rstrip("/")
        self.num_ctx = s.num_ctx
        self.timeout_s = timeout_s

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        host = self.url.split("//", 1)[-1].split("/", 1)[0].split(":")[0]
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise GenerationError("unavailable", f"model endpoint {host!r} is not local; refusing to send text to it")
        body = {"model": self.model, "stream": False, "think": False, "format": schema,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": 0.0, "num_ctx": self.num_ctx, "num_predict": 900}}
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{self.url}/api/chat", json=body, timeout=self.timeout_s)
        except httpx.TimeoutException as exc:
            raise GenerationError("timeout", f"the model did not answer within {self.timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise GenerationError("unavailable", f"cannot reach the model server at {self.url}: {exc}") from exc
        if r.status_code != 200:
            raise GenerationError("unavailable", f"model server returned {r.status_code}: {r.text[:200]}")
        data = r.json()
        try:
            parsed = json.loads((data.get("message") or {}).get("content") or "")
        except json.JSONDecodeError as exc:
            raise GenerationError("bad_response", "the model did not return JSON") from exc
        if not isinstance(parsed, dict):
            raise GenerationError("bad_response", "the model returned JSON that is not an object")
        return parsed, {"prompt_tokens": data.get("prompt_eval_count") or 0,
                        "completion_tokens": data.get("eval_count") or 0,
                        "seconds": round(time.perf_counter() - t0, 2)}
