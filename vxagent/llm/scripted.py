"""A scripted LLM for hermetic tests and failure injection.

Each scripted step is an ``LLMResponse``, an exception to raise, or a callable
``f(messages, tools) -> LLMResponse``. When the script runs out the last step repeats,
which is how the infinite-loop tests drive the agent.
"""
from __future__ import annotations

from typing import Any, Callable

from .base import LLMResponse, ToolCall, Usage

Step = LLMResponse | Exception | Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], LLMResponse]


def call(name: str, **arguments: Any) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=arguments)],
                       usage=Usage(50, 10))


def say(text: str) -> LLMResponse:
    return LLMResponse(content=text, usage=Usage(50, 10))


class ScriptedLLM:
    model = "scripted"

    def __init__(self, steps: list[Step]):
        self.steps = list(steps)
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             json_schema: dict[str, Any] | None = None, temperature: float = 0.0,
             timeout_s: float | None = None, max_tokens: int | None = None) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        step = self.steps[min(len(self.calls) - 1, len(self.steps) - 1)]
        if isinstance(step, Exception):
            raise step
        if callable(step):
            return step(messages, tools)
        return step
