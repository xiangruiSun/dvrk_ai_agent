"""Provider-neutral LLM interface.

Messages are plain dicts:
    {"role": "system" | "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "arguments"}]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}
Each provider adapter converts this shape to its own wire format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    model: str = ""

    def as_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls]
        return msg


class LLMError(Exception):
    """The model call failed. ``kind`` is 'timeout', 'unavailable' or 'bad_response'."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class LLMClient(Protocol):
    model: str

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             json_schema: dict[str, Any] | None = None, temperature: float = 0.0,
             timeout_s: float | None = None, max_tokens: int | None = None) -> LLMResponse: ...


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for budgeting before a call (about 4 characters per token)."""
    return max(1, len(text) // 4)


def messages_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        total += estimate_tokens(str(m.get("content") or "")) + 4
        for c in m.get("tool_calls") or []:
            total += estimate_tokens(str(c.get("arguments"))) + 8
    return total
