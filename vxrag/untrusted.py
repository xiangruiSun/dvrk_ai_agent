"""Text from threads and notes is written by anyone. It is evidence, never instruction.

Two things are done about that, both by code:

  scan         finds sentences that address an automated reader ("ignore your instructions",
               "AI assistants must ...", a line that pretends to be a system message). A unit
               that contains one is not put in front of the model at all.
  wrap         every piece of evidence is handed to the model inside a block that names its
               source and its author's standing, after anything that could close the block
               early has been removed.

Neither stops a plain false statement written by a person ("the fix is to run X"). That is
handled by trust: a step that rests only on a community reply is marked as unconfirmed
(vxrag/answer.py).
"""
from __future__ import annotations

import re
from typing import Any

_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("override", re.compile(
        r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}\b(?:previous|above|prior|earlier|all|your|any)\b"
        r"[^.\n]{0,30}\b(?:instructions?|rules|prompts?|guidelines)\b", re.I)),
    ("addressed_to_model", re.compile(
        r"\b(?:AI|LLM|language model|chatbot|AI assistants?|automated assistants?)\b[^.\n]{0,40}"
        r"\b(?:must|should|shall|need to|are required to|have to)\b", re.I)),
    # Content on the same line is required: a sentence that merely ends in "system:" is not a role line.
    ("role_line", re.compile(r"(?im)^[ \t]*(?:system|assistant|developer)[ \t]*(?:message|prompt|note)?[ \t]*:[ \t]+\S")),
    # "your instructions" alone is how a user addresses a maintainer, so it is not a rule.
    ("prompt_talk", re.compile(
        r"\b(?:system prompt|new instructions for|instructions? (?:to|for) (?:the |any )?(?:AI|assistant|model))\b",
        re.I)),
    ("conceal", re.compile(
        r"\bdo not (?:tell|mention|reveal|inform|warn)\b[^.\n]{0,40}\b(?:user|operator|engineer|reader)\b", re.I)),
    ("markup", re.compile(r"<\|[a-z_]+\|>|\[/?INST\]|^#{2,}\s*(?:instruction|system)\b", re.I | re.M)),
]
_BLOCK = re.compile(r"</?\s*evidence\b[^>]*>", re.I)


def scan(text: str) -> list[dict[str, Any]]:
    """Sentences that address an automated reader. Empty for ordinary engineering text."""
    found = []
    for rule, pattern in _RULES:
        for m in pattern.finditer(text):
            start = max(text.rfind("\n", 0, m.start()) + 1, m.start() - 80)
            found.append({"rule": rule, "at": m.start(), "excerpt": text[start:m.end() + 40].strip()[:160]})
    return sorted(found, key=lambda f: f["at"])


def wrap(unit_id: str, header: str, text: str) -> str:
    """One piece of evidence as the model sees it."""
    safe = _BLOCK.sub("[tag removed]", text)
    return f'<evidence id="{unit_id}" {header}>\n{safe}\n</evidence>'
