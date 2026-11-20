"""Working context: what is actually sent to the model on each step.

The full message history is kept in the run state. Before every model call it is fitted to a
token budget: the system prompt and the task are always kept, the most recent turns are kept
verbatim, and older tool observations are replaced by one-line stubs. If that is not enough the
verbatim window shrinks, down to the last exchange. The task is repeated as an anchor on long
runs so the goal does not scroll out of view.
"""
from __future__ import annotations

from typing import Any

from ..llm.base import messages_tokens


def fit_context(messages: list[dict[str, Any]], budget_tokens: int, keep_recent: int = 6,
                anchor_task: str | None = None, iteration: int = 0, reanchor_every: int = 4) -> list[dict[str, Any]]:
    msgs = [dict(m) for m in messages]
    head = 2 if len(msgs) > 1 and msgs[0]["role"] == "system" else 1
    keep = keep_recent
    while messages_tokens(msgs) > budget_tokens and keep >= 2:
        cutoff = max(head, len(msgs) - keep)
        # Pass 1: stub old tool observations. Pass 2: stub old assistant text as well.
        for roles in (("tool",), ("tool", "assistant")):
            for i in range(head, cutoff):
                if messages_tokens(msgs) <= budget_tokens:
                    break
                m = msgs[i]
                if m["role"] in roles and len(m.get("content") or "") > 160:
                    label = f"tool {m.get('name', '')}" if m["role"] == "tool" else "assistant note"
                    m["content"] = f"[earlier {label} output shortened] " + (m["content"] or "")[:120]
        keep -= 2
    if anchor_task and iteration > 0 and iteration % reanchor_every == 0:
        msgs.append({"role": "user", "content": "Reminder of the task you are completing: " + anchor_task
                     + "\nIf you already have what you need, give the final answer now."})
    return msgs
