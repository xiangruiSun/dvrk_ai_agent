"""Golden traces: runs of the operator's copilot kept as cases for the evaluation gate.

A case holds the task, who asked, the state the node started in, and every tool call of the run
with what the tool answered, in order. It is recorded from a live run on the twin with the
baseline configuration. A candidate configuration is later run against the same case with the
tools answered from this recording (vxagent/evals/replay.py).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .. import chat
from ..agent.loop import AgentConfig


@dataclass
class Case:
    id: str
    task: str
    role: str
    site: str
    start: dict[str, Any]                 # how the node was prepared: {"ready": bool, "event": str | None}
    asserts: dict[str, Any]               # see assertions.py
    calls: list[dict[str, Any]] = field(default_factory=list)      # {name, arguments, observation}
    approvals: list[str] = field(default_factory=list)             # the operator's decisions, in order
    recorded: dict[str, Any] = field(default_factory=dict)         # model, date, stop reason, answer
    live: bool = False                    # also run live on the twin for every candidate

    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.id}.json"
        path.write_text(json.dumps(asdict(self), indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
        return path


def load_cases(folder: Path) -> list[Case]:
    return [Case(**json.loads(p.read_text(encoding="utf-8"))) for p in sorted(Path(folder).glob("*.json"))]


def calls_of(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool calls of a run with what each returned, in the order the results came back."""
    asked: dict[str, dict[str, Any]] = {}
    out = []
    for m in messages:
        if m["role"] == "assistant":
            for c in m.get("tool_calls") or []:
                asked[c["id"]] = c
        elif m["role"] == "tool":
            c = asked.get(m.get("tool_call_id", ""), {})
            out.append({"name": m.get("name"), "arguments": c.get("arguments", {}), "observation": m["content"]})
    return out


def prepare(node: Any, start: dict[str, Any]) -> None:
    """Put a node into the state a case starts from."""
    if start.get("ready", True):
        node.robot.enable()
        node.robot.home()
    if start.get("event"):
        from ..memory import site as site_memory
        site_memory.record_event(node.db, node.name, start["event"], "recording", "the start state of a case")


def run_live(node: Any, who: Any, task: str, cfg: AgentConfig, approve: bool = True, max_approvals: int = 6) -> dict[str, Any]:
    """One run of the copilot on the node, with the operator's decisions made by the harness.
    Returns the final view, the messages and the decisions that were made."""
    from ..profiles import COPILOTS
    saved = COPILOTS[who.role]
    COPILOTS[who.role] = cfg
    decisions: list[str] = []
    try:
        out = chat.start(node, who, task, session=f"case-{time.time_ns()}")
        while out["status"] == "awaiting_approval" and len(decisions) < max_approvals:
            decisions.append("approved" if approve else "rejected")
            out = chat.decide(node, who, out["approval"]["approval_id"], approve)
    finally:
        COPILOTS[who.role] = saved
    box = node.boxes[out["run_id"]]
    composed = box.state.messages[0]["content"]
    return {"view": out, "messages": box.state.messages, "decisions": decisions, "state": box.state,
            "prompt_suffix": composed[len(cfg.system_prompt):] if composed.startswith(cfg.system_prompt) else ""}
