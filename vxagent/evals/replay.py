"""Replay: run a candidate configuration against a recorded case, with the tools answered from
the recording.

Nothing real is touched. Each tool keeps its name, schema and tier, takes its description from
the candidate, and answers with what the recording holds for that call. A call the recording
has no answer for (another tool, other arguments, or one call too many) ends the replay as
"diverged" at that step: from there on nobody knows what the tools would have said, so nothing
after it is judged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..agent.loop import AgentConfig, AgentState, run_agent
from ..llm.base import LLMClient
from ..runtime.trace import Tracer
from ..store import Database
from ..tools.registry import Tool, ToolContext, ToolError, ToolRegistry
from .cases import Case, calls_of


@dataclass
class Candidate:
    """What a change may touch: the prompt, the model, tool descriptions."""
    name: str
    model: str | None = None                                 # None: the node's large model
    remove_from_prompt: list[str] = field(default_factory=list)   # exact sentences to take out
    add_to_prompt: str = ""
    tool_descriptions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Any) -> "Candidate":
        from pathlib import Path
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def config(self, base: AgentConfig) -> AgentConfig:
        import dataclasses
        prompt = base.system_prompt
        for s in self.remove_from_prompt:
            if s not in prompt:
                raise ValueError(f"candidate {self.name}: the prompt does not contain {s[:60]!r}")
            prompt = prompt.replace(s, "")
        return dataclasses.replace(base, system_prompt=prompt + ("\n" + self.add_to_prompt if self.add_to_prompt else ""))


def _norm(args: dict[str, Any]) -> str:
    return json.dumps(args, sort_keys=True, default=str)


def replay_registry(real: ToolRegistry, case: Case, cand: Candidate, cursor: dict[str, Any]) -> ToolRegistry:
    reg = ToolRegistry()

    def answer(name: str):
        def fn(a: Any, ctx: ToolContext) -> Any:
            i = cursor["i"]
            args = a.model_dump(mode="json", exclude_unset=True)
            rec = case.calls[i] if i < len(case.calls) else None
            same = rec is not None and rec["name"] == name and _norm({k: v for k, v in rec["arguments"].items()}) == _norm(args)
            if not same:
                cursor["diverged"] = {"at_call": i + 1, "called": name, "arguments": args,
                                      "recorded": {"name": rec["name"], "arguments": rec["arguments"]} if rec else None}
                ctx.stop_event.set()
                raise ToolError("diverged", "the recording has no answer for this call")
            cursor["i"] = i + 1
            obs = json.loads(rec["observation"])
            if obs.get("ok"):
                return obs.get("result")
            raise ToolError(obs.get("error_kind") or "failed", obs.get("error") or "", hint=obs.get("hint"), detail=obs.get("detail"))
        return fn

    for name in real.names():
        t = real.get(name)
        reg.register(Tool(name=t.name, description=cand.tool_descriptions.get(t.name, t.description), args_model=t.args_model,
                          fn=answer(t.name), tier=t.tier, kind="call", scopes=t.scopes, timeout_s=10, max_retries=0,
                          requires_approval=False, origin="replay"))
    return reg


def replay(case: Case, cand: Candidate, base: AgentConfig, real: ToolRegistry, llm: LLMClient) -> dict[str, Any]:
    """Run the candidate on one case. Returns the trace the assertions are judged on.

    The prompt is the candidate's prompt followed by what the node added when the case was
    recorded (the runbook index, the site, what the site remembered), so that the only
    differences from the recorded run are the candidate's own."""
    import dataclasses

    from ..profiles import principal
    cursor: dict[str, Any] = {"i": 0, "diverged": None}
    reg = replay_registry(real, case, cand, cursor)
    db = Database(":memory:")
    who = principal("replay", case.role, case.site)           # holds the role's tiers: nothing here can move anything
    ctx = ToolContext(principal=who, db=db, tracer=Tracer(db), run_id=f"replay_{case.id}")
    cfg = cand.config(base)
    cfg = dataclasses.replace(cfg, system_prompt=cfg.system_prompt + case.recorded.get("prompt_suffix", ""))
    state = run_agent(llm, reg, ctx, case.task, cfg)
    calls = calls_of(state.messages)
    if cursor["diverged"]:
        calls = calls[:cursor["i"]]
    return {"case": case.id, "candidate": cand.name, "diverged": cursor["diverged"], "calls": calls,
            "stop_reason": "diverged" if cursor["diverged"] else state.stop_reason, "answer": "" if cursor["diverged"] else state.answer,
            "model_calls": state.iterations, "tokens": state.total_tokens, "recorded_calls": len(case.calls),
            "guard_repairs": state.answer_repairs}


def from_live(case: Case, cand_name: str, run: dict[str, Any]) -> dict[str, Any]:
    """The same trace shape for a live run on the twin."""
    st: AgentState = run["state"]
    return {"case": case.id, "candidate": cand_name, "diverged": None, "calls": calls_of(run["messages"]),
            "stop_reason": st.stop_reason, "answer": st.answer, "model_calls": st.iterations, "tokens": st.total_tokens,
            "recorded_calls": len(case.calls), "guard_repairs": st.answer_repairs, "live": True}
