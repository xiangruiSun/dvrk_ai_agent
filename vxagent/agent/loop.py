"""The tool-calling agent loop (ReAct) for a live system: model response -> tool execution ->
observation -> next model response, with explicit stopping conditions.

Stop reasons
    final_answer             the model answered without requesting a tool (the normal ending)
    final_answer_unverified  the answer still failed the grounding guard after its repair turns
    max_iterations           the step limit was reached
    token_budget             the run used its token allowance
    wall_clock               the run exceeded its time allowance
    loop_detected            the same tool call gave the same result three times in a row
    tool_failures            too many consecutive tool failures
    safety_rejections        the safety filter rejected three motions in a row
    robot_fault              the robot reported a fault; the agent must not go on commanding it
    operator_stop            the operator pressed stop; running actions were canceled
    awaiting_approval        a motion needs the operator's decision; the run can be resumed
    llm_error                the model could not be reached or timed out
    empty_reply              the model returned neither text nor a tool call, three times

Whenever a run stops, any action still running is canceled first.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field

from ..llm.base import LLMClient, LLMError, ToolCall, messages_tokens
from ..memory.working import fit_context
from ..robot.motion import state_report
from ..store import dumps
from ..tools.executor import AwaitingApproval, OperatorStop, call_key, execute_tool
from ..tools.registry import ToolContext, ToolRegistry


@dataclass
class AgentConfig:
    name: str = "agent"
    system_prompt: str = "You are a careful assistant. Use the tools to get facts; do not guess."
    tool_names: list[str] | None = None
    max_iterations: int = 10
    token_budget: int = 40000
    wall_clock_s: float = 600.0
    max_consecutive_tool_failures: int = 3
    max_consecutive_safety_rejections: int = 3
    loop_repeat_limit: int = 3
    context_tokens: int = 6000
    temperature: float = 0.0
    finalize_on_limit: bool = True
    # Offer, besides tool_names, the tools the last find_tools call returned (task 05).
    dynamic_tools: bool = False
    # Put the robot's current state, read by code, in front of the model on every turn (task 06 R9).
    inject_robot_state: bool = False
    # Optional check of a proposed final answer: returns a list of problems (empty = accept).
    answer_check: Callable[["AgentState", ToolContext], list[str]] | None = None
    max_answer_repairs: int = 2


class AgentState(BaseModel):
    run_id: str
    task: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    iterations: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    status: str = "running"          # running | completed | stopped | awaiting_approval | failed
    stop_reason: str = ""
    answer: str = ""
    call_counts: dict[str, int] = Field(default_factory=dict)      # calls per tool, for reporting
    last_call_sig: str = ""
    seen_call_ids: list[str] = Field(default_factory=list)
    repeat_streak: int = 0
    consecutive_failures: int = 0
    consecutive_safety_rejections: int = 0
    robot_fault: bool = False
    pending_calls: list[dict[str, Any]] = Field(default_factory=list)
    pending_approval: dict[str, Any] | None = None
    elapsed_s: float = 0.0
    tool_log: list[dict[str, Any]] = Field(default_factory=list)
    answer_candidate: str = ""
    answer_repairs: int = 0
    guard_log: list[list[str]] = Field(default_factory=list)
    schema_chars: int = 0            # largest set of tool schemas sent in one turn, in characters
    context_tokens_sent: int = 0     # largest context sent in one turn, estimated tokens
    injected_states: list[str] = Field(default_factory=list)   # robot states shown to the model by the system
    empty_replies: int = 0           # replies with neither text nor a tool call, which were sent back

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def run_agent(llm: LLMClient, registry: ToolRegistry, ctx: ToolContext, task: str | None = None,
              config: AgentConfig | None = None, state: AgentState | None = None,
              on_step: Callable[[AgentState], None] | None = None) -> AgentState:
    cfg = config or AgentConfig()
    allowed: set[str] = set()
    tiers: dict[str, str] = {}
    schemas: list[dict[str, Any]] = []

    def offer() -> None:
        """The tools this turn may use. With dynamic_tools, what the last search returned is added."""
        names = list(cfg.tool_names) if cfg.tool_names is not None else None
        if cfg.dynamic_tools and names is not None:
            names += [n for n in ctx.extras.get("offered_tools", []) if n not in names]
        tools = registry.allowed_for(ctx.principal, names)
        allowed.clear()
        allowed.update(t.name for t in tools)
        tiers.clear()
        tiers.update({t.name: t.tier for t in tools})
        schemas[:] = [t.schema() for t in tools]

    offer()
    if state is None:
        if task is None:
            raise ValueError("task is required for a new run")
        state = AgentState(run_id=ctx.run_id, task=task, messages=[
            {"role": "system", "content": cfg.system_prompt}, {"role": "user", "content": task}])
    state.status, state.stop_reason, state.pending_approval = "running", "", None
    t_start = time.perf_counter() - state.elapsed_s

    def stop(status: str, reason: str, answer: str = "") -> AgentState:
        _cancel_running(ctx)                          # nothing keeps moving after the run has stopped
        state.status, state.stop_reason = status, reason
        if answer:
            state.answer = answer
        state.elapsed_s = time.perf_counter() - t_start
        if on_step:
            on_step(state)
        return state

    def observe(call: ToolCall, content: str, ok: bool, kind: str | None, **log: Any) -> None:
        state.messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content})
        state.tool_log.append({"tool": call.name, "tier": tiers.get(call.name, ""), "ok": ok, "error_kind": kind, **log})

    def run_calls(calls: list[dict[str, Any]]) -> bool:
        """Execute tool calls and append observations. Returns False when the run must pause."""
        # A motion is never one of several calls in a response: the model must see the result of
        # a motion before it decides anything else.
        if len(calls) > 1 and any(tiers.get(c["name"]) == "actuate" for c in calls):
            for c in calls:
                call = ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"])
                observe(call, dumps({"ok": False, "error_kind": "not_parallel",
                                     "error": "a motion must be the only tool call in a response; nothing was executed",
                                     "hint": "send one call, read its result, then decide the next"}),
                        False, "not_parallel")
            state.consecutive_failures += 1
            state.pending_calls = []
            return True
        for i, c in enumerate(calls):
            call = ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"])
            try:
                res = execute_tool(registry, call, ctx, allowed=allowed)
            except AwaitingApproval as wait:
                state.pending_calls = calls[i:]
                state.pending_approval = {"approval_id": wait.approval_id, "tool": wait.tool, "arguments": wait.arguments}
                return False
            except OperatorStop:
                # The call was interrupted. Record that, so the history never holds a request
                # without its outcome, then let the stop end the run.
                observe(call, dumps({"ok": False, "error_kind": "operator_stop",
                                     "error": "the operator pressed stop; the action was canceled",
                                     "detail": json.loads(_where(ctx) or "{}")}), False, "operator_stop")
                state.pending_calls = []
                raise
            observe(call, res.observation(), res.ok, res.error_kind, attempts=res.attempts, cached=res.cached,
                    latency_ms=round(res.latency_ms, 1))
            state.consecutive_failures = 0 if res.ok else state.consecutive_failures + 1
            if res.error_kind == "safety_rejected":
                state.consecutive_safety_rejections += 1
            elif res.tier == "actuate":
                state.consecutive_safety_rejections = 0
            if res.error_kind == "robot_fault":
                state.robot_fault = True
            # A loop is the same call giving the same result, back to back. Looking at the robot
            # again is not a loop when something else happened in between or the state changed.
            sig = call_key(call.name, {"a": call.arguments, "r": res.observation()})
            state.call_counts[call.name] = state.call_counts.get(call.name, 0) + 1
            state.repeat_streak = state.repeat_streak + 1 if sig == state.last_call_sig else 1
            state.last_call_sig = sig
            if state.repeat_streak == cfg.loop_repeat_limit - 1 and cfg.loop_repeat_limit > 2:
                state.messages.append({"role": "user", "content":
                                       f"You have already called {call.name} with these exact arguments. Use the "
                                       "result, try something different, or give the final answer."})
        state.pending_calls = []
        return True

    try:
        # Resume: finish the tool calls that were waiting (for example after the operator approved).
        if state.pending_calls and not run_calls(state.pending_calls):
            return stop("awaiting_approval", "awaiting_approval")

        while True:
            state.elapsed_s = time.perf_counter() - t_start
            limit = None
            if ctx.stop_event.is_set():
                raise OperatorStop("the operator pressed stop")
            if state.robot_fault:
                limit = "robot_fault"
            elif state.consecutive_safety_rejections >= cfg.max_consecutive_safety_rejections:
                limit = "safety_rejections"
            elif state.repeat_streak >= cfg.loop_repeat_limit:
                limit = "loop_detected"
            elif state.consecutive_failures >= cfg.max_consecutive_tool_failures:
                limit = "tool_failures"
            elif state.iterations >= cfg.max_iterations:
                limit = "max_iterations"
            elif state.total_tokens >= cfg.token_budget:
                limit = "token_budget"
            elif state.elapsed_s >= cfg.wall_clock_s:
                limit = "wall_clock"
            if limit:
                _cancel_running(ctx)
                return stop("stopped", limit, _wrap_up(llm, state, cfg, ctx, limit) if cfg.finalize_on_limit else "")

            if cfg.dynamic_tools:
                offer()
            state.schema_chars = max(state.schema_chars, len(dumps(schemas)))
            now = None
            if cfg.inject_robot_state and ctx.robot is not None:
                now = _where(ctx)
                state.injected_states = (state.injected_states + [now])[-3:]
                ctx.note_observed(ctx.robot.state().seq)       # a look by code is a look
            context = fit_context(state.messages, cfg.context_tokens, anchor_task=state.task, iteration=state.iterations,
                                  state_now=now)
            state.context_tokens_sent = max(state.context_tokens_sent, messages_tokens(context))
            try:
                with ctx.tracer.span(ctx.run_id, "llm", f"{cfg.name}.step{state.iterations + 1}",
                                     input={"messages": len(context), "tools": sorted(allowed)}) as sp:
                    resp = llm.chat(context, tools=schemas or None, temperature=cfg.temperature)
                    sp.prompt_tokens, sp.completion_tokens = resp.usage.prompt_tokens, resp.usage.completion_tokens
                    sp.output = {"content": resp.content[:800], "tool_calls": [c.name for c in resp.tool_calls]}
            except LLMError as exc:
                return stop("failed", "llm_error", f"The model call failed ({exc.kind}): {exc}")
            state.iterations += 1
            state.prompt_tokens += resp.usage.prompt_tokens
            state.completion_tokens += resp.usage.completion_tokens
            # A motion's one-use command id is derived from the call id, so call ids must be
            # unique within a run even if the model server repeats them.
            for c in resp.tool_calls:
                base, n = c.id, 1
                while c.id in state.seen_call_ids:
                    n += 1
                    c.id = f"{base}_{n}"
                state.seen_call_ids.append(c.id)
            state.messages.append(resp.as_message())

            if not resp.tool_calls:                       # stopping condition: a plain answer
                state.answer_candidate = resp.content.strip()
                if not state.answer_candidate and state.empty_replies < 2:
                    # Nothing at all is not an answer. (Seen with the real model on "Run the rig check".)
                    state.empty_replies += 1
                    state.messages.append({"role": "user", "content":
                                           "Your reply was empty. Call the tool the request needs, or write the answer."})
                    continue
                if not state.answer_candidate:
                    return stop("failed", "empty_reply", "The model returned an empty reply three times. Nothing was done "
                                                         "beyond what the trace shows.")
                if cfg.answer_check is not None and state.answer_repairs < cfg.max_answer_repairs:
                    with ctx.tracer.span(ctx.run_id, "guard", f"{cfg.name}.answer_check") as sp:
                        problems = cfg.answer_check(state, ctx)
                        sp.output = {"problems": problems}
                        if problems:
                            sp.fail("ungrounded_answer", "; ".join(problems)[:400])
                    if problems:
                        state.answer_repairs += 1
                        state.guard_log.append(problems)
                        state.messages.append({"role": "user", "content":
                                               "Your answer cannot be accepted yet:\n"
                                               + "\n".join(f"{i}. {p}" for i, p in enumerate(problems, 1))
                                               + "\nFix these, using tools if needed, then write your COMPLETE "
                                                 "answer again, not only the corrected part."})
                        continue
                if cfg.answer_check is not None:
                    left = cfg.answer_check(state, ctx)
                    if left:      # repairs exhausted: never pass an unverified statement off as verified
                        return stop("completed", "final_answer_unverified", state.answer_candidate
                                    + "\n\n[Not verified against tool results: " + " | ".join(left)[:500] + "]")
                return stop("completed", "final_answer", state.answer_candidate)
            calls = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in resp.tool_calls]
            if not run_calls(calls):
                return stop("awaiting_approval", "awaiting_approval")
            if on_step:
                on_step(state)
    except OperatorStop:
        where = _where(ctx)
        return stop("stopped", "operator_stop",
                    "Stopped by the operator. Running actions were canceled."
                    + (f" The arm is at {where}." if where else ""))


def _cancel_running(ctx: ToolContext) -> None:
    for h in list(ctx.actions.values()):
        if not h.done:
            h.cancel()
            h.wait(5.0)


def _where(ctx: ToolContext) -> str:
    """Where the arm is now, read by code. Used when a run is stopped, so the last word about the
    robot's position never comes from the model's memory of it."""
    if ctx.robot is None:
        return ""
    scene = getattr(ctx.site, "scene", None)
    return dumps(state_report(ctx.robot.state(), scene))


def _wrap_up(llm: LLMClient, state: AgentState, cfg: AgentConfig, ctx: ToolContext, reason: str) -> str:
    """One last call without tools so a limited run still returns something honest."""
    where = _where(ctx)
    note = {"role": "user", "content":
            f"Stop now ({reason.replace('_', ' ')}). Do not request any tool. Using only the observations above, say "
            "what was done, and what was not done or could not be verified"
            + (f", and where the arm is.\nThe robot's state right now, read by the system: {where}" if where else ".")}
    try:
        with ctx.tracer.span(ctx.run_id, "llm", f"{cfg.name}.wrap_up") as sp:
            resp = llm.chat(fit_context(state.messages + [note], cfg.context_tokens), tools=None, temperature=0.0)
            sp.prompt_tokens, sp.completion_tokens = resp.usage.prompt_tokens, resp.usage.completion_tokens
        state.prompt_tokens += resp.usage.prompt_tokens
        state.completion_tokens += resp.usage.completion_tokens
        return resp.content.strip()
    except LLMError:
        return ""


def tool_messages(state: AgentState) -> list[dict[str, Any]]:
    """Parsed tool observations, in order. Used by the guard and by the evaluation scripts."""
    out = []
    for m in state.messages:
        if m["role"] == "tool":
            try:
                out.append({"name": m.get("name"), **json.loads(m["content"])})
            except json.JSONDecodeError:
                out.append({"name": m.get("name"), "ok": False, "raw": m["content"]})
    return out
