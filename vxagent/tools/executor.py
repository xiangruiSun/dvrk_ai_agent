"""Tool execution engineering for a live system.

In order, for every call: discovery, authorization by scope and tier, argument validation, the
resource-level check, freshness of the robot state a motion is based on, the operator's approval
bound to that state, idempotency, and then execution with a retry policy that depends on the
tier. Reads may be retried. A motion is never retried and never executed twice. Every failure is
turned into an observation the model can act on; nothing is raised into the agent loop except a
wait for approval and the operator's stop.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ..llm.base import ToolCall
from ..robot.backend import BackendError
from ..store import Database, dumps
from .actions import ActionHandle
from .registry import Tool, ToolContext, ToolError, ToolRegistry, is_allowed, may_request

_POOL = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")
MAX_RESULT_CHARS = 6000


@dataclass
class ToolResult:
    name: str
    ok: bool
    content: Any = None
    error_kind: str | None = None
    error: str | None = None
    hint: str | None = None
    attempts: int = 1
    latency_ms: float = 0.0
    cached: bool = False
    tier: str = ""

    def observation(self) -> str:
        """The text the model reads back. Errors carry the kind and a hint so it can recover,
        and for a motion that stopped, where the arm is now."""
        if self.ok:
            return dumps({"ok": True, "result": self.content})
        body: dict[str, Any] = {"ok": False, "error_kind": self.error_kind, "error": self.error, "hint": self.hint}
        if self.content is not None:
            body["detail"] = self.content
        return dumps(body)


def call_key(name: str, arguments: dict[str, Any]) -> str:
    blob = json.dumps({"n": name, "a": arguments}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


class AwaitingApproval(Exception):
    """Raised to pause the run until the operator approves a motion."""
    is_pause = True

    def __init__(self, approval_id: int, tool: str, arguments: dict[str, Any]):
        super().__init__(f"tool {tool} is waiting for the operator's approval")
        self.approval_id, self.tool, self.arguments = approval_id, tool, arguments


class OperatorStop(Exception):
    """The operator pressed stop. Running actions have been canceled."""
    is_pause = True


def decide_approval(db: Database, approval_id: int, approved: bool, decided_by: str) -> None:
    """Record a person's decision. Called by the console, never by an agent."""
    db.execute("UPDATE approvals SET status=?, decided_by=?, decided=? WHERE id=? AND status='pending'",
               ("approved" if approved else "rejected", decided_by, time.time(), approval_id))


def execute_tool(registry: ToolRegistry, call: ToolCall, ctx: ToolContext,
                 allowed: set[str] | None = None, sleep=time.sleep) -> ToolResult:
    seq0 = ctx.robot.state().seq if ctx.robot is not None else None
    with ctx.tracer.span(ctx.run_id, "tool", call.name, input=call.arguments) as sp:
        sp.seq_start = seq0
        try:
            res = _execute(registry, call, ctx, allowed, sleep)
        finally:
            sp.seq_end = ctx.robot.state().seq if ctx.robot is not None else None
        sp.output, sp.attempts = (res.content if res.ok else {"error": res.error, "detail": res.content}), res.attempts
        if not res.ok:
            sp.fail(res.error_kind or "failed", res.error or "")
        return res


def _execute(registry: ToolRegistry, call: ToolCall, ctx: ToolContext, allowed: set[str] | None,
             sleep) -> ToolResult:
    t0 = time.perf_counter()
    tool = registry.get(call.name)

    def done(**kw: Any) -> ToolResult:
        return ToolResult(name=call.name, latency_ms=(time.perf_counter() - t0) * 1000,
                          tier=tool.tier if tool else "", **kw)

    def refuse(reason: str) -> None:
        ctx.db.audit(ctx.principal.user_id, ctx.principal.role, f"tool:{call.name}",
                     dumps(call.arguments)[:200], False, reason)

    # 1. Discovery: the model may only call tools that exist and were offered to it.
    if tool is None or (allowed is not None and call.name not in allowed):
        offered = sorted(allowed) if allowed is not None else registry.names()
        return done(ok=False, error_kind="unknown_tool", error=f"no tool named {call.name!r}",
                    hint=f"available tools: {', '.join(offered)}")

    # 2. Authorization by scope and tier, from the principal and not from the prompt.
    if not is_allowed(ctx.principal, tool):
        if not may_request(ctx.principal, tool):
            refuse("missing scope or tier")
            return done(ok=False, error_kind="unauthorized",
                        error=f"role {ctx.principal.role!r} may not use {tool.name} ({tool.tier} tier)",
                        hint="do not retry this tool")
        from .. import access                      # actuate is held per run, not by the person
        held, why = access.run_allows(ctx.db, ctx.run_id, ctx.principal.user_id, tool.name)
        if not held:
            refuse(why)
            return done(ok=False, error_kind="unauthorized", error=why, hint="do not retry this tool")

    # 3. Schema validation of the arguments the model produced.
    try:
        args = tool.args_model.model_validate(call.arguments)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in exc.errors())
        return done(ok=False, error_kind="invalid_arguments", error=problems,
                    hint="fix the arguments to match the tool schema and call again")

    # 4. Resource-level authorization (for example, a recording that belongs to another site).
    if tool.authorize is not None:
        try:
            tool.authorize(args, ctx)
        except ToolError as exc:
            refuse(str(exc))
            return done(ok=False, error_kind="unauthorized", error=str(exc), hint="do not retry this tool")

    key = call_key(tool.name, args.model_dump(mode="json"))
    approval = None
    if tool.requires_approval:
        approval = ctx.db.one("SELECT * FROM approvals WHERE run_id=? AND call_key=? ORDER BY id DESC LIMIT 1",
                              (ctx.run_id, key))

    # 5. Freshness. A motion must be based on a state this run has looked at, and that state must
    #    still be the robot's state. The current number is not given back: the model has to look.
    if tool.tier == "actuate" and tool.origin == "local":
        if ctx.robot is None:
            return done(ok=False, error_kind="failed", error="no robot is connected")
        based_on = int(args.based_on_seq)
        now_seq = ctx.robot.state().seq
        look_again = "call get_robot_state and base the request on the state_seq it returns"
        if based_on not in ctx.observed:
            return done(ok=False, error_kind="stale_observation", hint=look_again,
                        error=f"this run has not observed robot state {based_on}")
        if based_on != now_seq:
            return done(ok=False, error_kind="stale_observation", hint=look_again,
                        error="the robot state has changed since you looked; nothing was moved")
        age = time.time() - ctx.observed[based_on]
        if age > ctx.freshness_limit_s and not (approval and approval["status"] == "approved"):
            return done(ok=False, error_kind="stale_observation", hint=look_again,
                        error=f"your look at the robot is {age:.0f} s old (limit {ctx.freshness_limit_s:.0f} s)")

    # 6. The operator's approval, bound to the state it was asked for.
    if tool.requires_approval:
        seq = ctx.robot.state().seq if ctx.robot is not None else None
        version = getattr(getattr(ctx.site, "profile", None), "version", None)
        if approval is not None and approval["status"] == "approved":
            if time.time() - (approval["decided"] or 0) > ctx.approval_ttl_s:
                ctx.db.execute("UPDATE approvals SET status='expired' WHERE id=?", (approval["id"],))
                return done(ok=False, error_kind="approval_expired",
                            error="the operator's approval is too old to use; nothing was moved",
                            hint="look at the robot again and make the request again")
            if approval["robot_seq"] != seq or approval["profile_version"] != version:
                ctx.db.execute("UPDATE approvals SET status='expired' WHERE id=?", (approval["id"],))
                return done(ok=False, error_kind="approval_expired",
                            error="the state the operator approved is no longer the robot's state",
                            hint="look at the robot again and make the request again")
        elif approval is not None and approval["status"] == "rejected":
            return done(ok=False, error_kind="rejected", error="the operator did not approve this action",
                        hint="do not retry; say that the action was not approved")
        elif approval is not None and approval["status"] == "pending":
            raise AwaitingApproval(approval["id"], tool.name, call.arguments)
        else:                                   # none yet, or the last one expired
            aid = ctx.db.execute(
                "INSERT INTO approvals(run_id,call_key,tool,arguments,requested_by,status,created,robot_seq,"
                "profile_version) VALUES (?,?,?,?,?,?,?,?,?)",
                (ctx.run_id, key, tool.name, dumps(call.arguments), ctx.principal.user_id, "pending",
                 time.time(), seq, version))
            raise AwaitingApproval(aid, tool.name, call.arguments)

    # 7. Idempotency. A write that is not a motion happens once per run. A motion carries a
    #    one-use command id, so the same request can never move the arm twice.
    command_id = ""
    if tool.tier == "actuate" and tool.origin == "local":
        command_id = f"{ctx.run_id}:{call.id}"
        if ctx.db.one("SELECT command_id FROM commands WHERE command_id=?", (command_id,)):
            return done(ok=False, error_kind="duplicate_command",
                        error="this motion request was already executed; it was not executed again",
                        hint="look at the robot to see where it is")
        ctx.db.execute("INSERT INTO commands(command_id,run_id,tool,arguments,user_id,seq_before,status,created) "
                       "VALUES (?,?,?,?,?,?,?,?)",
                       (command_id, ctx.run_id, tool.name, dumps(call.arguments), ctx.principal.user_id,
                        ctx.robot.state().seq, "started", time.time()))
    elif tool.side_effect:
        prior = ctx.db.one("SELECT result FROM idempotency WHERE key=?", (f"{ctx.run_id}:{key}",))
        if prior:
            return done(ok=True, content=json.loads(prior["result"]), cached=True)
    elif tool.cacheable and key in ctx.memo:
        return done(ok=True, content=ctx.memo[key], cached=True)

    # 8. Execute. Only retryable failures are retried, and only as often as the tier allows.
    attempts, last = 0, None
    while attempts <= (tool.max_retries or 0):
        attempts += 1
        ctx.command_id = command_id
        try:
            if tool.kind == "action":
                out, ok, err = _run_action(tool, args, ctx)
            else:
                out, ok, err = _POOL.submit(tool.fn, args, ctx).result(timeout=tool.timeout_s), True, None
            _note_state(out, ctx)
            if command_id:
                ctx.db.execute("UPDATE commands SET status=?, seq_after=?, ended=? WHERE command_id=?",
                               ("done" if ok else (err.kind if err else "failed"), ctx.robot.state().seq,
                                time.time(), command_id))
            if not ok:
                assert err is not None
                return done(ok=False, content=out, error_kind=err.kind, error=str(err), hint=err.hint,
                            attempts=attempts)
            size = len(dumps(out))
            if size > MAX_RESULT_CHARS:
                return done(ok=False, error_kind="result_too_large", attempts=attempts,
                            error=f"the result is {size} characters; the limit is {MAX_RESULT_CHARS}",
                            hint="ask for a summary or a narrower window")
            if tool.side_effect:
                ctx.db.execute("INSERT OR REPLACE INTO idempotency(key,tool,result,created) VALUES (?,?,?,?)",
                               (f"{ctx.run_id}:{key}", tool.name, dumps(out), time.time()))
            elif tool.cacheable:
                ctx.memo[key] = out
            ctx.db.audit(ctx.principal.user_id, ctx.principal.role, f"tool:{tool.name}",
                         dumps(call.arguments)[:200], True)
            sink = ctx.extras.get("on_result")           # site memory is written here, by code, from the result
            if sink is not None:
                sink(tool.name, out, call.id)
            return done(ok=True, content=out, attempts=attempts)
        except (OperatorStop, AwaitingApproval) as pause:  # a runbook's inner motion waits for the operator
            if command_id and isinstance(pause, OperatorStop):      # the record says how the motion ended
                ctx.db.execute("UPDATE commands SET status='stopped', seq_after=?, ended=? WHERE command_id=?",
                               (ctx.robot.state().seq, time.time(), command_id))
            raise
        except cf.TimeoutError:
            last = ToolError("timeout", f"{tool.name} did not finish within {tool.timeout_s}s",
                             retryable=tool.tier == "observe", hint="try a narrower request or continue without it")
        except BackendError as exc:
            last = ToolError(exc.kind, str(exc), hint=_backend_hint(exc.kind))
        except ToolError as exc:
            last = exc
        except Exception as exc:  # a bug or an unexpected upstream failure
            last = ToolError("failed", f"{type(exc).__name__}: {exc}")
        if not last.retryable or attempts > (tool.max_retries or 0):
            break
        sleep(min(0.5 * 2 ** (attempts - 1), 4.0))
    assert last is not None
    if command_id:
        ctx.db.execute("UPDATE commands SET status=?, seq_after=?, ended=? WHERE command_id=?",
                       (last.kind, ctx.robot.state().seq, time.time(), command_id))
    return done(ok=False, content=last.detail, error_kind=last.kind, error=str(last), hint=last.hint,
                attempts=attempts)


def _backend_hint(kind: str) -> str:
    return {"robot_not_ready": "the arm must be enabled and homed first",
            "robot_fault": "do not command the arm; report the fault",
            "duplicate_command": "look at the robot to see where it is"}.get(kind, "")


def _note_state(out: Any, ctx: ToolContext) -> None:
    """A result that reports the robot's state counts as a look at the robot."""
    if isinstance(out, dict) and isinstance(out.get("state_seq"), int):
        ctx.note_observed(out["state_seq"])


def _run_action(tool: Tool, args: Any, ctx: ToolContext) -> tuple[dict[str, Any], bool, ToolError | None]:
    """Start an action and wait for it up to tool.wait_s. Returns (content, ok, error)."""
    handle: ActionHandle = tool.fn(args, ctx)
    ctx.actions[handle.id] = handle
    return wait_for_action(handle, ctx, tool.wait_s)


def wait_for_action(handle: ActionHandle, ctx: ToolContext, wait_s: float) -> tuple[dict[str, Any], bool, ToolError | None]:
    deadline = time.time() + wait_s
    while not handle.wait(0.02):
        if ctx.stop_event.is_set():
            handle.cancel()
            handle.wait(5.0)
            raise OperatorStop("the operator pressed stop")
        if time.time() >= deadline:
            break
    content = handle.digest()
    if handle.status == "succeeded":
        return content, True, None
    if handle.status == "running":
        content["note"] = "still running; call wait_action to wait longer or cancel_action to stop it"
        return content, True, None
    if handle.status == "canceled":
        return content, False, ToolError("canceled", "the action was canceled before it finished",
                                         hint="the arm is where 'measured' says it is")
    kind = handle.reason_kind or "failed"
    hints = {"safety_rejected": "the target was not sent; do not try to get around the limit",
             "tracking_error": "the arm stopped; report it and do not repeat the motion",
             "robot_fault": "do not command the arm; report the fault",
             "robot_not_ready": "the arm must be enabled and homed first"}
    return content, False, ToolError(kind, handle.reason or "the action was aborted", hint=hints.get(kind))
