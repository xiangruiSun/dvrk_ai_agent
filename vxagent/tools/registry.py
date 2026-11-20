"""Tool definitions: a name, a description, a Pydantic argument model (the JSON Schema the model
sees is generated from it), a tier and a kind, and the execution policy that follows from them.

Tier    observe    reads state, recordings or documents; may be retried; may run in parallel
        simulate   runs on a copy of the twin; changes nothing real
        actuate    moves the arm; never retried, never in parallel, needs a fresh look at the
                   robot and an operator's approval
Kind    call       returns a result
        action     starts, reports progress, and ends as succeeded, aborted or canceled
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

from ..runtime.trace import Tracer
from ..store import Database

TIERS = ("observe", "simulate", "actuate")


@dataclass(frozen=True)
class Principal:
    """Who the agent is acting for. Authorization is decided from this, never from the prompt."""
    user_id: str
    role: str                      # operator | service | safety | support | system
    scopes: frozenset[str]
    site: str = ""                 # the one site this principal belongs to ("" for none)
    tiers: frozenset[str] = frozenset({"observe"})
    display_name: str = ""
    # A console operator does not hold actuate. With per_run, a motion tool may be requested inside
    # an active run of this person that was registered with that tool (vxagent/access.py).
    per_run: bool = False


SYSTEM = Principal("system", "system", frozenset({"*"}), tiers=frozenset(TIERS), display_name="System")


class ToolError(Exception):
    """A tool failed in a way the model should be told about.

    kind: invalid_arguments | unknown_tool | unauthorized | timeout | failed | transient |
          stale_observation | safety_rejected | robot_not_ready | robot_fault | not_parallel |
          duplicate_command | approval_expired | rejected | canceled | result_too_large
    """

    def __init__(self, kind: str, message: str, retryable: bool = False, hint: str | None = None,
                 detail: Any = None):
        super().__init__(message)
        self.kind, self.retryable, self.hint, self.detail = kind, retryable, hint, detail


class TransientToolError(ToolError):
    def __init__(self, message: str):
        super().__init__("transient", message, retryable=True)


@dataclass
class ToolContext:
    principal: Principal
    db: Database
    tracer: Tracer
    run_id: str = "adhoc"
    robot: Any = None                      # RobotBackend
    site: Any = None                       # robot.site.Site
    memo: dict[str, Any] = field(default_factory=dict)        # per-run cache for repeatable reads
    extras: dict[str, Any] = field(default_factory=dict)
    # Robot states this run has actually looked at: sequence number -> time of the look.
    observed: dict[int, float] = field(default_factory=dict)
    actions: dict[str, Any] = field(default_factory=dict)     # action id -> ActionHandle
    stop_event: threading.Event = field(default_factory=threading.Event)   # the operator's stop
    freshness_limit_s: float = 30.0
    approval_ttl_s: float = 120.0
    command_id: str = ""                   # set by the executor for the motion being executed
    tick: Callable[[float], None] = time.sleep                # control-rate wait; tests replace it

    def note_observed(self, seq: int) -> None:
        self.observed[int(seq)] = time.time()


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[Any, ToolContext], Any]
    tier: str = "observe"
    kind: str = "call"
    scopes: frozenset[str] = frozenset()
    timeout_s: float = 30.0
    max_retries: int | None = None     # None: by tier (observe 2, simulate 1, actuate 0)
    cacheable: bool = False            # a read whose answer cannot change within a run
    side_effect: bool = False          # a write that is not a motion; executed once per run
    requires_approval: bool | None = None   # None: actuate tools need approval
    wait_s: float = 20.0               # how long the loop waits for an action before returning it as running
    authorize: Callable[[Any, ToolContext], None] | None = None
    schema_override: dict[str, Any] | None = None
    origin: str = "local"

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}")
        if self.kind not in ("call", "action"):
            raise ValueError(f"unknown kind {self.kind!r}")
        # A remote tool is validated, checked for freshness and approved by the server that owns it.
        if self.tier == "actuate" and self.origin == "local" and "based_on_seq" not in self.args_model.model_fields:
            raise ValueError(f"{self.name}: an actuate tool must take based_on_seq")
        if self.max_retries is None:
            self.max_retries = {"observe": 2, "simulate": 1, "actuate": 0}[self.tier]
        if self.tier == "actuate" and self.max_retries != 0:
            raise ValueError(f"{self.name}: a motion is never retried automatically")
        if self.requires_approval is None:
            self.requires_approval = self.tier == "actuate"

    def schema(self) -> dict[str, Any]:
        params = _clean(self.schema_override or self.args_model.model_json_schema())
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": params}}


def _clean(schema: Any) -> Any:
    """Drop Pydantic titles so the schema the model reads is as small as possible."""
    if isinstance(schema, dict):
        return {k: _clean(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_clean(v) for v in schema]
    return schema


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def allowed_for(self, principal: Principal, names: list[str] | None = None) -> list[Tool]:
        """Tools this principal may even see. Hiding a tool is a convenience; the executor
        checks scope and tier again on every call."""
        pool = [self._tools[n] for n in (names or self.names()) if n in self._tools]
        return [t for t in pool if is_allowed(principal, t) or may_request(principal, t)]

    def schemas(self, principal: Principal, names: list[str] | None = None) -> list[dict[str, Any]]:
        return [t.schema() for t in self.allowed_for(principal, names)]


def is_allowed(principal: Principal, tool: Tool) -> bool:
    if "*" in principal.scopes:
        return True
    return tool.scopes <= principal.scopes and tool.tier in principal.tiers


def may_request(principal: Principal, tool: Tool) -> bool:
    """A motion tool a per-run principal may ask for. Whether the run holds it is checked by the
    executor on every call."""
    return tool.tier == "actuate" and principal.per_run and tool.scopes <= principal.scopes
