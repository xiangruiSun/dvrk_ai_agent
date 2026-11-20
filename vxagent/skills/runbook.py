"""Runbooks: procedures as files.

A runbook is a YAML file an engineer can read, review and version. Its header says what it is
for, what it needs and what it changes; its steps each name one tool, the arguments, the check
that must pass afterwards, what to do if it does not, and where the step was written from.

A runbook is run by code. Every inner step goes through the tool executor like a call the
model made: validated, authorized, looked at before a motion, approved where a motion needs
it, traced. The checks live in the file, so they cannot be skipped by a model in a hurry.
Runbooks are linear with forward jumps; they have no loops.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..agent.facts import FACTS, RESULT_FIELDS, observe_facts
from ..llm.base import ToolCall
from ..tools.executor import AwaitingApproval, decide_approval, execute_tool
from ..tools.registry import Tool, ToolContext, ToolError, ToolRegistry

STEP_REF = re.compile(r"^\$([a-z][a-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$")
PARAM_REF = re.compile(r"^\$\{([a-z][a-z0-9_]*)\}$")


class RunbookError(ValueError):
    pass


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    equals: Any = None
    at_most: float | None = None
    at_least: float | None = None
    above_step: str | None = Field(default=None, description="'<step>.<field>': this field must exceed that one ...")
    by: float = 0.0                                    # ... by at least this much
    why: str = ""

    @model_validator(mode="after")
    def one_condition(self) -> "Check":
        if sum(x is not None for x in (self.equals, self.at_most, self.at_least, self.above_step)) != 1:
            raise ValueError("a check has exactly one of equals, at_most, at_least, above_step")
        return self

    def describe(self) -> str:
        if self.equals is not None:
            return f"{self.field} is {str(self.equals).lower() if isinstance(self.equals, bool) else self.equals}"
        if self.at_most is not None:
            return f"{self.field} is at most {self.at_most:g}"
        if self.at_least is not None:
            return f"{self.field} is at least {self.at_least:g}"
        return f"{self.field} is above {self.above_step}" + (f" by at least {self.by:g}" if self.by else "")


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    checks: list[Check] = Field(default_factory=list)
    on_fail: str = "stop"                              # stop | recover | goto:<step id>
    source: str = Field(min_length=3, description="Where the step was written from: a documentation unit id or a paper section.")
    say: str = ""


class Runbook(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    summary: str
    when_to_use: str
    tier: Literal["observe", "simulate", "actuate"]
    version: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}, "additionalProperties": False})
    roles: list[Literal["operator", "service"]] = Field(default_factory=lambda: ["operator", "service"],
                                                         description="Who may run it.")
    requires: list[str] = Field(default_factory=list)
    effects: dict[str, bool] = Field(default_factory=dict)
    recovery: str | None = None
    steps: list[Step] = Field(min_length=1, max_length=20)
    content_hash: str = ""
    text: str = ""

    def prose(self) -> str:
        """The same procedure as numbered sentences, for a prompt that has to carry it."""
        lines = [f"Procedure {self.name}: {self.summary} Use it: {self.when_to_use}"]
        for i, s in enumerate(self.steps, 1):
            args = ", ".join(f"{k}={v}" for k, v in s.args.items())
            line = f"  {i}. Call {s.tool}" + (f" with {args}" if args else "") + "."
            if s.checks:
                line += " It must hold that " + " and ".join(c.describe() for c in s.checks) + "."
                if s.on_fail == "stop":
                    line += " If not, stop and report."
                elif s.on_fail == "recover":
                    line += f" If not, run procedure {self.recovery} and report."
                else:
                    line += f" If not, continue at step {[x.id for x in self.steps].index(s.on_fail[5:]) + 1}."
            lines.append(line)
        return "\n".join(lines)


def parse(text: str, registry: ToolRegistry | None = None, known: set[str] | None = None) -> Runbook:
    try:
        rb = Runbook.model_validate(yaml.safe_load(text))
    except (yaml.YAMLError, ValidationError) as exc:
        raise RunbookError(f"not a valid runbook: {exc}") from exc
    rb.text = text
    rb.content_hash = hashlib.sha256(text.replace("\r\n", "\n").encode()).hexdigest()[:12]
    ids = [s.id for s in rb.steps]
    problems = []
    if len(set(ids)) != len(ids):
        problems.append("step ids must be unique")
    for f in list(rb.requires) + list(rb.effects):
        if f not in FACTS:
            problems.append(f"{f!r} is not a fact of the vocabulary ({', '.join(FACTS)})")
    params = set((rb.parameters.get("properties") or {}))
    for i, s in enumerate(rb.steps):
        tool = registry.get(s.tool) if registry is not None else None
        if registry is not None and tool is None:
            problems.append(f"{s.id}: no tool named {s.tool!r}")
        if tool is not None and rb.tier != "actuate" and tool.tier == "actuate":
            problems.append(f"{s.id}: {s.tool} moves the arm but the runbook's tier is {rb.tier}")
        fields = RESULT_FIELDS.get(s.tool, {})
        for c in s.checks:
            if c.field not in fields:
                problems.append(f"{s.id}: {s.tool} returns no field {c.field!r} to check")
            if c.above_step:
                m = STEP_REF.match("$" + c.above_step)
                if not m or m.group(1) not in ids[:i]:
                    problems.append(f"{s.id}: above_step {c.above_step!r} must name an earlier step and a field")
        for v in s.args.values():
            if isinstance(v, str) and v.startswith("$"):
                m, p = STEP_REF.match(v), PARAM_REF.match(v)
                if p and p.group(1) not in params:
                    problems.append(f"{s.id}: {v} is not a parameter of the runbook")
                elif m and (m.group(1) not in ids[:i] or m.group(2) not in RESULT_FIELDS.get(rb.steps[ids.index(m.group(1))].tool, {})):
                    problems.append(f"{s.id}: {v} must name a field of an earlier step's result")
                elif not m and not p:
                    problems.append(f"{s.id}: {v!r} is neither ${{parameter}} nor $step.field")
        if s.on_fail.startswith("goto:"):
            if s.on_fail[5:] not in ids[i + 1:]:
                problems.append(f"{s.id}: on_fail may only jump forward to a later step, not to {s.on_fail[5:]!r}")
        elif s.on_fail == "recover":
            if not rb.recovery:
                problems.append(f"{s.id}: on_fail is recover but the runbook names no recovery runbook")
            elif known is not None and rb.recovery not in known:
                problems.append(f"recovery runbook {rb.recovery!r} does not exist")
        elif s.on_fail != "stop":
            problems.append(f"{s.id}: on_fail must be stop, recover or goto:<step>")
    if problems:
        raise RunbookError(f"{rb.name}: " + "; ".join(problems))
    return rb


class Library:
    """The runbooks of a folder. Only the index is cheap; the text of one is read on demand."""

    def __init__(self, folder: Path | str, registry: ToolRegistry | None = None, strict: bool = True):
        """With strict=False a runbook that names a tool this registry does not have is left out
        and listed in `unavailable`: a set-up without the service tools has no service runbooks."""
        self.folder, self.registry = Path(folder), registry
        names = {p.stem for p in self.folder.glob("*.yaml")}
        self.books: dict[str, Runbook] = {}
        self.unavailable: dict[str, list[str]] = {}
        for p in sorted(self.folder.glob("*.yaml")):
            text = p.read_text(encoding="utf-8")
            if not strict and registry is not None:
                missing = sorted({s.get("tool", "") for s in (yaml.safe_load(text) or {}).get("steps", [])
                                  if isinstance(s, dict) and registry.get(s.get("tool", "")) is None})
                if missing:
                    self.unavailable[p.stem] = missing
                    continue
            self.books[p.stem] = parse(text, registry, names)
        for name, rb in self.books.items():
            if rb.name != name:
                raise RunbookError(f"{name}.yaml declares the name {rb.name!r}")

    def index(self, role: str | None = None) -> list[dict[str, str]]:
        return [{"name": rb.name, "summary": rb.summary, "when_to_use": rb.when_to_use, "tier": rb.tier}
                for rb in self.books.values() if role is None or role in rb.roles]

    def get(self, name: str) -> Runbook:
        if name not in self.books:
            raise ToolError("invalid_arguments", f"no runbook named {name!r}", hint=f"runbooks: {', '.join(self.books)}")
        return self.books[name]


def _passes(c: Check, result: dict[str, Any], results: dict[str, Any]) -> tuple[bool, Any]:
    seen = result.get(c.field)
    if c.equals is not None:
        return seen == c.equals, seen
    if seen is None:
        return False, None
    if c.at_most is not None:
        return float(seen) <= c.at_most, seen
    if c.at_least is not None:
        return float(seen) >= c.at_least, seen
    other_step, other_field = c.above_step.split(".")                       # type: ignore[union-attr]
    other = results.get(other_step, {}).get(other_field)
    return other is not None and float(seen) >= float(other) + c.by, {"now": seen, "then": other}


@dataclass
class _Progress:
    position: int = 0
    results: dict[str, Any] = None        # type: ignore[assignment]
    rows: list[dict[str, Any]] = None     # type: ignore[assignment]
    looks: dict[str, int] = None          # type: ignore[assignment]


def run(library: Library, name: str, params: dict[str, Any], registry: ToolRegistry, ctx: ToolContext, call_id: str,
        on_approval: Callable[[AwaitingApproval], bool] | None = None, _depth: int = 0) -> dict[str, Any]:
    """Run one runbook. Raises AwaitingApproval when an inner motion waits for the operator;
    calling again with the same call_id goes on from that step."""
    import jsonschema

    rb = library.get(name)
    role = ctx.principal.role
    if role not in rb.roles and "*" not in ctx.principal.scopes:
        ctx.db.audit(ctx.principal.user_id, role, f"runbook:{name}", "", False, "not a runbook for this role")
        raise ToolError("unauthorized", f"runbook {name} is for: {', '.join(rb.roles)}", hint="do not retry; open a ticket instead")
    try:
        jsonschema.validate(params, rb.parameters)
    except jsonschema.ValidationError as exc:
        raise ToolError("invalid_arguments", f"parameters of {name}: {exc.message}", hint="see load_runbook for the parameters")
    facts = observe_facts(ctx)
    unmet = [f for f in rb.requires if facts.get(f) is not True]
    store = ctx.extras.setdefault("runbook_progress", {})
    prog: _Progress = store.setdefault(call_id, _Progress(0, {}, [], {}))
    head = {"runbook": rb.name, "version": rb.version, "content_hash": rb.content_hash}
    if unmet and prog.position == 0:
        return {**head, "status": "not_started", "unmet_requirements": unmet, "steps": [],
                "note": "nothing was done: the runbook's requirements are not met"}
    ids = [s.id for s in rb.steps]

    def call(tool: str, args: dict[str, Any], cid: str) -> Any:
        while True:
            try:
                return execute_tool(registry, ToolCall(id=cid, name=tool, arguments=args), ctx)
            except AwaitingApproval as wait:
                if on_approval is None:
                    raise
                decide_approval(ctx.db, wait.approval_id, bool(on_approval(wait)), "operator")

    status, stopped_at = "completed", None
    while prog.position < len(rb.steps):
        s = rb.steps[prog.position]
        tool: Tool | None = registry.get(s.tool)
        args: dict[str, Any] = {}
        for k, v in s.args.items():
            m = STEP_REF.match(v) if isinstance(v, str) else None
            p = PARAM_REF.match(v) if isinstance(v, str) else None
            args[k] = prog.results[m.group(1)][m.group(2)] if m else (params.get(p.group(1), (rb.parameters["properties"][p.group(1)]).get("default")) if p else v)
        if tool is not None and tool.tier == "actuate":
            if s.id not in prog.looks:                 # the look is by code, and is the one the approval is bound to
                look = call("get_robot_state", {}, f"{call_id}:{s.id}:look")
                if not look.ok:
                    prog.rows.append({"id": s.id, "tool": s.tool, "ok": False, "error_kind": look.error_kind, "checks": []})
                    status, stopped_at = "stopped", s.id
                    break
                prog.looks[s.id] = look.content["state_seq"]
            args["based_on_seq"] = prog.looks[s.id]
        res = call(s.tool, args, f"{call_id}:{s.id}")
        row: dict[str, Any] = {"id": s.id, "tool": s.tool, "ok": res.ok, "error_kind": res.error_kind, "checks": [],
                               "source": s.source}
        if not res.ok:
            row["error"] = res.error
        else:
            prog.results[s.id] = res.content
            for c in s.checks:
                ok, seen = _passes(c, res.content, prog.results)
                row["checks"].append({"check": c.describe(), "passed": ok, "observed": seen})
        prog.rows.append(row)
        failed = (not res.ok) or any(not c["passed"] for c in row["checks"])
        if not failed:
            prog.position += 1
            continue
        if res.ok and s.on_fail.startswith("goto:"):
            row["then"] = s.on_fail
            prog.position = ids.index(s.on_fail[5:])
            continue
        status, stopped_at = "stopped", s.id
        if res.ok and s.on_fail == "recover" and rb.recovery and _depth == 0:
            rec = run(library, rb.recovery, {}, registry, ctx, f"{call_id}:recovery", on_approval, _depth=1)
            status = "recovered" if rec["status"] == "completed" else "recovery_failed"
            row["recovery"] = {"runbook": rec["runbook"], "status": rec["status"],
                               "steps": [(r["id"], r["ok"]) for r in rec["steps"]]}
        prog.position = len(rb.steps)
    after = observe_facts(ctx)
    return {**head, "status": status, "stopped_at": stopped_at, "steps": prog.rows,
            "checks_run": sum(len(r["checks"]) for r in prog.rows),
            "checks_failed": sum(1 for r in prog.rows for c in r["checks"] if not c["passed"]),
            "facts_after": {f: after[f] for f in rb.effects}}
