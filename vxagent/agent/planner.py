"""Planning with preconditions and data flow.

One structured model call turns a goal into a plan of at most ten steps. A step's argument is
a literal or a reference to a field of an earlier step's result ("$s1.calibration_id"). Before
anything runs, the plan is checked by code: tools exist and are allowed, arguments validate,
references point at a field of the right type of a step this one depends on, there is no
cycle, no two motions share a level, and the plan is executed symbolically from the observed
facts so that every step's needs are met when it is reached. A plan with motion is then
rehearsed on a copy of the twin. Only after that is it executed, level by level, through the
same tool executor as any other call. One re-plan is allowed when a step fails or the robot is
not in the state the step declared.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..llm.base import LLMClient, LLMError, ToolCall
from ..robot.backend import BackendError
from ..runtime.trace import Tracer
from ..store import Database, dumps
from ..tools.executor import AwaitingApproval, decide_approval, execute_tool, wait_for_action
from ..tools.registry import SYSTEM, ToolContext, ToolError, ToolRegistry, is_allowed
from .facts import RESULT_FIELDS, TOOL_FACTS, observe_facts

MAX_STEPS = 10
REF = re.compile(r"^\$(s\d+)\.([A-Za-z_][A-Za-z0-9_]*)$")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^s\d+$", description="s1, s2, ...")
    purpose: str
    tool: str
    arguments: dict[str, Any] = Field(description='Literals, or "$s1.field" for a field of step s1\'s result.')
    depends_on: list[str] = Field(description="Ids of the steps that must finish before this one.")


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: list[PlanStep] = Field(min_length=1, max_length=MAX_STEPS)


# ------------------------------------------------------------------------------ validation

def levels(plan: Plan) -> list[list[PlanStep]] | None:
    """Steps grouped so that every step comes after everything it depends on. None on a cycle."""
    by = {s.id: s for s in plan.steps}
    done: set[str] = set()
    out: list[list[PlanStep]] = []
    while len(done) < len(by):
        ready = [s for s in plan.steps if s.id not in done and all(d in done for d in s.depends_on if d in by)]
        if not ready:
            return None
        out.append(ready)
        done |= {s.id for s in ready}
    return out


def _ancestors(plan: Plan, sid: str) -> set[str]:
    by = {s.id: s for s in plan.steps}
    seen: set[str] = set()
    todo = list(by[sid].depends_on)
    while todo:
        d = todo.pop()
        if d in by and d not in seen:
            seen.add(d)
            todo += by[d].depends_on
    return seen


def _arg_types(tool: Any, name: str) -> set[str]:
    prop = tool.args_model.model_json_schema().get("properties", {}).get(name, {})
    kinds = {prop["type"]} if "type" in prop else {p.get("type") for p in prop.get("anyOf", [])}
    return {k for k in kinds if k}


_DUMMY = {"string": "x", "integer": 0, "number": 0.0, "boolean": True}


def validate_plan(plan: Plan, registry: ToolRegistry, principal: Any, allowed: set[str] | None,
                  facts: dict[str, bool | None]) -> list[str]:
    errors: list[str] = []
    ids = [s.id for s in plan.steps]
    if len(set(ids)) != len(ids):
        errors.append("step ids must be unique")
    by = {s.id: s for s in plan.steps}
    for s in plan.steps:
        for d in s.depends_on:
            if d not in by:
                errors.append(f"{s.id} depends on {d}, which is not a step of the plan")
            if d == s.id:
                errors.append(f"{s.id} depends on itself")
    lv = levels(plan)
    if lv is None:
        errors.append("the dependencies form a cycle")
    for s in plan.steps:
        tool = registry.get(s.tool)
        if tool is None or (allowed is not None and s.tool not in allowed):
            errors.append(f"{s.id}: no tool named {s.tool!r}; tools: {', '.join(sorted(allowed or registry.names()))}")
            continue
        if not is_allowed(principal, tool):
            errors.append(f"{s.id}: role {principal.role!r} may not use {s.tool} ({tool.tier} tier)")
            continue
        probe: dict[str, Any] = {}
        for name, value in s.arguments.items():
            m = REF.match(value) if isinstance(value, str) else None
            if isinstance(value, str) and value.startswith("$") and not m:
                errors.append(f"{s.id}.{name}: {value!r} is not a reference; write $s1.field")
                continue
            if not m:
                probe[name] = value
                continue
            src, fld = m.group(1), m.group(2)
            if src not in by:
                errors.append(f"{s.id}.{name}: refers to {src}, which is not a step of the plan")
                continue
            if lv is not None and src not in _ancestors(plan, s.id):
                errors.append(f"{s.id}.{name}: uses a result of {src} but {s.id} does not depend on {src}")
            fields = RESULT_FIELDS.get(by[src].tool, {})
            if fld not in fields:
                errors.append(f"{s.id}.{name}: the result of {by[src].tool} has no field {fld!r} "
                              f"(fields: {', '.join(fields) or 'none'})")
                continue
            want = _arg_types(tool, name)
            have = fields[fld]
            if want and have not in want and not (have == "integer" and "number" in want):
                errors.append(f"{s.id}.{name}: {src}.{fld} is {have} but {s.tool} takes {' or '.join(sorted(want))} there")
                continue
            probe[name] = _DUMMY[have]
        if "based_on_seq" in tool.args_model.model_fields:
            if "based_on_seq" in s.arguments:
                errors.append(f"{s.id}: do not give based_on_seq; the executor looks at the robot itself before a motion")
            probe["based_on_seq"] = 0
        try:
            tool.args_model.model_validate(probe)
        except ValidationError as exc:
            errors += [f"{s.id}.{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in exc.errors()]
    if lv is None or errors:
        return errors
    for level in lv:
        motions = [s.id for s in level if registry.get(s.tool).tier == "actuate"]       # type: ignore[union-attr]
        if len(motions) > 1:
            errors.append(f"{' and '.join(motions)} are motions that could run at the same time: make one depend on "
                          f"the other")
    # Symbolic execution from the observed facts.
    state = dict(facts)
    for level in lv:
        for s in level:
            needs, sets = TOOL_FACTS.get(s.tool, ((), {}))
            for f in needs:
                if state.get(f) is not True:
                    how = "not known from observation" if state.get(f) is None else "false"
                    errors.append(f"{s.id} ({s.tool}) needs '{f}', which is {how} at that point of the plan")
        for s in level:
            state.update(TOOL_FACTS.get(s.tool, ((), {}))[1])
    return errors


# ------------------------------------------------------------------------------ planning

SYSTEM_PROMPT = (
    "You turn a service request for a da Vinci Research Kit arm into a plan, as one JSON object with a list of "
    "steps. Each step: id (s1, s2, ...), purpose, tool, arguments, depends_on.\n"
    "Rules:\n"
    "1. Use only the tools listed, with the argument names listed.\n"
    "2. An argument is a literal, or \"$sN.field\" for a field of the result of step sN. A step that uses $sN must "
    "list sN in depends_on.\n"
    "3. Every tool lists the facts it needs. They must be true when the step is reached: either true now, or made "
    "true by an earlier step it depends on. If nothing can make a needed fact true, still return the shortest "
    "plan for the request; it will be refused with the reason.\n"
    "4. Motions run one after another: a motion depends on the motion before it.\n"
    "5. Never give based_on_seq. Never invent a pose name: use the named poses listed.\n"
    "6. At most 10 steps. No step that the request does not need.")


@dataclass
class Planned:
    plan: Plan | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.plan is not None


def describe_tools(registry: ToolRegistry, principal: Any, names: list[str]) -> str:
    rows = []
    for t in registry.allowed_for(principal, names):
        props = t.args_model.model_json_schema().get("properties", {})
        args = {k: v.get("type", "any") for k, v in props.items() if k != "based_on_seq"}
        needs, sets = TOOL_FACTS.get(t.name, ((), {}))
        rows.append(dumps({"tool": t.name, "does": t.description[:160], "motion": t.tier == "actuate", "arguments": args,
                           "needs": list(needs), "makes_true": [k for k, v in sets.items() if v],
                           "result_fields": RESULT_FIELDS.get(t.name, {})}))
    return "\n".join(rows)


def make_plan(llm: LLMClient, goal: str, registry: ToolRegistry, ctx: ToolContext, tool_names: list[str],
              facts: dict[str, bool | None], history: str = "", max_attempts: int = 3) -> Planned:
    out = Planned()
    allowed = {t.name for t in registry.allowed_for(ctx.principal, tool_names)}
    poses = ", ".join(getattr(ctx.site, "poses", {}) or [])
    user = (f"Request: {goal}\n\nFacts observed now: {dumps(facts)}\nNamed poses of this site: {poses}\n\nTools:\n"
            f"{describe_tools(registry, ctx.principal, tool_names)}" + (f"\n\nWhat happened so far:\n{history}" if history else ""))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    for n in range(1, max_attempts + 1):
        att: dict[str, Any] = {"attempt": n, "stage": "llm_error", "errors": []}
        out.attempts.append(att)
        try:
            with ctx.tracer.span(ctx.run_id, "llm", f"planner.attempt{n}") as sp:
                resp = llm.chat(messages, json_schema=Plan.model_json_schema(), temperature=0.0, max_tokens=1200)
                sp.prompt_tokens, sp.completion_tokens = resp.usage.prompt_tokens, resp.usage.completion_tokens
        except LLMError as exc:
            att["errors"] = [f"{exc.kind}: {exc}"]
            break
        out.tokens += resp.usage.total
        plan = None
        try:
            plan = Plan.model_validate(json.loads(resp.content))
        except json.JSONDecodeError as exc:
            att["stage"], att["errors"] = "json", [str(exc)]
        except ValidationError as exc:
            att["stage"] = "schema"
            att["errors"] = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()][:10]
        if plan is not None:
            att["plan"] = [{"id": s.id, "tool": s.tool, "arguments": s.arguments, "depends_on": s.depends_on} for s in plan.steps]
            att["errors"] = validate_plan(plan, registry, ctx.principal, allowed, facts)
            att["stage"] = "validation" if att["errors"] else "ok"
            if not att["errors"]:
                out.plan = plan
                return out
        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": "That plan was not accepted. Problems:\n"
                         + "\n".join(f"{i}. {e}" for i, e in enumerate(att["errors"], 1))
                         + "\nSend the corrected plan as one JSON object."})
    return out


# ------------------------------------------------------------------------------ rehearsal and execution

def _resolve(step: PlanStep, results: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name, value in step.arguments.items():
        m = REF.match(value) if isinstance(value, str) else None
        out[name] = results[m.group(1)][m.group(2)] if m else value
    return out


def rehearse(plan: Plan, registry: ToolRegistry, ctx: ToolContext) -> dict[str, Any]:
    """Run the plan's robot steps on a copy of the twin. Nothing real moves and no approval is used."""
    order = [s for level in (levels(plan) or []) for s in level]
    if not any(registry.get(s.tool).tier == "actuate" for s in order):        # type: ignore[union-attr]
        return {"needed": False, "clean": True, "steps": []}
    if not hasattr(ctx.robot, "clone"):
        return {"needed": True, "clean": False, "steps": [], "error": "this backend cannot be rehearsed"}
    db = Database(":memory:")
    dctx = ToolContext(principal=SYSTEM, db=db, tracer=Tracer(db), run_id="rehearsal", robot=ctx.robot.clone(),
                       site=ctx.site, tick=lambda dt: None, extras=dict(ctx.extras))
    results: dict[str, Any] = {}
    rows = []
    for s in order:
        tool = registry.get(s.tool)
        assert tool is not None
        if not any(sc.startswith("robot:") for sc in tool.scopes):
            continue
        row: dict[str, Any] = {"id": s.id, "tool": s.tool, "ok": True}
        try:
            raw = _resolve(s, results)
            if tool.tier == "actuate":
                raw["based_on_seq"] = dctx.robot.state().seq
            args = tool.args_model.model_validate(raw)
            dctx.command_id = f"rehearsal:{s.id}"
            if tool.kind == "action":
                content, ok, err = wait_for_action(tool.fn(args, dctx), dctx, 60.0)
                if not ok:
                    raise err or ToolError("failed", "the action did not finish")
            else:
                content = tool.fn(args, dctx)
            results[s.id] = content
        except (KeyError, ValidationError) as exc:
            row.update({"ok": False, "error_kind": "cannot_rehearse", "error": str(exc)[:200]})
        except (ToolError, BackendError) as exc:
            row.update({"ok": False, "error_kind": exc.kind, "error": str(exc)[:200],
                        "violations": (getattr(exc, "detail", None) or {}).get("violations")})
        rows.append(row)
        if not row["ok"]:
            break
    return {"needed": True, "clean": all(r["ok"] for r in rows), "steps": rows}


@dataclass
class PlanRun:
    status: str = "running"            # running | completed | step_failed | state_mismatch
    results: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    stopped_at: str | None = None
    why: str = ""
    approvals: int = 0


def execute_plan(plan: Plan, registry: ToolRegistry, ctx: ToolContext, allowed: set[str], plan_id: str,
                 on_approval: Callable[[AwaitingApproval], bool] | None = None) -> PlanRun:
    run = PlanRun()

    def call(name: str, arguments: dict[str, Any], cid: str, by_code: bool = False) -> Any:
        while True:
            try:
                return execute_tool(registry, ToolCall(id=cid, name=name, arguments=arguments), ctx,
                                    allowed=None if by_code else allowed)
            except AwaitingApproval as wait:
                if on_approval is None:
                    raise
                decide_approval(ctx.db, wait.approval_id, bool(on_approval(wait)), "operator")
                run.approvals += 1

    for level in levels(plan) or []:
        for s in level:
            tool = registry.get(s.tool)
            assert tool is not None
            args = _resolve(s, run.results)
            expected = None
            if tool.tier == "actuate":
                look = call("get_robot_state", {}, f"{plan_id}:{s.id}:look", by_code=True)       # a look by code, never an old one
                if not look.ok:
                    run.status, run.stopped_at, run.why = "step_failed", s.id, f"could not look at the robot: {look.error}"
                    return run
                args["based_on_seq"] = look.content["state_seq"]
                expected = {**observe_facts(ctx), **TOOL_FACTS.get(s.tool, ((), {}))[1]}
            res = call(s.tool, args, f"{plan_id}:{s.id}")
            row = {"id": s.id, "tool": s.tool, "ok": res.ok, "error_kind": res.error_kind, "error": res.error,
                   "arguments": {k: v for k, v in args.items() if k != "based_on_seq"}}
            run.steps.append(row)
            if not res.ok:
                run.status, run.stopped_at, run.why = "step_failed", s.id, f"{s.tool}: {res.error_kind}: {res.error}"
                return run
            run.results[s.id] = res.content
            if expected is not None:
                seen = observe_facts(ctx)
                wrong = {f: seen[f] for f in TOOL_FACTS.get(s.tool, ((), {}))[1] if seen[f] != expected[f]}
                row["facts_after"] = {f: seen[f] for f in TOOL_FACTS.get(s.tool, ((), {}))[1]}
                if wrong:
                    run.status, run.stopped_at = "state_mismatch", s.id
                    run.why = f"after {s.tool} the robot is not in the state the step declared: {dumps(wrong)}"
                    return run
    run.status = "completed"
    return run


# ------------------------------------------------------------------------------ the whole thing

def run_goal(llm: LLMClient, goal: str, registry: ToolRegistry, ctx: ToolContext, tool_names: list[str],
             on_approval: Callable[[AwaitingApproval], bool] | None = None) -> dict[str, Any]:
    """Plan, check, rehearse, execute, re-plan once if needed, and answer from the step results."""
    t0 = time.time()
    allowed = {t.name for t in registry.allowed_for(ctx.principal, tool_names)}
    out: dict[str, Any] = {"goal": goal, "rounds": [], "replans": 0, "tokens": 0, "status": ""}
    history, results = "", {}
    for round_no in (1, 2):
        facts = observe_facts(ctx)
        planned = make_plan(llm, goal, registry, ctx, tool_names, facts, history)
        out["tokens"] += planned.tokens
        rnd: dict[str, Any] = {"facts": facts, "planning_attempts": planned.attempts}
        out["rounds"].append(rnd)
        if not planned.ok:
            out["status"] = "no_valid_plan"
            out["why"] = planned.attempts[-1]["errors"]
            break
        assert planned.plan is not None
        rnd["dry_run"] = rehearse(planned.plan, registry, ctx)
        if not rnd["dry_run"]["clean"]:
            out["status"] = "rejected_in_dry_run"
            out["why"] = [f"{r['id']} {r['tool']}: {r.get('error_kind')}: {r.get('error')}" for r in rnd["dry_run"]["steps"] if not r["ok"]]
            break
        run = execute_plan(planned.plan, registry, ctx, allowed, f"{ctx.run_id}:p{round_no}", on_approval)
        rnd["execution"] = {"status": run.status, "steps": run.steps, "stopped_at": run.stopped_at, "why": run.why,
                            "approvals": run.approvals}
        results = run.results
        if run.status == "completed":
            out["status"] = "completed"
            break
        out["status"] = run.status
        out["why"] = [run.why]
        if round_no == 2:
            break
        out["replans"] = 1
        history = (f"A first plan was started. Steps finished: {', '.join(r['id'] + ' ' + r['tool'] for r in run.steps if r['ok']) or 'none'}. "
                   f"It stopped at {run.stopped_at}: {run.why}. Plan again from the facts observed now.")
    out["answer"], out["answer_source"], tok = synthesise(llm, goal, out, results, ctx)
    out["tokens"] += tok
    out["seconds"] = round(time.time() - t0, 1)
    return out


def synthesise(llm: LLMClient, goal: str, outcome: dict[str, Any], results: dict[str, Any], ctx: ToolContext
               ) -> tuple[str, str, int]:
    """The answer, written from the step results only. A number that is in no result sends the
    text back once; if it comes back the same way the answer is replaced by a plain statement."""
    facts = {"request": goal, "status": outcome["status"], "why": outcome.get("why"), "step_results": results}
    allowed = [float(x) for x in _NUM.findall(dumps(facts))]
    plain = (f"Status: {outcome['status']}." + (f" Reason: {'; '.join(map(str, outcome['why']))}." if outcome.get("why") else "")
             + (f" Step results: {dumps(results)[:600]}" if results else " Nothing was executed."))
    messages = [{"role": "system", "content":
                 "You report the outcome of a service request to the person who asked. Use only the facts given. If "
                 "nothing was executed, say so and say why. Copy numbers exactly. Three sentences at most."},
                {"role": "user", "content": dumps(facts)}]
    tokens = 0
    for attempt in range(2):
        try:
            with ctx.tracer.span(ctx.run_id, "llm", "planner.synthesis") as sp:
                resp = llm.chat(messages, temperature=0.0, max_tokens=250)
                sp.prompt_tokens, sp.completion_tokens = resp.usage.prompt_tokens, resp.usage.completion_tokens
        except LLMError:
            return plain, "plain statement (model unavailable)", tokens
        tokens += resp.usage.total
        foreign = [x for x in _NUM.findall(resp.content) if not any(abs(float(x) - a) <= 0.0501 for a in allowed)]
        if resp.content.strip() and not foreign:
            return resp.content.strip(), "model" if attempt == 0 else "model, after one correction", tokens
        messages += [{"role": "assistant", "content": resp.content},
                     {"role": "user", "content": f"These numbers are not in the facts: {', '.join(foreign)}. Write it again "
                                                 f"using only numbers from the facts."}]
    return plain, "plain statement (the model's text had numbers that are not in the results)", tokens
