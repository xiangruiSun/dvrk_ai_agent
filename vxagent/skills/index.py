"""The catalogue an agent can search, and the three skill tools: find_tools, load_runbook and
run_runbook. Also the two atomic tools that procedures about the needle need.

With the whole catalogue in the prompt the schemas are large and the choice gets harder.
find_tools returns the few entries that match a request, and only those are offered on the
next turn. A missed search is then a new way to fail, which the verification counts.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..tools.registry import Tool, ToolContext, ToolError, ToolRegistry
from . import runbook as rbk

ROBOT_READ = frozenset({"robot:read"})
_WORD = re.compile(r"[a-z][a-z0-9]+")
_STOP = {"the", "and", "for", "with", "that", "this", "its", "are", "not", "from", "then", "arm", "tool", "does"}


def _tokens(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower().replace("_", " ")):
        if w in _STOP:
            continue
        out.append(w[:-1] if len(w) > 4 and w.endswith("s") else w)
    return out


class Catalogue:
    """Tools and runbooks as searchable entries: BM25 over name, description and when-to-use."""

    def __init__(self, registry: ToolRegistry, library: rbk.Library | None, tool_names: list[str]):
        self.entries: list[dict[str, Any]] = []
        for n in tool_names:
            t = registry.get(n)
            if t is not None:
                self.entries.append({"name": t.name, "kind": "tool", "text": f"{t.name} {t.name} {t.description}",
                                     "summary": t.description[:140]})
        for rb in (library.books.values() if library else []):
            self.entries.append({"name": rb.name, "kind": "runbook", "summary": rb.summary,
                                 "text": f"{rb.name} {rb.name} {rb.summary} {rb.when_to_use}"})
        self._docs = [_tokens(e["text"]) for e in self.entries]
        self._avg = sum(len(d) for d in self._docs) / max(1, len(self._docs))
        df: dict[str, int] = {}
        for d in self._docs:
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        n = len(self._docs)
        self._idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        q = _tokens(query)
        scored = []
        for e, d in zip(self.entries, self._docs):
            s = 0.0
            for w in q:
                f = d.count(w)
                if f:
                    s += self._idf.get(w, 0.0) * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(d) / self._avg))
            if s > 0:
                scored.append((s, e))
        scored.sort(key=lambda x: -x[0])
        return [{"name": e["name"], "kind": e["kind"], "summary": e["summary"], "score": round(s, 2)} for s, e in scored[:k]]


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(Args):
    pass


class FindArgs(Args):
    query: str = Field(min_length=2, max_length=200, description="What you need to do, in a few words.")


class NameArgs(Args):
    name: str = Field(description="Name of a runbook, exactly as the index or find_tools gives it.")


class RunArgs(Args):
    name: str = Field(description="Name of a runbook, exactly as the index or find_tools gives it.")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Parameters of the runbook; {} if it has none.")


class HoldArgs(Args):
    seconds: float = Field(gt=0, le=5, description="How long to keep still, in seconds (at most 5).")


def observe_needle(a: NoArgs, ctx: ToolContext) -> dict[str, Any]:
    needle = ctx.extras.get("needle")
    if needle is None or ctx.robot is None:
        raise ToolError("failed", "no needle observation is available in this session")
    return {**needle.observe(ctx), "state_seq": ctx.robot.state().seq}


def hold(a: HoldArgs, ctx: ToolContext) -> dict[str, Any]:
    ctx.tick(a.seconds)
    needle = ctx.extras.get("needle")
    if needle is not None:
        needle.held_still(a.seconds)
    return {"seconds": a.seconds, "state_seq": ctx.robot.state().seq if ctx.robot is not None else None}


def register_bench(reg: ToolRegistry) -> ToolRegistry:
    reg.register(Tool("observe_needle", "Observe the needle: whether it is visible, whether it is held in the jaw, and "
                      "its height over the pad in mm.", NoArgs, observe_needle, scopes=ROBOT_READ, timeout_s=5))
    reg.register(Tool("hold", "Keep the arm still for a number of seconds, then return.", HoldArgs, hold,
                      scopes=ROBOT_READ, timeout_s=10, max_retries=0))
    return reg


def register_skills(reg: ToolRegistry, library: rbk.Library, catalogue_tools: list[str]) -> Catalogue:
    """Adds find_tools, load_runbook and run_runbook. Returns the searchable catalogue."""
    cat = Catalogue(reg, library, catalogue_tools)

    def find_tools(a: FindArgs, ctx: ToolContext) -> dict[str, Any]:
        hits = cat.search(a.query)
        offered = [h["name"] for h in hits if h["kind"] == "tool"]
        if any(h["kind"] == "runbook" for h in hits):
            offered += ["load_runbook", "run_runbook"]
        ctx.extras["offered_tools"] = offered
        ctx.extras.setdefault("searches", []).append({"query": a.query, "returned": [h["name"] for h in hits]})
        return {"matches": [{k: h[k] for k in ("name", "kind", "summary")} for h in hits],
                "note": "these tools can be called on your next turn; a runbook is run with run_runbook"}

    def load_runbook(a: NameArgs, ctx: ToolContext) -> dict[str, Any]:
        rb = library.get(a.name)
        ctx.extras.setdefault("runbooks_loaded", []).append(rb.name)
        return {"name": rb.name, "version": rb.version, "content_hash": rb.content_hash, "text": rb.text}

    def run_runbook(a: RunArgs, ctx: ToolContext) -> dict[str, Any]:
        call_id = f"rb:{ctx.run_id}:{a.name}:{len(ctx.extras.get('runbook_calls', []))}"
        pending = ctx.extras.get("runbook_pending")
        if pending and pending[0] == a.name:
            call_id = pending[1]                       # the same run, resumed after the operator's decision
        else:
            ctx.extras.setdefault("runbook_calls", []).append(call_id)
        ctx.extras["runbook_pending"] = (a.name, call_id)
        out = rbk.run(library, a.name, a.parameters, reg, ctx, call_id)
        ctx.extras["runbook_pending"] = None
        ctx.extras.setdefault("runbook_runs", []).append(out)
        return out

    reg.register(Tool("find_tools", "Search the catalogue of tools and runbooks for what you need to do. The matches "
                      "become available on your next turn.", FindArgs, find_tools, timeout_s=5))
    reg.register(Tool("load_runbook", "Read the full text of one runbook: its steps, checks and parameters.", NameArgs,
                      load_runbook, cacheable=True, timeout_s=5))
    reg.register(Tool("run_runbook", "Run a runbook. Its steps and checks are executed by the system in order; motions "
                      "inside it still need the operator's approval. Returns every step with the result of its checks.",
                      RunArgs, run_runbook, timeout_s=180, max_retries=0))
    return cat


def default_library(registry: ToolRegistry | None = None, strict: bool = False) -> rbk.Library:
    return rbk.Library(Path(__file__).resolve().parents[2] / "runbooks", registry, strict=strict)
