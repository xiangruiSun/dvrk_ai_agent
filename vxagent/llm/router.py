"""Which model does which step. The route is fixed in code, per step, never chosen at run time.

Two local models share one GPU. The small one is fast and is given the steps that are a choice
among a few named options; the large one is given everything that reasons or writes. Whether
the small model is good enough for a step is measured (scripts/verify_t14.py) before that step
is routed to it. The measured accuracy and latency of both models on these steps are in
verification/REPORT.md, section T14; a step the small model fails there must be moved to "large".
"""
from __future__ import annotations

import json
import time
from typing import Any

from .base import LLMClient, LLMError

ROUTE = {
    # classification: a choice among named options
    # A step goes to the small model only if, on the measurement, it was at least as accurate there
    # as the large one. The counts are in verification/REPORT.md, section T14 (model routing).
    "route_request": "large",        # whose desk a request belongs to: the small model was less accurate
    "label_ticket": "large",         # the class of a problem report: the small model was less accurate
    "choose_runbook": "small",       # one runbook from the index: as accurate as the large model
    # reasoning and writing
    "copilot": "large", "diagnose": "large", "plan": "large", "write": "large", "structured_patch": "large",
}
ROLES = {"operator": "running a trial, moving the arm, the state of the robot or of a run",
         "service": "a fault, a calibration, a configuration value, a repair, an error message",
         "safety": "approving or reviewing a change, a limit, an overridden verdict, access for support",
         "support": "several sites, fleet health, versions across sites, costs"}


class Router:
    def __init__(self, large: LLMClient, small: LLMClient | None = None):
        self.large, self.small = large, small or large

    def for_step(self, step: str) -> LLMClient:
        if step not in ROUTE:
            raise KeyError(f"no route for step {step!r}; add it to ROUTE with a measurement")
        return self.small if ROUTE[step] == "small" else self.large


def choose(llm: LLMClient, instruction: str, options: dict[str, str], text: str) -> dict[str, Any]:
    """One structured call: pick one key of `options`. Returns the choice with its cost; the
    choice is None if the model could not be reached or answered outside the options."""
    schema = {"type": "object", "properties": {"choice": {"type": "string", "enum": list(options)}}, "required": ["choice"],
              "additionalProperties": False}
    menu = "\n".join(f"- {k}: {v}" for k, v in options.items())
    t0 = time.perf_counter()
    try:
        resp = llm.chat([{"role": "system", "content": f"{instruction} Answer with JSON: {{\"choice\": one of the names}}."},
                         {"role": "user", "content": f"Options:\n{menu}\n\nText:\n{text[:1500]}"}],
                        json_schema=schema, temperature=0.0, max_tokens=40)
        pick = json.loads(resp.content).get("choice")
        tokens = resp.usage.total
    except (LLMError, ValueError, AttributeError) as exc:
        return {"choice": None, "error": type(exc).__name__, "ms": round((time.perf_counter() - t0) * 1000, 1), "tokens": 0}
    return {"choice": pick if pick in options else None, "ms": round((time.perf_counter() - t0) * 1000, 1), "tokens": tokens}


def route_request(llm: LLMClient, text: str) -> dict[str, Any]:
    return choose(llm, "Decide whose desk this request belongs to at a surgical-robot research site.", ROLES, text)


def label_ticket(llm: LLMClient, symptom: str, classes: dict[str, str]) -> dict[str, Any]:
    return choose(llm, "Name the class of this problem report.", classes, symptom)


def choose_runbook(llm: LLMClient, request: str, index: list[dict[str, str]]) -> dict[str, Any]:
    return choose(llm, "Pick the one procedure that fits the request.", {r["name"]: r["when_to_use"] for r in index}, request)
