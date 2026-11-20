"""The grounding guard: a deterministic comparison of a proposed final answer with what the
tools returned in this run.

  * a number with a unit (mm, degrees, seconds, hertz, mm/s) must be a value a tool returned
    under a key carrying that unit, within the rounding the answer uses; the calculator's
    results are accepted for any unit
  * a statement about the robot's operating state must match the last state a tool reported
  * a bracketed citation must be an id a tool returned
  * an announcement of intent is not an answer

Problems are sent back to the model as a numbered list and it gets a bounded number of repair
turns (AgentConfig.max_answer_repairs).
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Iterator

from ..tools.registry import ToolContext

_NUM = r"(-?\d+(?:[.,]\d+)?)"
_UNITS: list[tuple[str, re.Pattern[str]]] = [
    ("mm_per_s", re.compile(_NUM + r"\s*(?:mm\s*/\s*s|mm per second|millimet(?:er|re)s per second)\b", re.I)),
    ("mm", re.compile(_NUM + r"\s*(?:mm|millimet(?:er|re)s?)\b(?!\s*/\s*s)(?! per second)", re.I)),
    ("deg", re.compile(_NUM + r"\s*(?:°|deg(?:rees?)?\b)", re.I)),
    ("hz", re.compile(_NUM + r"\s*(?:hz|hertz)\b", re.I)),
    ("s", re.compile(_NUM + r"\s*(?:s|sec|secs|seconds?)\b(?!\s*/)", re.I)),
]
_INTENT = re.compile(r"\bI(?:'ll| will)\s+(?:now\s+)?(?:proceed|call|retry|re-?run|use the|check|look|move|wait|request)", re.I)
_ID = r"[A-Za-z][\w.\-]*[#:](?:[\w.\-/#]|\[\])+"            # doc:page#section, schema:file#arms[].name
_CITE = re.compile(r"\[(" + _ID + r")\]")
_QUOTED_ID = re.compile(r'"(' + _ID + r')"')
_SEARCH_TOOLS = ("search_docs", "search_threads", "search_by_log_line")
_NOTHING = re.compile(r"\b(?:could not|couldn't|cannot|can't|unable|not (?:be )?(?:found|reach|search)|nothing|"
                      r"no (?:result|match|information|relevant)|did not find|unavailable|do(?:es)? not (?:cover|contain|answer|address))",
                      re.I)
_STATE = re.compile(r"\b(?:arm|robot|psm1?|it)\s+(?:is|was|remains)\s+(now\s+)?(not\s+)?(enabled|homed)\b", re.I)


def _unit_of(key: str) -> str | None:
    k = key.lower()
    if k.endswith("mm_per_s"):
        return "mm_per_s"
    for u in ("mm", "deg", "hz"):
        if k.endswith("_" + u):
            return u
    if k.endswith("_s"):
        return "s"
    return None


def _walk(obj: Any, key: str = "") -> Iterator[tuple[str, float]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, str(k))
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield key, float(obj)


def collect(state: Any) -> tuple[dict[str, list[float]], list[float], dict[str, Any] | None, set[str]]:
    """Values by unit, calculator results, the last reported robot state, and ids returned."""
    by_unit: dict[str, list[float]] = {u: [] for u, _ in _UNITS}
    calc: list[float] = []
    last_state: dict[str, Any] | None = None
    ids: set[str] = set()
    for m in state.messages:
        if m["role"] != "tool":
            continue
        try:
            body = json.loads(m["content"])
        except json.JSONDecodeError:
            continue
        payload = body.get("result") if body.get("ok") else body.get("detail")
        if payload is None:
            continue
        if m.get("name") == "calculator" and isinstance(payload, dict) and "value" in payload:
            calc.append(float(payload["value"]))
        for key, value in _walk(payload):
            unit = _unit_of(key)
            if unit:
                by_unit[unit].append(value)
        if isinstance(payload, dict) and "enabled" in payload and "homed" in payload:
            last_state = payload
        ids.update(_QUOTED_ID.findall(m["content"]))
    # A value with its unit that the person wrote in their own request may be repeated back to them.
    # Only the request itself counts: later user-role messages are the guard's own corrections.
    task = getattr(state, "task", "") or ""
    for unit, pattern in _UNITS:
        for m in pattern.finditer(task):
            try:
                by_unit[unit].append(float(m.group(1).replace(",", ".")))
            except ValueError:
                pass
    for text in getattr(state, "injected_states", []):       # a state the system showed the model is a look too
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        for key, value in _walk(payload):
            unit = _unit_of(key)
            if unit:
                by_unit[unit].append(value)
        if "enabled" in payload and "homed" in payload:
            last_state = payload
    return by_unit, calc, last_state, ids


def retrieved(state: Any) -> tuple[int, set[str]]:
    """How many passages the search tools returned in this run, and every number written in
    them. A number an answer quotes from a retrieved passage is grounded in that passage."""
    found, numbers = 0, set()
    for m in state.messages:
        if m["role"] != "tool" or m.get("name") not in _SEARCH_TOOLS:
            continue
        try:
            body = json.loads(m["content"])
        except json.JSONDecodeError:
            continue
        for r in (body.get("result") or {}).get("results", []) if body.get("ok") else []:
            found += 1
            numbers.update(n.replace(",", ".") for n in re.findall(r"-?\d+(?:[.,]\d+)?", r.get("text", "")))
    return found, numbers


def _matches(stated: str, values: list[float]) -> bool:
    text = stated.replace(",", ".")
    decimals = len(text.split(".")[1]) if "." in text else 0
    x = float(text)
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(abs(v) - abs(x)) <= tol or abs(v - x) <= tol for v in values)


def grounding_check(state: Any, ctx: ToolContext) -> list[str]:
    answer = state.answer_candidate
    problems: list[str] = []
    by_unit, calc, last_state, ids = collect(state)
    passages, quoted = retrieved(state)

    # 1. Numbers with units.
    seen: set[tuple[str, str]] = set()
    for unit, pattern in _UNITS:
        for m in pattern.finditer(answer):
            stated = m.group(1)
            if (unit, stated) in seen:
                continue
            seen.add((unit, stated))
            if _matches(stated, by_unit[unit] + calc) or stated.replace(",", ".") in quoted:
                continue
            label = unit.replace("_per_", "/")
            problems.append(f"You wrote {stated} {label}, but no tool returned that value in this conversation. "
                            f"State only values a tool returned (with the same unit), or compute it with calculator.")

    # 2. Statements about the robot's operating state.
    for m in _STATE.finditer(answer):
        negated, what = bool(m.group(2)), m.group(3).lower()
        if last_state is None:
            problems.append(f"You said the arm is {'not ' if negated else ''}{what}, but you have not looked at "
                            "the robot in this conversation. Call get_robot_state.")
        elif bool(last_state.get(what)) == negated:
            problems.append(f"You said the arm is {'not ' if negated else ''}{what}, but the last state a tool "
                            f"reported has {what}={str(bool(last_state.get(what))).lower()}.")

    # 3. Citations must be ids that were actually returned.
    for cite in _CITE.findall(answer):
        if cite not in ids:
            problems.append(f"Citation [{cite}] is not an id returned by a tool. Cite only ids you retrieved.")

    # 3b. An answer built on retrieved passages says which ones.
    if passages and not _CITE.search(answer) and not _NOTHING.search(answer):
        problems.append("You searched and received passages, but the answer cites none. Put the id of the passage "
                        "each statement comes from in square brackets, or say that the passages do not answer the "
                        "question.")

    # 4. An announcement is not an answer.
    if _INTENT.search(answer) and len(answer) < 700:
        problems.append("You said what you will do next but did not do it. Call the tool now, or give the final answer.")
    return problems[:5]


def nearly(a: float, b: float, rel: float = 0.01, abs_tol: float = 0.05) -> bool:
    """Used by the verification scripts to score a stated figure against the computed one."""
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)
