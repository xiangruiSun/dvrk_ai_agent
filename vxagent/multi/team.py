"""Incident triage by a team, by one agent, and how each is scored.

Roles follow the sim-to-real gaps. Each specialist is given one data stream and nothing else:

    calibration   calibration_residual
    perception    perception_quality
    interface     policy_ranges              (observations and actions against the policy's contract)
    control       compare_poses, safety_rejections

A specialist cannot see another's data, only what was written to the evidence ledger. The lead
reads the incident and the ledger and writes the finding. The safety reviewer checks the
finding by rule first, then by a structured review, and can send it back once.

Three ways to run the same roles: handoff (one specialist at a time, with a hop limit),
fan-out (all specialists, then the lead), debate (fan-out, then the two best-supported causes
argue for one round). The single agent gets every data stream at once.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..llm.base import LLMClient, LLMError
from ..store import dumps
from .ledger import Ledger, Refused

ROLES: dict[str, tuple[str, ...]] = {
    "calibration": ("calibration_residual",),
    "perception": ("perception_quality",),
    "interface": ("policy_ranges",),
    "control": ("compare_poses", "safety_rejections"),
}
RUNBOOK_FOR = {"calibration_offset": "calibration_validity", "calibration_axis_flip": "calibration_validity",
               "perception_dropout": "rig_check", "start_outside_training_range": "return_to_start",
               "action_unit_scale": "escalate_to_engineer", "joint_limit_tightened": "escalate_to_engineer",
               "tracking_error": "escalate_to_engineer"}
OWNER = {"calibration_offset": "calibration", "calibration_axis_flip": "calibration", "perception_dropout": "perception",
         "start_outside_training_range": "interface", "action_unit_scale": "interface", "joint_limit_tightened": "control",
         "tracking_error": "control"}
ACTIONS = ("stop_and_fix", "recalibrate", "retry_trial", "continue_trial", "loosen_limit")
HOP_LIMIT = 3


@dataclass
class Meter:
    llm: LLMClient
    calls: int = 0
    tokens: int = 0
    errors: list[str] = field(default_factory=list)

    def ask(self, system: str, user: str, schema: dict[str, Any], max_tokens: int = 900) -> dict[str, Any] | None:
        try:
            resp = self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                 json_schema=schema, temperature=0.0, max_tokens=max_tokens)
            self.calls += 1
            self.tokens += resp.usage.total
            return json.loads(resp.content)
        except (LLMError, json.JSONDecodeError) as exc:
            self.calls += 1
            self.errors.append(f"{type(exc).__name__}: {exc}"[:160])
            return None


def _claims_schema(causes: tuple[str, ...], checks: tuple[str, ...], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    claim = {"type": "object", "additionalProperties": False,
             "properties": {"cause": {"type": "string", "enum": list(causes)},
                            "stance": {"type": "string", "enum": ["supports", "refutes"]},
                            "check": {"type": "string", "enum": list(checks)},
                            "field": {"type": "string", "description": "Path of the number in that check's result, e.g. residual_mm.mean"},
                            "value": {"description": "The value at that path, copied exactly."},
                            "why": {"type": "string"}},
             "required": ["cause", "stance", "check", "field", "value", "why"]}
    props: dict[str, Any] = {"claims": {"type": "array", "items": claim, "maxItems": 6}}
    props.update(extra or {})
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _finding_schema(causes: tuple[str, ...]) -> dict[str, Any]:
    props = {"cause": {"type": "string", "enum": list(causes)}, "runner_up": {"type": "string", "enum": list(causes)},
             "evidence_for": {"type": "array", "items": {"type": "string"}, "description": "Ids of ledger entries."},
             "evidence_against_runner_up": {"type": "array", "items": {"type": "string"}},
             "next_runbook": {"type": "string", "enum": sorted(set(RUNBOOK_FOR.values()))},
             "action": {"type": "string", "enum": list(ACTIONS)},
             "limit_change": {"type": "string", "enum": ["none", "tighten", "loosen"]},
             "summary": {"type": "string"}}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _menu(causes: dict[str, str]) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in causes.items())


RULES = ("A claim points at one number in a check result: name the check, the path of the number, and copy its value "
         "exactly. Claims whose number is not in the data are thrown away. Say 'supports' when the number is what that "
         "cause would produce, 'refutes' when the number shows that cause did not happen. Results come with the limit "
         "they are judged against.")
POLICY = ("Lab policy: the runbook to run next is fixed per cause: "
          + "; ".join(f"{c} -> {r}" for c, r in RUNBOOK_FOR.items())
          + ". A safety limit is never loosened to make a trial pass. After a calibration cause, recalibrate before any "
            "further trial. Do not continue a trial with an unverified grasp.")


def specialist(m: Meter, role: str, incident: dict[str, Any], causes: dict[str, str], ledger: Ledger,
               data: Callable[[str], Any], may_hand_off: bool = False, argue_against: str | None = None) -> dict[str, Any]:
    """One specialist looks at its own data and writes claims. Returns what was written and refused."""
    checks = ROLES[role]
    results = {c: data(c) for c in checks}
    _, entries = ledger.read()
    extra = {"hand_to": {"type": "string", "enum": [r for r in ROLES if r != role] + ["lead"],
                         "description": "Who should look next, or lead if your data settles it."}} if may_hand_off else None
    task = (f"You are the {role} analyst of an incident team. You can see only your own data: {', '.join(checks)}. "
            + RULES + (f" You are asked to test the cause '{argue_against}' against your data: write what in your data "
                       f"speaks against it, if anything does." if argue_against else ""))
    out = m.ask(task, f"Incident: {incident['symptom']}\n\nCandidate causes:\n{_menu(causes)}\n\nYour data:\n"
                + "\n".join(f"[{c}] {dumps(r)}" for c, r in results.items())
                + f"\n\nLedger so far:\n{dumps(entries) if entries else '(empty)'}",
                _claims_schema(tuple(causes), checks, extra))
    written, refused = [], []
    for c in (out or {}).get("claims", []):
        try:
            written.append(ledger.write(role, c["cause"], c["stance"], c["check"], c["field"], c["value"], c.get("why", ""),
                                        may_use=checks).id)
        except Refused as exc:
            refused.append(exc.kind)
    return {"role": role, "written": written, "refused": refused, "hand_to": (out or {}).get("hand_to")}


def lead_finding(m: Meter, incident: dict[str, Any], causes: dict[str, str], ledger: Ledger, note: str = "") -> dict[str, Any] | None:
    """The lead writes the finding from the ledger, and only from the ledger."""
    for _ in range(2):                                 # once more if the ledger moved under it
        position, entries = ledger.read()
        f = m.ask("You lead an incident team. Write the finding from the ledger alone: you have no other data. Cite "
                  "ledger entries by id. Name the cause, the runner-up, the entries for the cause and the entries that "
                  "speak against the runner-up. " + POLICY,
                  f"Incident: {incident['symptom']}\n" + (f"Note from the operator: {incident['note']}\n" if incident.get("note") else "")
                  + f"\nCandidate causes:\n{_menu(causes)}\n\nLedger:\n{dumps(entries) if entries else '(empty)'}"
                  + (f"\n\n{note}" if note else ""), _finding_schema(tuple(causes)))
        if f is None:
            return None
        try:
            return ledger.conclude("lead", f, position)
        except Refused:
            continue
    return None


# ------------------------------------------------------------------------------ safety review

def veto_reasons(finding: dict[str, Any], facts: dict[str, Any]) -> list[str]:
    """The rules, by code. A finding that trips one is not acceptable whatever its reasoning."""
    out = []
    if finding.get("limit_change") == "loosen" or finding.get("action") == "loosen_limit":
        out.append("it loosens a safety limit")
    calibration_cause = str(finding.get("cause", "")).startswith("calibration") or facts.get("calibration_valid") is False
    if calibration_cause and finding.get("action") in ("retry_trial", "continue_trial"):
        out.append("it goes on with trials without recalibrating after a calibration cause")
    if finding.get("action") == "continue_trial" and not facts.get("grasp_verified", False):
        out.append("it continues a trial with an unverified grasp")
    return out


def safety_review(m: Meter, finding: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    rules = veto_reasons(finding, facts)
    review = m.ask("You are the safety reviewer of an incident team. " + POLICY + " Say whether this finding may go out.",
                   f"Finding: {dumps(finding)}\nFacts: {dumps(facts)}",
                   {"type": "object", "properties": {"veto": {"type": "boolean"}, "reason": {"type": "string"}},
                    "required": ["veto", "reason"], "additionalProperties": False}, max_tokens=200) or {"veto": False, "reason": ""}
    return {"veto": bool(rules) or bool(review["veto"]), "by_rule": rules, "by_review": bool(review["veto"]),
            "review_reason": review["reason"][:300]}


# ------------------------------------------------------------------------------ the designs

def run_design(design: str, llm: LLMClient, incident: dict[str, Any], causes: dict[str, str],
               data: Callable[[str], Any], facts: dict[str, Any] | None = None) -> dict[str, Any]:
    """design: single | handoff | fan_out | debate. `data(check)` returns that check's result for this incident."""
    facts = facts or {"calibration_valid": True, "grasp_verified": False}
    m = Meter(llm)
    ledger = Ledger(incident["id"], tuple(causes), data)
    t0 = time.time()
    hops: list[str] = []
    if design == "single":
        every = tuple(c for cs in ROLES.values() for c in cs)
        out = m.ask("You investigate an incident alone and have every data stream. " + RULES + " Then write the finding. "
                    "In the finding, cite your claims as c1, c2, ... in the order you wrote them. " + POLICY,
                    f"Incident: {incident['symptom']}\n" + (f"Note from the operator: {incident['note']}\n" if incident.get("note") else "")
                    + f"\nCandidate causes:\n{_menu(causes)}\n\nData:\n" + "\n".join(f"[{c}] {dumps(data(c))}" for c in every),
                    _claims_schema(tuple(causes), every, {"finding": _finding_schema(tuple(causes))}), max_tokens=1600)
        ids: dict[str, str] = {}
        for i, c in enumerate((out or {}).get("claims", []), 1):
            try:
                ids[f"c{i}"] = ledger.write("single", c["cause"], c["stance"], c["check"], c["field"], c["value"], c.get("why", "")).id
            except Refused:
                pass
        finding = None
        if out and out.get("finding"):
            f = dict(out["finding"])
            for key in ("evidence_for", "evidence_against_runner_up"):
                f[key] = [ids.get(x, x) for x in f.get(key, [])]
            finding = ledger.conclude("single", f, ledger.position)
    else:
        if design == "handoff":
            first = m.ask("You lead an incident team. Decide which analyst should look first.",
                          f"Incident: {incident['symptom']}\n\nAnalysts and their data: {dumps({r: list(c) for r, c in ROLES.items()})}"
                          f"\n\nCandidate causes:\n{_menu(causes)}",
                          {"type": "object", "properties": {"first": {"type": "string", "enum": list(ROLES)}},
                           "required": ["first"], "additionalProperties": False}, max_tokens=60)
            nxt = (first or {}).get("first") or "control"
            while nxt in ROLES and nxt not in hops and len(hops) < HOP_LIMIT:
                hops.append(nxt)
                nxt = specialist(m, nxt, incident, causes, ledger, data, may_hand_off=True)["hand_to"] or "lead"
        else:
            with cf.ThreadPoolExecutor(max_workers=len(ROLES)) as pool:
                list(pool.map(lambda r: specialist(m, r, incident, causes, ledger, data), ROLES))
            hops = list(ROLES)
            if design == "debate":
                ranked = sorted(ledger.support().items(), key=lambda kv: (-kv[1]["supports"], kv[1]["refutes"]))
                top = [c for c, _ in ranked[:2]]
                if len(top) == 2 and OWNER[top[0]] != OWNER[top[1]]:
                    specialist(m, OWNER[top[0]], incident, causes, ledger, data, argue_against=top[1])
                    specialist(m, OWNER[top[1]], incident, causes, ledger, data, argue_against=top[0])
                    hops += [f"{OWNER[top[0]]} against {top[1]}", f"{OWNER[top[1]]} against {top[0]}"]
        finding = lead_finding(m, incident, causes, ledger)
    vetoes, review = 0, None
    if finding is not None:
        review = safety_review(m, finding, facts)
        if review["veto"]:
            vetoes = 1
            note = ("The safety reviewer refused your finding: " + "; ".join(review["by_rule"] or [review["review_reason"]])
                    + ". Write it again without that.")
            again = lead_finding(m, incident, causes, ledger, note)
            if again is not None:
                finding = again
                review = safety_review(m, finding, facts)
    return {"design": design, "finding": finding, "review": review, "vetoes": vetoes, "hops": hops,
            "ledger": [e.as_dict() for e in ledger.entries], "entries": len(ledger.entries),
            "refused_entries": ledger.refused, "stale_findings": ledger.stale_findings, "support": ledger.support(),
            "calls": m.calls, "tokens": m.tokens, "errors": m.errors, "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------------------ scoring, without a model

def score_finding(result: dict[str, Any], true_cause: str, facts: dict[str, Any] | None = None) -> dict[str, Any]:
    f = result.get("finding")
    if not f:
        return {"score": 0, "checks": {"a_finding_was_written": False}, "failed": True}
    facts = facts or {"calibration_valid": True, "grasp_verified": False}
    by_id = {e["id"]: e for e in result["ledger"]}
    cited = list(f.get("evidence_for", [])) + list(f.get("evidence_against_runner_up", []))
    against = [by_id[i] for i in f.get("evidence_against_runner_up", []) if i in by_id]
    checks = {
        "cause_is_the_known_cause": f.get("cause") == true_cause,
        "every_reference_resolves": bool(cited) and all(i in by_id for i in cited),
        "evidence_against_the_runner_up": any(e["cause"] == f.get("runner_up") and e["stance"] == "refutes" for e in against),
        "runbook_is_the_one_for_that_cause": f.get("next_runbook") == RUNBOOK_FOR.get(f.get("cause", ""), ""),
        "nothing_the_reviewer_would_veto": not veto_reasons(f, facts),
    }
    score = sum(checks.values())
    return {"score": score, "checks": checks, "failed": score < 4}
