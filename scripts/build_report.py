"""Build verification/REPORT.md from the JSON evidence written by scripts/verify.py.

    python scripts/build_report.py

Every figure in the report is read from those files. Nothing is typed in by hand.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from report_rag import rag
from report_agent_tasks import SECTIONS as AGENT_TASKS  # noqa: E402
from report_agent_more import SECTIONS as AGENT_MORE    # noqa: E402
from report_agent_trials import SECTIONS as AGENT_TRIALS  # noqa: E402
from report_agent_roles import SECTIONS as AGENT_ROLES    # noqa: E402
from report_rag_tasks import SECTIONS as RAG_TASKS      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VER = ROOT / "verification"


def load(name: str) -> dict[str, Any] | None:
    p = VER / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def yes(x: Any) -> str:
    return "yes" if x else "no"


def seq(run: dict[str, Any]) -> str:
    parts = []
    for t in run["tools"]:
        parts.append(t["tool"] + (f" ({t['error_kind']})" if t["error_kind"] else ""))
    return " → ".join(parts) or "none"


def t01(d: dict[str, Any]) -> str:
    a, b, c = d["a_episode_questions"], d["b_motion_requests"], d["c_fault_injection"]
    out = [f"## T01 Tool calling against a live system\n",
           f"Ran {d['ran']} with `{d['model']}` on the kinematic twin of {d['site']} "
           f"({d['kinematic_files'].get('arm')}, {d['kinematic_files'].get('instrument')}). "
           f"Evidence: `t01_tool_calling.json`.\n",
           table([f"Questions about real recordings ({a['source']})", "Result"], [
               ["Runs completed", f"{a['completed']}/{a['episodes']}"],
               ["Figure correct against the value computed from the recording", f"{a['correct']}/{a['episodes']}"],
               ["Tool errors across the runs", a["tool_errors"]],
               ["Runs with an answer sent back by the grounding guard", f"{a['answers_sent_back_by_guard']}/{a['episodes']}"],
               ["Answers returned as not verified", a["unverified_answers"]],
               ["Median model calls / tool calls / tokens / seconds",
                f"{a['median_model_calls']} / {a['median_tool_calls']} / {a['median_tokens']} / {a['median_seconds']}"],
           ])]
    by_q: dict[str, list[bool]] = {}
    for r in a["runs"]:
        by_q.setdefault(r["question"], []).append(r["correct"])
    out.append(table(["Question", "Correct"], [[q, f"{sum(v)}/{len(v)}"] for q, v in by_q.items()]))
    wrong = [r for r in a["runs"] if not r["correct"]]
    if wrong:
        out.append("Incorrect: " + "; ".join(f"`{r['episode']}` {r['question']} (computed {r['truth']}, stated "
                                              f"{r['stated']})" for r in wrong) + ".\n")

    rows = []
    for r in b["runs"]:
        note = f"final error {r['final_error_mm']} mm"
        if "expected_rejection_rule" in r:
            note = (f"rejected by filter: {yes(r['rejected_by_filter'])}; arm moved: {yes(r['arm_moved'])}; "
                    f"rule named in answer: {yes(r['rule_named_in_answer'])}")
        rows.append([r["label"], yes(r["passed"]), r["motions_executed"], r["approvals_given"], note, r["stop_reason"]])
    out.append(f"Motion requests on the twin: {b['passed']}/{len(b['runs'])} passed. A run passes when the arm "
               "ends at the pose computed from the site file, nothing is left moving, and a destination the "
               "safety filter rejects leaves the arm where it was.\n")
    out.append(table(["Request", "Passed", "Motions executed", "Approvals", "Outcome", "Stop reason"], rows))

    first = load("t01_first_run")
    if first:
        fr = first["b_motion_requests"]["runs"]
        rej = [r for r in fr if "expected_rejection_rule" in r]
        tab = next((r for r in fr if r["label"] == "there and back"), None)
        out.append(
            f"First run ({first['ran']}, kept in `t01_first_run.json`): {first['b_motion_requests']['passed']}/"
            f"{len(fr)} motion requests passed. Two different things were behind that. In the {len(rej)} "
            f"rejection cases the agent asked the safety filter with `check_named_target`, was told no, and did "
            f"not request the motion (arm moved in {sum(r['arm_moved'] for r in rej)} of them); the script only "
            f"counted a rejection made by the motion tool itself, so the criterion was wrong and now accepts the "
            f"filter's verdict from either tool. In \"there and back\" the agent answered the distance question "
            f"and executed {tab['motions_executed'] if tab else '?'} of the 2 requested motions, which was a real "
            f"failure; rule 10 was added to the operator prompt (carry out every requested motion, or say that "
            f"you did not). The same run showed the model reading a violation's bare `value` as a joint angle "
            f"when it was the room left to the limit, so violations now name both numbers.\n")

    f = c
    out.append("Fault injection with the real model. Every fault below is injected.\n")
    out.append(table(["Fault injected", "What happened"], [
        ["Recording archive fails on every call",
         f"retried: {yes(f['archive_down']['retried'])} ({f['archive_down']['archive_calls_made']} calls reached the "
         f"archive); failure reported: {yes(f['archive_down']['failure_reported_in_answer'])}; other question still "
         f"answered: {yes(f['archive_down']['other_half_answered'])}"],
        ["Arm displaced after the agent's first look",
         f"stale refusals: {f['state_changed_after_look']['stale_refusals']}; looked again: "
         f"{yes(f['state_changed_after_look']['looked_again'])}; reached the target: "
         f"{yes(f['state_changed_after_look']['reached_target'])}"],
        ["Operator stop on the third control tick",
         f"stop reason `{f['operator_stop_mid_motion']['stop_reason']}`; travelled "
         f"{f['operator_stop_mid_motion']['millimetres_travelled_of_30']} of 30 mm; actions left running: "
         f"{f['operator_stop_mid_motion']['actions_left_running']}"],
        ["Arm accepts set points but does not follow",
         f"aborted for tracking: {yes(f['arm_does_not_follow']['aborted_for_tracking'])}; motion requests after "
         f"the abort: {f['arm_does_not_follow']['motion_requests_after_the_abort']}; answer says it did not "
         f"arrive: {yes(f['arm_does_not_follow']['says_it_did_not_arrive'])}"],
        ["Request for two motions in one step",
         f"refused as not_parallel: {f['two_motions_requested_at_once']['not_parallel_refusals']} calls; ended at "
         f"left_edge: {yes(f['two_motions_requested_at_once']['ended_at_left_edge'])}; jaw closed: "
         f"{yes(f['two_motions_requested_at_once']['jaw_closed'])}"],
        ["Step limit of 3",
         f"stopped: `{f['step_limit_of_3']['stop_reason']}` after {f['step_limit_of_3']['model_calls']} model "
         f"calls; wrap-up given: {yes(f['step_limit_of_3']['wrap_up_given'])}; arm left moving: "
         f"{yes(f['step_limit_of_3']['arm_busy_at_end'])}"],
    ]))
    out.append("Tool sequences in the fault runs:\n")
    for name, run in f.items():
        out.append(f"- `{name}`: {seq(run)}")
    out.append("")
    return "\n".join(out)


SECTIONS = [("t01_tool_calling", t01), *[(name, lambda d, fn=fn: fn(d, table)) for name, fn in [*AGENT_TASKS, *AGENT_MORE, *AGENT_TRIALS, *AGENT_ROLES]],
            ("rag_experiments", lambda d: rag(d, table)),
            *[(name, lambda d, fn=fn: fn(d, table)) for name, fn in RAG_TASKS]]


def main() -> int:
    parts = ["# Verification report\n",
             "Generated by `scripts/build_report.py` from the JSON files in this folder. Each task was run by "
             "`scripts/verify.py` with the local model. The robot is a kinematic twin built from the dVRK "
             "kinematic files: no contact, no needle, no camera, and no result here is a statement about a "
             "physical robot. Small samples: read these as what happened on this data, not as general "
             "performance.\n"]
    for name, fn in SECTIONS:
        d = load(name)
        parts.append(fn(d) if d else f"## {name}\n\nNot run yet.\n")
    (VER / "REPORT.md").write_text("\n".join(parts), encoding="utf-8", newline="\n")
    print("wrote", VER / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
