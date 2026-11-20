"""The evaluation gate (task 15): does a change to a prompt, a model or a tool description
keep the behaviour the golden traces hold?

    python scripts/gate.py --candidate evals/configs/no_look_rule.json
    python scripts/gate.py --all        the baseline against itself and the three planted regressions
                                        ->  verification/gate_<name>.json and verification/t15_gate.json

For each configuration every case in evals/cases is replayed with the tools answered from the
recording, and the cases marked live are also run on the twin with that configuration. The
decision follows evals/gate.json. The exit code is 0 only if the candidate is accepted.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_bench import log, who                                             # noqa: E402
from knowledge_service import start_knowledge                              # noqa: E402
from record_cases import fresh_node                                        # noqa: E402
from vxagent.config import get_settings                                    # noqa: E402
from vxagent.evals import cases as cs                                      # noqa: E402
from vxagent.evals.assertions import judge                                 # noqa: E402
from vxagent.evals.compare import compare, gate, load_rule, summary        # noqa: E402
from vxagent.evals.replay import Candidate, from_live, replay              # noqa: E402
from vxagent.llm.ollama import get_llm                                     # noqa: E402
from vxagent.node import Node, build_registry                              # noqa: E402
from vxagent.profiles import OPERATOR_COPILOT                              # noqa: E402

OUT = ROOT / "verification"
CONFIGS = ROOT / "evals" / "configs"
PLANTED = ("no_look_rule", "wrong_argument", "small_model")


def run_candidate(cand: Candidate, cases: list[cs.Case], registry: Any, work: Path, knowledge: Any) -> dict[str, Any]:
    s = get_settings()
    llm = get_llm(cand.model or s.model)
    replayed = []
    for case in cases:
        trace = replay(case, cand, OPERATOR_COPILOT, registry, llm)
        replayed.append(judge(case.asserts, trace))
        r = replayed[-1]
        log(f"  replay {case.id}: {'pass' if r['passed'] else ('diverged' if r['diverged'] else 'FAIL')} {r['failed']} {r['calls']}")
    live = []
    for case in [c for c in cases if c.live]:
        node = fresh_node(work, f"{cand.name}_{case.id}", llm, knowledge)
        for name, text in cand.tool_descriptions.items():      # this node's copy of the tool: the candidate's description
            node.registry.get(name).description = text
        cs.prepare(node, case.start)
        run = cs.run_live(node, who("op.lee"), case.task, cand.config(OPERATOR_COPILOT))
        r = judge(case.asserts, from_live(case, cand.name, run))
        r["motions_executed_on_the_twin"] = len(node.db.query("SELECT 1 FROM commands WHERE status='done'"))
        live.append(r)
        log(f"  live   {case.id}: {'pass' if r['passed'] else 'FAIL'} {r['failed']} {r['calls']}")
    return {"candidate": cand.name, "about": cand.about, "model": cand.model or s.model, "replayed": replayed, "live": live}


def decide(name: str, base: dict[str, Any], cand: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    report = compare(base["replayed"], cand["replayed"])
    verdict = gate(report, cand["live"], rule)
    return {"task": f"gate_{name}", "ran": time.strftime("%Y-%m-%d %H:%M:%S"), "candidate": name, "about": cand["about"],
            "model": cand["model"], "baseline": base["candidate"], "cases": len(cand["replayed"]),
            "comparison": report, "live": {"cases": len(cand["live"]), "passed": sum(r["passed"] for r in cand["live"]),
                                            "summary": summary(cand["live"]) if cand["live"] else None},
            "gate": verdict, "replayed": cand["replayed"], "live_rows": cand["live"]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluation gate.")
    ap.add_argument("--baseline", default=str(CONFIGS / "baseline.json"))
    ap.add_argument("--candidate")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if not a.candidate and not a.all:
        ap.error("give --candidate <file> or --all")
    cases = cs.load_cases(ROOT / "evals" / "cases")
    if not cases:
        print("no cases in evals/cases: run scripts/record_cases.py first", file=sys.stderr)
        return 2
    rule = load_rule(ROOT / "evals" / "gate.json")
    s = get_settings()
    work = s.runtime_dir / "verify" / "gate"
    shutil.rmtree(work, ignore_errors=True)
    registry, _, _ = build_registry()
    probe = Node.open("site_a", runtime=work / "probe", llm=None)
    knowledge, server = start_knowledge(probe.site)
    OUT.mkdir(exist_ok=True)
    try:
        log("baseline")
        base = run_candidate(Candidate.load(a.baseline), cases, registry, work, knowledge)
        names = ["baseline", *PLANTED] if a.all else [Path(a.candidate).stem]
        results = {}
        for name in names:
            if name == "baseline":
                log("baseline against itself (a second, independent replay)")
                cand = run_candidate(Candidate.load(a.baseline), cases, registry, work / "again", knowledge)
            else:
                log(name)
                cand = run_candidate(Candidate.load(CONFIGS / f"{name}.json" if a.all else a.candidate), cases, registry, work, knowledge)
            results[name] = decide(name, base, cand, rule)
            (OUT / f"gate_{name}.json").write_text(json.dumps(results[name], indent=1, ensure_ascii=False), encoding="utf-8")
            g = results[name]["gate"]
            log(f"{name}: {'ACCEPTED' if g['accepted'] else 'REFUSED'} {g['reasons']}")
        if a.all:
            recorded = [{"case": c.id, "task": c.task, "live": c.live, "recorded_calls": [x["name"] for x in c.calls],
                         "passed_when_recorded": c.recorded.get("passed_when_recorded"), "failed_when_recorded": c.recorded.get("failed_when_recorded")}
                        for c in cases]
            t15 = {"task": "t15_gate", "ran": time.strftime("%Y-%m-%d %H:%M:%S"), "model": s.model, "rule": rule,
                   "cases": recorded, "cases_recorded_with": cases[0].recorded.get("model"),
                   "baseline_first_replay": summary(base["replayed"]),
                   "by_candidate": {n: {"about": r["about"], "model": r["model"], "comparison": r["comparison"], "live": r["live"],
                                         "accepted": r["gate"]["accepted"], "reasons": r["gate"]["reasons"],
                                         "per_case": [{"case": x["case"], "passed": x["passed"], "diverged": x["diverged"], "failed": x["failed"],
                                                       "safety_failed": x["safety_failed"], "calls": x["calls"]} for x in r["replayed"]],
                                         "live_per_case": [{"case": x["case"], "passed": x["passed"], "failed": x["failed"],
                                                            "safety_failed": x["safety_failed"], "calls": x["calls"],
                                                            "motions_executed_on_the_twin": x["motions_executed_on_the_twin"]} for x in r["live_rows"]]}
                                    for n, r in results.items()}}
            (OUT / "t15_gate.json").write_text(json.dumps(t15, indent=1, ensure_ascii=False), encoding="utf-8")
            log("wrote verification/t15_gate.json")
    finally:
        server.stop()
    last = results[names[-1]]["gate"]["accepted"]
    return 0 if (a.all or last) else 1


if __name__ == "__main__":
    raise SystemExit(main())
