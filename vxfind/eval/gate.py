"""The release gate: a change to search ships only if it does not make the measured things worse.

Rules live in evals/vxfind/gate.json and are of two kinds:

  no_loss   on a query set's headline metric the candidate must not be worse than the serving
            configuration by more than chance explains: it fails when the mean difference is
            negative AND the paired randomization test puts it outside noise (p < alpha), or
            when the drop is larger than `max_drop` whatever the test says.
  floor     a measured property must hold outright (for example: no address the live check calls
            dead is shown; no table of a restricted owner is shown to someone outside the group).

The gate reads the locked test queries. It answers "may this ship", not "is this better": a
change that is not significantly better passes, and the comparison table says by how much it
moved.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .harness import compare

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "evals" / "vxfind" / "gate.json"
DEFAULT_RULES = {
    "alpha": 0.05,
    "no_loss": {"A": {"metric": "fit@10", "max_drop": 0.05}, "B": {"metric": "mrr@10", "max_drop": 0.05},
                "C": {"metric": "ndcg@10", "max_drop": 0.05}},
    "floors": {"dead_addresses_shown_after_live_check": 0, "restricted_tables_shown_to_outsiders": 0},
}


def rules() -> dict[str, Any]:
    if not RULES.exists():
        RULES.parent.mkdir(parents=True, exist_ok=True)
        RULES.write_text(json.dumps(DEFAULT_RULES, indent=1), encoding="utf-8")
    return json.loads(RULES.read_text(encoding="utf-8"))


def decide(serving: dict[str, dict[str, dict[str, float]]], candidate: dict[str, dict[str, dict[str, float]]],
           floors: dict[str, Any] | None = None, r: dict[str, Any] | None = None) -> dict[str, Any]:
    r = r or rules()
    cmp = compare(serving, candidate)
    checks = []
    for kind, rule in r["no_loss"].items():
        c = cmp.get(kind)
        if c is None:
            checks.append({"check": f"{kind} {rule['metric']}", "passed": False, "why": "no queries of this set were scored"})
            continue
        significant = c["mean_difference"] < 0 and c["p_randomization"] < r["alpha"]
        large = c["mean_difference"] < -rule["max_drop"]
        checks.append({"check": f"{kind} {rule['metric']}", "passed": not (significant or large), "difference": c["mean_difference"],
                       "ci95": c["ci95"], "p": c["p_randomization"], "queries": c["n"],
                       "why": ("a loss beyond chance" if significant else "") + (" and " if significant and large else "")
                              + (f"a drop larger than {rule['max_drop']}" if large else "")})
    for name, limit in r["floors"].items():
        v = (floors or {}).get(name)
        checks.append({"check": name, "passed": v is not None and v <= limit, "value": v, "limit": limit,
                       "why": "" if v is not None and v <= limit else ("not measured" if v is None else "over the limit")})
    return {"passed": all(c["passed"] for c in checks), "checks": checks}
