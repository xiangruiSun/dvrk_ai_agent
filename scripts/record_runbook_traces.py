"""Record the reference outcome of every runbook case on the made-up test arm.

    python scripts/record_runbook_traces.py

Run this only after a change to a runbook has been reviewed: the test suite refuses a runbook
whose outcomes differ from the recorded ones.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from runbook_bench import CASES, TRACES, outcome, run_case            # noqa: E402


def main() -> int:
    TRACES.mkdir(exist_ok=True)
    for name in CASES:
        result, _, _ = run_case(name)
        body = {"case": name, "runbook": result["runbook"], "version": result["version"],
                "content_hash": result["content_hash"], "outcome": outcome(result)}
        (TRACES / f"{name}.json").write_text(json.dumps(body, indent=1), encoding="utf-8", newline="\n")
        print(name, result["status"], [s["id"] for s in result["steps"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
