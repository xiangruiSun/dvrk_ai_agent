"""One process of the trial runtime.

    python -m vxagent.pipelines.trial_worker --db FILE --plant FILE --action boot
    python -m vxagent.pipelines.trial_worker --db FILE --plant FILE --action start  --run ID --sheet FILE --operator ID
    python -m vxagent.pipelines.trial_worker --db FILE --plant FILE --action resume --run ID [--operator ID]

Every process starts the same way: it applies the restart rule (runs that were running when the
last process ended become needs_reverification) and looks at the arm, whose state lives in the
plant file and not in this process. It then does the one thing it was asked and prints one line
of JSON.

For verification a process can be made to die: --crash NODE:K ends it with os._exit(137) on the
K-th control tick inside NODE (K = 0: as soon as the node has started). No clean-up code runs,
which is what a kill is from the process's own point of view.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from ..llm.ollama import get_llm
from ..profiles import principal
from ..robot.site import load_site
from ..robot.twin import KinematicTwin
from ..runtime.graph import Limits, NotAllowed, Runner, apply_restart_rule
from ..store import Database
from ..structured.schemas import RunSheet
from . import trial


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--plant", required=True)
    ap.add_argument("--action", required=True, choices=("boot", "start", "resume"))
    ap.add_argument("--run")
    ap.add_argument("--sheet")
    ap.add_argument("--operator")
    ap.add_argument("--held", choices=("yes", "no"))
    ap.add_argument("--crash")
    ap.add_argument("--controller-restarted", action="store_true")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--site", default="site_a")
    a = ap.parse_args(argv)

    site = load_site(a.site)
    db = Database(a.db)
    restarted = apply_restart_rule(db)
    twin = KinematicTwin.restore(site.chain, site.home_q, a.plant, jaw=site.jaw_angles.get("open", 0.0),
                                 controller_restarted=a.controller_restarted)
    seq_at_boot = twin.state().seq
    out: dict[str, Any] = {"action": a.action, "pid": os.getpid(), "marked_needs_reverification": restarted,
                           "robot_seq_at_boot": seq_at_boot}

    crash_node, crash_tick = (a.crash.split(":") + ["0"])[:2] if a.crash else (None, "0")
    ticks = {"n": 0}
    stage: dict[str, Any] = {"now": None}

    def die() -> None:
        sys.stdout.flush()
        os._exit(137)

    def tick(dt: float) -> None:
        if crash_node and stage["now"] == crash_node and int(crash_tick) > 0:
            ticks["n"] += 1
            if ticks["n"] >= int(crash_tick):
                die()
        time.sleep(dt)

    def on_event(event: dict[str, Any], state: dict[str, Any]) -> None:
        if event["type"] == "node_started":
            stage["now"] = event["data"]["node"]
            if crash_node == stage["now"] and int(crash_tick) == 0:
                die()

    if a.action != "boot":
        env = trial.make_env(site, twin, db, llm=get_llm() if a.llm else None, tick=tick, on_event=on_event)
        who = principal(a.operator, "operator", site.name) if a.operator else None
        try:
            if a.action == "start":
                sheet = RunSheet.model_validate_json(Path(a.sheet).read_text(encoding="utf-8"))
                assert who is not None, "start needs --operator"
                res, _ = trial.start_trial(env, sheet, who, Limits(), a.run,
                                           held=None if a.held is None else a.held == "yes")
            else:
                res = Runner(trial.build_graph(), env, Limits()).resume(a.run, who)
            st = res.state
            out.update({"status": res.status, "outcome": res.outcome, "next": st["next"], "stage": st["task"]["stage"],
                        "waiting_for": st["waiting_for"], "why": st["task"]["why"],
                        "path": [f"{h['node']}:{h['route']}" for h in st["history"]]})
        except NotAllowed as exc:
            out.update({"refused": str(exc)})
    out["robot_seq_at_exit"] = twin.state().seq
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
