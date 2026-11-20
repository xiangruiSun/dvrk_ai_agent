"""A bench for running runbooks on the made-up test arm: registry, twin, declared needle and a
stored calibration record. Used by the tests and by scripts/record_runbook_traces.py."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vxagent.diagnostics.episode import calibration
from vxagent.pipelines.trial import Path as _Path          # noqa: F401  (keeps the import graph honest)
from vxagent.robot.kinematics import load_chain
from vxagent.robot.needle import DeclaredNeedle
from vxagent.robot.site import build_site
from vxagent.runtime.trace import Tracer
from vxagent.skills import runbook as rbk
from vxagent.skills.index import default_library, register_bench, register_skills
from vxagent.store import Database
from vxagent.structured import proposals
from vxagent.tools.builtin import register_builtin
from vxagent.tools.registry import Principal, ToolContext
from vxagent.tools.service import register_service, store_calibration_record

FIX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[1]
TRACES = Path(__file__).parent / "runbooks"
CATALOGUE = ["get_robot_state", "list_named_poses", "check_named_target", "rehearse_move", "enable_and_home",
             "move_to_named_pose", "set_jaw", "wait_action", "cancel_action", "list_episodes", "episode_summary",
             "read_signal_window", "calculator", "read_calibration_record", "calibration_residual", "observe_needle", "hold"]

# name -> (how the bench is prepared, parameters). Each is one recorded reference run.
CASES: dict[str, dict[str, Any]] = {
    "bring_up": {"runbook": "bring_up", "ready": False},
    "rig_check": {"runbook": "rig_check"},
    "calibration_validity": {"runbook": "calibration_validity"},
    "calibration_validity__invalidated": {"runbook": "calibration_validity", "calibration_valid": False},
    "verify_grasp_retention": {"runbook": "verify_grasp_retention", "at": "pregrasp"},
    "verify_grasp_retention__slips_on_lift": {"runbook": "verify_grasp_retention", "at": "pregrasp",
                                              "needle": {"slips_on_lift": True}},
    "verify_grasp_retention__slips_while_holding": {"runbook": "verify_grasp_retention", "at": "pregrasp",
                                                    "needle": {"slips_while_holding": True}},
    "verify_grasp_retention__grasp_misses": {"runbook": "verify_grasp_retention", "at": "pregrasp",
                                             "needle": {"grasp_succeeds": False}},
    "return_to_start": {"runbook": "return_to_start", "at": "left_edge", "jaw": "closed"},
}


def make_site():
    spec = json.loads((ROOT / "sites" / "site_a.json").read_text(encoding="utf-8"))
    spec["named_poses"]["wrist_limit"]["joints_si"] = [0.0, 0.0, 0.12, 0.0, 1.36, 0.0]
    return build_site(spec, load_chain(FIX / "test_arm.json", FIX / "test_tool.json"))


def make_bench(site=None, ready: bool = True, at: str | None = None, jaw: str | None = None,
               needle: dict[str, bool] | None = None, calibration_valid: bool = True):
    site = site or make_site()
    db = Database(":memory:")
    twin = site.twin()
    if ready:
        twin.enable()
        twin.home()
    if at:
        twin.set_joints(site.poses[at].q)
    if jaw:
        twin._jaw = site.jaw_angles[jaw]
    reg = register_bench(register_service(register_builtin()))
    library = default_library(reg)
    catalogue = register_skills(reg, library, CATALOGUE)
    who = Principal("op.test", "operator", frozenset({"robot:read", "robot:move", "episodes:read"}), site="site-a",
                    tiers=frozenset({"observe", "simulate", "actuate"}))
    ctx = ToolContext(principal=who, db=db, tracer=Tracer(db), run_id="run_rb", robot=twin, site=site, tick=lambda dt: None)
    ctx.extras["needle"] = DeclaredNeedle(site, **(needle or {})).attach(twin)
    cal = calibration(None, site.scene.workspace_center, np.random.default_rng(5))
    store_calibration_record(db, site.name, {"id": cal["id"], "T_est": cal["T_est"].tolist(), "seen_m": cal["seen_m"].tolist(),
                                             "touched_m": cal["touched_m"].tolist(), "method": "point pairs"})
    proposals.set_calibration(db, site.name, calibration_valid, "test")
    return reg, ctx, twin, library, catalogue


def outcome(result: dict[str, Any]) -> dict[str, Any]:
    """What a reference trace pins down: the steps taken, every check's result, and how it ended."""
    return {"status": result["status"], "stopped_at": result.get("stopped_at"),
            "steps": [{"id": r["id"], "tool": r["tool"], "ok": r["ok"], "checks": [c["passed"] for c in r["checks"]],
                       **({"recovery": r["recovery"]["status"]} if "recovery" in r else {})} for r in result["steps"]],
            "facts_after": result.get("facts_after", {})}


def run_case(name: str) -> tuple[dict[str, Any], Any, Any]:
    case = dict(CASES[name])
    book = case.pop("runbook")
    reg, ctx, twin, library, _ = make_bench(**case)
    result = rbk.run(library, book, {}, reg, ctx, f"case:{name}", on_approval=lambda wait: True)
    return result, twin, ctx
