"""Pre-flight of a trial, all in code. A trial may start only when every check here passes, and
each failed check comes back with the runbook that repairs it.

The checks follow the deployment framework this project builds on: the arm is ready, the
instrument is the one on the run sheet, the camera-to-robot calibration exists, is well formed,
has not been ended by an event and is accurate enough, the scene records belong to the same
session as the calibration, the policy's contract fits the site, the start and the target lie
in the range the policy was trained on and inside the workspace, and perception answers in time.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pydantic import ValidationError

from ..diagnostics.tools import calibration_residual
from ..structured.proposals import calibration as calibration_status
from ..structured.rules import rotation_problem
from ..structured.schemas import CalibrationRecord
from ..tools.service import load_calibration_record
from .motion import check_goal

PERCEPTION_LIMIT_S = 0.5
PROBLEM = {
    "arm is powered, enabled and homed, with no fault": "the arm is not ready ({detail})",
    "instrument is the one on the run sheet and has an instrument file": "the installed instrument is not the one on the run sheet, or has no instrument file ({detail})",
    "a calibration record exists": "there is no camera-to-robot calibration record for this site",
    "the calibration record is well formed": "the calibration record is malformed ({detail})",
    "the calibration has not been ended by an event": "the calibration is no longer valid: {detail}",
    "the calibration residual is under the site's limit": "the calibration residual is over the site's limit ({detail})",
    "pad-plane and site records are of the same session as the calibration": "the scene records are not of the same session as the calibration ({detail})",
    "the policy contract is compatible with the site": "the policy contract is not compatible with the site ({detail})",
    "start and target lie in the policy's training range and inside the workspace": "the start or the target is outside the policy's training range or the workspace ({detail})",
    "the perception source answered within its time limit": "the perception source did not answer within its time limit ({detail})",
}


def run_preflight(robot: Any, site: Any, sheet: dict[str, Any], db: Any, *, start_T: np.ndarray, goal_T: np.ndarray,
                  compat: dict[str, Any] | None, in_range: tuple[bool, float] | None,
                  perception: Callable[[], dict[str, Any]] | None, tool_dir: Path | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str, repair: str) -> None:
        row = {"check": name, "ok": bool(ok), "detail": detail, "repair": None if ok else repair}
        if not ok:
            # What is wrong, as a sentence written here. A check's name states the requirement, and a
            # reader (or a model) who repeats the name as the reason says the opposite of what happened.
            row["problem"] = PROBLEM.get(name, "a pre-flight requirement is not met: " + name).format(detail=detail)
        out.append(row)

    s = robot.state()
    check("arm is powered, enabled and homed, with no fault", s.ready,
          f"powered {s.powered}, enabled {s.enabled}, homed {s.homed}, fault {s.fault}", "bring_up")

    have = Path(site.chain.source.get("instrument", "")).stem
    file_ok = tool_dir is None or (tool_dir / f"{sheet['instrument']}.json").exists()
    check("instrument is the one on the run sheet and has an instrument file", sheet["instrument"] == have and file_ok,
          f"run sheet {sheet['instrument']}, installed {have}, file present {file_ok}", "escalate_to_engineer")

    rec = load_calibration_record(db, site.name)
    status = calibration_status(db, site.name)
    if rec is None:
        check("a calibration record exists", False, "no camera-to-robot calibration is stored for this site", "calibration_validity")
    else:
        try:
            CalibrationRecord(method=rec.get("method") or "unknown", frames={"parent": "robot base", "child": "camera"},
                              transform=rec["T_est"], residuals={"translation_rms_mm": 0.0, "rotation_rms_deg": None},
                              samples=len(rec["touched_m"]), valid_until=["instrument_change"])
            bad = rotation_problem(np.array(rec["T_est"], dtype=float)[:3, :3])
            formed, why = bad is None, bad or "a 4 x 4 transform with a proper rotation"
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            formed, why = False, str(exc)[:160]
        check("the calibration record is well formed", formed, why, "calibration_validity")
        check("the calibration has not been ended by an event", bool(status["valid"]), status["why"], "calibration_validity")
        if formed:
            res = calibration_residual({"calibration": rec})
            mean, limit = res["residual_mm"]["mean"], res["acceptance_mm"]
            check("the calibration residual is under the site's limit", mean <= limit,
                  f"{mean} mm mean on {res['held_out_points']} held-out points, limit {limit} mm", "calibration_validity")
        row = db.one("SELECT v FROM kv WHERE k=?", (f"scene_session:{site.name}",))
        scene = row["v"].strip('"') if row else None
        check("pad-plane and site records are of the same session as the calibration", scene is not None and scene == rec.get("session"),
              f"scene records: session {scene}; calibration: session {rec.get('session')}", "rig_check")

    if compat is None:
        check("the policy contract is compatible with the site", False, "no contract was given for this trial", "escalate_to_engineer")
    else:
        check("the policy contract is compatible with the site", compat["verdict"].startswith("compatible"),
              compat["verdict"] + (": " + "; ".join(compat["failing"]) if compat["failing"] else ""), "escalate_to_engineer")

    bad_s = check_goal(site.profile, site.scene, site.chain, start_T, s.q)
    bad_g = check_goal(site.profile, site.scene, site.chain, goal_T, s.q)
    inside = in_range[0] if in_range is not None else False
    far = in_range[1] * 1000 if in_range is not None else float("nan")
    check("start and target lie in the policy's training range and inside the workspace", inside and not bad_s and not bad_g,
          f"start to target is {far:.1f} mm outside the trained range; start: "
          f"{', '.join(v.rule for v in bad_s) or 'accepted'}; target: {', '.join(v.rule for v in bad_g) or 'accepted'}",
          "escalate_to_engineer")

    if perception is None:
        check("the perception source answered within its time limit", False, "no perception source is connected", "rig_check")
    else:
        t0 = time.perf_counter()
        try:
            seen = perception()
            took = float(seen.get("latency_s", time.perf_counter() - t0))
            ok = bool(seen.get("visible")) and took <= PERCEPTION_LIMIT_S
            detail = f"answered in {took:.2f} s (limit {PERCEPTION_LIMIT_S} s), needle visible {seen.get('visible')}"
        except Exception as exc:                        # a source that fails is a source that did not answer
            ok, detail = False, f"the perception source failed: {type(exc).__name__}"
        check("the perception source answered within its time limit", ok, detail, "rig_check")
    return out
