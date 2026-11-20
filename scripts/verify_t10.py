"""Verification of task 10 (policy onboarding and episode data).

    python scripts/verify.py t10     ->  verification/t10_policy_and_data.json
                                         verification/t10_dataset_card.md

Real data: the SurgicAI expert demonstrations and environment sources (MIT), the ROSMA
recordings (CC BY 4.0), and the dVRK kinematic files of a Classic and an Si PSM.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vxagent.config import get_settings                                    # noqa: E402
from vxagent.data import episodes as rosma                                 # noqa: E402
from vxagent.data import quality, surgicai                                 # noqa: E402
from vxagent.llm.base import ToolCall                                      # noqa: E402
from vxagent.llm.ollama import OllamaClient                                # noqa: E402
from vxagent.policy import compat                                          # noqa: E402
from vxagent.policy.adapter import Adapter, angle_between_rpy              # noqa: E402
from vxagent.policy.contract import FIELDS, Contract, contract_from_demonstrations, draft_contract   # noqa: E402
from vxagent.profiles import principal                                     # noqa: E402
from vxagent.robot.kinematics import load_chain                            # noqa: E402
from vxagent.robot.site import build_site, load_site                       # noqa: E402
from vxagent.runtime.trace import Tracer                                   # noqa: E402
from vxagent.store import Database                                         # noqa: E402
from vxagent.tools.builtin import register_builtin                         # noqa: E402
from vxagent.tools.dataset import register_dataset                         # noqa: E402
from vxagent.tools.executor import execute_tool                            # noqa: E402
from vxagent.tools.registry import ToolContext                             # noqa: E402

SMOKE = bool(os.environ.get("VX_SMOKE"))
REFERENCE = ROOT / "evals" / "t10_reference_contract.json"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def measured_contract(episodes: list[quality.Episode], draft: dict[str, dict[str, Any]] | None = None) -> tuple[Contract, dict[str, Any]]:
    return contract_from_demonstrations(episodes, draft, surgicai_version())


def surgicai_version() -> str:
    m = json.loads((surgicai.folder() / "MANIFEST.json").read_text(encoding="utf-8"))
    return m["tree"][:10]


def t10() -> dict[str, Any]:
    s = get_settings()
    site = load_site("site_a")
    out: dict[str, Any] = {"task": "t10_policy_and_data" + ("_smoke" if SMOKE else ""), "ran": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "model": s.model, "surgicai_tree": surgicai_version(),
                           "demonstrations_read_with": "a restricted unpickler that rebuilds arrays and plain containers only"}
    approach = quality.from_surgicai("Approach")

    # ---------------------------------------------------------------- a. contract extraction
    files = {"RL/Approach_env.py": (surgicai.folder() / "RL" / "Approach_env.py").read_text(encoding="utf-8"),
             "RL/subtask_env.py": (surgicai.folder() / "RL" / "subtask_env.py").read_text(encoding="utf-8")[:10000]}
    llm = OllamaClient(num_ctx=16384)
    draft = draft_contract(llm, files, max_chars=10000)
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))["fields"]
    rows = []
    for name in FIELDS:
        f, r = draft.fields.get(name, {"value": None, "status": "missing", "offered": None}), ref[name]
        value = str(f["value"]).lower() if f["value"] is not None else None
        if r["stated"]:
            outcome = ("correct" if value is not None and all(t in value for t in r["must_contain"]) else
                       "wrong value" if value is not None else "missed: left empty although the files state it")
        else:
            outcome = "rightly left for a person" if value is None else "filled although the files do not state it"
        rows.append({"field": name, "stated_in_the_files": r["stated"], "value": f["value"], "status": f["status"],
                     "offered_by_the_model": f.get("offered"), "file": f.get("file"), "quote": f.get("quote"), "outcome": outcome})
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    out["a_contract_extraction"] = {
        "files_given": {k: len(v) for k, v in files.items()}, "fields": len(rows), "model_calls": draft.attempts,
        "tokens": draft.tokens, "error": draft.error,
        "fields_with_a_quotation_found_in_the_file": len(draft.filled()),
        "fields_emptied_because_the_quotation_was_not_in_the_file": sum(r["status"].startswith("quotation not found") for r in rows),
        "fields_left_for_a_person": draft.for_a_person(), "against_the_reference": counts, "rows": rows}
    log(f"a contract: {counts}")
    contract, measured = measured_contract(approach, draft.fields)
    out["a_contract_extraction"]["filled_from_the_demonstrations"] = measured
    out["contract"] = contract.model_dump()

    # ---------------------------------------------------------------- b. compatibility with two sites
    spec = json.loads((s.sites_dir / "site_a.json").read_text(encoding="utf-8"))
    si_spec = {**copy.deepcopy(spec), "name": "site-a-si", "generation": "Si", "kinematic_file": "share/kinematic/PSM_Si.json",
               "instrument_file": "share/tool/LARGE_NEEDLE_DRIVER_420006.json"}
    si_site = build_site(si_spec, load_chain(s.dvrk_dir / si_spec["kinematic_file"], s.dvrk_dir / si_spec["instrument_file"]))
    moved = copy.deepcopy(spec)
    moved["named_poses"]["pregrasp"]["offset_mm"] = [35.0, 10.0, -22.0]          # the needle 20 mm further out
    moved_site = build_site(moved, site.chain)
    tight = copy.deepcopy(spec)
    tight["safety_profile"]["near_step_mm"] = 1.0
    tight_site = build_site(tight, site.chain)
    wrong_unit = contract.model_copy(update={"translation_unit": "m"})
    p0, g0 = approach[0].notes["poses"][0], approach[0].notes["goal"]
    seen = np.concatenate([e.position for e in approach])
    cases = {"Classic PSM (site-a)": (contract, site), "Si PSM, same poses": (contract, si_site),
             "site-a with the needle 20 mm further out": (contract, moved_site),
             "site-a with the near step tightened to 1 mm": (contract, tight_site),
             "site-a, contract declaring metres": (wrong_unit, site)}
    out["b_compatibility"] = {}
    for label, (c, st) in cases.items():
        res = compat.check(c, st, "start", "pregrasp", 10.0, policy_start=p0, policy_goal=g0, seen_positions=seen)
        out["b_compatibility"][label] = res
        log(f"b {label}: {res['verdict']} failing {res['failing']}")
    out["b_compatibility"]["kinematic_files"] = {"Classic": site.source, "Si": si_site.source}

    # ---------------------------------------------------------------- c. adapter round trip on every demonstration
    ad = Adapter(contract)
    per_task = {}
    for task in surgicai.TASKS:
        eps = quality.from_surgicai(task)
        pos_err = rot_err = replay = 0.0
        for e in eps:
            P = e.notes["poses"]
            for p in P:
                back = ad.pose_to_policy(ad.pose_to_site(p), p[6])
                pos_err = max(pos_err, float(np.abs(back[:3] - p[:3]).max()))
                rot_err = max(rot_err, angle_between_rpy(back, p))
            cur = P[0].copy()
            for a in e.actions:
                cur = ad.step(cur, a)
            replay = max(replay, float(np.linalg.norm(cur[:3] - P[-1][:3])))
        per_task[task] = {"episodes": len(eps), "poses": int(sum(len(e.notes["poses"]) for e in eps)),
                          "largest_position_error_cm": pos_err, "largest_rotation_error_rad": rot_err,
                          "actions_replayed_from_the_start_end_within_cm": round(replay, 6),
                          "all_end_in_success": all(e.notes["success"] for e in eps)}
    out["c_adapter_round_trip"] = {"policy_to_site_and_back": per_task,
                                   "note": "replaying the recorded actions with the estimated step size from the recorded "
                                           "start reproduces the recorded end pose, which also checks the step size"}
    log(f"c round trip: {[(t, v['largest_position_error_cm'], v['actions_replayed_from_the_start_end_within_cm']) for t, v in per_task.items()]}")

    # ---------------------------------------------------------------- d. episode checks over both sources
    episodes = [quality.from_rosma(e) for e in rosma.list_episodes()]
    for task in surgicai.TASKS:
        episodes += quality.from_surgicai(task)
    limits = (site.chain.qmin, site.chain.qmax)
    rows_q = quality.check_all(episodes, limits)
    by_source = {}
    for src in ("ROSMA", "SurgicAI"):
        mine = [r for r in rows_q if r["source"] == src]
        flags: dict[str, int] = {}
        for r in mine:
            for f in r["flags"]:
                flags[f["check"]] = flags.get(f["check"], 0) + 1
        by_source[src] = {"episodes": len(mine), "accepted": sum(r["verdict"] == "accepted" for r in mine),
                          "accepted_with_flags": sum(r["verdict"] == "accepted with flags" for r in mine),
                          "excluded": sum(r["verdict"] == "excluded" for r in mine), "flags": flags,
                          "streams_absent": mine[0]["absent"] if mine else []}
    out["d_episode_checks"] = {"by_source": by_source,
                               "joint_limits_from": f"{site.source.get('arm')} and {site.source.get('instrument')}",
                               "flagged": [{"episode": r["episode_id"], "flags": r["flags"]} for r in rows_q if r["flags"]][:20]}
    log(f"d checks: {by_source}")

    # ---------------------------------------------------------------- e. negative controls
    base = episodes[0]                                               # a real ROSMA recording, altered four ways
    rng = np.random.default_rng(3)
    scaled = copy.copy(base)
    scaled.position, scaled.episode_id = base.position * 1000.0, "control_rescaled_by_1000"
    jumped = copy.copy(base)
    jumped.position = base.position.copy()
    jumped.position[len(base.position) // 2:] += [0.03, 0.0, 0.0]
    jumped.episode_id = "control_planted_jump"
    shuffled = copy.copy(base)
    order = rng.permutation(len(base.t))
    shuffled.t, shuffled.episode_id = base.t[order], "control_shuffled_time_stamps"
    twin_copy = copy.copy(base)
    twin_copy.episode_id = "control_duplicate"
    controls = quality.check_all([base, scaled, jumped, shuffled, twin_copy], limits)
    expect = {"control_rescaled_by_1000": "unit_magnitude", "control_planted_jump": "jump",
              "control_shuffled_time_stamps": "time_runs_backwards", "control_duplicate": "duplicate"}
    out["e_negative_controls"] = {"made_from": base.episode_id, "controls": [
        {"control": r["episode_id"], "must_be_caught_by": expect[r["episode_id"]],
         "caught": expect[r["episode_id"]] in [f["check"] for f in r["flags"]], "verdict": r["verdict"],
         "flags": [f["check"] for f in r["flags"]]} for r in controls[1:]]}
    out["e_negative_controls"]["all_caught"] = all(c["caught"] for c in out["e_negative_controls"]["controls"])
    log(f"e controls: {[(c['control'], c['caught'], c['verdict']) for c in out['e_negative_controls']['controls']]}")

    # ---------------------------------------------------------------- f. the split and the card
    split = quality.split_by_operator(rows_q)
    out["f_split"] = {k: v for k, v in split.items() if k not in ("train", "evaluation")}
    out["f_split"].update({"train_episodes": len(split["train"]), "evaluation_episodes": len(split["evaluation"]),
                           "overlap_of_operators": len(split["operators_on_both_sides"])})
    card = quality.card(rows_q, split, {
        "ROSMA": {"source": "Zenodo record 3932964, kinematics files of 12 recordings (4 operators, 3 tasks)", "licence": "CC BY 4.0",
                  "cannot_show": "jaw motion, commands, camera images, anything about suturing (the tasks are training exercises)"},
        "SurgicAI": {"source": f"github.com/surgical-robotics-ai/SurgicAI, RL/Expert_traj at tree {surgicai_version()}", "licence": "MIT",
                     "cannot_show": "timing, who or what produced each episode, variation of the start pose (every Approach "
                                    "episode starts from the same pose), anything measured on a physical robot"}})
    (ROOT / "verification" / "t10_dataset_card.md").write_text(card, encoding="utf-8", newline="\n")
    out["f_split"]["dataset_card"] = "verification/t10_dataset_card.md"

    # ---------------------------------------------------------------- authorization of the dataset tools
    db = Database(":memory:")
    reg = register_dataset(register_builtin())
    rows_a = []
    for who in (principal("op.lee", "operator", site.name), principal("eng.kim", "service", site.name),
                principal("saf.adler", "safety", site.name), principal("sup.wong", "support", "")):
        ctx = ToolContext(principal=who, db=db, tracer=Tracer(db), site=site, run_id="t10")
        res = execute_tool(reg, ToolCall("q", "episode_quality", {"episode_id": base.episode_id}), ctx)
        rows_a.append({"role": who.role, "episode_quality": "ok" if res.ok else res.error_kind})
    out["dataset_tool_by_role"] = rows_a
    return out
