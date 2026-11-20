"""The sections of verification/REPORT.md for agent tasks 02 and 03, built from the evidence
files written by scripts/verify.py."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

Table = Callable[[list[str], list[list[Any]]], str]
VER = Path(__file__).resolve().parents[1] / "verification"


def yes(x: Any) -> str:
    return "yes" if x else "no"


def _load(name: str) -> dict[str, Any] | None:
    path = VER / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _val(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.6g}"
    if isinstance(x, list):
        return "[" + ", ".join(_val(v) for v in x) + "]"
    return "not set" if x is None else str(x)


# ------------------------------------------------------------------------------ T02

def t02(d: dict[str, Any], table: Table) -> str:
    a, b, c, auth, e = (d["a_baseline"], d["b_change_requests"], d["c_negative_controls"], d["d_authorization"],
                        d["e_dry_application_and_apply"])
    out = ["## T02 Structured change proposals\n",
           f"Ran {d['ran']} with `{d['model']}`. Files: the dVRK project's `share/arm`, `share/system` and "
           f"`share/schemas` at commit `{d['dvrk_commit']}`, and this project's declared site file. Change requests "
           f"and their reference patches: `{d['requests_file']}` (written for this project). Evidence: "
           f"`t02_structured_output.json`.\n",
           f"Baseline. {a['files']} published arm and system files, {a['schemas']} schemas. "
           f"{a['name_a_schema_of_the_set']} files name a schema of the set and were validated against it as they "
           f"are: {a['pass']} pass, {len(a['fail'])} fail. {len(a['name_no_schema_of_the_set'])} files name no schema "
           f"of the set (setup-joint files, older console files, files of other components) and were not validated.\n"]
    if a["fail"]:
        out.append(table(["Published file that fails the project's own schema", "Error"],
                         [[f"`{f['file']}`", "; ".join(f["errors"])] for f in a["fail"]]))
        out.append("So a patched file is not required to be free of schema errors. It is required to have no "
                   "schema error the file did not already have, and the errors already there are reported.\n")

    rows = []
    for mode, m in b["by_mode"].items():
        stages = ", ".join(f"{k}: {v}" for k, v in m["failed_attempts_by_stage"].items()) or "none"
        rows.append([mode, f"{m['valid_on_first_attempt']}/{m['requests']}", m["accepted_after_repair"],
                     m["nothing_proposed"], f"{m['equals_reference']}/{m['requests']}", stages, m["median_tokens"],
                     m["median_seconds"]])
    out.append(f"Change requests in words ({b['requests']}, at most {b['max_attempts']} attempts each). A patch "
               "equals the reference when the patched file is the same and the limit declaration is the same.\n")
    out.append(table(["Mode", "Valid on first attempt", "Accepted after repair", "Nothing proposed",
                      "Equals the reference patch", "Failed attempts by stage", "Median tokens", "Median seconds"], rows))
    failed_first = [r for r in b["runs"] if not r["valid_on_first_attempt"]]
    if failed_first:
        seen: dict[str, list[str]] = {}
        for r in failed_first:
            seen.setdefault(r["attempt_log"][0]["errors"][0].split(":")[0], []).append(f"{r['id']} ({r['mode']})")
        out.append("First attempts that failed, by the first error they got:\n")
        out.append(table(["Error", "Requests"], [[k, ", ".join(v)] for k, v in seen.items()]))
    units = [(r["id"], x["given"], x["given_in"], x["after"], x["stored_in"])
             for r in e["effects"] for x in r["file_values"] if x["given_in"] != x["stored_in"]
             and x["given_in"] not in ("none", "ratio")]
    if units:
        out.append("Values stated in one unit and stored in another. The conversion is done by code; the model "
                   "names the unit and copies the number:\n")
        out.append(table(["Request", "Given", "Unit given", "Written to the file", "Unit of the file"],
                         [[i, _val(g), gu, _val(aft), su] for i, g, gu, aft, su in units]))

    # A number in the model's free-text reason that is neither what the request gave, nor what
    # the file held, nor what the code wrote: the model converted in its head.
    by_id = {x["id"]: x for x in e["effects"]}
    number = r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?(?![A-Za-z0-9_])"
    asked: dict[str, list[float]] = {}
    req_file = VER.parent / d["requests_file"]
    if req_file.exists():
        for q in json.loads(req_file.read_text(encoding="utf-8"))["requests"]:
            text = q["request"] + " " + " ".join(q["records"].values()) + " " + json.dumps(q["reference"])
            asked[q["id"]] = [float(t) for t in re.findall(r"-?\d+(?:\.\d+)?", text)]
    own = []
    for r in b["runs"]:
        if not r.get("patch"):
            continue
        known: list[float] = list(asked.get(r["id"], []))
        for x in by_id[r["id"]]["file_values"]:
            for v in (x["given"], x["before"], x["after"]):
                flat = v if isinstance(v, list) else [v]
                known += [float(n) for row in flat for n in (row if isinstance(row, list) else [row])
                          if isinstance(n, (int, float)) and not isinstance(n, bool)]
        written = [x["after"] for x in by_id[r["id"]]["file_values"]]
        for tok in re.findall(number, r["patch"]["reason"]):
            n = float(tok)
            if not any(abs(n - k) <= 0.001 * max(1.0, abs(k)) for k in known):
                own.append([f"{r['id']} ({r['mode']})", tok, ", ".join(_val(w) for w in written)])
    if own:
        out.append("The `reason` of a patch is free text and is not checked. Numbers in it that are not in the "
                   "request or its records, not the value the file held and not the value the code wrote, so "
                   "that the model must have computed them itself:\n")
        out.append(table(["Request", "Number in the model's reason", "Value the code wrote to the file"], own))

    out.append(f"Negative controls, hand-written. Accepted by the patch schema and by the file's schema, refused by "
               f"a physical rule: {c['accepted_by_both_schemas_refused_by_a_physical_rule']}. Refused by the "
               f"allowlist: {c['refused_by_the_allowlist']}. Accepted: {len(c['any_accepted'])}.\n")
    out.append(table(["Control", "Refused at", "First error"],
                     [[x["id"], x["refused_at"], x["errors"][0] if x["errors"] else ""] for x in c["controls"]]))

    out.append(f"Authorization, the same valid patch and different people: {auth['as_expected']}/{auth['cases']} "
               f"cases as expected; {auth['refusals_written_to_the_audit_log']} refusals in the audit log.\n")
    out.append(table(["Case", "Allowed", "Refused because"],
                     [[m["case"], yes(m["allowed"]), m["refusal"] or ""] for m in auth["matrix"]]))

    moving = [x for x in e["effects"] if x["poses_that_move"] or x["newly_rejected_by_the_safety_filter"]
              or x["largest_change_of_height_over_pad_mm"]]
    out.append(f"Dry application of the {len(e['effects'])} reference patches to the twin's inputs. "
               f"{len(moving)} have a geometric effect on the site's named poses:\n")
    out.append(table(["Request", "Named poses that move", "Largest move (mm)", "Largest turn (deg)",
                      "Largest change of height over the pad (mm)", "Poses the safety filter would now reject",
                      "Calibration invalidated"],
                     [[x["id"], x["poses_that_move"], x["largest_move_mm"], x["largest_turn_deg"],
                       x["largest_change_of_height_over_pad_mm"],
                       ", ".join(x["newly_rejected_by_the_safety_filter"]) or "none", yes(x["invalidates_calibration"])]
                      for x in moving]))
    ap = e["applied"]
    out.append("Applied after approval: " + "; ".join(
        f"`{x['id']}` became version {x['version']} of `{x['target']}`"
        + (f" (envelope version {x['envelope_version']})" if x["envelope_version"] else "")
        + (", calibration marked invalid" if x["calibration_invalidated"] else "") for x in ap)
        + f". Source files unchanged afterwards: {yes(e['source_files_unchanged'])} ({e['source_files_checked']} "
        f"checked by hash). A proposal that had been checked against an older version of its file: "
        f"`{e['proposal_checked_against_an_older_version']}`, not applied.\n")
    return "\n".join(out)


# ------------------------------------------------------------------------------ T03

def _route(path: list[str]) -> str:
    return " → ".join(p.split(":")[0] for p in path)


def t03(d: dict[str, Any], table: Table) -> str:
    a, b, c, st, e, f, ts = (d["a_trials"], d["b_crash_and_resume"], d["c_watchdog"], d["d_stale_approvals"],
                             d["e_stops_limits_faults"], d["f_state_from_events"], d["recorded_time_stamps"])
    out = ["## T03 Event-driven trial runtime\n",
           f"Ran {d['ran']} on the kinematic twin of {d['site']} at {d['control_rate_hz']:.0f} Hz, in real time. "
           f"`{d['model']}` writes two texts and decides nothing. Approach paths: {d['approach_paths']}. Needle held: "
           f"{d['needle_held']}. Every fault below is injected. Evidence: `t03_runtime.json`.\n",
           f"Trials end to end: {a['trials']} (one per recording; the needle is declared held in every second one). "
           f"{a['success']} ended as success and {a['recovered']} as recovered, {a['as_expected']}/{a['trials']} as "
           f"the declared outcome implies; arm back at the start pose in {a['arm_back_at_start_pose']}. Median "
           f"{a['median_events']} events and {a['median_checkpoints']} checkpoints per run. The fit of the recorded "
           f"stretches needed a scale of {a['scale_of_the_path_fit_min_max'][0]} to "
           f"{a['scale_of_the_path_fit_min_max'][1]}; the mapped paths leave the straight line by "
           f"{a['largest_distance_from_the_straight_line_mm_min_max'][0]} to "
           f"{a['largest_distance_from_the_straight_line_mm_min_max'][1]} mm. Summaries written by the model: "
           f"{a['summaries_written_by_the_model']}; replaced by the template because of a number that was not among "
           f"the facts, or an empty reply: {a['summaries_replaced_by_the_template']}.\n",
           table(["Recording", "Declared held", "Route", "Outcome", "Waypoints", "Tip travel (mm)", "Events",
                  "Checkpoints", "Approval bound to robot state / envelope"],
                 [[r["episode"], yes(r["needle_held_declared"]), _route(r["path"]), r["outcome"],
                   r["path_fit"]["waypoints"], r["tip_travel_mm"], r["events"], r["checkpoints"],
                   f"{r['approval']['bound_to_robot_seq']} / {r['approval']['bound_to_envelope']}"] for r in a["runs"]])]
    one = a["runs"][0]
    out.append(f"Time per node in the first trial (seconds): "
               + ", ".join(f"{k} {v}" for k, v in one["seconds_per_node"].items())
               + f". Spans written for it: {', '.join(f'{v} {k}' for k, v in sorted(one['spans'].items()))}.\n")

    out.append(f"Crash and resume, {b['cases']} cases. How: {b['how']}. Marked needs_reverification by the next "
               f"process: {b['marked_needs_reverification']}/{b['cases']}. Resume without an operator refused: "
               f"{b['resume_without_an_operator_refused']}/{b['cases']}. Arm moved before the operator resumed: "
               f"{b['arm_moved_before_the_operator_resumed']}. First thing after the resume was a look at the "
               f"robot: {b['looked_at_the_robot_first_after_resume']}/{b['cases']}. Arm at the start pose at the end: "
               f"{b['arm_at_start_pose_at_end']}/{b['cases']}.\n")
    out.append(table(["Killed in", "Arm then (mm from start)", "Requirement that failed on resume",
                      "Times the node was entered", "Place entered after resume", "Motions after resume", "Outcome"],
                     [[x["case"], x["arm_when_killed"]["mm_from_start"],
                       ", ".join(x["precondition_failed_after_resume"] or []) or "none",
                       x["times_the_killed_node_was_entered"], yes(x["place_entered_after_resume"]),
                       ", ".join(x["motions_after_resume"]), x["outcome"]] for x in b["runs"]]))

    out.append(f"Watchdog. {c['injected'].capitalize()}. Fired: {yes(c['watchdog_fired'])}, after "
               f"{c['feedback_silent_s_when_cancelled']} s of silence (limit {c['watchdog_limit_s']} s); the reach was "
               f"{c['reach_action_status']} after {c['control_steps_before_cancel']} of "
               f"{c['planned_control_steps']} planned control steps, so the arm went on moving for the length of the "
               f"limit before it was stopped. Grasp entered: {yes(c['grasp_entered'])}. Route: {_route(c['path'])}. "
               f"Outcome {c['outcome']}, arm at the start pose: {yes(c['arm_at_start_pose_at_end'])}.\n")

    m, env, old = st["arm_moved_after_approval"], st["envelope_changed_after_approval"], st["approval_older_than_its_limit"]
    out.append("Approvals bound to the state they were given for:\n")
    out.append(table(["Between approval and use", "What happened"], [
        [m["injected"], f"approval refused ({m['why']}: given for robot state {m['approved_for_robot_seq']}, state at "
                        f"use {m['robot_seq_at_use']}); motions commanded {m['motions_commanded']}; operator asked "
                        f"again: {yes(m['asked_again'])}; with the new approval: {m['after_a_new_approval']}"],
        [env["change"], f"run stopped at `{env['at_node']}` by the requirement "
                        f"{', '.join(env['requirement_that_failed'])} (run sheet written for envelope "
                        f"{env['run_sheet_written_for_envelope']}); earlier approval used: "
                        f"{yes(env['approval_given_before_the_change_was_used'])}; operator asked again: "
                        f"{yes(env['operator_asked_again'])}; motions commanded {env['motions_commanded']}; outcome "
                        f"`{env['outcome']}`"],
        [f"{old['used_after_s']} s pass, with a limit of {old['limit_s']} s",
         f"approval refused ({old['why']}); motions commanded {old['motions_commanded']}; operator asked again: "
         f"{yes(old['asked_again'])}"]]))
    first = _load("t03_first_run")
    if first and "after_a_new_approval" in first["d_stale_approvals"]["envelope_changed_after_approval"]:
        fe = first["d_stale_approvals"]["envelope_changed_after_approval"]
        out.append(f"First run ({first['ran']}, kept in `t03_first_run.json`): after the envelope change the old "
                   f"approval was refused ({fe['why']}), the operator was asked again, and with the new approval the "
                   f"run ended as `{fe['after_a_new_approval']}` with {fe['motions_commanded_in_the_whole_run']} "
                   f"motions. Nothing moved, but a person was asked for a decision that could not matter. The "
                   f"envelope is now a requirement of the go node itself, so the run stops before anyone is asked. "
                   f"The approval's own binding to the envelope version is still there and is covered by a test.\n")

    rows = []
    for name, r in e.items():
        note = []
        if "arm_stopped_at" in r:
            note.append(f"stopped {r['arm_stopped_at']['mm_from_start']} mm from start; moved while waiting: "
                        f"{yes(r['arm_moved_while_waiting_for_the_operator'])}; recovery needed an approval although "
                        f"recovery was pre-approved: {yes(r['recovery_still_needed_an_approval'])}")
        if "motions_commanded" in r:
            note.append(f"motions commanded {r['motions_commanded']}")
        if "steps_used" in r:
            note.append(f"limit of {r['limit']}; {r['steps_used']} entries in all, the ones after the limit being "
                        f"recovery, verdict and summary")
        elif "limit" in r:
            note.append(f"limit of {r['limit']}, checked {r['checked']}; tip travel in all {r['tip_travel_mm']} mm, "
                        f"the travel after the limit being the recovery")
        if "reach" in r and isinstance(r.get("reach"), dict):
            note.append(f"reach {r['reach']['status']} ({r['reach']['reason']})")
        if "motions_commanded_after_the_fault" in r:
            note.append(f"motions after the fault {r['motions_commanded_after_the_fault']}")
        if "failed_checks" in r:
            note.append(f"failed: {', '.join(r['failed_checks'])}; explanation by {r['explanation_source']}")
        if "rejections" in r:
            v = r["rejections"][0]["violations"][0]
            note.append(f"rejected in {r['rejections'][0]['stage']} by rule {v['rule']}")
        rows.append([r.get("injected") or name.replace("_", " "), _route(r["path"]), r["outcome"],
                     yes(r["arm_at_start_pose_at_end"]), "; ".join(note)])
    out.append("Stops, rejections, limits and faults:\n")
    out.append(table(["Case", "Route", "Outcome", "Arm at start pose at the end", "Notes"], rows))
    ex = e.get("calibration_invalid_at_preflight", {})
    if ex.get("explanation"):
        out.append(f"The explanation given for the failed pre-flight ({ex['explanation_source']}): \"{ex['explanation']}\"\n")

    out.append(f"State from the event log: {f['checkpoints_equal_to_the_state_rebuilt_from_events']}/"
               f"{f['checkpoints_compared']} checkpoints of {f['runs']} runs equal the state rebuilt from the events "
               f"up to that point ({f['events_in_all_runs']} events); the state a new process loads equals the "
               f"rebuilt state in {f['runs_whose_loaded_state_equals_the_rebuilt_state']}/{f['runs']} runs.\n")
    out.append(f"Found while building this task: {ts['note']} {ts['samples_moved_by_one_second']} samples in "
               f"{ts['episodes_with_a_late_seconds_field']} of {ts['episodes']} recordings were moved by one second; "
               f"time now runs forward in every recording: {yes(ts['time_runs_forward_in_every_episode'])}. Figures "
               f"of task 01 that are unchanged by the correction: {ts['figures_of_task_01_unchanged_by_the_correction']}.\n")
    return "\n".join(out)


SECTIONS = [("t02_structured_output", t02), ("t03_runtime", t03)]
