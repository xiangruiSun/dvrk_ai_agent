"""The sections of verification/REPORT.md for agent tasks 08 to 10, built from the evidence
files written by scripts/verify.py."""
from __future__ import annotations

from typing import Any

from report_agent_tasks import Table, yes


# ------------------------------------------------------------------------------ T08

def t08(d: dict[str, Any], table: Table) -> str:
    a, b, c, v = (d["a_designs_on_the_injected_set"], d["b_two_faults_at_once"], d["c_ledger_without_a_model"],
                  d["d_easy_fix_is_to_loosen_a_limit"])
    out = ["## T08 Multi-agent incident triage\n",
           f"Ran {d['ran']} with `{d['model']}`. Incidents: the injected faults of task 04, {a['incidents']} runs "
           f"({a['runs_per_cause']} per cause). Scoring uses no model: five checks, below 4 is a failure. "
           f"{d['one_model_one_gpu'].capitalize()}. Evidence: `t08_multi_agent.json`.\n",
           table(["Design", "Mean score of 5", "Perfect", "Failed", "Cause correct", "Mean model calls", "Mean tokens",
                  "Mean seconds", "Mean ledger entries", "Entries refused", "Vetoes"],
                 [[k, x["mean_score"], f"{x['perfect']}/{x['incidents']}", f"{x['failed_below_4']}/{x['incidents']}",
                   f"{x['cause_correct']}/{x['incidents']}", x["mean_calls"], x["mean_tokens"], x["mean_seconds"],
                   x["mean_ledger_entries"], x["entries_refused"], x["vetoes"]] for k, x in a["by_design"].items()])]
    names = list(next(iter(a["by_design"].values()))["checks_passed"])
    out.append(table([f"Check passed (of {a['incidents']})"] + list(a["by_design"]),
                     [[n.replace("_", " ")] + [a["by_design"][k]["checks_passed"][n] for k in a["by_design"]] for n in names]))
    causes = list(next(iter(a["by_design"].values()))["cause_correct_by_cause"])
    out.append(table([f"Cause correct, by cause (of {a['runs_per_cause']})"] + list(a["by_design"]),
                     [[cz] + [a["by_design"][k]["cause_correct_by_cause"][cz] for k in a["by_design"]] for cz in causes]))
    out.append(f"Two faults at once ({' and '.join(b['injected'])}). {b['note'].capitalize()}.\n")
    out.append(table(["Design", "Cause", "Runner-up", "Names both faults", "Causes with supporting ledger entries", "Model calls"],
                     [[k, x["cause"], x["runner_up"], yes(x["names_both_faults"]),
                       ", ".join(x["causes_with_supporting_entries"]) or "none", x["calls"]] for k, x in b["by_design"].items()]))
    out.append(f"The ledger without a model: {c['writers']} concurrent writers attempted {c['entries_attempted']} entries; "
               f"{c['entries_in_the_ledger']} are in the ledger, {c['entries_lost']} lost, positions consecutive: "
               f"{yes(c['positions_are_consecutive'])}. Entries with a reference that does not resolve or a value that is not in "
               f"the data: {c['entries_with_a_bad_reference_or_value_refused']} refused of {c['expected_refused']} planted. Findings "
               f"written against an older ledger: {c['findings_written_against_an_older_ledger_refused']} refused, "
               f"{c['findings_accepted']} accepted. An entry pointing at another role's data: refused "
               f"{yes(c['entry_pointing_at_another_roles_data_refused'])}.\n")
    out.append(f"The easy fix is to loosen a limit. The operator's note in the incident: \"{v['operator_note']}\"\n")
    out.append(table(["Design", "Vetoes", "Vetoed by rule", "Final action", "Final limit change", "Final finding loosens a limit",
                      "Score"],
                     [[k, x["vetoes"], "; ".join((x["review"] or {}).get("by_rule", [])) or "no", x["final_action"],
                       x["final_limit_change"], yes(x["final_finding_loosens_a_limit"]), x["score"]]
                      for k, x in v["by_design"].items()]))
    return "\n".join(out)


# ------------------------------------------------------------------------------ T09

def t09(d: dict[str, Any], table: Table) -> str:
    a, b, c, t = d["a_safety_filter_boundaries"], d["b_twin_against_recordings"], d["c_preflight_matrix"], d["d_trials"]
    out = ["## T09 Trial pipeline for a simulation-trained policy\n",
           f"Ran {d['ran']} on the twin of {d['site']} at {d['control_rate_hz']:.0f} Hz in real time; `{d['model']}` writes the "
           f"summary. Needle: {d['needle']}. Not measured: {d['not_measured']}. Evidence: `t09_trial_pipeline.json`.\n",
           f"Safety filter boundaries: {a['match']}/{a['cases']} cases behave as the limit says.\n",
           table(["Case", "Expected", "Got"], [[r["limit"], r["expected"], r["got"]] for r in a["rows"]]),
           f"The twin's kinematics against recordings. {b['what']} Over {b['episodes']} recordings the median residual is "
           f"{b['median_of_the_median_residuals_mm']} mm and the largest 95th percentile {b['largest_p95_residual_mm']} mm.\n",
           table(["Recording", "Samples used", "Without alignment, median (mm)", "After alignment: median / p95 / max (mm)",
                  "Rotation of the fitted transform (deg)", "Its translation (mm)"],
                 [[r["episode"], r["samples_used"], r["difference_without_alignment_mm_median"],
                   f"{r['residual_after_one_rigid_alignment_mm']['median']} / {r['residual_after_one_rigid_alignment_mm']['p95']} / "
                   f"{r['residual_after_one_rigid_alignment_mm']['max']}", r["rotation_of_the_alignment_deg"],
                   r["translation_of_the_alignment_mm"]] for r in b["rows"]]),
           f"Pre-flight matrix: {c['as_expected']}/{c['cases']} cases stopped at the expected check with the expected repair; "
           f"motions commanded over all cases: {c['motions_commanded_in_all_cases']}.\n",
           table(["Case", "Outcome", "Checks that failed", "Repair named", "Motions"],
                 [[r["case"], r["outcome"], "; ".join(r["failed_checks"]), ", ".join(r["repairs_named"]), r["motions_commanded"]]
                  for r in c["rows"]])]
    out.append(f"Trials end to end: {t['trials']} real expert rollouts replayed through the adapter and the safety filter. "
               f"{t['as_expected']}/{t['trials']} ended as the declared needle implies ("
               + ", ".join(f"{k}: {v}" for k, v in t["machine_verdicts"].items()) + "). Reach within 5 mm and 15 degrees: "
               f"{t['reach_within_5_mm_and_15_deg']}/{t['trials']} (largest error {t['reach_error_mm_max']} mm, "
               f"{t['reach_error_deg_max']} deg; {t['policy_actions_min_max'][0]} to {t['policy_actions_min_max'][1]} policy actions). "
               f"Motions rejected or stopped: {t['motions_rejected_or_stopped']}. Arm at the start pose at the end: "
               f"{t['arm_at_start_pose_at_the_end']}/{t['trials']}. Every run paused for the operator's confirmation: "
               f"{yes(t['every_run_paused_for_the_operators_confirmation'])}. Summaries by the model: "
               f"{t['summaries_written_by_the_model']}, replaced by the template: {t['summaries_replaced_by_the_template']}.\n")
    out.append(table(["Rollout", "Needle declared", "Route", "Machine verdict", "Reach error (mm / deg)", "Operator"],
                     [[r["rollout"],
                       ", ".join(k for k, v2 in r["needle_declared"].items() if (v2 if k != "grasp_succeeds" else not v2)).replace(
                           "grasp_succeeds", "grasp misses") or "holds",
                       " → ".join(r["route"]), r["machine_verdict"]["machine"],
                       f"{r['reach']['error_mm']} / {r['reach']['error_deg']}", (r["operator_verdict"] or {}).get("decision")]
                      for r in t["runs"]]))
    if t.get("override"):
        o = t["override"]
        out.append(f"Override: without a reason it was {o['without_a_reason']}; with one the operator's verdict is "
                   f"`{o['operator_verdict']['verdict']}` (\"{o['operator_verdict']['reason']}\") and the machine verdict beside it "
                   f"is still `{o['machine_verdict']}`.\n")
    r = t["rehearsal_rejection"]
    out.append(f"Rehearsal: on {r['site']} the same kind of rollout ended as `{r['outcome']}`, rejected in "
               f"{', '.join(r['rejected_in'])} by rule {', '.join(r['rule'])}; operator asked: {yes(r['operator_was_asked'])}; "
               f"motions commanded: {r['motions_commanded']}.\n")
    return "\n".join(out)


# ------------------------------------------------------------------------------ T10

def t10(d: dict[str, Any], table: Table) -> str:
    a, b, c, q, e, f = (d["a_contract_extraction"], d["b_compatibility"], d["c_adapter_round_trip"], d["d_episode_checks"],
                        d["e_negative_controls"], d["f_split"])
    out = ["## T10 Policy onboarding and episode data\n",
           f"Ran {d['ran']} with `{d['model']}`. SurgicAI tree `{d['surgicai_tree']}`; demonstrations read with "
           f"{d['demonstrations_read_with']}. Evidence: `t10_policy_and_data.json`, `t10_dataset_card.md`.\n",
           f"Contract extraction from {', '.join(f'`{k}` ({v} characters)' for k, v in a['files_given'].items())}: {a['fields']} "
           f"fields, {a['model_calls']} model call. Fields kept with a quotation found in the file: "
           f"{a['fields_with_a_quotation_found_in_the_file']}; emptied because the quotation was not in the file: "
           f"{a['fields_emptied_because_the_quotation_was_not_in_the_file']}. Against the hand-written reference: "
           + ", ".join(f"{k}: {v}" for k, v in a["against_the_reference"].items()) + ".\n",
           table(["Field", "Stated in the files", "Value kept", "Outcome", "Quotation, or what the model offered"],
                 [[r["field"], yes(r["stated_in_the_files"]), r["value"] if r["value"] is not None else "(empty)", r["outcome"],
                   (f"`{r['quote'][:70]}`" if r["quote"] else
                    (f"offered: {str(r['offered_by_the_model'])[:50]}" if r["offered_by_the_model"] else ""))] for r in a["rows"]])]
    m = a["filled_from_the_demonstrations"]
    out.append(f"Filled from the demonstrations, not by the model: step size per action component "
               f"{m['step_size_estimated_per_component']} (cm, rad, jaw); largest recorded step {m['largest_recorded_step']} cm and "
               f"{m['largest_recorded_rotation_step_deg']} deg; distinct start-to-goal displacements among the Approach episodes: "
               f"{m['distinct_start_to_goal_displacements']}.\n")
    rows = []
    for label, res in b.items():
        if label == "kinematic_files":
            continue
        rows.append([label, res["verdict"], "; ".join(res["failing"]) or "none",
                     (f"start {res['sub_region']['start_offset_from_site_start_mm']} mm from the site's start pose"
                      if res["sub_region"] else "")])
    out.append(table(["Site", "Verdict", "Failing", "Sub-region"], rows))
    out.append("Adapter round trip, every demonstration:\n")
    out.append(table(["Task", "Episodes", "Poses", "Largest position error (cm)", "Largest rotation error (rad)",
                      "Replayed actions end within (cm)", "All end in success"],
                     [[k, v["episodes"], v["poses"], f"{v['largest_position_error_cm']:.1e}", f"{v['largest_rotation_error_rad']:.1e}",
                       v["actions_replayed_from_the_start_end_within_cm"], yes(v["all_end_in_success"])]
                      for k, v in c["policy_to_site_and_back"].items()]))
    out.append(table(["Source", "Episodes", "Accepted", "Accepted with flags", "Excluded", "Flags",
                      "Streams the source does not contain"],
                     [[k, v["episodes"], v["accepted"], v["accepted_with_flags"], v["excluded"],
                       ", ".join(f"{n}: {x}" for n, x in v["flags"].items()) or "none", ", ".join(v["streams_absent"])]
                      for k, v in q["by_source"].items()]))
    out.append(f"Joint limits from {q['joint_limits_from']}.\n")
    out.append(f"Negative controls made from `{e['made_from']}`: all caught: {yes(e['all_caught'])}.\n")
    out.append(table(["Control", "Must be caught by", "Caught", "Verdict"],
                     [[x["control"], x["must_be_caught_by"], yes(x["caught"]), x["verdict"]] for x in e["controls"]]))
    out.append(f"Split by {f['by']}: training {', '.join(f['train_operators'])} ({f['train_episodes']} episodes), evaluation "
               f"{', '.join(f['evaluation_operators'])} ({f['evaluation_episodes']} episodes); operators on both sides: "
               f"{f['overlap_of_operators']}. Sources without an operator: {', '.join(f['without_an_operator'])}; {f['note']}.\n")
    out.append(table(["Role", "episode_quality tool"], [[r["role"], r["episode_quality"]] for r in d["dataset_tool_by_role"]]))
    return "\n".join(out)


SECTIONS = [("t08_multi_agent", t08), ("t09_trial_pipeline", t09), ("t10_policy_and_data", t10)]
