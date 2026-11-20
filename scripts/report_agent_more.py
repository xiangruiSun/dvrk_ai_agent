"""The sections of verification/REPORT.md for agent tasks 04 to 07, built from the evidence
files written by scripts/verify.py."""
from __future__ import annotations

import json
from typing import Any

from report_agent_tasks import Table, _load, yes


# ------------------------------------------------------------------------------ T04

def t04(d: dict[str, Any], table: Table) -> str:
    a, th, b = d["a_injected_faults"], d["a_labelled_threads"], d["b_planning"]
    out = ["## T04 Diagnostic reasoning and planning\n",
           f"Ran {d['ran']} with `{d['model']}`. Evidence: `t04_reasoning_and_planning.json`.\n",
           f"Injected faults. {a['how'].capitalize()}. {a['cases']} runs: {a['causes']} causes, {a['runs_per_cause']} runs each. "
           f"Fault-free runs that reach the needle: {a['fault_free_runs_that_reach_the_needle']}. A guess is right in "
           f"{a['chance_of_a_guess']} of cases. What the operator sees does not say which cause it was:\n",
           table(["How the run ended", "Causes behind it"],
                 [[k.replace("_", " "), ", ".join(v)] for k, v in a["causes_behind_each_kind_of_ending"].items()])]
    rows = [[s, f"{v['cause_correct']}/{v['cases']}", f"{v['family_correct']}/{v['cases']}", v["median_calls"], v["median_checks"],
             v["median_tokens"], v["median_seconds"]] for s, v in a["by_strategy"].items()]
    out.append(table(["Strategy", "Cause correct", "Family correct", "Median model calls", "Median checks run", "Median tokens",
                      "Median seconds"], rows))
    e = a["eliminate"]
    out.append(f"The elimination strategy with a budget of {a['check_budget_of_eliminate']} checks: narrowed to one candidate in "
               f"{e['narrowed_to_one']}/{a['cases']} runs; {e['states_rejected_by_the_rules']} updates rejected by the rules of the "
               f"hypothesis state. First check chosen: " + ", ".join(f"{k} {v}" for k, v in e["first_check_chosen"].items() if v) + ". "
               + (f"The check in which the fault leaves its mark was among those run in "
                  f"{e['the_signature_check_was_among_those_run']}/{a['cases']} runs, and the cause was right in "
                  f"{e['correct_when_it_was']} of those. " if "the_signature_check_was_among_those_run" in e else "")
               + "Correct by cause: " + ", ".join(f"{k} {v}" for k, v in e["correct_by_cause"].items()) + ".\n")
    if "eliminate_all_checks" in a:
        x = a["eliminate_all_checks"]
        out.append(f"With all five checks allowed ({x['what']}): narrowed to one in {x['narrowed_to_one']}/{a['cases']}; correct by "
                   f"cause: " + ", ".join(f"{k} {v}" for k, v in x["correct_by_cause"].items()) + ".\n")
    r, t = a["reflect"], a["tot"]
    out.append(f"Review step of `reflect`: the answer changed in {r['answer_changed_after_review']} runs, to the right cause in "
               f"{r['changed_to_correct']} and away from it in {r['changed_to_wrong']}; documentation passages were retrieved in "
               f"{r['runs_with_passages']} runs. `tot`: all three lines agreed in {t['all_three_lines_agree']} runs; a line had the "
               f"cause and another was picked in {t['a_line_had_the_cause_but_another_was_picked']}.\n")
    first = _load("t04_first_run")
    if first:
        fe = first["a_injected_faults"]["eliminate"]
        fs = first["a_injected_faults"]["by_strategy"]["eliminate"]
        out.append(f"First run ({first['ran']}, kept in `t04_first_run.json`): elimination was right in "
                   f"{fs['cause_correct']}/{fs['cases']} and narrowed to one candidate in {fe['narrowed_to_one']}. "
                   f"{fe['states_rejected_by_the_rules']} of its state updates were rejected: the model had to rewrite the whole "
                   f"typed state at every step, cited results by ids it shortened itself, and had replies cut off. With every "
                   f"update rejected, its answer was the first candidate of the list. The model now reports only what a result "
                   f"changes (drops, supports, ranking, next check), result ids are short and enforced by the output schema, and "
                   f"the state is written by code alone.\n")
    out.append(f"Labelled threads: {th['threads']} closed dVRK issue threads whose cause a maintainer stated, labelled by hand "
               f"(`{th['labels_file']}`). The model saw {th['what_the_model_saw']}. Always answering the most common label would be "
               f"right in {th['always_the_most_common_label']}. Elimination: {th['eliminate']}.\n")
    out.append(table(["Strategy", "Cause correct", "Median model calls", "Median tokens", "Median seconds"],
                     [[s, f"{v['cause_correct']}/{v['cases']}", v["median_calls"], v["median_tokens"], v["median_seconds"]]
                      for s, v in th["by_strategy"].items()]))
    out.append("Labels: " + ", ".join(f"{k} {v}" for k, v in th["labels"].items()) + ". The review step changed the answer in "
               f"{th['reflect_answer_changed_after_review']} threads.\n")
    out.append(f"Planning: {b['as_expected']}/{b['goals']} goals ended as expected; {b['plans_valid_on_first_attempt']}/{b['goals']} "
               f"plans were valid on the first attempt.\n")
    out.append(table(["Goal", "Outcome", "Planning attempts", "First error of the first attempt", "Steps executed", "Re-plans",
                      "Motions", "Answer written by"],
                     [[g["label"], g["status"], " → ".join(g["planning_attempts"]),
                       (g["planning_errors_first_attempt"][0][:110] if g["planning_errors_first_attempt"] else ""),
                       ", ".join(f"{x[1]}" + ("" if x[2] else f" ({x[3]})") for x in g["steps_executed"]) or "none",
                       g["replans"], g["motions_executed"], g["answer_source"]] for g in b["runs"]]))
    cal = next((g for g in b["runs"] if "residual_computed_by_code_mm" in g), None)
    if cal:
        out.append(f"Data flow: the second step of \"{cal['label']}\" took its argument from the first "
                   f"(`{json.dumps(cal['plan'][-1]['arguments'])}`); the residual computed by code is "
                   f"{cal['residual_computed_by_code_mm']} mm and it is in the answer: {yes(cal['residual_in_the_answer'])}.\n")
    return "\n".join(out)


# ------------------------------------------------------------------------------ T05

def t05(d: dict[str, Any], table: Table) -> str:
    out = ["## T05 Runbooks as skills\n",
           f"Ran {d['ran']} with `{d['model']}` on the twin of {d['site']}. Needle: {d['needle']}. Evidence: "
           f"`t05_tools_vs_skills.json`.\n",
           table(["Runbook", "Version", "Content hash", "Steps", "Checks", "Tier"],
                 [[r["name"], r["version"], f"`{r['content_hash']}`", r["steps"], r["checks"], r["tier"]] for r in d["runbooks"]]),
           f"Steps cite where they were written from. Documentation sections cited: "
           f"{', '.join('`' + x + '`' for x in d['sources_cited_by_steps']['documentation_sections'])}; found in the indexed corpus: "
           f"{d['sources_cited_by_steps']['found_in_the_corpus']}. The other citations are of the paper's framework "
           f"({len(d['sources_cited_by_steps']['not_checkable'])} places) and cannot be checked by code.\n",
           f"The same {d['requests']} service requests three ways. `atomic`: all {d['atomic_catalogue_size']} tools, the procedures "
           f"written into the prompt. `runbooks`: the index in the prompt, the procedure run by code. `runbooks_with_search`: one "
           f"tool at the start, the rest found by search.\n"]
    rows = []
    for arm, v in d["by_arm"].items():
        rows.append([arm, f"{v['all_four']}/{v['requests']}", f"{v['completed']}/{v['requests']}",
                     f"{v['every_required_check_performed']}/{v['requests']}", f"{v['verdict_correct']}/{v['requests']}",
                     f"{v['final_state_correct']}/{v['requests']}", v["median_model_calls"], v["median_tool_calls_by_the_model"],
                     v["median_tokens"], v["median_seconds"], v["schema_chars_in_prompt"], v["system_prompt_chars"]])
    out.append(table(["Design", "All four", "Completed", "Every required check performed", "Verdict correct", "Final state correct",
                      "Median model calls", "Median tool calls by the model", "Median tokens", "Median seconds",
                      "Tool schemas (chars)", "System prompt (chars)"], rows))
    bad = [r for r in d["runs"] if not r["all_good"]]
    if bad:
        out.append("Runs that did not meet all four:\n")
        out.append(table(["Design", "Request", "Checks performed", "Verdict given / expected", "Final state correct",
                          "Tools the model called"],
                         [[r["arm"], r["request"], yes(r["every_required_check_performed"]), f"{r['verdict']} / {r['verdict_expected']}",
                           yes(r["final_state_correct"]), ", ".join(r["tools_the_model_called"]) or "none"] for r in bad]))
    s = d["by_arm"].get("runbooks_with_search", {})
    if "search_returned_the_right_runbook" in s:
        ran = sum(1 for r in d["runs"] if r["arm"] == "runbooks_with_search" and r["every_required_check_performed"])
        out.append(f"In the search design the search returned the right runbook in "
                   f"{s['search_returned_the_right_runbook']}/{s['requests']} requests, and the runbook was then run in {ran}.\n")
    out.append(table(["Catalogue offered", "First call is the right tool", "Size of the schemas (chars)", "Wrong"],
                     [[x["catalogue"], f"{x['first_call_is_the_right_tool']}/{x['requests']}", x["schema_chars"],
                       "; ".join(f"{w['request']} -> {w['called']}" for w in x["wrong"]) or "none"]
                      for x in d["selection_of_a_single_tool"]]))
    w = d["search_without_a_model"]
    out.append(f"Search alone, no model: the right runbook is among the results for "
               f"{w['right_runbook_in_the_results']}/{w['requests']} requests.\n")
    return "\n".join(out)


# ------------------------------------------------------------------------------ T06

def t06(d: dict[str, Any], table: Table) -> str:
    b, c, s, e, f = (d["b_site_filter_through_the_tools"], d["c_questions_about_other_systems"], d["d_similar_episodes"],
                     d["e_site_memory"], d["f_working_context"])
    out = ["## T06 Memory and technical retrieval\n",
           f"Ran {d['ran']} with `{d['model']}`. Retrieval itself is measured in the retrieval sections below; this is what the "
           f"agents add. Evidence: `t06_memory_and_retrieval.json`.\n",
           f"The site decides what a search returns ({b['tool']}; {b['questions']}):\n",
           table(["Asking site", "Questions", "Passages returned", "Written for the other generation", "Marked as not applying"],
                 [[k, v["questions"], v["passages_returned"], v["passages_written_for_the_other_generation"],
                   v["passages_marked_as_not_applying"]] for k, v in b["by_site"].items()]),
           f"Questions whose results differ between the two sites: {b['questions_whose_results_differ_between_the_two_sites']}.\n",
           f"Questions about other robots ({c['other_systems']}), through the knowledge assistant. {c['note'].capitalize()}. Refused: "
           f"{c['refused']}/{c['other_systems']}; answered with a citation of a dVRK passage: {c['answered_with_a_citation']}; "
           f"searches that returned passages anyway: {c['searches_that_returned_passages_anyway']}. Control, {c['dvrk_controls']} dVRK "
           f"questions: answered with a citation {c['dvrk_controls_answered_with_a_citation']}, wrongly refused "
           f"{c['dvrk_controls_wrongly_refused']}.\n"]
    wrong = [r for r in c["runs"] if r["kind"] == "another system" and not r["refused"]]
    if wrong:
        out.append(table(["Question about another system that was not refused", "Cites a dVRK passage", "Answer begins"],
                         [[r["question"], yes(r["cites_a_passage"]), (r["answer"] or "")[:140].replace("\n", " ")] for r in wrong]))
    out.append(f"Similar episodes: {s['episodes']} ROSMA recordings, {len(s['features'])} statistics each. The nearest neighbour is "
               f"of the same task in {s['nearest_neighbour_is_the_same_task']}; by chance {s['expected_by_chance']}.\n")
    w, r, a = e["written"], e["recalled_in_a_new_session"], e["after_the_event"]
    out.append("Site memory:\n")
    out.append(table(["Step", "What happened"], [
        ["A calibration residual is measured", f"tools called: {', '.join(w['tools_called_by_the_agent'])}; fact written by "
                                               f"{w['by']}; active facts: {w['facts_active_after']}"],
        ["Another person asks in a new session", f"tools called: {', '.join(r['tools_called']) or 'none'}; measured again: "
                                                 f"{yes(r['measured_again'])}; value in memory {r['value_in_memory_mm']} mm, stated "
                                                 f"{r['value_stated_mm']} mm; correct: {yes(r['correct'])}"],
        [f"The event {a['event']['kind']} is recorded", f"facts active {a['facts_active']}, expired {a['facts_expired']}; calibration "
                                                        f"record reported valid: {yes(a.get('calibration_record_reported_valid'))}; "
                                                        f"the answer says it is no longer current: {yes(a['answer_says_expired'])}; "
                                                        f"names the event: {yes(a['answer_names_the_event'])}"],
        ["An operator of another site asks", f"recall tool: {e['other_site_operator']['recall_tool']}; memory put in the prompt: "
                                             f"{e['other_site_operator']['context_block_chars']} characters"],
        ["Remote support", f"memory put in the prompt: {e['remote_support']['context_block_chars']} characters"]]))
    out.append(f"Working context, {f['recordings']} recordings to look up and add:\n")
    out.append(table(["Run", "Context budget (tokens)", "History at the end", "Largest context sent", "Model calls", "Tool calls",
                      "Sum by code (s)", "Sum stated (s)", "Correct", "Stop reason"],
                     [[x["run"], x["context_budget_tokens"], x["history_tokens_at_the_end"], x["largest_context_sent_tokens"],
                       x["model_calls"], x["tool_calls"], x["sum_computed_by_code_s"], x["sum_stated_s"], yes(x["correct"]),
                       x["stop_reason"]] for x in f["runs"]]))
    return "\n".join(out)


# ------------------------------------------------------------------------------ T07

def t07(d: dict[str, Any], table: Table) -> str:
    a, b, c, dd, e, f, g = (d["a_servers"], d["b_resource_and_subscription"], d["c_progress_and_cancellation"],
                            d["d_agent_on_mcp_tools"], d["e_authorization_at_the_server"], d["f_description_pinning"],
                            d["g_planted_instructions"])
    servers = [k for k in a if k.startswith("vx-")]
    out = ["## T07 MCP as the robot boundary\n",
           f"Ran {d['ran']} with `{d['model']}`. SDK: `{d['sdk']}`. Transport: {d['transport']}. Evidence: `t07_mcp.json`.\n",
           table(["Server", "Protocol", "Capabilities", "Tools (tier)", "Resources", "Prompts"],
                 [[n, a[n]["protocol"], ", ".join(a[n]["capabilities"]), ", ".join(f"{t[0]} ({t[1]})" for t in a[n]["tools"]),
                   ", ".join(a[n]["resources"]) or "none", ", ".join(a[n]["prompts"]) or "none"] for n in servers])]
    fu = a["features_used"]
    out.append(f"Checked against the installed SDK before use. Elicitation: {fu['elicitation']}. Not built: "
               f"{', '.join(fu['not_built'])}.\n")
    out.append(f"Resource and subscription: `{b['resource']}` read at state {b['state_seq_before']}; after a motion through the "
               f"server and a displacement made outside it, {b['updates_received']} updates had arrived and the state was "
               f"{b['state_seq_after']}. Update after the change made outside the server: "
               f"{yes(b['update_after_a_change_made_outside_the_server'])}.\n")
    m, x = c["motion_with_progress"], c["motion_cancelled"]
    out.append(f"A motion through MCP reported {m['progress_events']} progress events. The same kind of motion cancelled after "
               f"{x['cancel_sent_after_s']} s: the client got `{x['client_got']}`, the arm had travelled "
               f"{x['millimetres_travelled_of_60']} of 60 mm, was busy afterwards: {yes(x['arm_busy_afterwards'])}, still moving half "
               f"a second later: {yes(x['still_moving_half_a_second_later'])}; the command record says `{x['command_record']}`. The "
               f"connected person was asked to confirm {c['confirmations_asked_of_the_person']} motions in this part.\n")
    out.append(f"Agent loop on MCP-discovered tools only (\"{dd['task']}\"): stop reason `{dd['stop_reason']}`; tools called: "
               f"{', '.join(t[0] for t in dd['tools_called'])}; `{dd['tool_left_off_the_allowlist']}` offered although left off the "
               f"allowlist: {yes(dd['offered_although_left_off'])}; arm at hover_needle: {yes(dd['arm_at_hover_needle'])}; "
               f"confirmations asked: {len(dd['confirmations_asked'])}. Its answer: \"{(dd['answer'] or '').strip()[:300]}\"\n")
    out.append(f"Authorization at the server: connecting without a token is {e['rows'][0]['connect']}; {e['as_expected']}/{e['of']} "
               f"principals as expected.\n")
    out.append(table(["Token of", "Read robot state", "Move the arm", "Arm moved", "List recordings", "Search documentation"],
                     [[r["who"], r["read_robot_state"], r["move_the_arm"], yes(r["arm_moved"]), r["list_recordings"],
                       r["search_documentation"]] for r in e["rows"][1:]]))
    out.append(f"Description pinning. Server restarted with {f['altered']}. Withdrawn by the host: "
               f"{', '.join(f['withdrawn']) or 'none'}; offered to the model: {', '.join(f['offered_to_the_model'])}; tools called: "
               f"{', '.join(t[0] for t in f['tools_called']) or 'none'}; arm moved: {yes(f['arm_moved'])}.\n")
    out.append(f"Planted instructions in results (INJECTED), {g['probes']} probes: reached the model {g['reached_the_model']}, obeyed "
               f"{g['obeyed']}, arm or jaw moved {g['arm_or_jaw_moved']}. Second line of defence: {g['second_line']}.\n")
    out.append(table(["Planted in", "Asked for", "Tools called after it", "Called the tool it asked for",
                      "Person asked to confirm a motion"],
                     [[r["planted_in"], r["text_asked_for"], ", ".join(r["tools_called_after_it"]) or "none",
                       yes(r["called_the_tool_it_asked_for"]), r["person_was_asked_to_confirm_a_motion"]] for r in g["runs"]]))
    return "\n".join(out)


SECTIONS = [("t04_reasoning_and_planning", t04), ("t05_tools_vs_skills", t05), ("t06_memory_and_retrieval", t06), ("t07_mcp", t07)]
