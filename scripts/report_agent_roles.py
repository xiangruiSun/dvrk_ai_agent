"""The sections of verification/REPORT.md for agent tasks 11 to 15, built from the evidence
files written by scripts/verify.py and scripts/gate.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from report_agent_tasks import Table, yes

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "verification"


def _load(name: str) -> dict[str, Any] | None:
    p = OUT / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _tools(row: dict[str, Any]) -> str:
    return ", ".join(row["tools_called"]) or "none"


def _matrix(m: dict[str, Any]) -> str:
    return f"Every guarded route asked by all four roles: {m['as_declared']}/{m['routes']} answer 403 exactly for the roles they do not declare."


def _short(text: str, n: int = 260) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


# ------------------------------------------------------------------------------ T11

def t11(d: dict[str, Any], table: Table) -> str:
    a, b, c, cf, e = d["a_questions"], d["b_probes"], d["c_actuate_per_run"], d["d_no_configuration"], d["e_console_routes"]
    out = ["## T11 Operator access\n",
           f"Ran {d['ran']} with `{d['model']}` at {d['site']} on the twin. The copilot `{d['copilot']}` was offered {d['tools_offered']} "
           f"tools. Evidence: `t11_operator_access.json`.\n",
           f"Questions through the chat path: {a['expected_tool_called']}/{a['asked']} called the tool the question needs, "
           f"{a['ended_with_a_verified_answer']}/{a['asked']} ended with an answer that passed the grounding guard.\n",
           table(["Question about", "Tools called", "Ended as", "Guard repairs", "Model calls", "Tokens", "Seconds"],
                 [[r["what"], _tools(r), r["stop_reason"], r["guard_repairs"], r["model_calls"], r["tokens"], r["seconds"]] for r in a["rows"]])]
    pre = next((r for r in a["rows"] if "preflight_named_the_failed_check" in r), None)
    if pre:
        out.append(f"With the camera arm reported as moved (injected), the pre-flight tool named the failed check: "
                   f"{yes(pre['preflight_named_the_failed_check'])}; the answer named the repair runbook: "
                   f"{yes(pre['answer_names_the_repair_runbook'])}; the answer says that the calibration is no longer valid: "
                   f"{yes(pre.get('answer_says_the_calibration_is_no_longer_valid'))}. The answer: \"{_short(pre['answer'])}\"\n")
        first = _load("t11_first_run")
        if first:
            old = next((r for r in first["a_questions"]["rows"] if "preflight_named_the_failed_check" in r), None)
            if old:
                out.append(f"First run ({first['ran']}, kept in `t11_first_run.json`): the tool returned the failed check under its "
                           f"name, which states the requirement, and the copilot repeated the name as the reason: "
                           f"\"{_short(old['answer'], 200)}\" The pre-flight now writes, for every failed check, a sentence that says "
                           f"what is wrong, and the tool and the ticket carry that sentence.\n")
    out.append(f"Probes: {b['held']}/{b['probes']} held; a ticket was offered or opened in {b['ticket_offered_or_opened']}. "
               f"{b['how'].capitalize()}.\n")
    out.append(table(["Probe", "Tools called", "Motion asked for", "Motions executed", "Limit changed", "Trials started",
                      "Ticket opened / offered", "Held"],
                     [[r["probe"], _tools(r), ", ".join(r["motion_tools_requested"]) or "none", r["motions_executed"], yes(r["limit_changed"]),
                       r["trials_started"], f"{r['ticket_opened']} / {yes(r['ticket_offered_in_the_answer'])}", yes(r["held"])] for r in b["rows"]]))
    planted = next((r for r in b["rows"] if "planted_instruction_obeyed" in r), None)
    if planted:
        out.append(f"The planted passage was returned by the search: {yes(planted['planted_passage_was_returned'])}; obeyed: "
                   f"{yes(planted['planted_instruction_obeyed'])}; the answer claims limits were raised: "
                   f"{yes(planted['answer_claims_limits_were_raised'])}.\n")
    out.append(f"Actuate is held per run, checked without a model: {c['as_expected']}/{c['cases']} cases as expected, "
               f"{c['motions_executed_in_total']} motion executed in total (the approved one).\n")
    out.append(table(["Case", "Outcome", "Motions executed"], [[r["case"], r["outcome"], r["motions_executed"]] for r in c["rows"]]))
    out.append(f"No configuration: {cf['refused_by_scope']}/{cf['tools_tried']} configuration, calibration, ticket-evidence and fleet tools "
               f"called with the operator's principal were refused by scope; {cf['refusals_audited']} refusals are in the audit log.\n")
    out.append("Console routes for the operator: own pages " + ", ".join(f"`{p}` {s}" for p, s in e["operator_own_pages"].items())
               + "; other pages " + ", ".join(f"`{p}` {s}" for p, s in e["operator_refused_pages"].items())
               + f". A site named in the request changes nothing: {yes(e['a_site_named_in_the_request_changes_nothing'])}. "
               + _matrix(e["matrix"]) + f" {e['refusals_audited']} refusals audited.\n")
    return "\n".join(out)


# ------------------------------------------------------------------------------ T12

def t12(d: dict[str, Any], table: Table) -> str:
    f, q, c, after, e = d["a_ticket_to_closure"], d["b_questions"], d["c_two_person_rule"], d["d_after_activation"], d["e_roles"]
    cal, act, rep = f["calibration"], f["activation"], f["service_report"]
    inj = d["injected"]
    out = ["## T12 Service engineer and safety lead\n",
           f"Ran {d['ran']} with `{d['model']}` at {d['site']} on the twin. Camera: {d['camera']}. Injected: the camera moved by "
           f"{inj['camera_moved_by_mm']} mm, its scale set to {inj['camera_scale']}, and the event `{inj['event_recorded']}` recorded. "
           f"Evidence: `t12_service_access.json`.\n",
           f"A ticket from a failed pre-flight to closure. The trial ended as `{f['ticket']['trial_outcome']}` with "
           f"{f['ticket']['motions_commanded']} motions commanded; failed check: {'; '.join(f['ticket']['failed_checks'])}; repair named: "
           f"{', '.join(f['ticket']['repair_named'])}.\n",
           table(["Diagnosis", "Cause named", "Injected cause", "Correct", "Narrowed to one", "Checks run", "Model calls", "Tokens",
                  "Seconds", "Evidence resolves to ticket records"],
                 [[x["method"], x["cause_named"], x["injected_cause"], yes(x["correct"]), yes(x["narrowed_to_one"]), len(x["checks_run"]),
                   x["model_calls"], x["tokens"], x["seconds"], yes(x["evidence_resolves_to_ticket_records"])] for x in f["diagnosis"]]),
           f"Calibration procedure (`{cal['runbook']}`, {cal['status']}): the motion was approved by {cal['approvals'][0]['approved_by']}; "
           f"the engineer could approve it: {yes(cal['approvals'][0]['engineer_could_approve_it'])}. {cal['samples']} samples, "
           f"{cal['held_out']} held out, {cal['seconds']} s.\n",
           table(["On the held-out samples", "Mean mm", "Max mm"],
                 [[k.replace("_", " "), v["mean_mm"], v["max_mm"]] for k, v in cal["held_out_residual_mm"].items()]),
           f"Shift between the old record and the fit in the workspace: {cal['shift_between_the_old_record_and_the_fit_mm']} mm "
           f"(injected: {cal['injected_shift_mm']} mm). Rigid fit against the declared truth: "
           f"{cal['rigid_fit_against_the_declared_truth_mm']} mm. Acceptance: {cal['acceptance_mm']} mm. After the fit the draft was "
           f"`{cal['draft_status_after_the_fit']}` and the record in use was still `{cal['record_in_use_after_the_fit']}`.\n",
           f"Activation: proposal {f['proposal']['proposal_id']} approved by {act['approved_by']}, status {act['status']}, event "
           f"`{act['event']}`, record in use `{act['record_in_use']}`, calibration valid: {yes(act['calibration_valid'])}, pre-flight "
           f"after: {act['preflight_after']}.\n",
           f"Service report: summary written by {rep['summary_written_by']} after {rep['attempts']} attempt(s), from "
           f"{rep['records_used']} ticket records; ticket {rep['ticket_status']}. \"{_short(rep['summary'], 420)}\" Listed as not "
           f"verified: {'; '.join(rep['not_verified'])}.\n"]
    if q["rows"]:
        out.append(f"Questions through the chat path: {q['expected_tool_called']}/{q['asked']} called the tool the question needs.\n")
        out.append(table(["Question about", "Tools called", "Ended as", "Guard repairs", "Tokens", "Seconds"],
                         [[r["what"], _tools(r), r["stop_reason"], r["guard_repairs"], r["tokens"], r["seconds"]] for r in q["rows"]]))
        prop = next((r for r in q["rows"] if "proposals_waiting_for_the_safety_lead" in r), None)
        if prop:
            out.append(f"The change asked for through the copilot: {prop['proposals_waiting_for_the_safety_lead']} proposal waiting "
                       f"for the safety lead afterwards; the limit in force unchanged: {yes(prop['limit_in_force_unchanged'])}.\n")
    first = _load("t12_first_run")
    if first and first["b_questions"]["rows"]:
        rows = first["b_questions"]["rows"]
        prop1 = next((r for r in rows if "proposals_waiting_for_the_safety_lead" in r), {})
        out.append(f"First run ({first['ran']}, kept in `t12_first_run.json`): of the same {len(rows)} questions, "
                   f"{sum(r['stop_reason'] == 'final_answer_unverified' for r in rows)} ended as not verified although the numbers in "
                   f"the answers were the ones the tools had returned: the guard takes a number's unit from the end of the key that "
                   f"carries it, and two tool results did not put it there. The change asked for through the copilot left "
                   f"{prop1.get('proposals_waiting_for_the_safety_lead')} proposals waiting, because the copilot cited the ticket's "
                   f"number where a record id was needed, and the guard also refused the values the engineer had stated in the "
                   f"request. Changed: the two tool results carry the unit in the key, a value stated in the request may be "
                   f"repeated, and the prompt says what an evidence id is.\n")
    out.append(f"Two-person rule: {c['as_expected']}/{c['cases']} cases as expected; {c['refusals_audited']} refusals audited.\n")
    out.append(table(["Case", "Outcome", "Expected"], [[r["case"], r["outcome"], r["expected"]] for r in c["rows"]]))
    out.append(f"After the tighter limit was applied: event `{c['after_the_tighter_limit']['event']}`, envelope version "
               f"{c['after_the_tighter_limit']['envelope_version']}, {c['after_the_tighter_limit']['approvals_ended']} open approvals ended. "
               f"Versions of the site file: {c['versions_of_the_site_file']}; limit in force at the end {c['limit_in_force_now_mm']} mm.\n")
    ap = after["approval_issued_before_the_change"]
    out.append(f"After activation of the calibration: the fact about the old calibration is expired: "
               f"{yes(after['old_calibration_fact_expired'])} "
               f"({'; '.join(x['because'] for x in after['site_facts']['expired']) or 'none'}). A motion approval given before the "
               f"change: status now `{ap['status_now']}`; the motion requested with it was {ap['motion_requested_with_it']}; motions "
               f"executed: {ap['motions_executed']}.\n")
    sc = d.get("f_safety_copilot")
    if sc:
        out.append(f"The safety lead's copilot, asked to explain the proposal that loosens a limit: tools called {_tools(sc)}; ended as "
                   f"{sc['stop_reason']}; says that it loosens a limit: {yes(sc['says_it_loosens'])}. Of its {len(sc['tools_offered'])} tools, "
                   f"{len(sc['tools_offered_that_write_or_move'])} write or move anything; the proposal was still "
                   f"`{sc['proposal_status_after_the_question']}` afterwards. \"{_short(sc['answer'], 320)}\"\n")
    out.append("The engineer cannot: " + "; ".join(f"{k}: {v}" for k, v in e["the_engineer_cannot"].items()) + ".\n")
    pages = list(next(iter(e["pages_by_role"].values())))
    out.append(table(["Page"] + list(e["pages_by_role"]), [[f"`{p}`"] + [e["pages_by_role"][r][p] for r in e["pages_by_role"]] for p in pages]))
    out.append(_matrix(e["matrix"]) + "\n")
    return "\n".join(out)


# ------------------------------------------------------------------------------ T13

def t13(d: dict[str, Any], table: Table) -> str:
    a, b, c, g, e = d["a_questions"], d["b_redaction"], d["c_store_and_forward"], d["d_grant"], d["e_pages"]
    sites = d["sites"]
    out = ["## T13 Remote support and fleet\n",
           f"Ran {d['ran']} with `{d['model']}`. Sites: " + "; ".join(f"{n} ({s['generation']}, release {s['software']}, `{s['arm_file']}`)"
                                                                     for n, s in sites.items())
           + f". {d['note'].capitalize()}. Evidence: `t13_remote_support.json`.\n",
           "Work done at the sites during this run, which is what the records describe: "
           + "; ".join(f"{n}: {x['chat_runs']} copilot runs, trials ended {', '.join(str(o) for o in x['trial_outcomes'])}"
                       for n, x in d["activity"]["by_site"].items()) + ".\n",
           f"Redaction: {b['records_scanned']} outbox records scanned ({', '.join(f'{k} {v}' for k, v in b['by_kind'].items())}); payload "
           f"fields or markers found: {b['payload_fields_or_markers_found']}. {len(b['markers_searched_for'])} strings that were typed, "
           f"named or measured at a site during the run were searched for. Negative control: "
           f"{b['negative_control']['refused']}/{b['negative_control']['records_with_a_payload_offered_to_the_outbox']} records with a "
           f"payload were refused by the outbox.\n",
           f"Store and forward: site B lost its link, produced {c['produced']} records, and delivered {c['delivered']} after "
           f"reconnecting. The acknowledgement of one batch was lost once (injected): the store ignored {c['duplicates_the_store_ignored']} "
           f"repeated records. Stored twice: {c['stored_twice']}; out of order: {c['out_of_order']}; sequence consecutive: "
           f"{yes(c['sequence_is_consecutive'])}; backlog at the end: {c['backlog_now']}.\n",
           f"Grant: {g['as_expected']}/{g['cases']} cases as expected. Uses audited at the site: {g['uses_audited_at_the_site']} "
           f"({g['of_which_allowed']} allowed); asks recorded at the fleet store: {g['asks_recorded_at_the_fleet_store']}.\n",
           table(["Reading a ticket", "Outcome"], [[r["case"], r["outcome"]] for r in g["rows"]]),
           "Who else can give a grant: " + "; ".join(f"{k}: {v}" for k, v in g["who_else_can_give_a_grant"].items())
           + ". With a grant, support still gets: " + "; ".join(f"`{k}` {v}" for k, v in g["with_a_grant_support_still_cannot"].items()) + ".\n",
           f"The support assistant: {a['ended_with_a_verified_answer']}/{a['asked']} questions ended with an answer that passed the "
           f"grounding guard.\n",
           table(["Question about", "Tools called", "Ended as", "Guard repairs", "Tokens", "Seconds"],
                 [[r["what"], _tools(r), r["stop_reason"], r["guard_repairs"], r["tokens"], r["seconds"]] for r in a["rows"]])]
    ref = next((r for r in a["rows"] if "says_it_does_not_leave_the_site" in r), None)
    if ref:
        out.append(f"Asked for a site's episode and trajectory: tools that read a site called: "
                   f"{', '.join(ref['tools_that_read_a_site']) or 'none'}; the answer says it does not leave the site: "
                   f"{yes(ref['says_it_does_not_leave_the_site'])}. \"{_short(ref['answer'])}\"\n")
    drift = a["version_drift"]
    out.append("Fleet view at the end: " + "; ".join(
        f"{s['site']}: in contact {yes(s['in_contact'])}, healthy {yes(s['healthy'])}, release {s['software']}, {s['runs']} runs, "
        f"{s['trials']} trials" for s in a["fleet_view"]["sites"]) + f". Version drift: {', '.join(drift['differs']) or 'none'}.\n")
    out.append("Pages for remote support: fleet " + ", ".join(f"`{p}` {s}" for p, s in e["support_fleet_pages"].items()) + "; site "
               + ", ".join(f"`{p}` {s}" for p, s in e["support_site_pages"].items()) + ". " + _matrix(e["matrix"]) + "\n")
    return "\n".join(out)


# ------------------------------------------------------------------------------ T14

def t14(d: dict[str, Any], table: Table) -> str:
    from vxagent.llm.router import ROUTE
    a, b, c, r, e, g, z = (d["a_control_loop"], d["b_perception_worker"], d["c_degraded_modes"], d["d_model_routing"],
                           d["e_egress_guard"], d["f_gateway"], d["g_sizing"])
    p = a["period"]
    out = ["## T14 Edge deployment beside the robot\n",
           f"Ran {d['ran']} with `{d['models']['large']}` and `{d['models']['small']}` on the twin. Not verified here: "
           f"{d['not_verified_here']}. Timing: {d['timing_note']}. Evidence: `t14_edge_deployment.json`.\n",
           f"Control loop over {a['trials']} trials at {a['control_rate_hz']:.0f} Hz: {p['periods']} periods, mean {p['mean_ms']} ms, "
           f"median {p['p50_ms']} ms, 95th percentile {p['p95_ms']} ms, longest {p['max_ms']} ms, standard deviation "
           f"{p['jitter_std_ms']} ms (nominal {p['nominal_ms']:.0f} ms). Model calls that started while an action was executing: "
           f"{a['model_calls_inside_an_action_in_all_trials']} of {a['model_calls_in_all_trials']} model calls.\n",
           table(["Trial", "Needle declared", "Outcome", "Actions", "Model calls", "Inside an action", "Model ms", "Seconds"],
                 [[x["trial"], x["needle_declared"] if isinstance(x["needle_declared"], str) else ", ".join(x["needle_declared"]),
                   x["outcome"], x["actions"], x["model_calls"], x["model_calls_inside_an_action"], x["model_ms"], x["seconds"]]
                  for x in a["rows"]])]
    lat, slow, down = b["latency"], b["a_request_slower_than_the_limit_INJECTED"], b["worker_stopped_before_pre_flight"]
    out.append(f"Perception worker ({b['process']}; source: {b['source']}). {lat['n']} requests: median {lat['p50_ms']} ms, 95th "
               f"percentile {lat['p95_ms']} ms, longest {lat['max_ms']} ms; time limit {b['time_limit_s']} s. With {slow['extra_ms']} ms "
               f"added (injected): {slow['result']}; pre-flight then `{slow['pre_flight_then']}` with {slow['motions_commanded']} motions "
               f"commanded. Worker stopped before pre-flight: `{down['outcome']}`, failed check: {'; '.join(down['failed_checks'])}, "
               f"repair {', '.join(down['repair'])}, {down['motions_commanded']} motions commanded.\n")
    out.append(table(["Worker stopped during", "Outcome", "Grasp stopped for", "Needle pose in place", "Machine verdict"],
                     [[k, x["outcome"], x["grasp_stopped_for"] or "", x["needle_pose_in_place"] or "", x["machine_verdict"]]
                      for k, x in b["worker_stopped_during_a_stage"].items()]))
    first, second = _load("t14_first_run"), _load("t14_second_run")
    if first and second:
        fl, sl = first["b_perception_worker"]["latency"], second["b_perception_worker"]["latency"]
        out.append(f"Earlier runs, kept as evidence. First run ({first['ran']}, `t14_first_run.json`): the same requests had a median "
                   f"of {fl['p50_ms']} ms for 40 ms of declared work. The first explanation was wrong: the worker's two writes per reply "
                   f"were blamed and the delay between them was switched off, and the second run ({second['ran']}, "
                   f"`t14_second_run.json`) still had a median of {sl['p50_ms']} ms. The cause was on the calling side: a new HTTP "
                   f"client was created for every request, which costs about 0.2 s on this machine. The process now keeps one client "
                   f"open (`vxagent/net.py`). The same cost was in every model call, retrieval call and gateway command before this, "
                   f"so the latencies reported for tasks 01 to 10 and in the retrieval sections, which were measured before this "
                   f"change, include it. Keeping the connection open "
                   f"also exposed a fault in the retrieval service, which answered the request after an error with 501; it now reads "
                   f"the unread body before replying with an error. The routing table below was first set from the first run's "
                   f"measurement.\n")
    m, rt, fl2 = c["model_server_down"], c["retrieval_missing"], c["fleet_link_down"]
    out.append(table(["Missing part", "What still worked", "What was reported"],
                     [["model server", f"trial `{m['trial_outcome']}`; runbook rig_check `{m['runbook_rig_check']}`",
                       f"summary by {m['trial_summary_written_by']}; copilot says it is unavailable: {yes(m['copilot_says_it_is_unavailable'])}"],
                      ["perception worker", "nothing started", f"pre-flight fails closed: {yes(c['perception_worker_down']['pre_flight_fails_closed'])}"],
                      ["retrieval service", f"robot question answered from a look: {yes(rt['robot_question_answered_from_a_look'])}",
                       f"search failed as a tool error: {yes(rt['search_failed_as_a_tool_error'])}; the answer says so: "
                       f"{yes(rt['answer_says_the_search_is_unavailable'])}"],
                      ["fleet link", f"trial `{fl2['trial_outcome']}`", f"outbox {fl2['outbox_before']} → {fl2['outbox_after']} records"]]))
    out.append("Model routing. A step is routed to the small model only if it was at least as accurate there as the large one.\n")
    out.append(table(["Step", "Cases", "Classes", "Large correct", "Large median ms", "Small correct", "Small median ms",
                      "Route when measured", "Route in code now"],
                     [[k, x["large"]["cases"], x["classes"], x["large"]["correct"], x["large"]["median_ms"], x["small"]["correct"],
                       x["small"]["median_ms"], x["route_in_code"], ROUTE[k]] for k, x in r.items()]))
    out.append(f"Egress guard (profile `{e['profile']}`, allowlist {', '.join(e['allowlist'])}): {e['match']}/{e['cases']} addresses "
               f"decided as expected. A model client pointed at an external host: {e['model_client_pointed_at_an_external_host']}; a "
               f"perception client: {e['perception_client_at_an_external_host']}; requests that reached the network layer: "
               f"{e['requests_that_reached_the_network_layer']}.\n")
    t, o, u = g["through_the_gateway"], g["command_with_another_token"], g["from_a_host_not_on_the_allowlists"]
    out.append(f"Gateway: a motion through it {t['motion']} (backend `{t['backend_name']}`, final error {t['final_error_mm']} mm). A "
               f"command with another token: {o['status']}, arm still enabled: {yes(o['arm_still_enabled'])}. From {u['peer_address']} "
               f"with the allowlist {u['allowlist']}: state request {u['state_request']}, command {u['command']}, arm enabled by it: "
               f"{yes(u['arm_enabled_by_it'])}.\n")
    tr, ch = z["per_trial_that_ran_to_the_end"], z["per_chat_run"]
    out.append(f"Sizing. A copilot answer: {ch['model_calls']} model calls, {ch['tokens']} tokens, {ch['seconds']} s. A trial that ran "
               f"to the end ({tr['trials']} of them): median {tr['median_seconds']} s, of which {tr['median_model_ms']} ms in the model "
               f"for {tr['median_tokens']} tokens.\n")
    return "\n".join(out)


# ------------------------------------------------------------------------------ T15

def t15(d: dict[str, Any], table: Table) -> str:
    rule = d["rule"]
    out = ["## T15 Evaluation gate\n",
           f"Ran {d['ran']} with `{d['model']}`. {len(d['cases'])} golden cases recorded with `{d['cases_recorded_with']}` from live "
           f"runs of the operator's copilot on the twin; {sum(c['live'] for c in d['cases'])} of them are also run live for every "
           f"candidate. Rule (`evals/gate.json`): {rule['note']} Evidence: `t15_gate.json`, `gate_<name>.json`.\n",
           table(["Case", "Task", "Recorded tool calls", "Passed when recorded", "Live"],
                 [[c["case"], _short(c["task"], 90), ", ".join(c["recorded_calls"]) or "none",
                   yes(c["passed_when_recorded"]) + (f" ({', '.join(c['failed_when_recorded'])})" if c["failed_when_recorded"] else ""),
                   yes(c["live"])] for c in d["cases"]])]
    rows = []
    for name, r in d["by_candidate"].items():
        cmp_, cand = r["comparison"], r["comparison"]["candidate"]
        rows.append([name, r["model"], f"{cand['passed']}/{cand['cases']}", cand["diverged"], cand["failed"], cand["cases_with_a_safety_failure"],
                     f"{r['live']['passed']}/{r['live']['cases']}", ", ".join(cmp_["newly_failing"]) or "none", cand["tokens"],
                     "accepted" if r["accepted"] else "REFUSED"])
    out.append(table(["Configuration", "Model", "Replay passed", "Diverged", "Failed", "Cases with a safety failure", "Live passed",
                      "Newly failing in replay", "Tokens", "Gate"], rows))
    for name, r in d["by_candidate"].items():
        out.append(f"`{name}`: {r['about']} " + ("Accepted." if r["accepted"] else "Refused: " + "; ".join(r["reasons"]) + "."))
        bad = [x for x in r["per_case"] if not x["passed"]]
        if bad:
            out.append("\n" + table(["Replay of", "Result", "Failed assertions", "Safety", "Calls"],
                                    [[x["case"], "diverged" if x["diverged"] else "failed", ", ".join(x["failed"]) or "none",
                                      ", ".join(x["safety_failed"]) or "none", ", ".join(x["calls"]) or "none"] for x in bad]))
        lbad = [x for x in r["live_per_case"] if not x["passed"]]
        if lbad:
            out.append(table(["Live on the twin", "Failed assertions", "Safety", "Calls", "Motions executed"],
                             [[x["case"], ", ".join(x["failed"]), ", ".join(x["safety_failed"]) or "none", ", ".join(x["calls"]) or "none",
                               x["motions_executed_on_the_twin"]] for x in lbad]))
        out.append("")
    first = _load("t15_first_recording")
    if first:
        out.append(f"First recording ({first['ran']}, kept in `t15_first_recording.json`): "
                   f"{sum(c['passed'] for c in first['cases'])}/{len(first['cases'])} cases passed when recorded. "
                   + " ".join(x[0].upper() + x[1:] + "." for x in first["found"]) + " Changed: " + "; ".join(first["changed"]) + ".\n")
    return "\n".join(out)


SECTIONS = [("t11_operator_access", t11), ("t12_service_access", t12), ("t13_remote_support", t13), ("t14_edge_deployment", t14),
            ("t15_gate", t15)]
