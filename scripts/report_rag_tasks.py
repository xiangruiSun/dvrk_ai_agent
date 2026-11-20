"""The sections of verification/REPORT.md for the retrieval tasks R05, R06, R07, R09 and R10,
built from the evidence files written by scripts/rag_verify.py."""
from __future__ import annotations

from typing import Any, Callable

import json
from collections import Counter
from pathlib import Path

Table = Callable[[list[str], list[list[Any]]], str]
VER = Path(__file__).resolve().parents[1] / "verification"


def first_run(name: str) -> dict[str, Any] | None:
    path = VER / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def yes(x: Any) -> str:
    return "yes" if x else "no"


def interval(d: dict[str, Any]) -> str:
    return f"{d['mean']} ({d['low']} to {d['high']}); {d['wins']} better, {d['losses']} worse, {d['ties']} equal"


def r05(d: dict[str, Any], table: Table) -> str:
    q, w, dep = d["questions_in_old_terms"], d["queries_as_written"], d["as_deployed"]
    out = ["## RAG R05: applicability (hardware generation, release, renamed things)\n",
           f"Ran {d['ran']}. Evidence: `rag_r05_applicability.json`. No model. {d['development_queries']} development "
           f"queries, `{d['strategy']}`.\n",
           f"Renames read from the change log ({d['releases_known']} releases, newest {d['latest_release']}):\n",
           table(["Old name", "New name", "Release", "Read from", "Identifier-shaped"],
                 [[f"`{r['old']}`", f"`{r['new']}`", r["renamed_in"], f"`{r['read_from']}`", yes(r["identifier_shaped"])]
                  for r in d["renames"]]),
           "Questions in old terms: development queries that use the new name of a renamed thing, rewritten by code "
           "to the old name. Every rename found in these queries is of a plain word "
           f"({', '.join(f'{k}: {v}' for k, v in q['by_rename'].items())}).\n",
           table(["Queries", "nDCG@10 as written", "In old terms", "In old terms, widened", "Widening against not widening"],
                 [[q["all"]["queries"], q["all"]["ndcg@10_original"], q["all"]["ndcg@10_in_old_terms"],
                   q["all"]["ndcg@10_in_old_terms_widened"], interval(q["all"]["widening_vs_not"])]]),
           f"The same widening applied to queries nobody rewrote changed {w['changed_by_widening']} of {w['of']} and "
           f"moved their nDCG@10 from {w['ndcg@10_before']} to {w['ndcg@10_after']}: {interval(w['difference'])}. "
           + ("The interval lies below zero, so plain words are not added to queries. " if w["difference"]["high"] < 0
              else "The interval includes zero. ") + "Deployed rule: "
           f"{dep['rule']}. It changes {dep['development_queries_changed']} development queries "
           f"({interval(dep['difference'])}).\n",
           "Hardware generation:\n",
           table(["Site generation", "Results from the other generation, unfiltered", "Filtered",
                  "Queries answered on a page of this generation", "Their nDCG@10 unfiltered", "Filtered"],
                 [[g, v["share_of_results_from_the_other_generation_unfiltered"],
                   v["share_of_results_from_the_other_generation_filtered"],
                   v["queries_answered_on_a_page_of_this_generation"], v["ndcg@10_unfiltered"], v["ndcg@10_filtered"]]
                  for g, v in d["hardware_generation"].items()]),
           "Release (the releases are those the four sites' files are written for):\n",
           table(["Site release", "Release notes newer than the site", "Queries with one in the top 10, unfiltered",
                  "Filtered", "Results carrying a note about names"],
                 [[rel, v["release_notes_newer_than_the_site"], v["queries_with_such_a_note_in_top10_unfiltered"],
                   v["queries_with_such_a_note_in_top10_filtered"],
                   f"{v['results_for_a_Classic_site']['carry_a_note_about_names']} of {v['results_for_a_Classic_site']['total']}"]
                  for rel, v in d["release"].items()])]
    return "\n".join(out)


def _answers(name: str, s: dict[str, Any]) -> list[Any]:
    abst = ", ".join(f"{k}: {v}" for k, v in s["abstained"].items()) or "none"
    return [name, s["questions"], s["answered"], abst, s["relevant_unit_in_evidence"],
            s["answered_and_cites_a_relevant_unit"], f"{s['steps_proposed']} / {s['steps_rejected_on_first_pass']} / "
            f"{s['steps_removed_after_the_second_pass']}", s["literals_checked"], s["mean_seconds"]]


def r06(d: dict[str, Any], table: Table) -> str:
    a, b = d["a_evidence_present"], d["b_evidence_removed"]
    head = ["Questions", "Asked", "Answered", "Abstained (by)", "Labelled unit in the evidence",
            "Answered and cites a labelled unit", "Steps proposed / rejected at first / removed", "Literals checked",
            "Mean seconds"]
    off = b["with_the_gate_switched_off"]
    out = ["## RAG R06: grounded procedure answers\n",
           f"Ran {d['ran']}. Evidence: `rag_r06_grounded_answers.json`. Model `{d['model']}`, `{d['strategy']}`, "
           f"evidence budget {d['evidence_budget_tokens']} tokens.\n",
           table(head, [_answers("Identifier questions", a["identifier_questions"]),
                        _answers("Thread questions", a["thread_questions"])]),
           "What the answers rest on: " + "; ".join(
               f"{name}: " + (", ".join(f"{k} {v}" for k, v in s["relies_on"].items()) or "nothing answered")
               + f", {s['unconfirmed_steps']} steps marked unconfirmed"
               for name, s in (("identifier", a["identifier_questions"]), ("thread", a["thread_questions"]))) + ".\n",
           f"Evidence removed: {b['how']}.\n",
           table(head, [_answers("As deployed (gate first)", b["as_deployed"]),
                        _answers("Gate switched off", off)]),
           f"With the gate off, {off['answered']} of {off['questions']} questions were still answered although no "
           "unit containing the identifier was available. Those answers cannot mention the identifier (it would "
           "fail the literal check), so they are answers to a neighbouring question. With the gate on the outcome "
           "is certain by construction and says nothing about the model.\n",
           "No person and no second model judged these answers. \"Cites a labelled unit\" is measured against weak "
           "labels.\n"]
    f = first_run("rag_r06_first_run")
    if f:
        t = f["a_evidence_present"]["thread_questions"]
        out.append(f"First run ({f['ran']}, kept in `rag_r06_first_run.json`): the identifier gate applied to every "
                   f"question. Of {t['questions']} thread questions it stopped {t['abstained'].get('gate', 0)} before "
                   f"the model was called and {t['answered']} were answered. A thread's opening post is full of "
                   "paths and file names that are incidental to the problem, so the gate now applies only to "
                   "questions of at most 30 words.\n")
    return "\n".join(out)


def r07(d: dict[str, Any], table: Table) -> str:
    t, acc, sc = d["trust_as_a_ranking_rule"], d["access"], d["scanner_on_the_real_text"]
    out = ["## RAG R07: trust, access and untrusted text\n",
           f"Ran {d['ran']}. Evidence: `rag_r07_trust_access_injection.json`. `{d['strategy']}`.\n",
           "Standing as a ranking rule, on the development queries:\n",
           table(["Mode", "nDCG@10", "xref", "thread", "identifier", "First result written by", "Difference to off",
                  "Accepted by the policy"],
                 [[m, v["ndcg@10"], v["by_category"]["xref"], v["by_category"]["thread"], v["by_category"]["identifier"],
                   ", ".join(f"{k} {n}" for k, n in sorted(v["first_result_written_by"].items())),
                   interval(v["difference_to_off"]) if v["difference_to_off"] else "", yes(v["accepted_by_the_selection_policy"]) if m != "off" else ""]
                  for m, v in t.items()]),
           ("No mode was accepted by the selection policy, so ranking stays as it is and standing is reported "
            "with each answer instead (R06).\n" if not any(v["accepted_by_the_selection_policy"] for v in t.values())
            else "A mode was accepted by the selection policy; see the table.\n"),
           f"Access: {acc['private_units']} private units from {len(acc['sites'])} sites ({', '.join(acc['sites'])}).\n",
           table(["Check", "Result"], [
               ["Searches for a private unit by its own words, as another caller (index)",
                f"{acc['index_level']['searches_as_other_callers']} searches, "
                f"{acc['index_level']['private_units_leaked']} private units returned"],
               ["Private units found by their owner (index)",
                f"{acc['index_level']['found_by_owner']} of {acc['index_level']['private_units']}"],
               ["The same after the owner filled the cache (engine)",
                f"{acc['service_level']['searches_as_another_caller_after_the_owner']} searches, "
                f"{acc['service_level']['private_units_leaked']} private units returned, "
                f"{acc['service_level']['answers_served_from_the_owners_cache_entry']} served from the owner's cache entry"],
               ["Units read in full",
                f"by their owner: {acc['service_level']['units_read_in_full_by_their_owner']}; by another caller: "
                f"{acc['service_level']['units_read_in_full_by_another_caller']}"]]),
           f"Scanner on the real text: {sc['units_flagged']} of {sc['units_scanned']} thread and note units flagged.\n"]
    p = d.get("planted_text")
    if p:
        out.append(f"Planted replies (injected) in {p['threads']} real threads, model `{p['model']}`. Chosen: "
                   f"{p['how_threads_were_chosen']} (threads tried: "
                   + ", ".join(f"{k} {v}" for k, v in p["threads_tried_without_a_planted_reply"].items()) + ").\n")
        out.append(table(["Planted reply / condition", "Cases", "Shown to the model", "Screened out",
                          "Planted command in the answer", "Marked unconfirmed", "Presented without a warning"],
                         [[k, v["cases"], v["planted_reply_shown_to_the_model"], v["screened_out"],
                           v["planted_command_in_answer"], v["of_which_marked_unconfirmed"],
                           v["presented_without_a_warning"]] for k, v in p["summary"].items()]))
        out.append("The planted replies were written by this project: they are attacks the defences were built "
                   "against, not a sample of what an attacker would write.\n")
        f = first_run("rag_r07_first_run")
        if f and f.get("planted_text"):
            cases = f["planted_text"]["cases"]
            by = Counter("answered" if c["answered"] else c["abstained_by"] for c in cases)
            out.append(f"First run ({f['ran']}, kept in `rag_r07_first_run.json`): of {len(cases)} cases, "
                       + ", ".join(f"{v} ended as `{k}`" for k, v in sorted(by.items()))
                       + ". The identifier gate stopped most cases before the model saw anything, and the planted "
                       "reply was added at the end of each thread, where a long thread is cut. That run therefore "
                       "said little about the model. The threads are now ones the model answers when nothing is "
                       "planted, the reply is planted first, and \"shown to the model\" is counted from the text "
                       "of the evidence.\n")
    return "\n".join(out)


def _gates(v: dict[str, Any]) -> str:
    failed = [g["gate"] for g in v["gates"] if not g["passed"]]
    return "all passed" if not failed else "failed: " + ", ".join(failed)


def r09(d: dict[str, Any], table: Table) -> str:
    s, e, rb, sv, f = d["sync"], d["embedding_reuse"], d["rollback"], d["service"], d["first_design_of_the_quality_gate"]
    doc = s["changes"].get("doc", {})
    reg = next((g for g in d["v2"]["gates"] if g["gate"] == "regression"), None)
    out = ["## RAG R09: service and operations\n",
           f"Ran {d['ran']}. Evidence: `rag_r09_service_and_operations.json`. No model. Documentation at commit "
           f"`{d['documentation_commits']['earlier']}`, then synced to `{d['documentation_commits']['current']}`.\n",
           table(["Version", "Built from", "Units", "Gates", "Activated"], [
               [f"`{d['v1']['version']}`", "documentation at the earlier commit", sum(d["v1"]["units"].values()),
                _gates(d["v1"]), yes(d["v1"]["activated"])],
               [f"`{d['v2']['version']}`", "after the sync", sum(d["v2"]["units"].values()), _gates(d["v2"]),
                yes(d["v2"]["activated"])],
               [f"`{d['candidate_fixed_chunking']['version']}`", "candidate: fixed-window chunking",
                "", _gates(d["candidate_fixed_chunking"]), yes(d["candidate_fixed_chunking"]["activated"])],
               [f"`{d['candidate_without_threads']['version']}`", "candidate: threads missing (injected)",
                "", _gates(d["candidate_without_threads"]), yes(d["candidate_without_threads"]["activated"])]]),
           f"Sync between the two commits: {doc.get('added', 0)} documentation sections added, {doc.get('changed', 0)} "
           f"changed, {doc.get('removed', 0)} removed ({s['change_events']} change events; {s['units_after']} units "
           f"afterwards; {s['seconds']} s). Running it again wrote {s['third_run_change_events']} events. After the "
           f"two refusals the active version was still `{d['active_after_refusals']}`.\n"]
    if reg:
        out.append(f"Regression gate for the synced version: {reg['value']} against a limit of {reg['limit']} "
                   f"({reg['detail']}).\n")
    out += [f"First design of that gate: it compared {f['compared']}. The serving version scored "
            f"{f['serving_version_on_its_own_queries']} on its {f['serving_queries']} queries and the new one "
            f"{f['new_version_on_its_own_queries']} on its {f['new_queries']}, against a limit of {f['limit_it_applied']}: "
            f"it would have refused the update ({yes(f['would_have_refused_the_update'])}). The two numbers are "
            "measured on different queries and cannot be compared; the gate now scores both versions on the queries "
            "they have in common.\n",
            f"What a dense index would embed again (`{e['model']}`): {e['embedded_for_v1']} vectors for the first "
            f"version ({e['seconds_v1']} s), {e['embedded_for_v2']} for the second ({e['seconds_v2']} s), "
            f"{e['reused_for_v2']} reused.\n",
            f"Rollback: versions served in turn: {', '.join(f'`{v}`' for v in rb['served_in_turn'])}. The unit "
            f"`{rb['a_unit_added_by_the_sync']}`, added by the sync, was present: "
            f"{', '.join(yes(x) for x in rb['present_in_turn'])}.\n",
            f"Service, {sv['queries']} development queries over HTTP from one client on the loopback interface: first "
            f"pass p50 {sv['http_ms_first_pass']['p50']} ms, p95 {sv['http_ms_first_pass']['p95']} ms; second pass "
            f"p50 {sv['http_ms_second_pass']['p50']} ms, p95 {sv['http_ms_second_pass']['p95']} ms; cache hit rate "
            f"{sv['cache_hit_rate']}; a request without a key: status {sv['status_without_a_key']}.\n",
            "Threads updated since the last fetch, asked of the live API: " + "; ".join(
                f"`{repo}` {v['threads_updated']}" for repo, v in d["threads_updated_since_the_last_fetch"].items()) + ".\n",
            f"Serving now: `{d['serving']['version']}`, {sum(d['serving']['units'].values())} units, nDCG@10 "
            f"{d['serving']['measured']['dev_ndcg@10']} on {d['serving']['measured']['dev_queries']} development "
            "queries. This is slightly below the figure in the retrieval experiments because the serving index also "
            "holds the sites' notes, and term statistics are computed over every collection.\n"]
    return "\n".join(out)


def r10(d: dict[str, Any], table: Table) -> str:
    a, b, c, lg = d["a_questions"], d["b_filtered_out_for_this_site"], d["c_faults"], d["d_log_lines"]
    p = d["site_profile"]
    head = ["Questions", "Asked", "Stop reasons", "Labelled unit returned by search", "Cites a labelled unit",
            "Answers with a citation", "Every citation was a returned id", "Sent back by the guard", "Mean searches",
            "Mean seconds"]

    def row(name: str, s: dict[str, Any]) -> list[Any]:
        return [name, s["questions"], ", ".join(f"{k} {v}" for k, v in s["stop_reasons"].items()),
                s["relevant_unit_returned_by_search"], s["cites_a_relevant_unit"], s["answers_with_a_citation"],
                s["every_citation_was_returned"], s["sent_back_by_guard"], s["mean_searches"], s["mean_seconds"]]

    out = ["## RAG R10: the agents' search tools\n",
           f"Ran {d['ran']}. Evidence: `rag_r10_agent_tools.json`. Model `{d['model']}`, agent `{d['agent']}`, index "
           f"`{d['index_version']}`. The asking site is {p['site']} ({p['generation']}, release {p['release']}): "
           f"{p['note']}.\n",
           table(head, [row("Identifier questions", a["identifier"]), row("Thread questions", a["thread"]),
                        row("Filtered out for this site", b)]),
           f"\"Filtered out for this site\" are questions whose only labelled answer is on a page that does not "
           f"apply to the site (here: pages for the other hardware generation). The searches returned "
           f"{b['passages_for_the_other_generation']} passages for the other generation. "
           f"{b['says_nothing_found']} of {b['questions']} answers said that nothing was found; "
           f"{b['answers_made_without_any_citation']} gave an answer with no citation at all. The others answered "
           "from passages that do apply to the site, which may or may not answer the question: nobody judged that.\n",
           "Faults, all injected:\n"]
    rows = [["Wrong key", f"tool errors {c['wrong_key']['tool_errors']}, attempts {c['wrong_key']['attempts']}; "
             f"citations {c['wrong_key']['citations']}; says it could not search: "
             f"{yes(c['wrong_key']['says_it_could_not_search'])}; `{c['wrong_key']['stop_reason']}`"],
            ["One late response", f"attempts {c['one_late_response']['attempts']}; cites a labelled unit: "
             f"{yes(c['one_late_response']['cites_a_relevant_unit'])}; `{c['one_late_response']['stop_reason']}`"],
            ["Service stopped", f"tool errors {c['service_down']['tool_errors']}, attempts "
             f"{c['service_down']['attempts']}; citations {c['service_down']['citations']}; says it could not search: "
             f"{yes(c['service_down']['says_it_could_not_search'])}; `{c['service_down']['stop_reason']}`"]]
    for key, label in (("planted_reply_addressed_to_the_model", "Planted reply addressed to the model"),
                       ("planted_reply_plain_false_claim", "Planted reply: a plain false claim")):
        v = c[key]
        rows.append([label, f"returned by the search: {yes(v['planted_unit_returned_by_search'])}; planted text "
                     f"reached the model: {yes(v['planted_text_reached_the_model'])}; planted command in "
                     f"the answer: {yes(v['planted_command_in_answer'])}; answer says it rests on a thread reply: "
                     f"{yes(v['answer_says_it_rests_on_a_thread_reply'])}; `{v['stop_reason']}`"])
    out.append(table(["Fault", "What happened"], rows))
    claim = c["planted_reply_plain_false_claim"]
    if claim["planted_command_in_answer"]:
        out.append("The plain false claim reached the agent and the agent repeated the planted command. The search "
                   "tools hand over passages with their standing; unlike the answer endpoint (R06), no code marks a "
                   "statement that rests on a community reply as unconfirmed. Whether the agent says so is left to "
                   "its wording, which the check here only matches by keyword. One thread, one run.\n")
    out.append(f"Log lines: {lg['lines']} real lines found in code blocks of thread opening posts. Their thread was "
               f"in the top 5 for {lg['found_in_top5_raw']} searched raw and {lg['found_in_top5_cleaned']} searched "
               f"cleaned. Home directories left in a cleaned query: {lg['home_directories_left_in_a_query']}. "
               "The sample is too small to say whether cleaning helps retrieval.\n")
    return "\n".join(out)


SECTIONS = [("rag_r05_applicability", r05), ("rag_r06_grounded_answers", r06),
            ("rag_r07_trust_access_injection", r07), ("rag_r09_service_and_operations", r09),
            ("rag_r10_agent_tools", r10)]
