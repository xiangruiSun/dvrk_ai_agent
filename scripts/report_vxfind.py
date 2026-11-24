"""Report sections for the training-data finder (vxfind). Every figure is read from
verification/vxfind_*.json; nothing is typed in."""
from __future__ import annotations

from typing import Any, Callable

T = Callable[[list[str], list[list[Any]]], str]
SETS = {"A": "A attribute needs (fit@10)", "B": "B known-item needs (mrr@10)", "C": "C written needs (ndcg@10)"}
HEAD = {"A": "fit@10", "B": "mrr@10", "C": "ndcg@10"}


def cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def kv(d: Any) -> str:
    """A small dictionary as words: 'A 31, B 57'."""
    return ", ".join(f"{str(k).replace('_', ' ')} {v}" for k, v in d.items()) if isinstance(d, dict) else str(d)


def pct(x: Any, d: int = 1) -> str:
    return "n/a" if x is None else f"{100 * float(x):.{d}f}%"


def score(s: dict[str, Any], kind: str, metric: str | None = None) -> str:
    v = (s.get(kind) or {}).get(metric or HEAD[kind])
    return "n/a" if not v else f"{v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"


def diff(c: dict[str, Any] | None, kind: str) -> str:
    v = (c or {}).get(kind)
    if not v:
        return ""
    star = " *" if v["p_randomization"] < 0.05 else ""
    return f"{v['mean_difference']:+.3f} (p {v['p_randomization']:.3f}){star}"


def write_path(d: dict[str, Any], table: T) -> str:
    snaps = d["source"]["snapshots"]
    old, new = d["spark_job"]["old"], d["spark_job"]["new"]
    out = ["## Write path: from the catalog dump to a searchable index\n",
           f"Evidence: `vxfind_write_path.json`. Source: {d['source']['what']}. Old snapshot published "
           f"{snaps['old']['published'][:10]}, new {snaps['new']['published'][:10]}; "
           f"{d['source'].get('dump_revisions_available')} daily revisions of the dump exist.\n"]
    rows = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        if isinstance(a, (int, float, str)) or isinstance(b, (int, float, str)):
            rows.append([k.replace("_", " "), a if a is not None else "", b if b is not None else ""])
    out.append(table(["Spark job", "old snapshot", "new snapshot"], rows))
    for k in sorted(set(old) | set(new)):
        b = new.get(k) if isinstance(new.get(k), dict) else old.get(k)
        if isinstance(b, dict) and all(isinstance(v, (int, float, str)) for v in b.values()) and len(b) <= 14:
            out.append(f"{k.replace('_', ' ')} (new snapshot): " + ", ".join(f"{x} {y}" for x, y in b.items()) + ".\n")
    emb = []
    for snap in ("old", "new"):
        for m, e in d["embedding"].get(snap, {}).items():
            emb.append([snap, m, e["tables"], e["computed"], e["reused_from_cache"], e["model_seconds"], e.get("per_second"), e["file_mb"]])
    out.append(table(["Embedding", "model", "tables", "vectors computed", "reused from cache", "model seconds", "per second", "file MB"], emb))
    ib, sy, now = d.get("index_build") or {}, d.get("incremental_sync") or {}, d["index_now"]
    rows = [["Full build: documents / failed", f"{ib.get('documents')} / {ib.get('failed')}"],
            ["Full build: bulk seconds (documents per second)", f"{ib.get('bulk_seconds')} ({ib.get('documents_per_second')})"],
            ["Full build: merge seconds", ib.get("force_merge_seconds")],
            ["Index now: tables / MB / bytes per table", f"{now['tables']} / {now['store_mb']} / {now['bytes_per_table']}"],
            ["Catalog database (newer snapshot), MB", (d.get("catalog_database") or {}).get("file_mb")]]
    if sy:
        rows += [["Incremental sync: written / deleted", f"{sy.get('written_ok')} of {sy.get('written')} / {sy.get('deleted_ok')} of {sy.get('deleted')}"],
                 ["Incremental sync: write seconds / delete seconds", f"{sy.get('write_seconds')} / {sy.get('delete_seconds')}"],
                 ["Tables in the synced index afterwards", sy.get("count_after")],
                 ["Was the synced index the serving one", "yes" if sy.get("is_the_serving_index") else "no: a copy, so that the evaluation keeps measuring the index that is behind"]]
    else:
        rows.append(["Incremental sync", "not run yet (it is run after the freshness measurement, on purpose)"])
    out.append(table(["Index", "Value"], rows))
    if d.get("engine_rrf_on_this_licence"):
        out.append(f"The engine's own rank fusion on this licence: {d['engine_rrf_on_this_licence']}. Fusion is done by the caller.\n")
    return "\n".join(out)


def queries(d: dict[str, Any], table: T) -> str:
    rf = d.get("robot_families", {})
    out = ["## Query sets\n", "Evidence: `vxfind_queries.json`; the sets are in `evals/vxfind/queries/`.\n",
           table(["Set", "Truth comes from", "Queries", "dev", "test"], [
               ["A attribute needs", "computed by code from the catalog", d["A"]["queries"], d["A"]["dev"], d["A"]["test"]],
               ["B known-item needs", f"the table each need was written from (written by `{d['B']['written_by']}`)", d["B"]["queries"], d["B"]["dev"], d["B"]["test"]],
               ["C written needs", "graded judgments of pooled results", d["C"]["queries"], d["C"]["dev"], d["C"]["test"]],
               ["D no-answer needs", "nothing fits, confirmed by judging the pool", d["D"]["queries"], d["D"]["dev"], d["D"]["test"]]])]
    if rf:
        out.append(f"Robot families: {rf['rules']} hand-written rules cover {rf['tables_in_a_family']} of {rf['tables']} tables "
                   f"({rf['distinct_values_matched']} of {rf['distinct_robot_type_values']} spellings); "
                   f"{rf['tables_with_a_robot_type_no_rule_matches']} tables name a robot no rule matches and "
                   f"{rf['tables_without_robot_type']} name none. A table outside every family is never counted as a match.\n")
    a, b = d["A"], d["B"]
    out.append(f"Set A: constraints per query: {kv(a['constraints_per_query'])}; used: {kv(a['constraints_used'])}; the median query has "
               f"{a['matching_tables_median']} fully matching tables, {pct(a['share_of_robots_tables_that_match_median'])} of its robot's tables.\n")
    out.append(f"Set B: {b['eligible_tables']} tables are eligible sources (described, not scratch, one per family); {b['tables_tried']} "
               f"were tried, discarded: {kv(b['discarded']) or 'none'}; spread over robots: {kv(b['robots'])}.\n")
    out.append(f"Set C: {d['C']['author']}. Set D kinds: {kv(d['D']['kinds'])}.\n")
    return "\n".join(out)


def retrieval(d: dict[str, Any], table: T) -> str:
    j = d["judging"]
    out = ["## Retrieval: which configuration, measured on three kinds of truth\n",
           f"Evidence: `vxfind_retrieval.json`. Index: {d['index']['tables']} tables (`{d['index']['index']}`, engine {d['index']['engine']}). "
           f"Queries: dev {kv(d['queries']['dev'])}; test {kv(d['queries']['test'])}. Judge: `{d['judge']}`; pool depth {j['pool_depth']} over "
           f"{j['configurations_pooled_dev']} configurations on dev and {j['configurations_pooled_test']} on test; "
           f"{j['dev_pool']['pairs']} + {j['test_pool']['pairs']} pairs judged ({j['dev_pool']['failed']} failed), "
           f"{j['pairs_per_C_query_mean']} per written need; grades given: {kv(j['grades_given_to_C_pairs'])}.\n",
           "Each cell is the mean with its 95% bootstrap interval. \"vs rrf\" is the paired difference to plain fusion with the "
           "randomization p-value; * marks p < 0.05. With this many comparisons about one in twenty is expected to be starred by chance.\n",
           "### Development queries\n"]
    fr = d.get("first_run_with_the_first_rubric")
    if fr:
        out.insert(-1, f"This is the second full run. The first used the judge's first rubric ({fr['judge_rubric']}), which passed tables that "
                       f"contradicted the request: it gave the grades {kv(fr['grades_given_to_C_pairs'])} to the written needs' pairs, put plain "
                       f"fusion at {fr['C_ndcg@10_of_rrf_on_dev']:.3f} on set C, and called {fr['no_answer_needs_that_judge_called_answerable']} of the "
                       f"no-answer needs answerable. The selection rule took the same settings then "
                       f"({', '.join(fr['settings_the_rule_took'] or []) or 'none'}), because the choice is driven by set B, whose truth needs "
                       f"no judge. Kept as `{fr['evidence']}`. Rubric in use now: {d.get('judge_rubric')}.\n")
    rows = []
    for name, v in d["dev"].items():
        s, c = v["scores"], v.get("vs_rrf")
        rows.append([f"`{name}`", score(s, "A"), diff(c, "A"), score(s, "B"), diff(c, "B"), score(s, "C"), diff(c, "C"),
                     f"{(s.get('C') or {}).get('repeat@10', {}).get('mean', 0):.2f}"])
    out.append(table(["Configuration", "A fit@10", "vs rrf", "B mrr@10", "vs rrf", "C ndcg@10", "vs rrf", "copies in C top 10"], rows))
    sv = d["serving"]
    why = sv["why"]
    out.append("### The serving configuration\n")
    out.append("Rule (written before the run): take a setting when it is significantly better than plain fusion on at least one "
               "query set and significantly worse on none; combine what was taken; keep the combination only if it is not "
               "significantly worse anywhere.\n")
    out.append(f"Settings taken: {', '.join('`' + x + '`' for x in why.get('settings_taken', [])) or 'none'}. "
               + (f"Combined change: {kv(why.get('combined_change') or {})}. Combination kept: {'yes' if why.get('combination_kept') else 'no'}. "
                  f"Top-up judgments for it: {why.get('top_up_judgments')}.\n" if why.get("settings_taken") else f"{why.get('note', '')}\n"))
    cfg = sv["config"]
    out.append(f"Serving: retriever `{cfg['retriever']}`, embedding `{cfg['embedding']}`, window {cfg['window']}, rank constant "
               f"{cfg['rank_constant']}, collapse {cfg['collapse']}, scratch x{cfg['demote_scratch']}, described x{cfg['boost_described']}.\n")
    out.append("### Locked test queries (scored after the choice)\n")
    rows = []
    for name, v in d["test"].items():
        s, c = v["scores"], v.get("vs_rrf")
        rows.append([f"`{name}`", score(s, "A"), diff(c, "A"), score(s, "B"), diff(c, "B"), score(s, "C"), diff(c, "C"),
                     score(s, "C", "precision@10"), f"{(s.get('C') or {}).get('unjudged@10', {}).get('mean', 0):.3f}"])
    out.append(table(["Configuration", "A fit@10", "vs rrf", "B mrr@10", "vs rrf", "C ndcg@10", "vs rrf", "C precision@10", "C unjudged@10"], rows))
    sd, st = sv["dev_scores"], d["test"]["serving"]["scores"]
    out.append(table(["Serving configuration, other metrics", "dev", "test"], [
        ["A: share of top 10 with the right robot", score(sd, "A", "robot_right@10"), score(st, "A", "robot_right@10")],
        ["B: target found at rank 1 / in 5 / in 10", " / ".join(f"{sd['B'][m]['mean']:.2f}" for m in ("hit@1", "hit@5", "hit@10")),
         " / ".join(f"{st['B'][m]['mean']:.2f}" for m in ("hit@1", "hit@5", "hit@10"))],
        ["C: at least one usable table in the top 10", score(sd, "C", "success@10"), score(st, "C", "success@10")],
        ["C: precision@10", score(sd, "C", "precision@10"), score(st, "C", "precision@10")]]))
    m = d["returning_more_than_ten"]
    ks = [k for k in m["dev_serving"]]
    out.append("### Returning more than ten\n")
    out.append(table(["Known-item target found within the first"] + [k.rsplit("_", 1)[1] for k in ks],
                     [["dev"] + [pct(m["dev_serving"][k], 0) for k in ks], ["test"] + [pct(m["test_serving"][k], 0) for k in ks]]))
    lat = d["latency_ms"]
    out.append(table(["Latency per search, ms (dev run, one caller)", "p50", "p95", "max"],
                     [[f"`{n}`", v["total"].get("p50_ms"), v["total"].get("p95_ms"), v["total"].get("max_ms")] for n, v in lat.items()]))
    na = d["no_answer_set"]
    out.append(f"No-answer set: best grade found in the pool, needs per grade: {kv(na['best_grade_found_in_pool'])}. Needs that turned out to have "
               f"a usable table and were moved out: " + ("; ".join(f"\"{x['need']}\"" for x in na["turned_out_to_have_a_usable_table"]) or "none") + ".\n")
    return "\n".join(out)


def judge(d: dict[str, Any], table: T) -> str:
    a = d["against_labels_computed_by_code"]
    big, small = d["judge"], d["second_judge"]
    out = ["## The judge: how far its grades can be trusted\n",
           f"Evidence: `vxfind_judge.json`. Judge `{big}`, second judge `{small}`, prompt version {d['prompt_version']}, one table per call.\n",
           f"### Against truth computed by code\n\n{a['what']}. \"Relevant\" for the judge is grade 2 or 3; truth is \"every constraint matches\".\n"]
    rows = []
    for m in (big, small):
        c = a[m]
        cf = c["confusion"]
        rows.append([f"`{m}`", c["pairs"], pct(c["agreement"]), c["kappa"], pct(c["judge_precision"]), pct(c["judge_recall"]),
                     pct(c["false_positive_rate"]), f"{cf['true_pos']} / {cf['false_pos']} / {cf['false_neg']} / {cf['true_neg']}"])
    out.append(table(["Judge", "Pairs", "Agreement", "Kappa", "Precision", "Recall", "False-positive rate", "TP / FP / FN / TN"], rows))
    held = a.get("held_out_pairs_of_the_test_attribute_needs")
    if held:
        rows = []
        for m in held:
            c = held[m]
            cf = c["confusion"]
            rows.append([f"`{m}`", c["pairs"], pct(c["agreement"]), c["kappa"], pct(c["judge_precision"]), pct(c["judge_recall"]),
                         pct(c["false_positive_rate"]), f"{cf['true_pos']} / {cf['false_pos']} / {cf['false_neg']} / {cf['true_neg']}"])
        out.append(f"Held out ({a.get('note', '')}):\n")
        out.append(table(["Judge, held-out pairs", "Pairs", "Agreement", "Kappa", "Precision", "Recall", "False-positive rate", "TP / FP / FN / TN"], rows))
    g = a[big]["judge_grade_by_true_grade"]
    out.append(table([f"`{big}`: true grade", "judged 0", "judged 1", "judged 2", "judged 3"],
                     [[k] + [v.get(str(x), 0) for x in range(4)] for k, v in g.items()]))
    out.append("Examples where the judge is wrong:\n")
    out.append(table(["Need", "Table", "True", "Judge", "Judge's reason"],
                     [[e["need"][:90], f"`{e['table_id']}`", e["true_grade"], e["judge_grade"], e["judge_reason"]] for e in a["examples_where_the_judge_is_wrong"][:6]]))
    k = d["need_written_from_a_table_judged_against_that_table"]
    t = d["two_judges_on_pooled_pairs_of_written_needs"]
    s = d["same_pair_judged_twice"]
    out.append(table(["Other checks", "Result"], [
        ["A need written from a table, judged against that table: usable or better", f"{pct(k['share_judged_usable_or_better'])} of {k['pairs']} (grades: {kv(k['grades'])})"],
        [f"Two judges on {t['pairs']} pooled pairs of written needs: same grade / within one", f"{pct(t['exact'])} / {pct(t['within_one'])}"],
        ["Weighted kappa / kappa on relevant-or-not", f"{t['weighted_kappa']} / {t['kappa_relevant']}"],
        [f"Mean grade: `{big}` / `{small}`", f"{t['mean_first']} / {t['mean_second']}"],
        ["Same pair judged twice: same grade", f"{s['same_grade']} of {s['pairs']}"],
        ["Seconds per judgment", ", ".join(f"{k2.rsplit('_', 1)[-1] if False else k2.replace('seconds_per_judgment_', '')}: {v}" for k2, v in d["cost"].items() if k2.startswith("seconds"))],
        ["Judgments stored", d["cost"]["judgments_stored"]]]))
    h = d["hand_labels"]
    out.append(f"### Human labels\n\nSheet: `{h['sheet']}`, {h['pairs']} pairs, filled by a person: **{h['filled_by_a_person']}**. "
               + (h.get("status", "") if not h.get("judge_vs_person") else f"Judge against the person: {h['judge_vs_person']}") + "\n")
    return "\n".join(out)


def judge_iterations(d: dict[str, Any], table: T) -> str:
    out = ["## The judge's rubric, version by version\n",
           f"Evidence: `vxfind_judge_iterations.json` (and the first full run with version g2 in `vxfind_judge_first_run.json` and "
           f"`vxfind_retrieval_first_run.json`). {cap(d['what'])}. {d['pinned_pairs']} pinned pairs; the computed truth of "
           f"{d['pairs_whose_computed_truth_was_corrected']} of them was corrected on the way: {d['truth_correction']}. In use: {d['in_use']}.\n"]
    rows = []
    for v, ms in d["versions"].items():
        for m, c in ms.items():
            rows.append([v, d["changes"].get(v, ""), f"`{m}`", c["pairs"], pct(c["agreement"]), c["kappa"], pct(c["judge_precision"]),
                         pct(c["judge_recall"]), pct(c["false_positive_rate"])])
    out.append(table(["Version", "What changed", "Judge", "Pairs", "Agreement", "Kappa", "Precision", "Recall", "False-positive rate"], rows))
    out.append("The rubric was revised while reading these pairs, so the last rows flatter it; the held-out rows in the section "
               "above are the figure to quote.\n")
    return "\n".join(out)


def llm_stages(d: dict[str, Any], table: T) -> str:
    out = ["## Model stages: what understanding, reranking and selection buy, and cost\n",
           f"Evidence: `vxfind_llm_stages.json`. Small model `{d['small_model']}`, large model `{d['large_model']}`. Dev queries "
           f"{kv(d['queries'])} (a fixed half of set B). Top-up judgments: {d['top_up_judgments']['judged_now']}. Differences are to the "
           "serving configuration, paired; * marks p < 0.05. Set C here is judged by the same model that reranks and selects, "
           "so sets A and B are the stronger evidence.\n"]
    rows = []
    for name, v in d["configurations"].items():
        s, c, lat = v["scores"], v.get("vs_serving"), v["latency_ms"]
        rows.append([f"`{name}`", score(s, "A"), diff(c, "A"), score(s, "B"), diff(c, "B"), score(s, "C"), diff(c, "C"),
                     f"{lat.get('total', {}).get('p50_ms', '')} / {lat.get('total', {}).get('p95_ms', '')}"])
    out.append(table(["Configuration", "A fit@10", "vs serving", "B mrr@10", "vs serving", "C ndcg@10", "vs serving", "ms p50 / p95"], rows))
    rd = d["reading_of_attribute_needs"]
    first = d.get("first_run_with_the_first_reading_prompt") or {}
    cols = {m: r for m, r in rd.items()}
    f1 = first.get("reading_by_" + d["small_model"])
    if f1:
        cols = {f"{d['small_model']}, first prompt (first run)": f1, **{f"{m}, prompt {d.get('reading_prompt', '')}": r for m, r in rd.items()}}
    keys = [("min_episodes", "minimum episodes"), ("fps", "frame rate"), ("min_cameras", "minimum cameras"), ("format", "format")]
    rows = [[label] + [f"{r.get(k + '_read_right', 0)} of {r.get(k + '_stated', 0)}" for r in cols.values()] for k, label in keys]
    rows.append(["a robot was read at all"] + [f"{r.get('robot_read', 0)} of {r.get('robot_stated', 0)}" for r in cols.values()])
    rows.append(["the robot read is the right family"] + [f"{r['robot_read_as_the_right_family']} of {r.get('robot_stated', 0)}" if "robot_read_as_the_right_family" in r else "not measured"
                                                         for r in cols.values()])
    rows.append(["fields filled although the need does not state them"] + [str(sum(v for k, v in r.items() if k.endswith("_filled_though_not_stated")))
                                                                          if any(k.endswith("_stated") for k in r) and "robot_read_as_the_right_family" in r else "not measured"
                                                                          for r in cols.values()])
    out.append("### Reading the need\n")
    out.append(table(["Attribute needs: what the need states, read with the right value"] + [f"`{c}`" for c in cols], rows))
    if first.get("headline_scores"):
        out.append("The first reading prompt told the model not to guess a robot, and it then left the robot empty for almost every need that "
                   f"names one. That run is kept in `{first['evidence']}`; its headline scores were "
                   + "; ".join(f"`{n}`: " + ", ".join(f"{k} {v:.3f}" for k, v in sc.items()) for n, sc in first["headline_scores"].items())
                   + ". The reading prompt was rewritten field by field, and a robot's name now boosts and no longer filters, because a name "
                     "matched as text cannot bridge spellings (\"Universal Robots\" and \"ur5e\" share no word).\n")
    stage = []
    for name, v in d["configurations"].items():
        for s in ("understand", "rerank", "select"):
            if s in v["latency_ms"] and v["latency_ms"][s]["p50_ms"] > 0:
                stage.append([f"`{name}`", s, v["latency_ms"][s]["p50_ms"], v["latency_ms"][s]["p95_ms"]])
    out.append(table(["Configuration", "Stage", "p50 ms", "p95 ms"], stage))
    return "\n".join(out)


def answers(d: dict[str, Any], table: T) -> str:
    out = ["## Reasons and \"nothing fits\"\n",
           f"Evidence: `vxfind_answers.json`. Selection by `{d['selection_model']}`. Dev queries: {kv(d['queries'])}. {cap(d['note'])}.\n"]
    rows = []
    for name, v in d["configurations"].items():
        ab = v["abstention"]
        dd = ab.get("D", {})
        ans = [ab[k] for k in "ABC" if k in ab]
        n = sum(x["queries"] for x in ans)
        rows.append([f"`{name}`", f"{dd.get('showed_nothing')} of {dd.get('queries')}", dd.get("showed_nothing_absent", ""), dd.get("showed_nothing_size", ""),
                     dd.get("said_what_is_missing"), f"{sum(x['showed_nothing'] for x in ans)} of {n}",
                     " / ".join(f"{k} {ab[k]['mean_tables_shown']}" for k in "ABC" if k in ab)])
    out.append(table(["Configuration", "No-answer needs: showed nothing", "kind absent", "kind size", "said what is missing",
                      "Answerable needs: showed nothing (wrongly)", "mean tables shown"], rows))
    for name, v in d["configurations"].items():
        gs = {k.replace("groundedness_checked_by_", ""): g for k, g in v.items() if k.startswith("groundedness") and g}
        if not gs:
            continue
        first = next(iter(gs.values()))
        out.append(f"Reasons written by `{name}`: {first['reasons']} ({first['empty_reasons']} empty). With a number that is in "
                   f"neither the table's text nor the need: {first['with_a_number_not_in_the_table_text']} "
                   f"({pct(first['share_with_unsupported_number'])}).\n")
        out.append(table(["Checked by", "Reasons checked", "Called unsupported", "Share"],
                         [[f"`{m}`" + (" (the model that wrote them)" if m == d["selection_model"] else ""), g["checked_by_model"],
                           g["model_says_unsupported"], pct(g["share_model_says_unsupported"])] for m, g in gs.items()]))
        ex = [e for g in gs.values() for e in g["examples"]][:6]
        out.append(table(["Table", "Reason shown", "What is not supported"],
                         [[f"`{e['table_id']}`", e["reason"][:150], e.get("model_says_unsupported") or ("numbers " + ", ".join(e["numbers_not_in_text"]))] for e in ex]))
    return "\n".join(out)


def meta(d: dict[str, Any], table: T) -> str:
    out = ["## Can these numbers be trusted\n", "Evidence: `vxfind_meta.json`. Planted faults are copies of the serving configuration's "
           "stored results made worse on purpose; \"seen\" means a negative difference with p < 0.05.\n"]
    for split in ("dev", "test"):
        f = d[split]["planted_faults"]
        rows = []
        for how, v in f.items():
            cells = []
            for k in "ABC":
                x = v.get(k)
                cells.append(f"{x['difference']:+.3f} {'seen' if x['seen_as_a_loss'] else 'NOT seen'}" if x else "")
            ctrl = v.get("control_fit@10_changed")
            rows.append([how.replace("_", " ")] + cells + ["" if ctrl is None else ("CHANGED" if ctrl else "unchanged")])
        out.append(table([f"Planted fault ({split}: {kv(d[split]['queries'])})", "A fit@10", "B mrr@10", "C ndcg@10", "control: A fit@10 under reordering"], rows))
    rows = []
    for split in ("dev", "test"):
        for k, v in d[split]["power_from_serving_vs_rrf"].items():
            rows.append([split, f"{k} {v['metric']}", v["queries"], v["sd_of_paired_differences"], v["smallest_detectable_difference"],
                         v["queries_needed_for_0.05"], v["queries_needed_for_0.02"]])
    out.append(table(["Split", "Set", "Queries", "SD of paired differences", "Smallest detectable difference (80% power)",
                      "Queries needed for 0.05", "for 0.02"], rows))
    ag = d["agreement_between_query_sets_dev"]
    out.append(f"Agreement between the query sets in how they rank the {ag['configurations']} configurations (Kendall's tau): "
               + ", ".join(f"{k.replace('tau_', '').replace('_', ' and ')}: {v}" for k, v in ag.items() if k.startswith("tau")) + ". "
               f"Best three by set: {ag['best_by_set']}.\n")
    return "\n".join(out)


def fresh(d: dict[str, Any], table: T) -> str:
    w, lc = d["what_was_shown"], d["live_checks"]
    out = ["## Freshness: dead, moved and changed addresses, before and after the live check\n",
           f"Evidence: `vxfind_fresh.json`. The index was built from the catalog of {d['index_built_from_catalog_of'][:10]}; the site was "
           f"asked on {d['site_asked_on']}. Nothing is injected here. {d['distinct_tables_shown']} distinct tables are shown for "
           f"{d['queries']} queries. The site's answers for {lc['tables_on_record']} tables are on record (first {lc['first_answer']}, "
           f"last {lc['last_answer']}); the first pass took {lc['minutes_from_first_to_last_answer_of_the_first_pass']} minutes under a "
           f"budget of {lc['budget']}, most of it waiting for the budget. States: {kv(lc['states'])}.\n",
           table(["What the serving configuration showed (top 10)", "Value"], [
               ["Addresses shown / checked / not checked", f"{w['addresses_shown']} / {w['addresses_checked']} / {w['not_checked']}"],
               ["Dead", f"{w['dead']} ({pct(w['share_dead'])})"], ["Moved to another address", f"{w['moved']} ({pct(w['share_moved'])})"],
               ["Changed since indexed", f"{w['changed_since_indexed']} ({pct(w['share_changed'])})"],
               ["Answers with at least one dead address", f"{w['answers_with_a_dead_address']} of {w['answers']} ({pct(w['share_answers_with_a_dead_address'])})"],
               ["Answers with a dead, moved or changed address", f"{w['answers_with_dead_moved_or_changed']} of {w['answers']}"],
               ["After the live check: dead addresses shown", w["after_live_check"]["dead_addresses_shown"]],
               ["After the live check: tables removed per answer (mean)", w["after_live_check"]["mean_removed_per_answer"]],
               ["After the live check: answers still full after refilling", f"{w['after_live_check']['answers_still_full_after_refill']} of {w['answers']}"]]),
           table(["Query set", "Addresses checked", "Dead", "Moved", "Changed", "Answers with a dead address"],
                 [[k, v["addresses_checked"], f"{v['dead']} ({pct(v['share_dead'])})", v["moved"], v["changed_since_indexed"],
                   f"{v['answers_with_a_dead_address']} of {v['answers']}"] for k, v in d["by_query_set"].items()])]
    src = d.get("the_source_does_not_report_deletions")
    if src:
        out.append(f"Why so many: of {src['dead_tables_found']} dead tables found, {src['of_which_still_in_the_newer_dump']} are still in the "
                   f"newer dump a month later; the newer dump dropped {src['tables_the_newer_dump_dropped']} tables and added "
                   f"{src['tables_the_newer_dump_added']}. {cap(src['reading'])}\n")
    r = d["random_tables_of_the_index"]
    out.append(f"Random tables of the index: {r['dead']} of {r['checked']} dead ({pct(r['share_dead'])}, 95% interval "
               f"{pct(r['ci95'][0])} to {pct(r['ci95'][1])}); {r['moved']} moved.\n")
    out.append(table(["Year created", "Shown tables checked", "Dead", "Share"],
                     [[y, v["checked"], v["dead"], pct(v["share_dead"])] for y, v in d["dead_share_by_year_created"].items()]))
    ls = d["listing_as_a_cheap_signal"]
    if "precision" in ls:
        out.append(f"Absence from the site's public listing as a sign of a dead table, against the live check on {ls['checked_tables']} "
                   f"tables: {ls['not_in_listing']} not in the listing, of which {ls['dead_and_not_in_listing']} dead and "
                   f"{ls['alive_but_not_in_listing']} alive; {ls['dead_but_in_listing']} dead tables are in the listing. "
                   f"Precision {pct(ls['precision'])}, recall {pct(ls['recall'])}.\n")
    c = d["seconds_added_to_one_answer"]
    out.append(f"Seconds added to one answer by checking {c['tables_per_answer']} tables: one after another median "
               f"{c['one_after_another_s']['median']} (max {c['one_after_another_s']['max']}); four at a time median "
               f"{c['four_at_a_time_s']['median']} (max {c['four_at_a_time_s']['max']}).\n")
    km = d["kinds_of_move"]
    out.append(f"Of the moved tables, {km['moved_to_another_owner']} went to another owner and {km['renamed_under_the_same_owner']} were renamed "
               "under the same owner.\n")
    return "\n".join(out)


def access(d: dict[str, Any], table: T) -> str:
    p, ap = d["probes_by_an_outsider"], d["approval"]
    out = ["## Access: can anyone see, or infer, a hidden table\n",
           f"Evidence: `vxfind_access.json`. Policy (DECLARED): {d['policy']['restricted_owners']} owners, the ones that appear most often "
           f"in results, are treated as restricted to one group. They are not named here or in git. {d['queries']} queries.\n",
           table(["Check", "Result"], [
               ["Without the rule: restricted tables in the top 10 (the test can fail)", f"{d['without_the_rule']['restricted_tables_in_top_10']} in {d['without_the_rule']['queries_with_one']} queries"],
               ["Outsider: restricted tables in any result", d["outsider"]["restricted_tables_in_any_result"]],
               ["Outsider: restricted tables among the first 100 candidates", d["outsider"]["restricted_tables_among_the_first_100_candidates"]],
               ["Member of the group: restricted tables in the top 10", d["insider_of_the_group"]["restricted_tables_in_top_10"]],
               [f"Probes of {p['hidden_tables_probed']} hidden tables through {len(p['tools_probed'])} tools: replies identical to a made-up id",
                f"{p['replies_identical_to_a_nonexistent_table']} of {p['replies_compared']}"],
               ["Requests the server sent to the hosting site for hidden or made-up ids", p["requests_sent_to_the_site_for_hidden_or_fake_ids"]],
               ["Newcomer decides their own request", f"{ap['newcomer_decides_own_request']['error_kind']}: {ap['newcomer_decides_own_request']['error']}"],
               ["Newcomer reads the mentor's queue", ap["newcomer_reads_the_queue"]["error_kind"]],
               ["Mentor decides their own request", f"{ap['mentor_decides_own_request']['error_kind']}: {ap['mentor_decides_own_request']['error']}"],
               ["Mentor decides the newcomer's request -> newcomer's status", f"{ap['mentor_decides_newcomers_request']} -> {ap['newcomer_status_after']}"],
               ["Connection without a valid token", d["no_token"]],
               ["Mentor tools offered to the assistant", d["mentor_tools_offered_to_the_assistant"] or "none"],
               ["Audit rows written", d["audit_rows"]]])]
    return "\n".join(out)


def agent(d: dict[str, Any], table: T) -> str:
    models = [k for k, v in d.items() if isinstance(v, dict) and "summary" in v]
    out = ["## Agent: evaluated on what it did\n", f"Evidence: `vxfind_agent.json`; cases in `{d['cases_file']}`. {cap(d['world'])}. One run per case per model.\n"]
    rows = []
    for m in models:
        s = d[m]["summary"]
        rows.append([f"`{m}`", f"{s['passed']} of {s['cases']}", ", ".join(s["failed"]) or "none", s["answers_sent_back_by_the_check"],
                     s["mean_tool_calls"], s["mean_tokens"], f"{s['median_seconds']} / {s['max_seconds']}"])
    fp = d.get("fixed_pipeline_same_needs")
    if fp:
        rows.append(["fixed pipeline (first 10 cases, no model call)", f"gone shown {fp['gone_shown']}, old address shown {fp['old_address_shown']}", "", "", "", 0,
                     f"{fp['median_ms'] / 1000:.2f} (median)"])
    out.append(table(["Model", "Cases passed", "Failed cases", "Answers sent back by the check", "Tool calls (mean)", "Tokens (mean)", "Seconds median / max"], rows))
    fr = d.get("first_run")
    if fr:
        out.append(f"First run (`{fr['evidence']}`): {fr['what_it_showed']} Runs that ended with no table shown and no live check: "
                   + "; ".join(f"`{m}` {v['ended_with_nothing_shown_and_no_live_check']} of {v['cases']} (and {v['passed']} counted as passed)"
                               for m, v in fr["models"].items()) + ".\n")
    names = sorted({p for m in models for p in d[m]["summary"]["properties"]})
    out.append(table(["Property (held / applies)"] + [f"`{m}`" for m in models],
                     [[n.replace("_", " ")] + [f"{d[m]['summary']['properties'][n]['held']} / {d[m]['summary']['properties'][n]['applies']}"
                                               + (f" (failed: {', '.join(d[m]['summary']['properties'][n]['failed_in'])})" if d[m]["summary"]["properties"][n]["failed_in"] else "")
                                               for m in models] for n in names]))
    m0 = models[0]
    out.append(table([f"Case (`{m0}`)", "Injected", "Fault met by the agent", "Tools called", "Tables shown", "Result"],
                     [[c["id"], (", ".join(sorted(set(c["injected"].values()))) if isinstance(c["injected"], dict) else c["injected"]) or "",
                       ("yes" if all(c["injected_table_was_found_by_the_agent"].values()) else "no") if c["injected_table_was_found_by_the_agent"] else "",
                       " → ".join(c["tools"]) or "none", len(c["listed"]), "pass" if c["passed"] else "FAIL: " + ", ".join(k for k, v in c["properties"].items() if v is False)]
                      for c in d[m0]["cases"]]))
    return "\n".join(out)


def rollout(d: dict[str, Any], table: T) -> str:
    s, t, g = d["scripted_sessions"], d["triage"], d["gate"]
    om = s["online_measures"]
    out = ["## Roll-out: feedback to cases, and the release gate\n", f"Evidence: `vxfind_rollout.json`. **{s['what']}**\n",
           table(["Online measures (SCRIPTED sessions)", "Value"], [[k.replace("_", " "), v] for k, v in om.items()]),
           f"Triage: {t['candidate_cases']} candidate cases ({kv(t['reasons'])}; by query set: {kv(t['by_query_set'])}), written to `{t['written_to']}`. "
           f"{cap(t['status'])}.\n",
           f"### The gate\n\nRules: no significant loss (alpha {g['rules']['alpha']}) and no drop larger than "
           + ", ".join(f"{v['max_drop']} on {k} {v['metric']}" for k, v in g["rules"]["no_loss"].items())
           + f"; floors: {kv(g['rules']['floors'])}. Floors as measured: {kv(g['floors_measured'])}. Candidates are compared with the serving "
             "configuration on the locked test queries.\n"]
    rows = []
    for name, dec in g["decisions"].items():
        cells = []
        for c in dec["checks"]:
            if "difference" in c:
                cells.append(f"{c['check']}: {c['difference']:+.3f} (p {c['p']:.3f}) {'ok' if c['passed'] else 'FAIL'}")
        floors = "; ".join(f"{c['check'].replace('_', ' ')}: {c.get('value')} {'ok' if c['passed'] else 'FAIL ' + c['why']}" for c in dec["checks"] if "limit" in c)
        rows.append([name, "PASS" if dec["passed"] else "FAIL", "; ".join(cells), floors])
    out.append(table(["Candidate", "Decision", "No-loss checks", "Floors"], rows))
    return "\n".join(out)


def capacity(d: dict[str, Any], table: T) -> str:
    out = ["## Capacity: searches per second, latency, storage, growth\n",
           f"Evidence: `vxfind_capacity.json`. {cap(d['machine'])}. Index: {d['index']['tables']} tables. {d['needs_cycled']} needs cycled, closed loop.\n",
           table(["Retriever", "Callers", "Searches per second", "p50 ms", "p95 ms", "p99 ms", "Errors"],
                 [[f"`{r['retriever']}`", r["workers"], r["per_second"], r["p50_ms"], r["p95_ms"], r["p99_ms"], r["errors"]] for r in d["load"]])]
    out.append("Throughput stops growing between 4 and 16 callers; beyond that, more callers only wait longer. The load generator is "
               "Python threads in one process on the same machine, and it was not separated whether it or the engine is the limit there.\n")
    fr = d.get("first_run") or {}
    if fr.get("serving_32_callers"):
        f = fr["serving_32_callers"]
        out.append(f"First run (`{fr['evidence']}`): {fr['what_it_showed']}; for `serving` at 32 callers it measured {f['per_second']} per second "
                   f"with p50 {f['p50_ms']} ms and p99 {f['p99_ms']} ms. The client now opens {d.get('client_connections')} connections.\n")
    c = d["query_vector_computed_per_search"]
    out.append(f"With the query vector computed for every search (`{c['retriever']}`, {c['workers']} callers): {c['per_second']} per second, "
               f"p50 {c['p50_ms']} ms, p95 {c['p95_ms']} ms.\n")
    s, g = d["storage"], d["growth"]
    out.append(table(["Storage and growth", "Value"], [
        ["Index on disk, MB / bytes per table", f"{s['index_mb']} / {s['bytes_per_table']}"],
        ["Raw vector bytes per table (two models)", s["vector_bytes_per_table_raw"]],
        ["Data the catalog describes (declared by the tables), TB", f"{s['data_the_catalog_describes_tb']} over {s['tables_declaring_size']} tables"],
        ["Frames described", s["frames_described"]],
        ["Tables created per month, last full months", kv(g["created_per_month_recent"])],
        ["Mean per month / share of the catalog", f"{g['mean_created_per_month_recent']} / {pct(g['monthly_growth_of_catalog'])}"],
        ["Mean per month in the six months before", g.get("mean_created_per_month_in_the_six_months_before")],
        ["Projected tables / index MB in 12 months", f"{g['projected_tables_in_12_months']} / {g['projected_index_mb_in_12_months']}"]]))
    out.append(f"{cap(g['note'])}.\n")
    return "\n".join(out)


def failures(d: dict[str, Any], table: T) -> str:
    out = ["## Where the serving configuration fails (development queries)\n", "Evidence: `vxfind_failures.json`. Counts that code can make; the lists say where to read.\n"]
    keys = sorted({k for v in d["counts_by_query_set"].values() for k in v})
    out.append(table(["Count over the top 10"] + list(d["counts_by_query_set"]),
                     [[k.replace("_", " ")] + [v.get(k, "") for v in d["counts_by_query_set"].values()] for k in keys]))
    for kind, rows in d["lowest_scoring_queries"].items():
        out.append(table([f"Lowest-scoring set {kind} needs", "Score", "First results and what is wrong with them"],
                         [[r["need"][:110], next(v for k, v in r.items() if "@" in k),
                           "; ".join(f"`{t['table_id']}`" + (f" ({t['note'][:70]})" if t["note"] else "") for t in r["top_5"][:3])] for r in rows[:4]]))
    return "\n".join(out)


SECTIONS = [("vxfind_queries", queries), ("vxfind_write_path", write_path), ("vxfind_retrieval", retrieval), ("vxfind_judge", judge),
            ("vxfind_judge_iterations", judge_iterations),
            ("vxfind_llm_stages", llm_stages), ("vxfind_answers", answers), ("vxfind_meta", meta), ("vxfind_failures", failures),
            ("vxfind_fresh", fresh), ("vxfind_access", access), ("vxfind_agent", agent), ("vxfind_rollout", rollout),
            ("vxfind_capacity", capacity)]
