"""The retrieval evaluation, stage by stage. Every stage writes verification/vxfind_<stage>.json.

    python scripts/vxfind_eval.py retrieval     configurations on dev, pooling, judging, choice of the serving one, test
    python scripts/vxfind_eval.py judge         how far the judge can be trusted
    python scripts/vxfind_eval.py llm_stages    understanding, reranking and selection by a model
    python scripts/vxfind_eval.py answers       reasons (groundedness) and "nothing fits" (abstention)
    python scripts/vxfind_eval.py meta          planted faults, power, agreement between the query sets

Stages read each other's stored runs and judgments; nothing is searched or judged twice.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random

from _vxfind import EVALS, JUDGE_MODEL, SMALL_MODEL, Q, engine, evidence, index_info, llm, log, read, records, save_serving, serving

from vxfind.eval import answers as ANS
from vxfind.eval import harness as H
from vxfind.eval import meta as META
from vxfind.eval import metrics as M
from vxfind.eval.judge import PROMPT_VERSION, Judge, sources
from vxfind.search.engine import describe

TEST_CONFIGS = ["bm25", "knn_nomic", "rrf", "rrf_no_collapse", "bm25_name_only"]


def judge_pool(judge: Judge, eng, queries, pools) -> dict:
    """Judge every pooled (need, table) pair of the C and D queries that has no judgment yet."""
    by = {q["qid"]: q for q in queries if q["kind"] in "CD"}
    ids = sorted({t for qid in by for t in pools.get(qid, [])})
    srcs = sources(eng.es, eng.index, ids)
    pairs = [(qid, by[qid]["need"], t, srcs[t]) for qid in sorted(by) for t in pools.get(qid, []) if t in srcs]
    before = judge.calls
    judge.grade_many(pairs, workers=4, log=log)
    return {"pairs": len(pairs), "judged_now": judge.calls - before, "failed": judge.failures,
            "seconds_per_judgment": round(judge.seconds / judge.calls, 2) if judge.calls else None}


def scored(eng, cfg, queries, recs, qrels, **kw):
    r = H.run(eng, cfg, queries, log=log, **kw)
    return r, H.per_query(queries, r, recs, qrels)


def flat(summary: dict) -> dict:
    return {k: summary[k][m]["mean"] for k, m in H.HEADLINE.items() if k in summary and m in summary[k]}


# ---------------------------------------------------------------------------- retrieval

def retrieval() -> None:
    eng = engine()
    recs = records()
    dev, test = Q.all_queries("dev"), Q.all_queries("test")
    judge = Judge(llm(JUDGE_MODEL))
    log(f"index {index_info(eng.es)}; dev {len(dev)} queries, test {len(test)}")

    # 1. every configuration on the development queries
    runs = {name: H.run(eng, cfg, dev, log=log) for name, cfg in H.CONFIGS.items()}
    pools = H.pool(runs, 10)
    judged_dev = judge_pool(judge, eng, dev, pools)
    log(f"dev pool judged: {judged_dev}")
    qrels = judge.qrels(dev + test)
    pq = {name: H.per_query(dev, r, recs, qrels) for name, r in runs.items()}
    table = {name: H.summarise(p) for name, p in pq.items()}
    versus = {name: H.compare(pq["rrf"], p) for name, p in pq.items() if name != "rrf"}

    # 2. choose the serving configuration, by a rule written before the numbers were seen:
    #    start from plain fusion; take a setting when it is significantly better on at least one
    #    query set and significantly worse on none; combine what was taken; keep the combination
    #    only if it is not significantly worse than plain fusion anywhere.
    def verdict(c: dict) -> tuple[int, int]:
        wins = sum(v["mean_difference"] > 0 and v["p_randomization"] < 0.05 for v in c.values())
        losses = sum(v["mean_difference"] < 0 and v["p_randomization"] < 0.05 for v in c.values())
        return wins, losses

    single = {"embedding": ["rrf_bge"], "rank_constant": ["rrf_c10", "rrf_c100"], "window": ["rrf_w20", "rrf_w50", "rrf_w200"],
              "mess": ["rrf_scratch_0.5", "rrf_scratch_0.2", "rrf_described_1.5", "rrf_described_2", "rrf_scratch_0.5_described_1.5"]}
    taken, change = [], {}
    for group, names in single.items():
        best = None
        for n in names:
            w, l = verdict(versus[n])
            gain = sum(v["mean_difference"] for v in versus[n].values())
            if w >= 1 and l == 0 and (best is None or gain > best[1]):
                best = (n, gain)
        if best:
            taken.append(best[0])
            base, var = describe(H.CONFIGS["rrf"]), describe(H.CONFIGS[best[0]])
            change.update({k: var[k] for k in var if k != "name" and var[k] != base[k]})
    if change:
        change.pop("fields", None)
        cand = H.CONFIGS["rrf"].but(name="serving", **change)
        r_c, pq_c = scored(eng, cand, dev, recs, qrels)
        top_up = judge_pool(judge, eng, dev, H.pool({"serving": r_c}, 10))
        qrels = judge.qrels(dev + test)
        pq_c = H.per_query(dev, r_c, recs, qrels)
        cmp_c = H.compare(pq["rrf"], pq_c)
        w, l = verdict(cmp_c)
        chosen = cand if l == 0 else H.CONFIGS[max(taken, key=lambda n: sum(v["mean_difference"] for v in versus[n].values()))].but(name="serving")
        why = {"settings_taken": taken, "combined_change": change, "combination_vs_rrf_on_dev": cmp_c,
               "combination_kept": l == 0, "top_up_judgments": top_up["judged_now"]}
    else:
        chosen = H.CONFIGS["rrf"].but(name="serving")
        why = {"settings_taken": [], "note": "no setting was significantly better than plain fusion on any query set"}
    save_serving(chosen, why)
    log(f"serving configuration: {why.get('settings_taken')} -> {describe(chosen)}")

    # 3. the locked test queries: the chosen configuration and a few reference points
    test_cfgs = {n: H.CONFIGS[n] for n in TEST_CONFIGS} | {"serving": chosen}
    truns = {n: H.run(eng, c, test, log=log, tag="_test") for n, c in test_cfgs.items()}
    judged_test = judge_pool(judge, eng, test, H.pool(truns, 10))
    r_dev_serving = H.run(eng, chosen, dev, log=log)
    qrels = judge.qrels(dev + test)
    tpq = {n: H.per_query(test, r, recs, qrels) for n, r in truns.items()}
    ttable = {n: H.summarise(p) for n, p in tpq.items()}
    tversus = {n: H.compare(tpq["rrf"], p) for n, p in tpq.items() if n != "rrf"}

    # 4. how much does returning more than ten help (known-item needs; the target's rank in the first 100)
    def more(run_, qs):
        out = {}
        for k in (1, 5, 10, 20, 50, 100):
            hits = []
            for q in qs:
                if q["kind"] != "B" or q["qid"] not in run_:
                    continue
                v = run_[q["qid"]]
                lst = (v["ids"] + [t for t in v["deep"] if t not in v["ids"]])[:k] if k > len(v["ids"]) else v["ids"][:k]
                fams = dict(zip(v["ids"], v["families"]))
                hits.append(float(q["target"] in lst or any(fams.get(t) == q["target_family"] for t in lst)))
            out[f"found_in_first_{k}"] = round(M.mean(hits), 4)
        return out

    # 5. the no-answer set: does anything usable turn up after all
    d_all = [q for q in dev + test if q["kind"] == "D"]
    d_found = {q["qid"]: max(qrels.get(q["qid"], {}).values(), default=0) for q in d_all}
    answerable = sorted(k for k, v in d_found.items() if v >= 2)
    (EVALS / "queries" / "D_answerable.json").write_text(json.dumps(answerable), encoding="utf-8")

    grades = collections.Counter(g for q in dev + test if q["kind"] == "C" for g in qrels.get(q["qid"], {}).values())
    fr = read("retrieval_first_run")
    first_run = {"evidence": "vxfind_retrieval_first_run.json", "judge_rubric": "g2 (the first one)",
                 "settings_the_rule_took": (fr.get("serving") or {}).get("why", {}).get("settings_taken"),
                 "grades_given_to_C_pairs": (fr.get("judging") or {}).get("grades_given_to_C_pairs"),
                 "C_ndcg@10_of_rrf_on_dev": ((fr.get("dev") or {}).get("rrf") or {}).get("scores", {}).get("C", {}).get("ndcg@10", {}).get("mean"),
                 "no_answer_needs_that_judge_called_answerable": len((fr.get("no_answer_set") or {}).get("turned_out_to_have_a_usable_table", []))} if fr else None
    evidence("retrieval", {
        "first_run_with_the_first_rubric": first_run, "judge_rubric": PROMPT_VERSION,
        "index": index_info(eng.es), "judge": JUDGE_MODEL,
        "queries": {"dev": dict(collections.Counter(q["kind"] for q in dev)), "test": dict(collections.Counter(q["kind"] for q in test))},
        "headline_metrics": H.HEADLINE,
        "judging": {"dev_pool": judged_dev, "test_pool": judged_test, "pool_depth": 10, "configurations_pooled_dev": len(H.CONFIGS) + 1,
                    "configurations_pooled_test": len(test_cfgs), "grades_given_to_C_pairs": {str(k): grades[k] for k in sorted(grades)},
                    "pairs_per_C_query_mean": round(sum(len(qrels.get(q["qid"], {})) for q in dev + test if q["kind"] == "C")
                                                    / max(1, sum(q["kind"] == "C" for q in dev + test)), 1)},
        "dev": {name: {"config": {k: v for k, v in describe(H.CONFIGS[name]).items() if k not in ("name", "fields")},
                       "scores": table[name], "vs_rrf": versus.get(name)} for name in H.CONFIGS},
        "serving": {"config": describe(chosen), "why": why, "dev_scores": H.summarise(H.per_query(dev, r_dev_serving, recs, qrels))},
        "test": {n: {"scores": ttable[n], "vs_rrf": tversus.get(n)} for n in test_cfgs},
        "returning_more_than_ten": {"dev_serving": more(r_dev_serving, dev), "test_serving": more(truns["serving"], test)},
        "latency_ms": {n: {"total": H.latency(runs[n]), "bm25": H.latency(runs[n], "bm25"), "knn": H.latency(runs[n], "knn")}
                       for n in ("bm25", "knn_nomic", "rrf", "rrf_no_collapse")},
        "no_answer_set": {"queries": len(d_all), "best_grade_found_in_pool": dict(collections.Counter(str(v) for v in d_found.values())),
                          "turned_out_to_have_a_usable_table": [{"qid": q["qid"], "need": q["need"]} for q in d_all if q["qid"] in answerable]},
    })
    log("retrieval evidence written")


# ---------------------------------------------------------------------------- judge

PINNED = EVALS / "labels" / "calibration_pairs.json"


def calibration_pairs(eng, recs, dev, rng, pinned=None):
    """(need, table) pairs of the dev attribute needs with their computed grade: per need, tables that
    meet every constraint, tables of the right robot that miss one, tables of another robot that
    search returned, and random tables. The pairs are pinned in a file after the first sampling, so
    that every version of the rubric is measured on the same pairs; the grade is computed afresh."""
    pinned = pinned or PINNED
    by = {q["qid"]: q for q in dev if q["kind"] == "A"}
    if pinned.exists():
        chosen = [(x["qid"], x["table_id"]) for x in json.loads(pinned.read_text(encoding="utf-8")) if x["qid"] in by]
    else:
        by_fam = collections.defaultdict(list)
        for r in recs.values():
            by_fam[r["robot_family"]].append(r)
        chosen = []
        run_rrf = H.run(eng, H.CONFIGS["rrf"], dev, tag="_test" if pinned != PINNED else "")
        for q in by.values():
            fam = by_fam[q["spec"]["family"]]
            g3 = [r for r in fam if Q.grade_attribute(q["spec"], r) == 3]
            g1 = [r for r in fam if Q.grade_attribute(q["spec"], r) == 1]
            retrieved = [recs[t] for t in run_rrf[q["qid"]]["ids"] if t in recs]
            hard0 = [r for r in retrieved if Q.grade_attribute(q["spec"], r) == 0]          # other robot, but search liked it
            easy0 = rng.sample(list(recs.values()), 2)
            for r in rng.sample(g3, min(3, len(g3))) + rng.sample(g1, min(3, len(g1))) + hard0[:2] + [e for e in easy0 if Q.grade_attribute(q["spec"], e) == 0]:
                chosen.append((q["qid"], r["table_id"]))
        pinned.parent.mkdir(parents=True, exist_ok=True)
        pinned.write_text(json.dumps([{"qid": q, "table_id": t} for q, t in chosen], indent=0), encoding="utf-8")
    srcs = sources(eng.es, eng.index, sorted({t for _, t in chosen}))
    keep = [(q, t) for q, t in chosen if t in srcs and t in recs]
    return [(q, by[q]["need"], t, srcs[t]) for q, t in keep], [Q.grade_attribute(by[q]["spec"], recs[t]) for q, t in keep]


def rubric_versions(pairs, truth) -> dict:
    """Every version of the rubric that has stored judgments for the pinned pairs, measured against
    the computed truth as it is now. No model is called."""
    from vxfind.eval.judge import key
    out = {}
    for model in (JUDGE_MODEL, SMALL_MODEL):
        store = Judge(llm(model)).store
        for version in ("g2", "g3", "g4"):
            got = [(t, store[k]["grade"]) for (qid, need, tid, src), t in zip(pairs, truth)
                   if (k := key(model, need, src.get("content_hash") or "", version)) in store]
            if len(got) >= 30:
                c = META.calibrate([a for a, _ in got], [b for _, b in got])
                out.setdefault(version, {})[model] = {k2: c[k2] for k2 in ("pairs", "agreement", "kappa", "judge_precision", "judge_recall",
                                                                         "false_positive_rate", "confusion")}
    return out


def calibrate_only() -> None:
    """The judge against computed truth, and nothing else: the quick check after a change of the rubric."""
    eng = engine()
    pairs, truth = calibration_pairs(eng, records(), Q.all_queries("dev"), random.Random(41))
    for model in (JUDGE_MODEL, SMALL_MODEL):
        j = Judge(llm(model))
        out = j.grade_many(pairs, log=log)
        ok = [i for i in range(len(pairs)) if out[i]]
        c = META.calibrate([truth[i] for i in ok], [out[i]["grade"] for i in ok])
        log(f"{model} prompt {PROMPT_VERSION}: " + json.dumps({k: c[k] for k in ("pairs", "agreement", "kappa", "judge_precision", "judge_recall", "false_positive_rate")}))
        log(f"   judge grade by true grade: {c['judge_grade_by_true_grade']}")
    first = json.loads(PINNED.read_text(encoding="utf-8"))
    changed = sum(1 for x, t in zip(first, truth) if "truth_as_first_computed" in x and x["truth_as_first_computed"] != t)
    evidence("judge_iterations", {
        "what": "each version of the judge's rubric against truth computed by code, on the same pinned pairs of the dev attribute "
                "needs; a version is listed for the pairs whose need text it was run on",
        "pinned_pairs": len(first), "pairs_whose_computed_truth_was_corrected": changed,
        "truth_correction": "a table with no episodes is not a match (found while reading the judge's disagreements)",
        "changes": {"g2": "four grades listed; the order of the checks left to the model",
                    "g3": "the checks as a procedure; a contradicted requirement caps the grade at 1 before anything else",
                    "g4": "spellings of robot names count as the robot; an empty dataset is grade 0"},
        "versions": rubric_versions(pairs, truth), "in_use": PROMPT_VERSION})



def judge_stage() -> None:
    eng = engine()
    recs = records()
    rng = random.Random(41)
    big, small = Judge(llm(JUDGE_MODEL)), Judge(llm(SMALL_MODEL))
    dev = Q.all_queries("dev")

    # (a) against labels computed by code: attribute needs, where truth needs no judge
    pairs, truth = calibration_pairs(eng, recs, dev, rng)
    out_big = big.grade_many(pairs, log=log)
    out_small = small.grade_many(pairs, log=log)
    ok = [i for i in range(len(pairs)) if out_big[i] and out_small[i]]
    cal_big = META.calibrate([truth[i] for i in ok], [out_big[i]["grade"] for i in ok])
    cal_small = META.calibrate([truth[i] for i in ok], [out_small[i]["grade"] for i in ok])
    wrong = [{"need": pairs[i][1], "table_id": pairs[i][2], "true_grade": truth[i], "judge_grade": out_big[i]["grade"],
              "judge_reason": out_big[i]["reason"]} for i in ok if (truth[i] >= 3) != (out_big[i]["grade"] >= 2)][:8]
    log(f"calibration: {cal_big['agreement']} agreement, kappa {cal_big['kappa']} on {cal_big['pairs']} pairs")

    # (a2) held out. The rubric was revised while looking at the pairs above. The same measurement
    # on pairs of the TEST attribute needs, which were never looked at, says how much of the gain is real.
    hp, ht = calibration_pairs(eng, recs, Q.all_queries("test"), random.Random(43), EVALS / "labels" / "calibration_pairs_test.json")
    hb, hs = big.grade_many(hp, log=log), small.grade_many(hp, log=log)
    held = {}
    for name, res in ((JUDGE_MODEL, hb), (SMALL_MODEL, hs)):
        okh = [i for i in range(len(hp)) if res[i]]
        held[name] = META.calibrate([ht[i] for i in okh], [res[i]["grade"] for i in okh])
    log(f"held out: {held[JUDGE_MODEL]['agreement']} agreement, kappa {held[JUDGE_MODEL]['kappa']} on {held[JUDGE_MODEL]['pairs']} pairs")
    # the first rubric on the same held-out pairs: the improvement, where it was not tuned
    from vxfind.eval.judge import RUBRIC_G2
    first = Judge(llm(JUDGE_MODEL), version="g2", rubric=RUBRIC_G2).grade_many(hp, log=log)
    okf = [i for i in range(len(hp)) if first[i]]
    held[JUDGE_MODEL + " with the first rubric (g2)"] = META.calibrate([ht[i] for i in okf], [first[i]["grade"] for i in okf])

    # (b) the need written from a table must be judged to fit that table
    bq = [q for q in dev if q["kind"] == "B"]
    bsrc = sources(eng.es, eng.index, [q["target"] for q in bq])
    bres = big.grade_many([(q["qid"], q["need"], q["target"], bsrc[q["target"]]) for q in bq if q["target"] in bsrc], log=log)
    bg = collections.Counter(j["grade"] for j in bres if j)

    # (c) two judges on the same pooled pairs of the written needs
    cq = {q["qid"]: q for q in dev if q["kind"] == "C"}
    cj = [j for j in big.store.values() if j["qid"] in cq and j["prompt"] == PROMPT_VERSION and j["judge"] == JUDGE_MODEL]
    rng.shuffle(cj)
    sample = cj[:300]
    ssrc = sources(eng.es, eng.index, sorted({j["table_id"] for j in sample}))
    spairs = [(j["qid"], cq[j["qid"]]["need"], j["table_id"], ssrc[j["table_id"]]) for j in sample if j["table_id"] in ssrc]
    s_big = big.grade_many(spairs)
    s_small = small.grade_many(spairs, log=log)
    both = [(a["grade"], b["grade"]) for a, b in zip(s_big, s_small) if a and b]
    two = META.agreement([a for a, _ in both], [b for _, b in both])

    # (d) the same pair judged twice gives the same grade (temperature 0)
    again = 0
    recheck = spairs[:60]
    fresh = Judge(llm(JUDGE_MODEL), name="recheck_scratch", folder=EVALS / "qrels" / "scratch")
    fresh.store.clear()
    if fresh.path.exists():
        fresh.path.unlink()
    r2 = fresh.grade_many(recheck)
    again = sum(1 for a, b in zip(s_big[:60], r2) if a and b and a["grade"] == b["grade"])

    # (e) the sheet a person fills in: 200 pairs, spread over the judge's grades
    by_grade = collections.defaultdict(list)
    for j in cj:
        by_grade[j["grade"]].append(j)
    sheet = []
    for g in (3, 2, 1, 0):
        sheet += by_grade[g][:50]
    rng.shuffle(sheet)
    lsrc = sources(eng.es, eng.index, sorted({j["table_id"] for j in sheet}))
    path = EVALS / "labels" / "hand_labels.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    filled = 0
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            old = list(csv.DictReader(f))
        filled = sum(1 for r in old if r.get("your_grade", "").strip() in ("0", "1", "2", "3"))
    if filled == 0:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["qid", "need", "table_id", "url", "table_text", "your_grade"])
            for j in sheet:
                s = lsrc.get(j["table_id"], {})
                # A short excerpt only: the sheet is in git and the tables' own text is not ours to republish.
                w.writerow([j["qid"], cq[j["qid"]]["need"], j["table_id"], "https://huggingface.co/datasets/" + j["table_id"],
                            (s.get("summary") or "")[:300], ""])
        from vxfind.eval.judge import doc_text
        full = EVALS.parents[1] / "data" / "vxfind" / "eval" / "hand_labels_full_text.csv"       # local only: exactly what the judge read
        full.parent.mkdir(parents=True, exist_ok=True)
        with full.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["qid", "table_id", "text_the_judge_read"])
            for j in sheet:
                w.writerow([j["qid"], j["table_id"], doc_text(lsrc.get(j["table_id"], {}))])
        hand = {"sheet": str(path.relative_to(EVALS.parents[1])).replace("\\", "/"), "pairs": len(sheet), "filled_by_a_person": 0,
                "status": "NOT DONE. The sheet hides the judge's grade. Until a person fills it in, agreement between the "
                          "judge and a person on the written needs is unknown; the comparison with computed truth above is the only comparison with truth."}
    else:
        jg = {(j["qid"], j["table_id"]): j["grade"] for j in cj}
        xs = [(int(r["your_grade"]), jg[(r["qid"], r["table_id"])]) for r in old
              if r.get("your_grade", "").strip() in ("0", "1", "2", "3") and (r["qid"], r["table_id"]) in jg]
        hand = {"sheet": str(path.relative_to(EVALS.parents[1])).replace("\\", "/"), "pairs": len(old), "filled_by_a_person": len(xs),
                "judge_vs_person": META.agreement([a for a, _ in xs], [b for _, b in xs]) if xs else None}

    evidence("judge", {
        "judge": JUDGE_MODEL, "second_judge": SMALL_MODEL, "prompt_version": PROMPT_VERSION, "one_table_per_call": True,
        "against_labels_computed_by_code": {"what": "attribute needs (dev): for each need, tables that meet every constraint, tables "
                                                    "of the right robot that miss one, tables of another robot that search returned, "
                                                    "and random tables", JUDGE_MODEL: cal_big, SMALL_MODEL: cal_small,
                                            "examples_where_the_judge_is_wrong": wrong,
                                            "held_out_pairs_of_the_test_attribute_needs": held,
                                            "note": "the rubric was revised while reading the dev pairs (see vxfind_judge_iterations.json); "
                                                    "the held-out pairs were not looked at"},
        "need_written_from_a_table_judged_against_that_table": {"pairs": sum(bg.values()), "grades": {str(k): bg[k] for k in sorted(bg)},
                                                                "share_judged_usable_or_better": round((bg[2] + bg[3]) / max(1, sum(bg.values())), 4)},
        "two_judges_on_pooled_pairs_of_written_needs": two,
        "same_pair_judged_twice": {"pairs": len(recheck), "same_grade": again},
        "hand_labels": hand,
        "cost": {"seconds_per_judgment_" + JUDGE_MODEL: round(big.seconds / big.calls, 2) if big.calls else None,
                 "seconds_per_judgment_" + SMALL_MODEL: round(small.seconds / small.calls, 2) if small.calls else None,
                 "judgments_stored": len(big.store),
                 "replies_that_could_not_be_read_" + JUDGE_MODEL: big.failures, "replies_that_could_not_be_read_" + SMALL_MODEL: small.failures},
    })
    log("judge evidence written")


# ---------------------------------------------------------------------------- model stages

def llm_stages() -> None:
    eng = engine(with_models=True)
    recs = records()
    rng = random.Random(3)
    dev = Q.all_queries("dev")
    b = [q for q in dev if q["kind"] == "B"]
    rng.shuffle(b)
    sub = [q for q in dev if q["kind"] in "ACD"] + b[:30]
    judge = Judge(llm(JUDGE_MODEL))
    base = serving()
    cfgs = {"serving": base} | {n.replace("rrf_", "serving+"): base.but(name=n.replace("rrf_", "serving+"), understand=c.understand, rerank=c.rerank, select=c.select,
                                                                         understand_model=c.understand_model, understand_version=c.understand_version)
                                for n, c in H.LLM_CONFIGS.items()}
    runs = {}
    for n, c in cfgs.items():
        runs[n] = H.run(eng, c, sub, log=log, tag="_llm")
        log(f"{n}: done")
    top = judge_pool(judge, eng, sub, H.pool(runs, 10))
    qrels = judge.qrels(dev)
    pq = {n: H.per_query(sub, r, recs, qrels) for n, r in runs.items()}
    # how often the reading of an attribute need recovers what the need states, per reader
    from vxfind.eval.families import family_of

    def reading_quality(run_) -> dict:
        ok = collections.Counter()
        for q in sub:
            if q["kind"] != "A":
                continue
            rd = run_[q["qid"]].get("reading") or {}
            cons = q["spec"]["constraints"]
            for k, field in (("min_episodes", "min_episodes"), ("fps", "fps"), ("min_cameras", "cameras"), ("format", "format")):
                if k in cons:
                    ok[k + "_stated"] += 1
                    try:
                        ok[k + "_read_right"] += rd.get(field) is not None and float(str(rd[field]).lstrip("v")) == float(str(cons[k]).lstrip("v"))
                    except ValueError:                  # the model wrote something that is not a number: not read right
                        pass
                elif rd.get(field) is not None:
                    ok[k + "_filled_though_not_stated"] += 1
            ok["robot_stated"] += 1
            ok["robot_read"] += bool(rd.get("robot"))
            ok["robot_read_as_the_right_family"] += family_of(str(rd.get("robot") or "").replace("-", "").replace(" ", "_")) == q["spec"]["family"] \
                or family_of(str(rd.get("robot") or "")) == q["spec"]["family"]
        return dict(ok)

    reading = {SMALL_MODEL: reading_quality(runs["serving+understand_hard"]), JUDGE_MODEL: reading_quality(runs["serving+understand_hard_large"])}
    first = read("llm_stages_first_run")
    evidence("llm_stages", {
        "small_model": SMALL_MODEL, "large_model": JUDGE_MODEL, "base": describe(base),
        "queries": dict(collections.Counter(q["kind"] for q in sub)), "top_up_judgments": top,
        "configurations": {n: {"scores": H.summarise(pq[n]), "vs_serving": H.compare(pq["serving"], pq[n]) if n != "serving" else None,
                               "latency_ms": {s: H.latency(runs[n], s) for s in ("total", "understand", "rerank", "select") if H.latency(runs[n], s)},
                               "notes": dict(collections.Counter(x.split(" (")[0].split(":")[0] for v in runs[n].values() for x in v["notes"]).most_common(5))}
                           for n in cfgs},
        "reading_of_attribute_needs": reading, "reading_prompt": base.understand_version,
        "first_run_with_the_first_reading_prompt": {"evidence": "vxfind_llm_stages_first_run.json",
                                                    "reading_by_" + SMALL_MODEL: first.get("reading_of_attribute_needs"),
                                                    "headline_scores": {n: {k: v["scores"][k][m]["mean"] for k, m in H.HEADLINE.items()}
                                                                        for n, v in (first.get("configurations") or {}).items() if "understand" in n}},
    })
    log("llm_stages evidence written")


# ---------------------------------------------------------------------------- answers

def answers() -> None:
    eng = engine(with_models=True)
    dev = Q.all_queries("dev")
    base = serving()
    moved = set(json.loads((EVALS / "queries" / "D_answerable.json").read_text(encoding="utf-8"))) if (EVALS / "queries" / "D_answerable.json").exists() else set()
    rng = random.Random(3)
    b = [q for q in dev if q["kind"] == "B"]
    rng.shuffle(b)
    sub = [q for q in dev if q["kind"] in "ACD" and q["qid"] not in moved] + b[:30]
    out = {}
    for n in ("serving+select", "serving+rerank_large_select", "serving+understand_hard_large_select"):
        cfg = base.but(name=n, select=True, rerank="large" if "rerank" in n else "off", understand="hard" if "understand" in n else "off",
                       understand_model="large" if "understand" in n else "small")
        r = H.run(eng, cfg, sub, log=log, tag="_llm")
        ids = sorted({t for v in r.values() for t in v["ids"]})
        srcs = sources(eng.es, eng.index, ids)
        # The model checks of the reasons are slow and depend only on the stored run: they are kept
        # with a fingerprint of the reasons and done again only when the reasons changed.
        import hashlib
        mark = hashlib.sha256(json.dumps({q: v.get("reasons") for q, v in sorted(r.items())}, sort_keys=True).encode()).hexdigest()[:12]
        prev = (read("answers").get("configurations") or {}).get(n) or {}
        if n != "serving+select":
            g = g2 = None
        elif prev.get("reasons_fingerprint", mark) == mark and prev.get("groundedness_checked_by_" + SMALL_MODEL)                 and prev["groundedness_checked_by_" + SMALL_MODEL]["reasons"] == sum(len(v.get("reasons") or {}) for v in r.values()):
            g, g2 = prev["groundedness_checked_by_" + SMALL_MODEL], prev["groundedness_checked_by_" + JUDGE_MODEL]
        else:
            g, g2 = ANS.groundedness(r, sub, srcs, llm(SMALL_MODEL)), ANS.groundedness(r, sub, srcs, llm(JUDGE_MODEL))
        if g:                                           # the part done by code is cheap: always from the code as it is now
            code = ANS.number_check(r, sub, srcs)
            g, g2 = {**g, **{k: code[k] for k in list(code)[1:]}}, {**g2, **{k: code[k] for k in list(code)[1:]}}
            needs = {q["qid"]: q["need"] for q in sub}
            for block in (g, g2):                       # examples kept from an earlier run are checked again by today's code
                fresh = []
                for e in block["examples"]:
                    nums = ANS.unsupported_numbers(e["reason"], {**srcs.get(e["table_id"], {}), "table_id": e["table_id"]}, needs.get(e["qid"], ""))
                    if nums or e.get("model_says_unsupported"):
                        fresh.append({**e, "numbers_not_in_text": nums})
                block["examples"] = fresh
        out[n] = {"abstention": ANS.abstention(r, sub), "reasons_fingerprint": mark,
                  "groundedness_checked_by_" + SMALL_MODEL: g, "groundedness_checked_by_" + JUDGE_MODEL: g2}
        log(f"{n}: {out[n]['abstention']}")
    evidence("answers", {"selection_model": JUDGE_MODEL, "queries": dict(collections.Counter(q["kind"] for q in sub)),
                         "no_answer_needs_moved_out_because_a_usable_table_exists": sorted(moved), "configurations": out,
                         "note": "the reasons were written by " + JUDGE_MODEL + "; a check by the same model is a self-check and is "
                                 "reported beside the check by the other model and the check by code"})
    log("answers evidence written")


# ---------------------------------------------------------------------------- meta

def meta() -> None:
    eng = engine()
    recs = records()
    judge = Judge(llm(JUDGE_MODEL))
    ret = read("retrieval")
    all_ids = list(recs)
    out = {}
    for split in ("dev", "test"):
        qs = Q.all_queries(split)
        qrels = judge.qrels(qs)
        base_run = H.run(eng, serving(), qs, tag="_test" if split == "test" else "")
        base = H.per_query(qs, base_run, recs, qrels)
        faults = {}
        for how in ("one_random", "half_random", "second_page", "shuffled_top10", "reversed_top10"):
            bad = H.per_query(qs, META.degraded(base_run, how, all_ids), recs, qrels)
            faults[how] = META.detected(H.compare(base, bad))
            if how.endswith("top10"):       # control: a metric that ignores order must not move when only the order changes
                faults[how]["control_fit@10_changed"] = H.compare({"A": {"fit@10": base["A"]["fit@10"]}}, {"A": {"fit@10": bad["A"]["fit@10"]}}) != {} and \
                    any(abs(base["A"]["fit@10"][k] - bad["A"]["fit@10"][k]) > 1e-9 for k in base["A"]["fit@10"])
        other = H.per_query(qs, H.run(eng, H.CONFIGS["rrf"], qs, tag="_test" if split == "test" else ""), recs, qrels)
        out[split] = {"queries": dict(collections.Counter(q["kind"] for q in qs)), "planted_faults": faults,
                      "power_from_serving_vs_rrf": META.power(other, base)}
    table = {n: {k: v["scores"][k][m]["mean"] for k, m in H.HEADLINE.items() if k in v["scores"] and m in v["scores"][k]}
             for n, v in ret.get("dev", {}).items()}
    out["agreement_between_query_sets_dev"] = META.set_agreement(table)
    out["configurations_ranked"] = {k: sorted(table, key=lambda n: -table[n].get(k, 0))[:6] for k in "ABC"}
    evidence("meta", out)
    log("meta evidence written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["retrieval", "judge", "calibrate", "llm_stages", "answers", "meta"])
    a = ap.parse_args()
    {"retrieval": retrieval, "judge": judge_stage, "calibrate": calibrate_only, "llm_stages": llm_stages, "answers": answers, "meta": meta}[a.stage]()
