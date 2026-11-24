"""Where to read: the serving configuration's worst development queries and failure counts (E01 R8, E06 R10).

    python scripts/vxfind_failures.py        -> verification/vxfind_failures.json

It counts the failure types that code can detect and lists the lowest-scoring queries with what
was returned and why the judge marked each wrong table down. It does not replace reading the
results; it says where to start.
"""
from __future__ import annotations

import collections

from _vxfind import JUDGE_MODEL, Q, engine, evidence, llm, log, records, serving

from vxfind.eval import harness as H
from vxfind.eval.judge import Judge, sources


def main() -> None:
    eng = engine()
    recs = records()
    dev = Q.all_queries("dev")
    judge = Judge(llm(JUDGE_MODEL))
    qrels = judge.qrels(dev)
    reasons = {(j["qid"], j["table_id"]): j["reason"] for j in judge.store.values()}
    cfg = serving()
    run = H.run(eng, cfg, dev)
    pq = H.per_query(dev, run, recs, qrels)
    by = {q["qid"]: q for q in dev}
    src = sources(eng.es, eng.index, sorted({t for v in run.values() for t in v["ids"][:10]}))

    counts: dict[str, collections.Counter] = {k: collections.Counter() for k in "ABCD"}
    for q in dev:
        ids, fams = run[q["qid"]]["ids"][:10], run[q["qid"]]["families"][:10]
        c = counts[q["kind"]]
        c["queries"] += 1
        c["results"] += len(ids)
        scratch = sum(bool(src.get(t, {}).get("is_scratch")) for t in ids)
        undesc = sum(not src.get(t, {}).get("has_description") for t in ids)
        c["scratch_tables_in_top_10"] += scratch
        c["queries_with_a_scratch_table"] += scratch > 0
        c["tables_with_no_description_in_top_10"] += undesc
        c["repeated_family_in_top_10"] += round(H.repeat_at(fams) * len(fams))
        if q["kind"] == "A":
            g = [Q.grade_attribute(q["spec"], recs[t]) for t in ids if t in recs]
            c["F3_wrong_robot"] += sum(x == 0 for x in g)
            c["F6_right_robot_constraint_missed"] += sum(x == 1 for x in g)
            c["queries_with_no_fully_matching_table"] += not any(x == 3 for x in g)
        elif q["kind"] == "B":
            hit = pq["B"]["hit@10"][q["qid"]]
            c["F12_target_not_in_top_10"] += hit == 0
            c["F12_target_within_first_100_but_not_top_10"] += hit == 0 and pq["B"]["hit@100"][q["qid"]] == 1
        elif q["kind"] in "CD":
            g = [qrels.get(q["qid"], {}).get(t) for t in ids]
            c["unjudged"] += sum(x is None for x in g)
            c["grade_0"] += sum(x == 0 for x in g)
            c["grade_1_related_only"] += sum(x == 1 for x in g)
            c["queries_with_no_usable_table"] += not any((x or 0) >= 2 for x in g)

    worst = {}
    for kind, metric in H.HEADLINE.items():
        order = sorted(pq[kind][metric].items(), key=lambda kv: kv[1])[:6]
        rows = []
        for qid, score in order:
            q = by[qid]
            shown = []
            for t in run[qid]["ids"][:5]:
                if kind == "A":
                    note = {3: "fits", 1: "right robot, a constraint is missed", 0: "another robot"}[Q.grade_attribute(q["spec"], recs[t])] if t in recs else ""
                elif kind == "B":
                    note = "the target" if t == q["target"] else ""
                else:
                    g = qrels.get(qid, {}).get(t)
                    note = f"grade {g}: {reasons.get((qid, t), '')}" if g is not None else "unjudged"
                shown.append({"table_id": t, "note": note})
            rows.append({"qid": qid, "need": q["need"], metric: round(score, 3), "expected": q.get("target") or q.get("spec"),
                         "top_5": shown})
        worst[kind] = rows
    evidence("failures", {"serving": cfg.name, "split": "dev", "counts_by_query_set": {k: dict(v) for k, v in counts.items()},
                          "lowest_scoring_queries": worst})
    log("failures evidence written")


if __name__ == "__main__":
    main()
