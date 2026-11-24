"""Build the evaluation's query sets (see vxfind/eval/queries.py for what each one is).

    python scripts/vxfind_queries.py A            attribute needs, labels computed by code
    python scripts/vxfind_queries.py CD           stamp the hand-written needs with ids and splits
    python scripts/vxfind_queries.py B --n 110    known-item needs written by a local model

Writes evals/vxfind/queries/{A,B,C,D}.jsonl and verification/vxfind_queries.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vxfind.eval import queries as Q                                   # noqa: E402
from vxfind.eval.families import FAMILIES, family_of                   # noqa: E402

OUT = ROOT / "verification" / "vxfind_queries.json"


def note(section: str, data: dict) -> None:
    cur = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    cur[section] = data
    OUT.write_text(json.dumps(cur, indent=1, ensure_ascii=False), encoding="utf-8")


def written(name: str) -> list[str]:
    return [l.strip() for l in (Q.QDIR / name).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["A", "A_recount", "B", "CD"])
    ap.add_argument("--n", type=int, default=110)
    ap.add_argument("--model", default="qwen3:14b")
    a = ap.parse_args()

    if a.stage == "CD":
        c = Q.stamp("C", [{"need": n, "source": "written by hand before any result was seen"} for n in written("C_written.txt")])
        d = Q.stamp("D", [{"need": l.split("|", 1)[1], "why_no_answer": l.split("|", 1)[0]} for l in written("D_written.txt")])
        Q.save("C", c)
        Q.save("D", d)
        note("C", {"queries": len(c), "dev": sum(q["split"] == "dev" for q in c), "test": sum(q["split"] == "test" for q in c),
                   "author": "one person, the builder of the system; not a newcomer"})
        note("D", {"queries": len(d), "dev": sum(q["split"] == "dev" for q in d), "test": sum(q["split"] == "test" for q in d),
                   "kinds": dict(collections.Counter(q["why_no_answer"] for q in d))})
        print(len(c), "C,", len(d), "D")
        return

    recs = Q.records("old")
    if a.stage == "A_recount":
        # Set A keeps its ids and specs. Two things found while calibrating the judge are repaired:
        # the wording of one robot family, and the count of matching tables now that a table with
        # no episodes is not a match.
        from vxfind.eval.families import label
        old_label = "Universal Robots arm (UR3, UR5, UR10)"
        qs, reworded, fewer, thin = Q.load("A"), 0, 0, []
        for q in qs:
            if old_label in q["need"]:
                q["need"] = q["need"].replace(old_label, label("ur"))
                reworded += 1
            pool = [r for r in recs if r["robot_family"] == q["spec"]["family"]]
            full = [r for r in pool if Q.grade_attribute(q["spec"], r) == 3]
            fewer += len(full) < q["matching_tables"]
            q["matching_tables"], q["matching_groups"] = len(full), len({r["family"] for r in full})
            if len(full) < 5:
                thin.append(q["qid"])
        Q.save("A", qs)
        note("A_repaired", {"needs_reworded": reworded, "why_reworded": "the family label listed three models and was read as excluding the others",
                            "queries_with_fewer_matching_tables": fewer, "why_fewer": "a table with no episodes no longer counts as a match",
                            "queries_now_with_fewer_than_5_matches": thin,
                            "matching_tables_median": sorted(q["matching_tables"] for q in qs)[len(qs) // 2]})
        print("A repaired:", reworded, "reworded;", fewer, "with fewer matches;", thin)
        return
    if a.stage == "A":
        fam = collections.Counter(r["robot_family"] for r in recs)
        qs = Q.build_attribute(recs, n=60)
        Q.save("A", qs)
        note("robot_families", {"rules": len(FAMILIES), "tables": len(recs), "tables_in_a_family": sum(v for k, v in fam.items() if k),
                                "tables_without_robot_type": sum(1 for r in recs if not r["robot_type"]),
                                "tables_with_a_robot_type_no_rule_matches": sum(1 for r in recs if r["robot_type"] and not r["robot_family"]),
                                "distinct_robot_type_values": len({r["robot_type"] for r in recs if r["robot_type"]}),
                                "distinct_values_matched": len({r["robot_type"] for r in recs if r["robot_family"]}),
                                "per_family": {k: v for k, v in fam.most_common() if k}})
        note("A", {"queries": len(qs), "dev": sum(q["split"] == "dev" for q in qs), "test": sum(q["split"] == "test" for q in qs),
                   "families_used": len({q["spec"]["family"] for q in qs}),
                   "constraints_used": dict(collections.Counter(k for q in qs for k in q["spec"]["constraints"])),
                   "constraints_per_query": dict(collections.Counter(len(q["spec"]["constraints"]) for q in qs)),
                   "matching_tables_median": sorted(q["matching_tables"] for q in qs)[len(qs) // 2],
                   "share_of_robots_tables_that_match_median": round(sorted(q["matching_tables"] / q["tables_of_robot"] for q in qs)[len(qs) // 2], 3),
                   "examples": [q["need"] for q in qs[:4]]})
        print(len(qs), "A queries")
        return

    from vxagent.llm.ollama import get_llm
    llm = get_llm(a.model)
    cands = Q.candidates_known_item(recs, int(a.n * 1.25))
    out, rejected = [], collections.Counter()
    t0 = time.time()
    for i, r in enumerate(cands, 1):
        need, why = Q.write_known_item(llm, r)
        if need is None:
            rejected[why] += 1
        else:
            out.append(Q.known_item_query(r, need))
        if i % 20 == 0:
            print(time.strftime("%H:%M:%S"), f"{i}/{len(cands)} written, {len(out)} kept", flush=True)
        if len(out) >= a.n:
            break
    Q.save("B", out)
    note("B", {"queries": len(out), "dev": sum(q["split"] == "dev" for q in out), "test": sum(q["split"] == "test" for q in out),
               "written_by": a.model, "tables_tried": i, "discarded": dict(rejected), "seconds": round(time.time() - t0, 1),
               "eligible_tables": sum(1 for r in recs if r["has_description"] and not r["is_scratch"] and r["is_head"] and len(r["description"] or "") >= 200),
               "robots": dict(collections.Counter(q["robot_family"] or "other" for q in out).most_common()),
               "examples": [q["need"] for q in out[:4]]})
    print(len(out), "B queries;", dict(rejected))


if __name__ == "__main__":
    main()
