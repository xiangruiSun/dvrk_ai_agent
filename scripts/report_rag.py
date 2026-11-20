"""The retrieval section of verification/REPORT.md, built from verification/rag_experiments.json."""
from __future__ import annotations

from typing import Any, Callable

HEAD = ["Strategy", "Chunking", "nDCG@10", "Recall@10", "MRR@10", "xref", "thread", "identifier", "p95 ms"]
NAMES = {"ident_vs_plain_tokenizer": "identifier tokenizer over plain tokenizer (BM25, section chunks)",
         "section_ctx_vs_fixed_chunking": "section chunks with breadcrumb over fixed windows (best lexical)",
         "fusion_vs_best_lexical": "fusion over the best lexical retriever",
         "fusion_vs_best_dense": "fusion over the best dense retriever"}


def _rows(runs: list[dict[str, Any]]) -> list[list[Any]]:
    return [[f"`{r['strategy']}`", r["chunking"], r["ndcg@10"], r["recall@10"], r["mrr@10"],
             r["by_category"]["xref"]["ndcg@10"], r["by_category"]["thread"]["ndcg@10"],
             r["by_category"]["identifier"]["ndcg@10"], r["latency_ms"]["p95"]] for r in runs]


def _why_not(x: dict[str, Any]) -> str:
    if not x["gates_passed"]:
        return "failed a gate"
    if not x["credible_improvement"]:
        return "interval includes zero"
    return "over the latency budget"


def rag(d: dict[str, Any], table: Callable[[list[str], list[list[Any]]], str]) -> str:
    c, q, src = d["corpus"], d["queries"], d["corpus"]["sources"]
    out = [
        "## RAG: retrieval over the dVRK documentation, threads, change log and schemas\n",
        f"Ran {d['ran']}. Evidence: `rag_experiments.json`. Corpus hash `{c['hash']}`.\n",
        table(["Source", "Units", "Notes"], [
            ["Documentation sections", c["units"].get("doc"),
             f"{src['doc']['files']} pages, {src['doc']['stub_sections']} stubs, {src['doc']['code_blocks']} code blocks"],
            ["Issue threads", c["units"].get("thread"),
             f"{src['thread']['threads']} threads ({src['thread']['closed']} closed), {src['thread']['replies']} "
             f"replies, {src['thread']['maintainer_replies']} by maintainers"],
            ["Change log", c["units"].get("changelog"), f"{src['changelog']['releases']} releases"],
            ["Configuration schemas", c["units"].get("schema"),
             f"{src['schema']['schemas']} schemas, {src['schema']['properties']} properties, "
             f"{src['schema']['without_description']} without a description"],
        ]),
        f"Cross-references: {c['validation']['references']}, pointing at an unknown label: "
        f"{c['validation']['references_to_unknown_labels']}.\n",
        f"Queries: {q['total']}, labelled by code ({q['labels']}). "
        + ", ".join(f"{k}: {v}" for k, v in q["by_category_and_split"].items()) + ".\n",
        "Development queries. The columns xref, thread and identifier are nDCG@10 for that query category.\n",
        "Baseline:\n", table(HEAD, _rows([d["baseline"]])),
        "Stage E, tokenizers and embedding models:\n", table(HEAD, _rows(d["stage_E"])),
        "Stage C, chunking:\n", table(HEAD, _rows(d["stage_C"])),
        "Stage R, fusion:\n", table(HEAD, _rows(d["stage_R"])),
        "Paired differences in nDCG@10 (95% bootstrap interval over queries):\n",
        table(["Comparison", "Mean difference", "Interval", "Wins / losses / ties"],
              [[NAMES.get(k, k), v["mean"], f"{v['low']} to {v['high']}", f"{v['wins']} / {v['losses']} / {v['ties']}"]
               for k, v in d["comparisons"].items()]),
    ]
    s = d["selection"]
    accepted = [x for x in s["decisions"] if x["accepted"]]
    refused = [x for x in s["decisions"] if not x["accepted"]]
    out.append(f"Selection policy: baseline `{s['baseline']}` (nDCG@10 {s['baseline_ndcg@10']}), latency budget "
               f"{s['latency_budget_ms']} ms p95. {len(accepted)} of {len(s['decisions'])} candidates were a "
               f"credible improvement within budget. Selected: `{s['selected']}`.\n")
    if refused:
        out.append("Not accepted: " + "; ".join(f"`{x['candidate']}` ({_why_not(x)})" for x in refused) + ".\n")
    f = d["final_test"]
    out.append("Held-out test queries, scored once for the selected configuration (baseline shown for reference):\n")
    out.append(table(["Configuration", "Queries", "nDCG@10", "Recall@5", "Recall@10", "MRR@10", "Unlabelled@10"],
                     [[f"`{r['strategy']}` @ {r['chunking']}", r["queries"], r["ndcg@10"], r["recall@5"],
                       r["recall@10"], r["mrr@10"], r["unlabelled@10"]] for r in (f["selected"], f["baseline"])]))
    df = f["difference"]
    out.append(f"Difference on the test queries: {df['mean']} ({df['low']} to {df['high']}); {df['wins']} wins, "
               f"{df['losses']} losses, {df['ties']} ties.\n")
    out.append("The labels are weak: they record what the authors linked or answered, so absolute scores "
               "understate quality, and the identifier category favours lexical retrieval by construction. "
               "Hardest development queries for the selected configuration:\n")
    for h in d["hardest_dev_queries_for_selected"][:6]:
        query = " ".join(h["query"].split())[:110]
        got = (h["returned_top3"] or ["nothing"])[0]
        out.append(f"- `{h['category']}`, nDCG {h['ndcg']}: \"{query}\". Expected `{h['expected'][0]}`, "
                   f"first result `{got}`.")
    out.append("")
    return "\n".join(out)
