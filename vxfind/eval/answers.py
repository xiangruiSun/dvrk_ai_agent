"""Evaluating what the person reads: the reasons, and the honesty of "nothing fits".

Retrieval metrics score a list of ids. The person does not read ids; they read a sentence per
table saying why it fits, and they trust it. Two properties are measured here:

  groundedness   every reason states only what the table's own text says. Checked twice: by code
                 (every number in the reason must occur in the table's text or metadata) and by
                 a model asked whether the text supports the sentence.
  abstention     for a need the catalog cannot meet, the selection shows nothing and says what is
                 missing; for a need it can meet, it does not give up.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .judge import doc_text

NUM = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(k\b)?", re.I)
SUPPORT_PROMPT = ("You check one sentence against a dataset's text. Answer whether the text states or directly implies "
                  "everything the sentence claims about the dataset. A claim the text does not mention is not supported. "
                  "Give the unsupported claim in at most 12 words, or an empty string.")
SUPPORT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["unsupported_claim", "supported"],
                  "properties": {"unsupported_claim": {"type": "string"}, "supported": {"type": "boolean"}}}


def numbers(text: str) -> set[str]:
    out = set()
    for m in NUM.finditer(text or ""):
        v = float(m.group(1).replace(",", ""))
        if m.group(2):
            v *= 1000
        out.add(f"{v:g}")
    return out


def unsupported_numbers(reason: str, src: dict[str, Any], need: str = "") -> list[str]:
    """Numbers a reason states that neither the table's text, its metadata nor the request holds.
    A number repeated from the request ("you asked for 200 episodes") is not a claim about the table."""
    known = numbers(doc_text(src)) | numbers(need)
    for k in ("fps", "total_episodes", "total_frames", "n_cameras", "total_tasks"):
        if src.get(k) is not None:
            known.add(f"{float(src[k]):g}")
    for c in src.get("cameras") or []:
        known |= numbers(c)
    known |= numbers(str(src.get("robot_type") or "")) | numbers(str(src.get("codebase_version") or ""))
    # Digits inside an identifier ("..._48h", "so101", "cam_2") are the table's own, wherever they sit in the word.
    for ident in [src.get("table_id") or "", str(src.get("robot_type") or "")] + list(src.get("cameras") or []):
        known |= {f"{float(x):g}" for x in re.findall("[0-9]+", ident)}
    return sorted(numbers(reason) - known)


def number_check(run: dict[str, dict[str, Any]], queries: list[dict[str, Any]], srcs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The check by code alone, over every reason of a run: no model, so it is done afresh every time."""
    needs = {q["qid"]: q["need"] for q in queries}
    total = bad = 0
    for qid, v in run.items():
        for tid, reason in (v.get("reasons") or {}).items():
            if tid in srcs and reason.strip():
                total += 1
                bad += bool(unsupported_numbers(reason, {**srcs[tid], "table_id": tid}, needs.get(qid, "")))
    return {"reasons_with_text": total, "with_a_number_not_in_the_table_text": bad,
            "share_with_unsupported_number": round(bad / total, 4) if total else None}


def supported(llm: Any, reason: str, src: dict[str, Any]) -> dict[str, Any] | None:
    try:
        r = llm.chat([{"role": "system", "content": SUPPORT_PROMPT},
                      {"role": "user", "content": f"Dataset text:\n{doc_text(src)}\n\nSentence:\n{reason[:400]}"}],
                     json_schema=SUPPORT_SCHEMA, temperature=0.0, max_tokens=70)
        out = json.loads(r.content)
        return {"supported": bool(out["supported"]), "claim": str(out.get("unsupported_claim", ""))[:120]}
    except Exception:
        return None


def groundedness(run: dict[str, dict[str, Any]], queries: list[dict[str, Any]], srcs: dict[str, dict[str, Any]], llm: Any) -> dict[str, Any]:
    needs = {q["qid"]: q["need"] for q in queries}
    total = bad_num = checked = unsupported = empty = 0
    examples: list[dict[str, Any]] = []
    for qid, v in run.items():
        for tid, reason in (v.get("reasons") or {}).items():
            src = srcs.get(tid)
            if src is None:
                continue
            total += 1
            if not reason.strip():
                empty += 1
                continue
            nums = unsupported_numbers(reason, {**src, "table_id": tid}, needs.get(qid, ""))
            bad_num += bool(nums)
            s = supported(llm, reason, src)
            if s is not None:
                checked += 1
                unsupported += not s["supported"]
            if (nums or (s and not s["supported"])) and len(examples) < 8:
                examples.append({"qid": qid, "table_id": tid, "reason": reason, "numbers_not_in_text": nums,
                                 "model_says_unsupported": s["claim"] if s and not s["supported"] else None})
    return {"reasons": total, "empty_reasons": empty,
            "with_a_number_not_in_the_table_text": bad_num, "share_with_unsupported_number": round(bad_num / total, 4) if total else None,
            "checked_by_model": checked, "model_says_unsupported": unsupported,
            "share_model_says_unsupported": round(unsupported / checked, 4) if checked else None, "examples": examples}


def abstention(run: dict[str, dict[str, Any]], queries: list[dict[str, Any]]) -> dict[str, Any]:
    """How often the selection shows nothing, by query set, and how many tables it shows otherwise."""
    out: dict[str, Any] = {}
    for kind in sorted({q["kind"] for q in queries}):
        qs = [q for q in queries if q["kind"] == kind and q["qid"] in run]
        if not qs:
            continue
        shown = [len(run[q["qid"]]["ids"]) for q in qs]
        said = [any(n.startswith("missing:") for n in run[q["qid"]]["notes"]) for q in qs]
        out[kind] = {"queries": len(qs), "showed_nothing": sum(s == 0 for s in shown),
                     "share_showed_nothing": round(sum(s == 0 for s in shown) / len(qs), 4),
                     "mean_tables_shown": round(sum(shown) / len(qs), 2), "said_what_is_missing": sum(said)}
        if kind == "D":
            for why in sorted({q.get("why_no_answer", "") for q in qs}):
                sub = [q for q in qs if q.get("why_no_answer") == why]
                out[kind][f"showed_nothing_{why}"] = f"{sum(len(run[q['qid']]['ids']) == 0 for q in sub)}/{len(sub)}"
    return out
