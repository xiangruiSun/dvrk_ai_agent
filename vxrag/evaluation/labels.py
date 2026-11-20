"""Relevance labels built by code from the structure of the sources. Nothing is typed by hand
and no model is asked for a judgement.

    xref        A sentence in the documentation that links to another section with words of its
                own choosing. Query: the sentence as a reader sees it. Relevant: the section it
                links to (grade 2), and, when the link is to the top of a page, the other
                sections of that page (grade 1). The section the sentence comes from is removed
                from the candidates, because it contains the query word for word.
    thread      A closed issue thread that has replies. Query: its title and the start of its
                opening post. Relevant: the replies of that thread (grade 2). The opening post is
                removed from the candidates.
    identifier  An identifier written as a literal in the documentation that occurs in at most
                three units of the whole corpus. Query: the identifier in a short question.
                Relevant: the units that contain it (grade 2).

These are weak labels: they say what the authors linked or answered, not everything a domain
expert would accept. They are complete enough to compare configurations on the same queries and
are reported with the share of results that have no label at all.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from ..store import Unit

LITERAL = re.compile(r"`([^`\s]{5,60})`")
IDENT_SHAPE = re.compile(r"^[A-Za-z/][\w\-./:]*$")
CATEGORIES = ("xref", "thread", "identifier")


@dataclass
class Query:
    qid: str
    category: str
    text: str
    relevant: dict[str, int]
    exclude: set[str] = field(default_factory=set)
    split: str = "dev"
    generation: str = "both"          # the hardware generation of the page the answer is on


def _split(qid: str) -> str:
    return "test" if int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 5 == 0 else "dev"


def _looks_like_identifier(s: str) -> bool:
    if not IDENT_SHAPE.match(s) or s.lower() in ("https", "http"):
        return False
    camel = bool(re.search(r"[a-z][A-Z]", s))
    return camel or any(c in s for c in "_-./:")


def build_queries(units: list[Unit], max_identifier: int = 150) -> list[Query]:
    by_id = {u.unit_id: u for u in units}
    label_to_unit = {lb: u.unit_id for u in units for lb in u.labels}
    by_doc: dict[str, list[Unit]] = {}
    for u in units:
        by_doc.setdefault(u.doc_id, []).append(u)
    out: dict[str, Query] = {}

    # xref
    for u in units:
        if u.source != "doc":
            continue
        for r in u.refs:
            target_id = label_to_unit.get(r["target"])
            if r["text"] is None or target_id is None or target_id == u.unit_id:
                continue
            target = by_id[target_id]
            if target.meta.get("stub") or target.doc_id == u.doc_id:
                continue
            sentence = re.sub(r"\s+", " ", r["sentence"]).strip()
            if len(sentence.split()) < 6:
                continue
            qid = "xref:" + hashlib.sha256(f"{sentence}|{target_id}".encode()).hexdigest()[:12]
            relevant = {target_id: 2}
            if target.ord == 0:                         # a link to the top of a page is a link to the page
                for other in by_doc[target.doc_id]:
                    if other.unit_id != target_id and not other.meta.get("stub"):
                        relevant[other.unit_id] = 1
            out[qid] = Query(qid, "xref", sentence, relevant, {u.unit_id}, _split(qid), target.generation)

    # thread
    for u in units:
        if u.source == "thread" and u.kind == "replies" and u.meta.get("state") == "closed":
            q = by_id.get(f"{u.doc_id}#q")
            if q is None:
                continue
            body = q.text[len(q.title):].strip()
            text = (q.title + "\n" + body[:500]).strip()
            qid = u.doc_id
            out[qid] = Query(qid, "thread", text, {u.unit_id: 2}, {q.unit_id}, _split(qid))

    # identifier
    literals: dict[str, None] = {}
    for u in units:
        if u.source == "doc":
            for m in LITERAL.finditer(u.text):
                s = m.group(1).strip(".,;:")
                if _looks_like_identifier(s):
                    literals[s] = None
    rare = []
    for ident in literals:
        holders = [u.unit_id for u in units if ident in u.text]
        if 1 <= len(holders) <= 3:
            rare.append((hashlib.sha256(ident.encode()).hexdigest(), ident, holders))
    for _, ident, holders in sorted(rare)[:max_identifier]:
        qid = "identifier:" + ident
        gen = by_id[holders[0]].generation
        out[qid] = Query(qid, "identifier", f"What is `{ident}` used for?", {h: 2 for h in holders}, set(),
                         _split(qid), gen)
    return sorted(out.values(), key=lambda q: q.qid)
