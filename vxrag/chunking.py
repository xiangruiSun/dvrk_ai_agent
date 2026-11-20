"""Chunking strategies behind one function. A chunk always belongs to exactly one unit, and
results are reported per unit, so strategies can be compared on the same labels.

    fixed         windows of a fixed number of characters with overlap, cut across a page with
                  no regard for structure (the baseline)
    section       one chunk per unit; a long unit is split between paragraphs and a code block is
                  never cut; the unit's own title is indexed with it
    section_ctx   as section, with the full breadcrumb (page > section > subsection) indexed in
                  front of the text
"""
from __future__ import annotations

from dataclasses import dataclass

from .store import Unit

STRATEGIES = ("fixed", "section", "section_ctx")


@dataclass
class Chunk:
    chunk_id: str
    unit_id: str
    source: str
    text: str            # what is shown and cited
    index_text: str      # what is tokenized and embedded


def _paragraph_pieces(text: str, max_chars: int) -> list[str]:
    """Split between paragraphs so that no piece exceeds max_chars, unless a single paragraph or
    code block is longer than that (it is then kept whole)."""
    blocks, cur, in_code = [], [], False
    for para in text.split("\n\n"):
        fences = para.count("```")
        cur.append(para)
        if fences % 2 == 1:
            in_code = not in_code
        if not in_code:
            blocks.append("\n\n".join(cur))
            cur = []
    if cur:
        blocks.append("\n\n".join(cur))
    pieces, acc = [], ""
    for b in blocks:
        if acc and len(acc) + len(b) + 2 > max_chars:
            pieces.append(acc)
            acc = b
        else:
            acc = f"{acc}\n\n{b}" if acc else b
    if acc:
        pieces.append(acc)
    return pieces or [""]


def _windows(text: str, size: int, overlap: int) -> list[tuple[int, str]]:
    out, start = [], 0
    while start < len(text):
        out.append((start, text[start:start + size]))
        if start + size >= len(text):
            break
        start += size - overlap
    return out or [(0, text)]


def chunk_units(units: list[Unit], strategy: str = "section", max_chars: int = 1600, window: int = 800,
                overlap: int = 100) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown chunking strategy {strategy!r}; choose from {STRATEGIES}")
    chunks: list[Chunk] = []
    if strategy == "fixed":
        by_doc: dict[str, list[Unit]] = {}
        for u in units:
            by_doc.setdefault(u.doc_id, []).append(u)
        for doc_id, us in by_doc.items():
            us.sort(key=lambda u: u.ord)
            page, spans = "", []
            for u in us:
                piece = (u.title + "\n" if u.source == "doc" and u.title else "") + u.text + "\n\n"
                spans.append((len(page), len(page) + len(piece), u))
                page += piece
            counters: dict[str, int] = {}
            for start, text in _windows(page, window, overlap):
                mid = start + len(text) // 2
                owner = next((u for a, b, u in spans if a <= mid < b), spans[-1][2])
                counters[owner.unit_id] = counters.get(owner.unit_id, 0) + 1
                chunks.append(Chunk(f"{owner.unit_id}~w{counters[owner.unit_id]}", owner.unit_id, owner.source,
                                    text.strip(), text.strip()))
        return [c for c in chunks if c.text]
    for u in units:
        head = " > ".join(u.path) if strategy == "section_ctx" else u.title
        pieces = _paragraph_pieces(u.text, max_chars) if u.text.strip() else [""]
        for i, piece in enumerate(pieces):
            cid = u.unit_id if len(pieces) == 1 else f"{u.unit_id}~{i + 1}"
            chunks.append(Chunk(cid, u.unit_id, u.source, piece, f"{head}\n{piece}".strip()))
    return chunks
