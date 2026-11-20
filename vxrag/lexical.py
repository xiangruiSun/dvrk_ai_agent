"""Lexical retrieval: two tokenizers and BM25.

Engineering text is full of identifiers: `measured_cp`, `sawRobotIO1394-PSM1-12345.json`,
`/PSM1/servo_cp`, `tool_detection`. A tokenizer that only splits on punctuation turns each of
them into common words, and the one token that would have found the right page is lost.

    plain    lower-case runs of letters and digits
    ident    the same, plus every identifier kept whole, plus the parts of camelCase words
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable

import numpy as np

WORD = re.compile(r"[A-Za-z0-9]+")
IDENT = re.compile(r"[A-Za-z0-9]+(?:[_\-./:][A-Za-z0-9]+)+|/[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)*")
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOP = frozenset("a an and are as at be by for from has have in is it its of on or that the this to was were will "
                 "with you your can if not we i do does how what which when".split())


def tokenize_plain(text: str) -> list[str]:
    return [w for w in (m.group(0).lower() for m in WORD.finditer(text)) if w not in STOP]


def tokenize_ident(text: str) -> list[str]:
    out = []
    for m in WORD.finditer(text):
        w = m.group(0)
        low = w.lower()
        if low not in STOP:
            out.append(low)
        parts = CAMEL.findall(w)
        if len(parts) > 1:                              # sawRobotIO1394 -> saw, robot, io, 1394
            out.extend(p.lower() for p in parts if len(p) > 1 and p.lower() not in STOP)
    for m in IDENT.finditer(text):
        ident = m.group(0).strip("./:-").lower()
        if any(c.isalpha() for c in ident) and len(ident) > 2:
            out.append(ident)                           # measured_cp kept whole
    return out


TOKENIZERS: dict[str, Callable[[str], list[str]]] = {"plain": tokenize_plain, "ident": tokenize_ident}


class BM25:
    def __init__(self, texts: list[str], tokenizer: str = "ident", k1: float = 1.2, b: float = 0.75):
        self.tokenize = TOKENIZERS[tokenizer]
        self.k1, self.b = k1, b
        self.n = len(texts)
        lengths = np.zeros(self.n)
        postings: dict[str, list[tuple[int, int]]] = {}
        for i, text in enumerate(texts):
            counts = Counter(self.tokenize(text))
            lengths[i] = sum(counts.values())
            for term, tf in counts.items():
                postings.setdefault(term, []).append((i, tf))
        self.lengths = lengths
        self.avg = float(lengths.mean()) if self.n else 0.0
        self.postings = {t: (np.array([i for i, _ in p]), np.array([tf for _, tf in p], dtype=float))
                         for t, p in postings.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n)
        for term, qtf in Counter(self.tokenize(query)).items():
            hit = self.postings.get(term)
            if hit is None:
                continue
            idx, tf = hit
            idf = math.log(1.0 + (self.n - len(idx) + 0.5) / (len(idx) + 0.5))
            norm = tf + self.k1 * (1.0 - self.b + self.b * self.lengths[idx] / (self.avg or 1.0))
            out[idx] += idf * tf * (self.k1 + 1.0) / norm * min(qtf, 3)
        return out
