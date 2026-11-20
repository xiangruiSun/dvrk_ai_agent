"""The evidence ledger: the only thing the agents of a team share.

It is append-only. An entry is a claim about one candidate cause: it supports or refutes it,
and it points at the data it rests on (which check of which run, which field, and the value
read there). An entry whose reference does not resolve to that data, or whose value is not the
value in the data, is refused when it is written. So nothing reaches the ledger that code has
not found in a measurement.

A finding is written against the ledger position its author last read. If the ledger has
grown since, the finding is refused and has to be redone from the newer ledger.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class Refused(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def dig(data: Any, path: str) -> Any:
    """The value at a dotted path, with [i] for list items. Raises KeyError if it is not there."""
    node = data
    for part in path.replace("]", "").replace("[", ".").split("."):
        if part == "":
            continue
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise KeyError(path)
    return node


def same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return type(a) is type(b) and a == b          # true is not 1.0
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 0.011 * max(1.0, abs(float(b)))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return str(a) == str(b)


@dataclass
class Entry:
    position: int
    author: str
    cause: str
    stance: str                 # supports | refutes
    check: str
    field: str
    value: Any
    why: str = ""
    at: float = 0.0

    @property
    def id(self) -> str:
        return f"e{self.position}"

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "author": self.author, "cause": self.cause, "stance": self.stance, "check": self.check,
                "field": self.field, "value": self.value, "why": self.why}


@dataclass
class Ledger:
    incident: str
    causes: tuple[str, ...]
    resolve: Callable[[str], Any]            # check name -> the measurement of this incident, or KeyError
    entries: list[Entry] = field(default_factory=list)
    refused: list[dict[str, Any]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    stale_findings: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def position(self) -> int:
        return len(self.entries)

    def write(self, author: str, cause: str, stance: str, check: str, field_path: str, value: Any, why: str = "",
              may_use: tuple[str, ...] | None = None) -> Entry:
        """Append a claim, or raise Refused. `may_use` limits the checks this author can point at."""
        try:
            if cause not in self.causes:
                raise Refused("unknown_cause", f"{cause!r} is not a candidate cause")
            if stance not in ("supports", "refutes"):
                raise Refused("bad_stance", "stance is supports or refutes")
            if may_use is not None and check not in may_use:
                raise Refused("not_your_data", f"{author} cannot refer to {check}: it is not that role's data")
            try:
                actual = dig(self.resolve(check), field_path)
            except (KeyError, IndexError, ValueError, TypeError):
                raise Refused("unresolved_reference", f"{check}.{field_path} is not in the data of this incident")
            if not same(value, actual):
                raise Refused("value_mismatch", f"{check}.{field_path} is {actual!r} in the data, not {value!r}")
        except Refused as exc:
            with self._lock:
                self.refused.append({"author": author, "kind": exc.kind, "reason": str(exc), "cause": cause, "check": check,
                                     "field": field_path})
            raise
        with self._lock:
            e = Entry(len(self.entries) + 1, author, cause, stance, check, field_path, actual, why[:300], time.time())
            self.entries.append(e)
            return e

    def read(self) -> tuple[int, list[dict[str, Any]]]:
        with self._lock:
            return len(self.entries), [e.as_dict() for e in self.entries]

    def conclude(self, author: str, finding: dict[str, Any], read_position: int) -> dict[str, Any]:
        """Record a finding written from the ledger as it was at read_position."""
        with self._lock:
            if read_position != len(self.entries):
                self.stale_findings += 1
                raise Refused("stale", f"the ledger has {len(self.entries)} entries; this finding was written from "
                                       f"the first {read_position}")
            row = {"author": author, "at_position": read_position, **finding}
            self.findings.append(row)
            return row

    def support(self) -> dict[str, dict[str, int]]:
        out = {c: {"supports": 0, "refutes": 0} for c in self.causes}
        for e in self.entries:
            out[e.cause][e.stance] += 1
        return out
