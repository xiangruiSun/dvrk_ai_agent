"""Turning a line pasted from a log into a search query.

A log line carries things that belong to one machine and one moment: a time stamp, a process
id, a serial number, a path under somebody's home directory. None of them helps to find the
page that explains the message, and a home directory usually contains a person's name, which
has no business in a search service's logs. They are removed here, before the line goes
anywhere. Identifiers, file names and the words of the message are kept.
"""
from __future__ import annotations

import re

_STEPS: list[tuple[str, re.Pattern[str], str]] = [
    ("ansi", re.compile(r"\x1b\[[0-9;]*[A-Za-z]"), ""),
    ("timestamp", re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), " "),
    ("timestamp", re.compile(r"\[\s*\d+\.\d+\s*\]"), " "),
    ("timestamp", re.compile(r"(?<![\w.])\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?(?![\w])"), " "),
    # /home/<name>/a/b/c/file.cpp -> c/file.cpp ; the user's name never leaves this function
    ("home_path", re.compile(r"(?:/home/|/Users/|[A-Za-z]:\\Users\\)[^\s:'\"()]+"), "__PATH__"),
    ("address", re.compile(r"\b0x[0-9a-fA-F]{4,}\b"), " "),
    ("level", re.compile(r"(?<![\w])-[EWDI]-(?![\w])|\[(?:ERROR|WARN(?:ING)?|INFO|DEBUG|FATAL)\]", re.I), " "),
    ("process", re.compile(r"\[(?:pid\s*)?\d+\]|\bpid[=: ]\s*\d+", re.I), " "),
    ("long_number", re.compile(r"(?<![\w.\-/])\d{4,}(?![\w.\-/])"), " "),
]


def normalise(line: str) -> tuple[str, list[str]]:
    """The query, and the kinds of thing that were removed."""
    removed: list[str] = []
    text = line.strip()[:2000]

    def short_path(m: re.Match[str]) -> str:
        parts = [p for p in re.split(r"[\\/]", m.group(0)) if p]
        return "/".join(parts[-2:]) if len(parts) > 3 else parts[-1]

    for name, pattern, repl in _STEPS:
        if not pattern.search(text):
            continue
        removed.append(name)
        text = pattern.sub(short_path if repl == "__PATH__" else repl, text)
    return re.sub(r"\s+", " ", text).strip(), sorted(set(removed))
