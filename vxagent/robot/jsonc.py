"""The dVRK configuration files are JSON with // and /* */ comments. This reads them."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def strip_comments(text: str) -> str:
    out: list[str] = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
            out.append(ch)
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def loads(text: str) -> Any:
    return json.loads(strip_comments(text))


def load(path: Path | str) -> Any:
    return loads(Path(path).read_text(encoding="utf-8"))
