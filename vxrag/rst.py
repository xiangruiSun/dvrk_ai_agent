"""A small reStructuredText reader for the dVRK documentation.

It does not render. It recovers what retrieval needs: the section tree with its headings, the
labels that other pages point at, the cross-references each section makes (with the words they
are made with), code blocks kept whole, and readable text with the markup removed. Figures,
tables of contents, raw HTML and includes (which are diagrams here) are dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ADORN = set("=-*#^~\"'`+:._")
LABEL = re.compile(r"^\.\. _([A-Za-z0-9][\w\-. ]*):\s*$")
DIRECTIVE = re.compile(r"^(\s*)\.\. ([A-Za-z][\w\-]*)::\s*(.*)$")
REF = re.compile(r":ref:`([^`<]+?)\s*<([^`>]+)>`|:ref:`([^`<]+)`")
LINK = re.compile(r"`([^`<]+?)\s*<(https?://[^`>]+)>`_+")
ROLE = re.compile(r":[a-z]+:`([^`]+)`")
DROP = {"figure", "image", "toctree", "raw", "include", "contents", "only", "tabs", "tab", "youtube", "video",
        "bibliography", "index"}
KEEP_BODY = {"note", "warning", "caution", "hint", "important", "tip", "admonition", "list-table", "csv-table",
             "rubric", "attention", "seealso"}
CODE = {"code-block", "code", "sourcecode", "parsed-literal"}


@dataclass
class Ref:
    target: str            # the label pointed at, lower case
    text: str | None       # the words the link is made with; None when the page lets Sphinx use the title
    sentence: str          # the sentence the link sits in, with the link shown as its words


@dataclass
class Section:
    title: str
    level: int
    path: list[str]                       # titles from the page title down to this section
    labels: list[str] = field(default_factory=list)
    text: str = ""                        # prose, markup removed, code blocks in place
    code: list[str] = field(default_factory=list)
    refs: list[Ref] = field(default_factory=list)


def _is_adornment(line: str) -> bool:
    s = line.rstrip()
    return len(s) >= 3 and s[0] in ADORN and len(set(s)) == 1


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def clean_inline(text: str) -> str:
    """Inline markup to plain words. Literals keep their backticks (one each side) so that
    identifiers stay recognisable to the tokenizer."""
    text = REF.sub(lambda m: (m.group(1) or m.group(3) or "").strip(), text)
    text = LINK.sub(lambda m: m.group(1).strip(), text)
    text = ROLE.sub(lambda m: m.group(1), text)
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"\1", text)
    text = re.sub(r"\|([A-Za-z][\w\- ]*)\|", r"\1", text)
    return text


def _sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?:;])\s+(?=[A-Z`:(\[])", paragraph) if s.strip()]


def parse(source: str) -> list[Section]:
    lines = source.replace("\r\n", "\n").replace("\t", "    ").split("\n")
    styles: list[str] = []                 # adornment styles in order of first use = heading levels
    sections: list[Section] = [Section(title="", level=0, path=[])]
    pending_labels: list[str] = []
    stack: list[Section] = []
    buf: list[str] = []                    # raw lines of the current section body

    def flush() -> None:
        _finish(sections[-1], buf)
        buf.clear()

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        m = LABEL.match(line)
        if m:
            pending_labels.append(m.group(1).strip().lower())
            i += 1
            continue
        # Heading: text line followed by an adornment line at least as long (optionally with an overline).
        over = _is_adornment(line) and i + 2 < n and lines[i + 1].strip() and _is_adornment(lines[i + 2]) \
            and lines[i + 2].rstrip()[0] == line.rstrip()[0]
        under = (not over and line.strip() and not line.startswith((" ", "..")) and i + 1 < n
                 and _is_adornment(lines[i + 1]) and len(lines[i + 1].rstrip()) >= len(line.rstrip())
                 and not _is_adornment(line))
        if over or under:
            title = (lines[i + 1] if over else line).strip()
            style = ("o" if over else "u") + (line.rstrip()[0] if over else lines[i + 1].rstrip()[0])
            if style not in styles:
                styles.append(style)
            level = styles.index(style) + 1
            flush()
            while stack and stack[-1].level >= level:
                stack.pop()
            sec = Section(title=clean_inline(title), level=level,
                          path=[s.title for s in stack] + [clean_inline(title)], labels=pending_labels)
            pending_labels = []
            stack.append(sec)
            sections.append(sec)
            i += 3 if over else 2
            continue
        buf.append(line)
        i += 1
    flush()
    if pending_labels:
        sections[-1].labels.extend(pending_labels)
    # A page that opens with text before any heading keeps that text in its first real section.
    if len(sections) > 1 and sections[0].text.strip():
        sections[1].text = (sections[0].text + "\n\n" + sections[1].text).strip()
        sections[1].refs = sections[0].refs + sections[1].refs
        sections[1].code = sections[0].code + sections[1].code
    if len(sections) > 1:
        sections[1].labels = sections[0].labels + sections[1].labels
        return sections[1:]
    return [s for s in sections if s.text.strip()]


def _finish(sec: Section, raw: list[str]) -> None:
    """Turn the raw body lines of a section into text, code blocks and references."""
    out: list[str] = []
    i, n = 0, len(raw)
    while i < n:
        line = raw[i]
        m = DIRECTIVE.match(line)
        literal = line.rstrip().endswith("::") and not m
        if m or literal:
            base = _indent(line)
            name = m.group(2).lower() if m else "literal"
            j = i + 1
            block: list[str] = []
            while j < n and (not raw[j].strip() or _indent(raw[j]) > base):
                block.append(raw[j])
                j += 1
            body = [b for b in block if not re.match(r"^\s*:[\w\-]+:", b)]        # drop option lines
            width = min((_indent(b) for b in body if b.strip()), default=0)
            body_text = "\n".join(b[width:] for b in body).strip("\n")
            if name in CODE or name == "literal":
                if literal:
                    lead = line.rstrip()[:-2].rstrip()
                    if lead:
                        out.append(lead + ":")
                if body_text.strip():
                    sec.code.append(body_text)
                    out.append("```\n" + body_text + "\n```")
            elif name in KEEP_BODY:
                head = m.group(3).strip() if m else ""
                inner = "\n".join(x for x in (head, body_text) if x)
                if inner.strip():
                    out.append(inner)
            elif name not in DROP and m and m.group(3).strip():
                out.append(m.group(3).strip())
            i = j
            continue
        if line.startswith(".. ") and not m:         # comment or substitution definition
            i += 1
            while i < n and raw[i].strip() and _indent(raw[i]) > 0:
                i += 1
            continue
        out.append(line)
        i += 1
    # Paragraphs: join wrapped lines, keep code fences intact.
    paragraphs: list[str] = []
    cur: list[str] = []
    in_code = False
    for line in out:
        if line.startswith("```"):
            if cur and not in_code:
                paragraphs.append(" ".join(x.strip() for x in cur))
                cur = []
            cur.append(line)
            if in_code or line.count("```") == 2:
                paragraphs.append("\n".join(cur))
                cur = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            cur.append(line)
            if line.rstrip().endswith("```"):
                paragraphs.append("\n".join(cur))
                cur, in_code = [], False
            continue
        if not line.strip():
            if cur:
                paragraphs.append(" ".join(x.strip() for x in cur))
                cur = []
        elif re.match(r"^\s*([*\-+]|\d+[.)]|#\.)\s+", line) and cur:
            paragraphs.append(" ".join(x.strip() for x in cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        paragraphs.append("\n".join(cur) if in_code else " ".join(x.strip() for x in cur))
    text_parts = []
    for p in paragraphs:
        if p.startswith("```"):
            text_parts.append(p)
            continue
        for s in _sentences(p):
            for m in REF.finditer(s):
                target = (m.group(2) or m.group(3)).strip().lower()
                words = m.group(1).strip() if m.group(1) else None
                sec.refs.append(Ref(target=target, text=words, sentence=clean_inline(s)))
        text_parts.append(clean_inline(p))
    sec.text = "\n\n".join(t for t in text_parts if t.strip()).strip()


def slug(title: str) -> str:
    """The anchor Sphinx gives a section title."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
