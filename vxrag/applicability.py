"""Applicability: which hardware generation and which release a piece of text is about, and
which names it uses.

A dVRK site runs one release on arms of one generation. The documentation describes the newest
release; a thread describes whatever was current when it was written; a release note describes
its own release. Three things follow, and all of them are decided by code from the sources:

  generation   a page under .../Classic/ or .../Si/ is for that generation only
  release      a release note of a release newer than the site's does not apply to the site;
               the release that was current when a thread was opened is its "era"
  renames      names change between releases ("console.json" became "system.json" in 2.4.0).
               The pairs are read from the change log's own wording. A question in old terms
               is widened with the new ones and the other way round, and a result says which
               name applies at the site's release.

Nothing here is typed by hand: the list of releases and the renames come from the change log.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .store import Unit

GENERATIONS = ("Classic", "Si")
_Q = r'(?:`([^`\n]+)`|"([^"\n]+)")'                      # a term in backticks or double quotes

# (pattern, which match is the old name, which is the new one). Groups come in pairs because a
# quoted term is captured by one of two alternatives.
_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("was_now", re.compile(r"\(was\s+([^,()]+?),\s*now\s+([^()]+?)\)"), "old_new_plain"),
    ("new_was_old", re.compile(_Q + r"\s*\(was\s+" + _Q + r"\)"), "new_old"),
    ("referring_to", re.compile(r"referring to\s+" + _Q + r",\s*now uses\s+" + _Q), "old_new"),
    ("is_now", re.compile(_Q + r"\s+is now\s+" + _Q), "old_new"),
    ("replaced_by", re.compile(_Q + r"[^`\"\n]{0,60}?has been replaced by\s+" + _Q), "old_new"),
    ("use_instead", re.compile(_Q + r"[^`\"\n]{0,60}?\b[Uu]se\s+(?:the\s+)?" + _Q + r"[^`\"\n]{0,80}?\binstead\b"), "old_new"),
    ("used_to_be", re.compile(_Q + r"\.\s+It used to be\s+" + _Q), "new_old"),
    ("extension", re.compile(r"(\.[a-z]{2,5}) replaced by (\.[a-z]{2,5})\b"), "old_new_plain"),
]


@dataclass(frozen=True)
class SiteProfile:
    """What a site runs. Either value may be unknown (None)."""
    generation: str | None = None
    release: str | None = None

    def __post_init__(self) -> None:
        if self.generation not in (None, *GENERATIONS):
            raise ValueError(f"unknown hardware generation {self.generation!r}")
        if self.release is not None and not re.fullmatch(r"\d+\.\d+(\.\d+)?", self.release):
            raise ValueError(f"a release looks like 2.3.1, not {self.release!r}")


@dataclass(frozen=True)
class Rename:
    old: str
    new: str
    version: str             # the release whose notes state the change
    unit_id: str             # the release note it was read from
    rule: str
    specific: bool           # identifier-shaped (a file name, a key); a plain word is ambiguous


def version_key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def newer(a: str, b: str) -> bool:
    return version_key(a) > version_key(b)


def _specific(term: str) -> bool:
    return bool(re.search(r"[._\-/:\[\]]|[a-z][A-Z]|\d|[A-Z]{2}", term))


def _term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])", re.I)


def releases(units: Iterable[Unit]) -> list[tuple[str, str]]:
    """(version, release date) in order of release, from the change log."""
    seen = {u.version: u.meta.get("released", "") for u in units if u.source == "changelog" and u.version}
    return sorted(seen.items(), key=lambda kv: version_key(kv[0]))


def release_at(date: str, known: list[tuple[str, str]]) -> str | None:
    """The release that was current on a date (ISO format)."""
    current = None
    for version, released in known:
        if released and released <= date[:10]:
            current = version
    return current


def extract_renames(units: Iterable[Unit]) -> list[Rename]:
    out: dict[tuple[str, str], Rename] = {}
    for u in sorted((u for u in units if u.source == "changelog"), key=lambda u: version_key(u.version)):
        for line in u.text.split("\n"):
            for rule, pattern, order in _PATTERNS:
                for m in pattern.finditer(line):
                    g = [x for x in m.groups() if x is not None]
                    if len(g) != 2:
                        continue
                    a, b = (x.strip().strip('`"') for x in g)
                    old, new = (a, b) if order.startswith("old_new") else (b, a)
                    if not old or not new or old.lower() == new.lower() or len(old) > 60 or len(new) > 60:
                        continue
                    if " " in old.strip() or " " in new.strip():
                        continue                              # a phrase, not a name
                    key = (old.lower(), new.lower())
                    out.setdefault(key, Rename(old, new, u.version, u.unit_id, rule, _specific(old) or _specific(new)))
    return sorted(out.values(), key=lambda r: (version_key(r.version), r.old))


class Applicability:
    """The releases and renames of one corpus, and the questions asked of them."""

    def __init__(self, units: Iterable[Unit]):
        units = list(units)
        self.releases = releases(units)
        self.latest = self.releases[-1][0] if self.releases else None
        self.renames = extract_renames(units)
        self._old = [(r, _term_pattern(r.old)) for r in self.renames]
        self._new = [(r, _term_pattern(r.new)) for r in self.renames]

    # ------------------------------------------------------------------ queries
    def expand(self, query: str, profile: SiteProfile | None = None,
               widen_words: bool = False) -> tuple[str, list[dict[str, Any]]]:
        """Add the other name of anything in the query that was renamed. Returns the widened
        query and, for each rename found, which name applies at the site's release.

        Only identifier-shaped names (a file name, a key) are added to the query. A rename of a
        plain word ("console" became "system") is reported as a note and not searched for:
        measured on the development queries, adding such words made retrieval worse
        (verification/rag_r05_applicability.json). `widen_words` exists to repeat that measurement."""
        extra: list[str] = []
        notes: list[dict[str, Any]] = []
        release = profile.release if profile else None
        for pairs, uses in ((self._old, "old"), (self._new, "new")):
            for r, pattern in pairs:
                if not pattern.search(query):
                    continue
                other = r.new if uses == "old" else r.old
                if other.lower() in (e.lower() for e in extra) or _term_pattern(other).search(query):
                    continue
                if any(n["old"] == r.old and n["new"] == r.new for n in notes):
                    continue
                if r.specific or widen_words:
                    extra.append(other)
                at_site = "unknown" if release is None else ("old" if newer(r.version, release) else "new")
                notes.append({"old": r.old, "new": r.new, "renamed_in": r.version, "query_uses": uses,
                              "name_at_site_release": {"old": r.old, "new": r.new}.get(at_site),
                              "ambiguous_word": not r.specific, "stated_in": r.unit_id})
        return (query + " " + " ".join(extra)).strip() if extra else query, notes

    # ------------------------------------------------------------------ results
    def era(self, unit: Unit) -> str | None:
        """The release a unit speaks about: its own for a release note, the newest for the
        documentation, the one current when it was opened for a thread."""
        if unit.source == "changelog":
            return unit.version or None
        if unit.source in ("doc", "schema"):
            return self.latest
        if unit.source == "thread":
            return release_at(unit.meta.get("created_at", ""), self.releases)
        if unit.source == "note":
            return unit.meta.get("release") or None
        return None

    def annotate(self, unit: Unit, profile: SiteProfile | None, text: str | None = None) -> dict[str, Any]:
        """Does this unit apply to the site, and what should a reader at that site know?
        `applies` is True, False or "check" (it applies, but names have changed)."""
        era = self.era(unit)
        out: dict[str, Any] = {"applies": True, "era": era, "notes": []}
        if profile is None:
            return out
        if profile.generation and unit.generation not in ("both", profile.generation):
            out["applies"] = False
            out["notes"].append(f"written for the {unit.generation} generation; this site is {profile.generation}")
        if profile.release is None:
            return out
        body = text if text is not None else unit.text
        if unit.source == "changelog" and era and newer(era, profile.release):
            out["applies"] = False
            out["notes"].append(f"describes release {era}; this site runs {profile.release}")
        elif era and newer(era, profile.release):
            # Written for a newer release than the site's: a name it uses may not exist there yet.
            for r, pattern in self._new:
                if r.specific and newer(r.version, profile.release) and not newer(r.version, era) and pattern.search(body):
                    out["notes"].append(f"uses `{r.new}`; at release {profile.release} this is `{r.old}` "
                                        f"(renamed in {r.version})")
        elif era and newer(profile.release, era):
            # Written before the site's release: a name it uses may have changed since.
            for r, pattern in self._old:
                if r.specific and newer(r.version, era) and not newer(r.version, profile.release) and pattern.search(body):
                    out["notes"].append(f"written at release {era} and uses `{r.old}`; since {r.version} this is "
                                        f"`{r.new}`")
        if out["applies"] is True and out["notes"]:
            out["applies"] = "check"
        return out

    def describe(self) -> dict[str, Any]:
        return {"releases": len(self.releases), "latest": self.latest,
                "renames": [asdict(r) for r in self.renames]}
