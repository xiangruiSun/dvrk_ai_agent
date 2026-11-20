"""Who is asking, and which collections they may read.

The public sources are one collection, "public". A site's own notes are a collection of their
own, "site:<name>". What a principal may read is decided here, from the API key the request
carried, and is applied inside the search (vxrag/retrieval.py) and inside the cache key
(vxrag/engine.py). A request cannot name its own site: nothing in a request body is consulted.

    operator, engineer, safety    the public collection and their own site's notes
    support                       the public collection only. Remote support helps many sites
                                  and reads no site's notes.
    (no key)                      nothing

Keys are stored as SHA-256 hashes. The token itself is shown once, when it is created.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path

from .applicability import SiteProfile
from .store import PUBLIC

SITE_ROLES = ("operator", "engineer", "safety")
ROLES = SITE_ROLES + ("support",)


@dataclass(frozen=True)
class Principal:
    key_id: str
    role: str
    site: str = ""
    generation: str | None = None        # what the site runs; used when a request does not say
    release: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}")
        SiteProfile(self.generation, self.release)          # validates both values

    @property
    def profile(self) -> SiteProfile:
        return SiteProfile(self.generation, self.release)


def visible_collections(principal: Principal | None) -> frozenset[str]:
    if principal is None:
        return frozenset()
    if principal.site and principal.role in SITE_ROLES:
        return frozenset({PUBLIC, f"site:{principal.site}"})
    return frozenset({PUBLIC})


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class KeyRing:
    """API keys and the principal each stands for."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else None
        self._by_digest: dict[str, Principal] = {}
        if self.path and self.path.exists():
            for digest, p in json.loads(self.path.read_text(encoding="utf-8")).items():
                self._by_digest[digest] = Principal(**p)

    def issue(self, principal: Principal) -> str:
        """Create a key for a principal and return the token. It cannot be read back later."""
        token = "vx_" + secrets.token_urlsafe(24)
        self._by_digest[_digest(token)] = principal
        if self.path:
            self.path.write_text(json.dumps({d: asdict(p) for d, p in self._by_digest.items()}, indent=1),
                                 encoding="utf-8")
        return token

    def resolve(self, token: str | None) -> Principal | None:
        if not token:
            return None
        return self._by_digest.get(_digest(token))

    def principals(self) -> list[Principal]:
        return list(self._by_digest.values())
