"""Public mode: what must be true before the console may be reached from outside.

The public demonstration serves the kinematic twin only. It listens on 127.0.0.1 and is reached
through a tunnel; it never binds a public address itself. Before it starts, this module checks
the settings and refuses, with the reason, if any of them would be unsafe in public.
"""
from __future__ import annotations

from typing import Any

from .. import access

WEAK = ("password", "demo", "admin", "changeme", "letmein", "operator", "123456")


class NotFitForPublic(RuntimeError):
    pass


def problems(console: Any, source_passwords: dict[str, str] | None = None) -> list[str]:
    from .serve import DEFAULT_SECRET
    out = []
    if console.secret == DEFAULT_SECRET or len(console.secret) < 24:
        out.append("the session secret is the default or shorter than 24 characters")
    for name, node in console.nodes.items():
        if node.backend != "twin" or getattr(node.robot, "name", "") != "twin":
            out.append(f"{name}: the robot backend is {getattr(node.robot, 'name', node.backend)!r}; public mode serves the twin only")
    rows = console.db.query("SELECT user_id, salt, pw_hash FROM users")
    for r in rows:
        for guess in list(WEAK) + [r["user_id"], r["user_id"].split(".")[0]] + list((source_passwords or {}).values()):
            if access.hash_password(guess, r["salt"]) == r["pw_hash"]:
                out.append(f"the account {r['user_id']} has a password that is in the source or easy to guess")
                break
    if console.ration.per_hour > 30 or console.ration.concurrent > 1:
        out.append("copilot jobs are not rationed per visitor")
    return out


def check_public(console: Any, source_passwords: dict[str, str] | None = None) -> None:
    bad = problems(console, source_passwords)
    if bad:
        raise NotFitForPublic("public mode refuses to start: " + "; ".join(bad))
