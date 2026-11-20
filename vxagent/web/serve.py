"""Putting a console together and serving it.

    python -m vxagent.web.serve                 the local demonstration on http://127.0.0.1:8300
    python -m vxagent.web.serve --public        the same behind a tunnel (see public.py for what it refuses)

Both sites of the demonstration run as edge nodes in this one process, with the fleet desk for
remote support. The accounts are demonstration accounts. Their passwords are generated on first
start and written to data/runtime/demo_accounts.json, a file that is not in the repository and
is never sent to a browser.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path
from typing import Any

from .. import access
from ..config import get_settings
from ..fleet.desk import FleetDesk
from ..fleet.outbox import Outbox
from ..fleet.store import FleetStore
from ..node import Node
from ..store import Database

DEFAULT_SECRET = "dev-secret-change-me"
SITES = ("site_a", "site_b")


def accounts_file() -> Path:
    return get_settings().runtime_dir / "demo_accounts.json"


def ensure_accounts(db: Database) -> Path:
    """Create the demonstration accounts once. Returns where their passwords were written."""
    path = accounts_file()
    have = db.query("SELECT user_id FROM users") if _has_users(db) else []
    if not have or not path.exists():
        path.write_text(json.dumps(access.seed_demo(db), indent=1), encoding="utf-8")
    return path


def _has_users(db: Database) -> bool:
    access.ready(db)
    return True


def session_secret() -> str:
    """The secret that signs sessions: from the environment, or generated once for this machine."""
    env = os.environ.get("VXAGENT_SESSION_SECRET")
    if env:
        return env
    path = get_settings().runtime_dir / "session_secret"
    if not path.exists():
        path.write_text(secrets.token_urlsafe(32), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


def build_console(public: bool = False, llm: Any = "default", knowledge: Any = None, perception: Any = None,
                  secret: str | None = None) -> Any:
    from ..llm.ollama import get_llm
    from .app import Console
    s = get_settings()
    if llm == "default":
        llm = get_llm()
    small = get_llm(s.small_model) if llm is not None and hasattr(llm, "base_url") else llm
    users = Database(s.runtime_dir / "console.sqlite")
    ensure_accounts(users)
    nodes: dict[str, Node] = {}
    for name in SITES:
        node = Node.open(name, llm=llm, small_llm=small, knowledge=knowledge, perception=perception)
        node.fleet = Outbox(node.db, node.name, node.registry, software=node.software,
                            config_version=str(node.store.current(node.site_target)[2]) if node.store else "0",
                            model=getattr(llm, "model", "none"))
        nodes[node.name] = node
    store = FleetStore(s.runtime_dir / "fleet.sqlite")
    desk = FleetDesk(store, Database(s.runtime_dir / "fleet_desk.sqlite"), llm=llm, knowledge=knowledge, edges=nodes)
    console = Console(users, nodes, desk, secret=secret or session_secret(), public=public)
    if public:
        from .public import check_public
        check_public(console)
    return console


def main() -> int:
    import uvicorn

    from .app import create_app
    ap = argparse.ArgumentParser(description="The dVRK agent tools console.")
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--public", action="store_true", help="public mode: refuses unsafe settings, rations jobs")
    a = ap.parse_args()
    console = build_console(public=a.public)
    print(f"console on http://127.0.0.1:{a.port}  (accounts: {accounts_file()})", flush=True)
    uvicorn.run(create_app(console), host="127.0.0.1", port=a.port, log_level="warning")     # never 0.0.0.0: a tunnel connects to it
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
