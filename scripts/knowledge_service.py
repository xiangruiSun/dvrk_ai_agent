"""Start the retrieval service (vxrag) in this process for a verification run, and hand back a
client for it. The agents only ever hold the client: a URL and a key."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def start_knowledge(site: Any, role: str = "operator") -> tuple[Any, Any]:
    """(client, server). The key in memory stands for a caller of this site and role; it is not
    written anywhere. Call server.stop() when done."""
    from vxagent.tools.knowledge import KnowledgeClient
    from vxrag import versions
    from vxrag.access import KeyRing, Principal
    from vxrag.engine import Engine
    from vxrag.service import EngineHolder, Server

    active = versions.pointer()["active"]
    if not active:
        raise SystemExit("no index version is active: run scripts/rag_verify.py r09 first")
    index, manifest = versions.load(active)
    keys = KeyRing()
    token = keys.issue(Principal(f"agent-{site.name}", role, site.name, site.generation, site.software_release))
    server = Server(EngineHolder(Engine(index, manifest["config"], version=active)), keys).start()
    return KnowledgeClient(server.url, token, timeout_s=10), server
