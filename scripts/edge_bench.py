"""What the verification scripts of tasks 11 to 15 share: real edge nodes on a fresh runtime
folder, the retrieval service for each of them, a console on top, and small helpers.

The nodes are built from the real dVRK files (site A: Classic PSM, site B: Si PSM). The model
is the local model. Each verification gets its own folder under data/runtime/verify/, emptied
first, so one run cannot read what another left behind.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knowledge_service import start_knowledge                              # noqa: E402
from vxagent import access                                                 # noqa: E402
from vxagent.config import get_settings                                    # noqa: E402
from vxagent.fleet.desk import FleetDesk                                   # noqa: E402
from vxagent.fleet.outbox import Outbox                                    # noqa: E402
from vxagent.fleet.store import FleetStore                                 # noqa: E402
from vxagent.llm.ollama import get_llm                                     # noqa: E402
from vxagent.node import Node                                              # noqa: E402
from vxagent.store import Database                                         # noqa: E402

PEOPLE = {u: (r, s) for u, r, s, _ in access.DEMO_ACCOUNTS}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def who(user_id: str) -> Any:
    role, site = PEOPLE[user_id]
    return access.principal_of({"user_id": user_id, "role": role, "site": site, "display_name": user_id})


class Bench:
    def __init__(self, tag: str, sites: tuple[str, ...] = ("site_a",), llm: Any = "default", knowledge: bool = True,
                 tick: Any = None):
        s = get_settings()
        self.folder = s.runtime_dir / "verify" / tag
        shutil.rmtree(self.folder, ignore_errors=True)
        self.folder.mkdir(parents=True)
        self.llm = get_llm() if llm == "default" else llm
        self.servers: list[Any] = []
        self.nodes: dict[str, Node] = {}
        for name in sites:
            kw: dict[str, Any] = {"llm": self.llm, "runtime": self.folder}
            if tick is not None:
                kw["tick"] = tick
            node = Node.open(name, **kw)
            if knowledge:
                client, server = start_knowledge(node.site)
                node.knowledge = client
                self.servers.append(server)
            node.fleet = Outbox(node.db, node.name, node.registry, software=node.software,
                                config_version=str(node.store.current(node.site_target)[2]), model=getattr(self.llm, "model", "none"))
            self.nodes[node.name] = node
        self.store = FleetStore(self.folder / "fleet.sqlite")
        desk_knowledge = None
        if knowledge:
            desk_knowledge, server = start_knowledge(SimpleNamespace(name="", generation=None, software_release=None), "support")
            self.servers.append(server)
        self.desk = FleetDesk(self.store, Database(self.folder / "desk.sqlite"), llm=self.llm, knowledge=desk_knowledge, edges=self.nodes)

    def console(self) -> tuple[Any, dict[str, str]]:
        """A console on these nodes with the demonstration accounts. The passwords exist only in
        this process."""
        from vxagent.web.app import Console
        users = Database(self.folder / "console.sqlite")
        passwords = access.seed_demo(users)
        return Console(users, self.nodes, self.desk, secret="verification-" + "x" * 30), passwords

    def client(self, console: Any, passwords: dict[str, str], user_id: str) -> Any:
        from starlette.testclient import TestClient

        from vxagent.web.app import create_app
        c = TestClient(create_app(console))
        r = c.post("/api/login", json={"user_id": user_id, "password": passwords[user_id]})
        assert r.status_code == 200, r.text
        return c

    def close(self) -> None:
        for s in self.servers:
            try:
                s.stop()
            except Exception:
                pass


def ask(place: Any, user_id: str, text: str, approve: bool | None = True, max_approvals: int = 6) -> dict[str, Any]:
    """One question through the chat path, with the operator's decisions made by the script.
    Returns what a verification needs to judge the run."""
    from vxagent import chat
    from vxagent.agent.loop import tool_messages
    p = who(user_id)
    t0 = time.time()
    out = chat.start(place, p, text, session=f"v{time.time_ns()}")
    decisions = []
    while out["status"] == "awaiting_approval" and approve is not None and len(decisions) < max_approvals:
        decisions.append({"tool": out["approval"]["tool"], "decision": "approved" if approve else "rejected"})
        out = chat.decide(place, who("op.lee") if place.name == "site-a" else who("op.b"), out["approval"]["approval_id"], approve)
    box = place.boxes[out["run_id"]]
    calls = [{"tool": m.get("name"), "ok": bool(m.get("ok")), "error_kind": m.get("error_kind")} for m in tool_messages(box.state)]
    return {"question": text, "asked_by": user_id, "status": out["status"], "stop_reason": out["stop_reason"],
            "tools": calls, "tools_called": [c["tool"] for c in calls], "approvals": decisions, "answer": out["answer"],
            "model_calls": out["trace"]["model_calls"], "tokens": out["trace"]["tokens"], "seconds": round(time.time() - t0, 1),
            "guard_repairs": box.state.answer_repairs, "run_id": out["run_id"],
            "messages": box.state.messages}


def strip(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "messages"}


def route_matrix(bench: Bench, console: Any, passwords: dict[str, str], users: dict[str, str]) -> dict[str, Any]:
    """Every guarded route asked by every role. A route answers 403 exactly when the role is not
    one it declares."""
    from vxagent.web.app import route_table
    rows, wrong = [], 0
    clients = {role: bench.client(console, passwords, u) for role, u in users.items()}
    for r in route_table():
        path = r["path"].replace("{run_id}", "none").replace("{ticket_id}", "999").replace("{approval_id}", "999") \
            .replace("{proposal_id}", "999").replace("{grant_id}", "999").replace("{site}", "site-a")
        if path in ("/api/chat", "/api/stop", "/api/trials") and r["method"] == "POST" or path.endswith("/diagnose"):
            status = {role: (403 if role not in r["roles"] else "not called (it would start work)") for role in clients}
            for role, c in clients.items():
                if role not in r["roles"]:
                    status[role] = c.post(path, json={}).status_code
        else:
            status = {role: (c.get(path) if r["method"] == "GET" else c.post(path, json={})).status_code for role, c in clients.items()}
        ok = all((status[role] == 403) == (role not in r["roles"]) for role in clients)
        if "fleet/tickets" in path:                       # for support this is a 403 too: there is no grant for that ticket
            ok = all(v == 403 for v in status.values())
        wrong += not ok
        rows.append({"route": f"{r['method']} {r['path']}", "declared_for": r["roles"], "status": status, "as_declared": ok})
    return {"routes": len(rows), "as_declared": len(rows) - wrong, "rows": rows}
