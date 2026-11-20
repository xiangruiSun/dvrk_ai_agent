"""The host's side of MCP: which servers and which tools it has approved, what it remembers
about each approved tool, and how a discovered tool becomes a tool of the agent loop.

A tool the server offers but the host has not approved is never shown to the model. For every
approved tool the host keeps a hash of its description and schema; if a server later presents
another description under the same name, the tool is withdrawn until a person approves it
again. Results are data: the host parses them and passes them on, it never follows them.
Execution safety for a remote tool is the server's (see servers.py); the host adds a time
limit and the result-size limit of its own executor.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from ..store import Database, dumps
from ..tools.registry import Tool, ToolContext, ToolError, ToolRegistry
from .client import Connection


class RemoteArgs(BaseModel):
    """Arguments of a remote tool are validated by the server that owns the tool."""
    model_config = ConfigDict(extra="allow")


def pin_of(tool: dict[str, Any]) -> str:
    blob = json.dumps({"n": tool["name"], "d": tool["description"], "s": tool["input_schema"]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Host:
    def __init__(self, db: Database, allow: dict[str, list[str]]):
        self.db = db
        self.allow = allow                               # server name -> tools the host has approved
        self.connections: dict[str, Connection] = {}
        self.report: dict[str, dict[str, Any]] = {}

    def attach(self, conn: Connection) -> dict[str, Any]:
        """Discover a connected server's tools and sort them into offered, not approved and withdrawn."""
        if conn.name not in self.allow:
            raise PermissionError(f"server {conn.name!r} is not on the host's list of servers")
        self.connections[conn.name] = conn
        found = conn.list_tools()
        offered, hidden, withdrawn, pinned_now = [], [], [], []
        for t in found:
            if t["name"] not in self.allow[conn.name]:
                hidden.append(t["name"])
                continue
            key, pin = f"mcp_pin:{conn.name}:{t['name']}", pin_of(t)
            row = self.db.one("SELECT v FROM kv WHERE k=?", (key,))
            if row is None:                              # first sight of an approved tool: remember what was approved
                self.db.execute("INSERT INTO kv(k,v) VALUES (?,?)", (key, pin))
                pinned_now.append(t["name"])
            elif row["v"] != pin:
                withdrawn.append({"tool": t["name"], "approved": row["v"], "presented": pin})
                continue
            offered.append(t)
        self.report[conn.name] = {"discovered": [t["name"] for t in found], "offered": [t["name"] for t in offered],
                                  "not_approved_by_the_host": hidden, "withdrawn_description_changed": withdrawn,
                                  "pinned_now": pinned_now}
        self._offered = getattr(self, "_offered", {})
        self._offered[conn.name] = offered
        return self.report[conn.name]

    def reapprove(self, server: str, tool: str) -> None:
        """A person has looked at the new description and accepts it."""
        t = next(x for x in self.connections[server].list_tools() if x["name"] == tool)
        self.db.execute("INSERT OR REPLACE INTO kv(k,v) VALUES (?,?)", (f"mcp_pin:{server}:{tool}", pin_of(t)))

    def register(self, registry: ToolRegistry, on_progress: Callable[[str, float, str], None] | None = None,
                 timeout_s: float = 60.0) -> list[str]:
        """Make every offered tool a tool of the registry, with the server's schema and tier."""
        names = []
        for server, tools in getattr(self, "_offered", {}).items():
            conn = self.connections[server]
            for t in tools:
                meta = t["meta"] or {}

                def call(a: RemoteArgs, ctx: ToolContext, conn: Connection = conn, name: str = t["name"]) -> Any:
                    body = conn.call_tool(name, a.model_dump(), timeout=timeout_s, cancel=ctx.stop_event,
                                          on_progress=(lambda f, m: on_progress(name, f, m)) if on_progress else None)
                    if not body.get("ok"):
                        raise ToolError(body.get("error_kind") or "failed", str(body.get("error") or "the server refused"),
                                        hint=body.get("hint"), detail=body.get("detail"),
                                        retryable=body.get("error_kind") in ("transient", "knowledge_unavailable"))
                    return body.get("result")

                registry.register(Tool(t["name"], t["description"], RemoteArgs, call, tier=meta.get("vx/tier", "observe"),
                                       kind="call", timeout_s=timeout_s + 5, schema_override=t["input_schema"],
                                       origin=f"mcp:{server}", requires_approval=False,
                                       max_retries=0 if meta.get("vx/motion") else None))
                names.append(t["name"])
        return names

    def summary(self) -> str:
        return dumps(self.report)
