"""The fleet desk: where the remote-support assistant runs. It has the fleet store and a way to
ask an edge node for a granted ticket. It has no robot, no site, no recordings and no
configuration: a support principal's tools have nothing here to read them from.

In this demonstration the edge nodes are objects in the same process. Between two real labs
the same call would be an HTTP request to the node, which checks the grant itself.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from ..node import Bus, build_registry
from ..runtime.trace import Tracer
from ..store import Database
from ..tools.registry import Principal, ToolContext


class FleetDesk:
    name = ""                       # the desk belongs to no site

    def __init__(self, store: Any, db: Database, *, llm: Any = None, knowledge: Any = None, edges: dict[str, Any] | None = None):
        self.store, self.db, self.llm, self.knowledge = store, db, llm, knowledge
        self.edges = edges if edges is not None else {}
        self.bus = Bus()
        self.tracer = Tracer(db)
        self.registry, _, _ = build_registry()
        self.library = None
        self.boxes: dict[str, Any] = {}
        self.fleet = None
        self.started = time.time()

    def ctx(self, who: Principal, run_id: str) -> ToolContext:
        ctx = ToolContext(principal=who, db=self.db, tracer=self.tracer, run_id=run_id)
        ctx.extras.update({"fleet_store": self.store, "edges": self.edges})
        if self.knowledge is not None:
            ctx.extras["knowledge"] = self.knowledge
        return ctx

    def new_run_id(self, prefix: str = "run") -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"
