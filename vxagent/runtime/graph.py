"""A small graph runtime for work on a physical arm.

Nodes are functions of the run. Edges are fixed or chosen by the label a node returns. What
makes it a runtime for a robot and not for a document:

  - before a node runs, what it declares as required is checked by looking at the robot, on
    first entry and on every resume. A node whose requirement no longer holds is neither re-run
    nor skipped: the run is routed to the graph's abort node (recovery);
  - a stop from the operator, a fault of the robot and a silent watchdog end a node from the
    outside and route the same way;
  - the state and the next node are written to SQLite after every node, and every change of
    state is an event, so a new process can pick the run up from its id;
  - after a restart nothing resumes by itself. Runs that were running are marked
    needs_reverification and wait for an operator.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..store import Database, dumps
from ..tools.registry import Principal
from .events import EventLog, rebuild, reduce
from .trace import Tracer

END = "END"


class GraphError(ValueError):
    pass


class Interrupt(Exception):
    """The node cannot go on until something outside happens (a person decides). The run is
    checkpointed and returned as waiting; the same node runs again on resume."""
    is_pause = True

    def __init__(self, waiting_for: dict[str, Any]):
        super().__init__(f"waiting for {waiting_for.get('kind')}")
        self.waiting_for = waiting_for


class Abort(Exception):
    """A node was ended from the outside. kind: operator_stop | robot_fault | watchdog"""

    def __init__(self, kind: str, detail: dict[str, Any] | None = None):
        super().__init__(kind)
        self.kind, self.detail = kind, detail or {}


@dataclass
class Step:
    route: str = "next"
    update: dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    name: str
    fn: Callable[["Runner"], Step]
    requires: tuple[str, ...] = ()


@dataclass
class Limits:
    max_steps: int = 30                  # nodes entered
    wall_clock_s: float = 600.0          # time spent inside nodes (waiting for a person is not counted)
    max_tokens: int = 6000
    max_path_mm: float = 400.0           # total distance the tip may travel in one run
    max_recoveries: int = 1


class Graph:
    def __init__(self, entry: str, abort_to: Callable[[dict[str, Any], str, Limits], str],
                 exempt: tuple[str, ...] = ()):
        self.entry = entry
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, dict[str, str]] = {}
        self.abort_to = abort_to          # where a run goes when a node cannot run or is ended from outside
        self.exempt = set(exempt)         # nodes that must be able to run whatever the limits say
        self.abort_targets: set[str] = set()

    def add_node(self, name: str, fn: Callable[["Runner"], Step], requires: tuple[str, ...] = ()) -> None:
        if name in self.nodes or name == END:
            raise GraphError(f"node {name!r} already exists")
        self.nodes[name] = Node(name, fn, requires)

    def add_edge(self, a: str, b: str) -> None:
        self.edges.setdefault(a, {})["next"] = b

    def add_conditional(self, a: str, routes: dict[str, str]) -> None:
        self.edges.setdefault(a, {}).update(routes)

    def validate(self, checks: dict[str, Any] | None = None) -> None:
        if self.entry not in self.nodes:
            raise GraphError(f"entry node {self.entry!r} does not exist")
        for name, node in self.nodes.items():
            if not self.edges.get(name):
                raise GraphError(f"node {name!r} has no way out")
            for label, target in self.edges[name].items():
                if target != END and target not in self.nodes:
                    raise GraphError(f"edge {name} --{label}--> {target}: no such node")
            for req in node.requires:
                if checks is not None and req not in checks:
                    raise GraphError(f"node {name!r} requires {req!r}, which nothing knows how to check")
        for name in self.edges:
            if name not in self.nodes:
                raise GraphError(f"edge from {name!r}, which is not a node")
        for target in self.abort_targets | self.exempt:
            if target not in self.nodes:
                raise GraphError(f"{target!r} is named as an abort or exempt node but does not exist")
        reach = lambda start: self._closure(start)                       # noqa: E731
        seen = reach(self.entry) | {t for a in self.abort_targets for t in reach(a) | {a}}
        for name in self.nodes:
            if name not in seen and name != self.entry:
                raise GraphError(f"node {name!r} cannot be reached")
            if END not in reach(name):
                raise GraphError(f"node {name!r} can never reach the end")

    def _closure(self, start: str) -> set[str]:
        seen: set[str] = set()
        todo = [start]
        while todo:
            n = todo.pop()
            for target in self.edges.get(n, {}).values():
                if target not in seen:
                    seen.add(target)
                    todo.append(target)
        return seen


@dataclass
class Env:
    robot: Any
    site: Any
    db: Database
    log: EventLog
    tracer: Tracer
    checks: dict[str, Callable[["Runner"], tuple[bool, str]]] = field(default_factory=dict)
    observe: Callable[["Runner"], dict[str, Any]] | None = None
    tick: Callable[[float], None] = time.sleep
    llm: Any = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    watchdog_s: float = 0.5               # longest silence of motion feedback before the action is cancelled
    approval_ttl_s: float = 120.0
    approval_wait_s: float = 300.0
    feedback_gate: Callable[[int], bool] | None = None    # INJECTED when set: False withholds a feedback event
    on_event: Callable[[dict[str, Any], dict[str, Any]], None] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    run_id: str
    status: str                           # finished | waiting | needs_reverification
    outcome: str | None
    state: dict[str, Any]

    @property
    def waiting_for(self) -> dict[str, Any] | None:
        return self.state.get("waiting_for")


class NotAllowed(Exception):
    pass


class Runner:
    def __init__(self, graph: Graph, env: Env, limits: Limits | None = None):
        self.graph, self.env = graph, env
        self.limits = limits or Limits()
        self.run_id = ""
        self.state: dict[str, Any] = {}
        graph.validate(env.checks)

    # ------------------------------------------------------------------ events and checkpoints
    def emit(self, kind: str, data: dict[str, Any]) -> dict[str, Any]:
        seq = self.env.robot.state().seq if self.env.robot is not None else None
        event = self.env.log.append(self.run_id, kind, data, seq)
        self.state = reduce(self.state or None, event)
        if self.env.on_event is not None:
            self.env.on_event(event, self.state)
        return event

    def checkpoint(self) -> None:
        db = self.env.db
        row = db.one("SELECT COALESCE(MAX(n),0) AS n FROM checkpoints WHERE run_id=?", (self.run_id,))
        db.execute("INSERT INTO checkpoints(run_id,n,next_node,state,at,events_n) VALUES (?,?,?,?,?,?)",
                   (self.run_id, (row["n"] if row else 0) + 1, self.state["next"], dumps(self.state), time.time(),
                    self.state["events"]))
        db.execute("UPDATE runs SET status=? WHERE run_id=?", (self.state["task"]["status"], self.run_id))

    @staticmethod
    def load(db: Database, run_id: str) -> dict[str, Any] | None:
        """The last checkpoint with every later event applied. With no checkpoint, the events alone."""
        log = EventLog(db)
        cp = db.one("SELECT * FROM checkpoints WHERE run_id=? ORDER BY n DESC LIMIT 1", (run_id,))
        if cp is None:
            return rebuild(log.events(run_id))
        state = json.loads(cp["state"])
        for e in log.events(run_id, after=cp["events_n"]):
            state = reduce(state, e)
        return state

    # ------------------------------------------------------------------ entry points
    def start(self, run_id: str, sheet: dict[str, Any], who: Principal) -> RunResult:
        self.run_id, self.state = run_id, {}
        self.env.tracer.start_run("trial", f"trial at {sheet['site']} on {sheet['arm']}", who.user_id, who.role,
                                  sheet["site"], run_id=run_id)
        self.emit("run_created", {"sheet": sheet, "limits": self.limits.__dict__, "entry": self.graph.entry,
                                  "started_by": who.user_id})
        self.checkpoint()
        return self._run()

    def resume(self, run_id: str, by: Principal | None = None) -> RunResult:
        self.run_id = run_id
        state = self.load(self.env.db, run_id)
        if state is None:
            raise KeyError(f"no run {run_id!r}")
        self.state = state
        status = state["task"]["status"]
        if status == "finished":
            return RunResult(run_id, "finished", state["task"]["outcome"], state)
        if status == "needs_reverification":
            # Nothing resumes by itself after a restart. An operator of this site has to ask.
            if by is None or "trial:run" not in by.scopes or by.site != state["task"]["site"]:
                raise NotAllowed("this run was interrupted by a restart: an operator of this site must resume it")
            had_go = any(h["node"] == "operator_go" and h["route"] == "go" for h in state["history"])
            self.emit("resumed", {"by": by.user_id, "grants_go": had_go})
        self.env.db.execute("UPDATE runs SET status='running' WHERE run_id=?", (run_id,))
        return self._run()

    # ------------------------------------------------------------------ the loop
    def _over_limit(self) -> tuple[str, Any] | None:
        u, lim = self.state["usage"], self.limits
        for name, value, cap in (("steps", u["steps"], lim.max_steps), ("wall_clock_s", u["seconds"], lim.wall_clock_s),
                                 ("tokens", u["tokens"], lim.max_tokens), ("path_mm", u["path_mm"], lim.max_path_mm)):
            if value >= cap:
                return name, value
        return None

    def _reroute(self, kind: str, node: str, data: dict[str, Any]) -> None:
        target = self.graph.abort_to(self.state, node, self.limits)
        self.emit(kind, {"node": node, **data, "next": target})
        self.checkpoint()

    def _run(self) -> RunResult:
        g, env = self.graph, self.env
        while True:
            node = self.state["next"]
            if node == END:
                outcome = self.state["data"].get("outcome") or "unknown"
                self.emit("run_finished", {"outcome": outcome})
                self.checkpoint()
                env.tracer.end_run(self.run_id, "finished", outcome)
                return RunResult(self.run_id, "finished", outcome, self.state)
            exempt = node in g.exempt
            if not exempt:
                over = self._over_limit()
                if over:
                    self._reroute("limit_reached", node, {"limit": over[0], "value": over[1], "reason": f"limit:{over[0]}"})
                    continue
                if env.stop_event.is_set():
                    self._reroute("operator_stop", node, {"reason": "operator_stop"})
                    continue
            # What the node requires is checked by looking, every time the node is entered.
            if env.observe is not None:
                self.emit("observed", env.observe(self))
            failed = []
            for name in g.nodes[node].requires:
                ok, detail = env.checks[name](self)
                if not ok:
                    failed.append({"check": name, "detail": detail})
            if failed:
                self._reroute("precondition_failed", node,
                              {"failed": failed, "reason": "precondition:" + ",".join(f["check"] for f in failed)})
                continue
            entry = 1 + sum(1 for h in self.state["history"] if h["node"] == node)
            self.emit("node_started", {"node": node, "entry": entry})
            t0 = time.time()
            try:
                with env.tracer.span(self.run_id, "node", node) as sp:
                    sp.seq_start = env.robot.state().seq
                    try:
                        step = g.nodes[node].fn(self)
                    finally:
                        sp.seq_end = env.robot.state().seq
                    sp.output = {"route": step.route}
            except Interrupt as exc:
                if self.state["task"]["status"] != "waiting":
                    self.emit("waiting", exc.waiting_for)
                self.checkpoint()
                return RunResult(self.run_id, "waiting", None, self.state)
            except Abort as exc:
                self._reroute(exc.kind, node, {**exc.detail, "reason": exc.kind})
                continue
            routes = g.edges[node]
            if step.route not in routes:
                raise GraphError(f"node {node!r} returned route {step.route!r}; it has {sorted(routes)}")
            self.emit("node_finished", {"node": node, "route": step.route, "next": routes[step.route],
                                        "update": step.update, "seconds": round(time.time() - t0, 3)})
            self.checkpoint()


def apply_restart_rule(db: Database) -> list[str]:
    """Called once when a process starts. Every run that was running when the last process ended
    is marked needs_reverification. Nothing is resumed and nothing is commanded."""
    log = EventLog(db)
    out = []
    for row in db.query("SELECT run_id FROM runs WHERE agent='trial' AND status='running'"):
        run_id = row["run_id"]
        state = rebuild(log.events(run_id))
        if state is None or state["task"]["status"] == "finished":
            continue
        event = log.append(run_id, "process_restarted", {})
        state = reduce(state, event)
        n = db.one("SELECT COALESCE(MAX(n),0) AS n FROM checkpoints WHERE run_id=?", (run_id,))
        db.execute("INSERT INTO checkpoints(run_id,n,next_node,state,at,events_n) VALUES (?,?,?,?,?,?)",
                   (run_id, (n["n"] if n else 0) + 1, state["next"], dumps(state), time.time(), state["events"]))
        db.execute("UPDATE runs SET status='needs_reverification' WHERE run_id=?", (run_id,))
        out.append(run_id)
    return out


# ------------------------------------------------------------------------------ approvals bound to state

def approval_gate(rt: Runner, key: str, grants_go: bool = False) -> str:
    """"approved", "rejected" or "timeout"; raises Interrupt while the decision is open.

    An approval is for one run, one node, one robot state and one version of the envelope. If
    the robot's sequence number or the envelope's version is no longer the one the approval was
    asked for, or the approval is older than its time limit, it is refused and asked for again."""
    env, db = rt.env, rt.env.db
    row = db.one("SELECT * FROM approvals WHERE run_id=? AND call_key=? ORDER BY id DESC LIMIT 1", (rt.run_id, key))
    seq, version, now = env.robot.state().seq, str(env.site.profile.version), time.time()

    def ask() -> None:
        aid = db.execute(
            "INSERT INTO approvals(run_id,call_key,tool,arguments,requested_by,status,created,robot_seq,"
            "profile_version) VALUES (?,?,?,?,?,?,?,?,?)",
            (rt.run_id, key, key, "{}", "trial runtime", "pending", now, seq, version))
        rt.emit("approval_requested", {"approval_id": aid, "node": key, "robot_seq": seq, "envelope_version": version,
                                       "wait_limit_s": env.approval_wait_s})
        raise Interrupt({"kind": "approval", "approval_id": aid})

    if row is None or row["status"] in ("expired", "used"):
        ask()
    assert row is not None
    if row["status"] == "pending":
        if now - row["created"] > env.approval_wait_s:
            db.execute("UPDATE approvals SET status='expired' WHERE id=?", (row["id"],))
            rt.emit("wait_timeout", {"approval_id": row["id"], "node": key, "waited_s": round(now - row["created"], 1)})
            return "timeout"
        raise Interrupt({"kind": "approval", "approval_id": row["id"]})
    if row["status"] == "rejected":
        rt.emit("approval_rejected", {"approval_id": row["id"], "node": key, "decided_by": row["decided_by"]})
        return "rejected"
    why = None
    if now - (row["decided"] or 0) > env.approval_ttl_s:
        why = "too_old"
    elif row["robot_seq"] != seq:
        why = "robot_state_changed"
    elif row["profile_version"] != version:
        why = "envelope_changed"
    if why:
        db.execute("UPDATE approvals SET status='expired' WHERE id=?", (row["id"],))
        rt.emit("approval_refused", {"approval_id": row["id"], "node": key, "why": why,
                                     "approved_for_robot_seq": row["robot_seq"], "robot_seq_now": seq,
                                     "approved_for_envelope": row["profile_version"], "envelope_now": version})
        ask()
    db.execute("UPDATE approvals SET status='used' WHERE id=?", (row["id"],))
    rt.emit("approval_used", {"approval_id": row["id"], "node": key, "decided_by": row["decided_by"],
                              "grants_go": grants_go})
    return "approved"
