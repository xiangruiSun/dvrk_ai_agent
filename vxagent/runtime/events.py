"""The event log of a run, and the state as a function of it.

Every change of a run's state is an appended event. The state object is never written to
directly: an event is appended and the state is the previous state with that event applied
(`reduce`). So the state can always be rebuilt from the events alone (`rebuild`), which is what
a new process does after a crash, and what the tests compare against every checkpoint.

State of a run (one JSON object):
    task          site, arm, stage, status, outcome, why, go (the operator's go for this run)
    observations  operating state, pose and needle_held, each with the robot sequence number
                  it was observed at
    history       one entry per node that finished: node, route taken, next node
    actions       one entry per motion: command id, node, status, path length
    usage         steps, tokens, path_mm, recoveries
    waiting_for   None, or what the run is waiting for (an approval, the operator)
    next          the node that runs next
"""
from __future__ import annotations

import copy
import json
import time
from typing import Any

from ..store import Database, dumps


def merge(into: dict[str, Any], update: dict[str, Any]) -> None:
    """Deep merge of a partial update. A dict merges; anything else replaces."""
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(into.get(k), dict):
            merge(into[k], v)
        else:
            into[k] = copy.deepcopy(v)


def reduce(state: dict[str, Any] | None, event: dict[str, Any]) -> dict[str, Any]:
    """The state after this event. Pure: it reads the event and nothing else."""
    kind, d = event["type"], event["data"]
    if kind == "run_created":
        return {"run_id": event["run_id"],
                "task": {"site": d["sheet"]["site"], "arm": d["sheet"]["arm"], "stage": None, "status": "running",
                         "outcome": None, "why": None, "go": False},
                "sheet": copy.deepcopy(d["sheet"]), "limits": copy.deepcopy(d["limits"]),
                "observations": {}, "history": [], "actions": [], "data": {},
                "usage": {"steps": 0, "tokens": 0, "path_mm": 0.0, "recoveries": 0, "seconds": 0.0},
                "waiting_for": None, "next": d["entry"], "events": event["n"]}
    if state is None:
        raise ValueError("the first event of a run must be run_created")
    s = copy.deepcopy(state)
    if kind == "node_started":
        s["task"]["stage"] = d["node"]
        s["task"]["status"] = "running"
        s["usage"]["steps"] += 1
        s["waiting_for"] = None
    elif kind == "node_finished":
        merge(s, d.get("update") or {})
        s["usage"]["seconds"] = round(s["usage"]["seconds"] + d.get("seconds", 0.0), 3)
        s["history"].append({"node": d["node"], "route": d["route"], "next": d["next"]})
        s["next"] = d["next"]
    elif kind == "observed":
        merge(s["observations"], d)
    elif kind == "action_started":
        s["actions"].append({"command_id": d["command_id"], "node": d["node"], "label": d["label"],
                             "status": "running", "path_mm": 0.0})
    elif kind == "action_result":
        for a in s["actions"]:
            if a["command_id"] == d["command_id"]:
                a["status"], a["path_mm"] = d["status"], d["path_mm"]
        s["usage"]["path_mm"] = round(s["usage"]["path_mm"] + d["path_mm"], 3)
    elif kind == "approval_requested":
        s["waiting_for"] = {"kind": "approval", **d}
        s["task"]["status"] = "waiting"
    elif kind == "waiting":
        s["waiting_for"] = copy.deepcopy(d)
        s["task"]["status"] = "waiting"
    elif kind == "approval_used":
        s["waiting_for"] = None
        if d.get("grants_go"):
            s["task"]["go"] = True
    elif kind in ("approval_refused", "approval_rejected", "wait_timeout"):
        s["waiting_for"] = None
    elif kind == "recovery_started":
        s["usage"]["recoveries"] += 1
        s["task"]["why"] = s["task"]["why"] or d["reason"]
    elif kind in ("precondition_failed", "watchdog", "operator_stop", "robot_fault", "limit_reached"):
        s["next"] = d["next"]
        s["task"]["why"] = s["task"]["why"] or d["reason"]
        s["task"]["stopped_at"] = s["task"].get("stopped_at") or d["node"]
    elif kind == "model_text":
        s["usage"]["tokens"] += d.get("tokens", 0)
    elif kind == "process_restarted":
        s["task"]["status"] = "needs_reverification"
        s["task"]["go"] = False                      # the go was for a process that no longer exists
        s["waiting_for"] = {"kind": "operator_resume"}
        for a in s["actions"]:
            if a["status"] == "running":
                a["status"] = "interrupted"          # nobody knows how far it got: it is observed, not assumed
    elif kind == "resumed":
        s["task"]["status"] = "running"
        s["task"]["go"] = bool(d.get("grants_go"))
        s["waiting_for"] = None
    elif kind == "run_finished":
        s["task"]["status"], s["task"]["outcome"] = "finished", d["outcome"]
        s["next"] = "END"
        s["waiting_for"] = None
    # check_passed, check_failed and anything else are facts for the record; they change no field.
    s["events"] = event["n"]
    return s


def rebuild(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    state = None
    for e in events:
        state = reduce(state, e)
    return state


class EventLog:
    def __init__(self, db: Database):
        self.db = db

    def append(self, run_id: str, kind: str, data: dict[str, Any], robot_seq: int | None = None) -> dict[str, Any]:
        row = self.db.one("SELECT COALESCE(MAX(n),0) AS n FROM events WHERE run_id=?", (run_id,))
        n = (row["n"] if row else 0) + 1
        at = time.time()
        payload = json.loads(dumps(data))            # what is stored is what is reduced
        self.db.execute("INSERT INTO events(run_id,n,type,data,at,robot_seq) VALUES (?,?,?,?,?,?)",
                        (run_id, n, kind, dumps(payload), at, robot_seq))
        return {"run_id": run_id, "n": n, "type": kind, "data": payload, "at": at, "robot_seq": robot_seq}

    def events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM events WHERE run_id=? AND n>? ORDER BY n", (run_id, after))
        return [{"run_id": r["run_id"], "n": r["n"], "type": r["type"], "data": json.loads(r["data"]),
                 "at": r["at"], "robot_seq": r["robot_seq"]} for r in rows]

    def started_at(self, run_id: str) -> float | None:
        row = self.db.one("SELECT MIN(at) AS at FROM events WHERE run_id=?", (run_id,))
        return row["at"] if row else None
