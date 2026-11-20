"""The outbox of an edge node: operational records waiting to be sent to the fleet store.

A record is built from the node's own tables, reduced to counts, kinds and timings, checked by
the redaction rules, and only then written to the outbox. Records are numbered per node. The
outbox does not care whether a connection exists: it grows while there is none.
"""
from __future__ import annotations

import json
import time
from typing import Any

from ..store import Database, dumps
from . import redact

SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
    site TEXT, seq INTEGER, kind TEXT, body TEXT, created REAL, sent REAL, PRIMARY KEY (site, seq));
"""


class Outbox:
    def __init__(self, db: Database, site: str, registry: Any = None, software: str = "", config_version: str = "",
                 backend: str = "twin", model: str = ""):
        self.db, self.site = db, site
        self.vocab = redact.vocabulary(registry)
        self.meta = {"software": software or "unknown", "config_version": config_version or "0", "backend": backend,
                     "model": model or "unknown"}
        self.refused = 0
        db._con.executescript(SCHEMA)

    # ------------------------------------------------------------------ the boundary

    def add(self, kind: str, body: dict[str, Any]) -> int | None:
        """Write one record, or refuse it. A refused record is counted and audited at the site."""
        record = {"kind": kind, "site": self.site, **body}
        try:
            redact.check(record, self.vocab)
        except redact.Leak as exc:
            self.refused += 1
            self.db.audit("system", "system", "outbox:refused", kind, False, str(exc))
            return None
        with self.db._lock:                           # numbered per site, without gaps
            seq = (self.db.one("SELECT COALESCE(MAX(seq),0) AS n FROM outbox WHERE site=?", (self.site,)) or {"n": 0})["n"] + 1
            self.db.execute("INSERT INTO outbox(site,seq,kind,body,created) VALUES (?,?,?,?,?)",
                            (self.site, seq, kind, dumps(record), time.time()))
        return seq

    def pending(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT seq, kind, body, created FROM outbox WHERE site=? AND sent IS NULL ORDER BY seq LIMIT ?",
                             (self.site, limit))
        return [{"seq": r["seq"], "kind": r["kind"], "created": r["created"], "body": json.loads(r["body"])} for r in rows]

    def mark_sent(self, up_to_seq: int) -> None:
        self.db.execute("UPDATE outbox SET sent=? WHERE site=? AND seq<=? AND sent IS NULL", (time.time(), self.site, up_to_seq))

    def backlog(self) -> int:
        return len(self.db.query("SELECT 1 FROM outbox WHERE site=? AND sent IS NULL", (self.site,)))

    def all_records(self) -> list[dict[str, Any]]:
        return [json.loads(r["body"]) for r in self.db.query("SELECT body FROM outbox WHERE site=? ORDER BY seq", (self.site,))]

    # ------------------------------------------------------------------ builders: counts, kinds, timings

    def record_run(self, run_id: str) -> int | None:
        run = self.db.one("SELECT * FROM runs WHERE run_id=?", (run_id,))
        if run is None:
            return None
        spans = self.db.query("SELECT kind, name, latency_ms, prompt_tokens, completion_tokens, ok, error_kind, attempts, output "
                              "FROM spans WHERE run_id=? ORDER BY id", (run_id,))
        llm = [s for s in spans if s["kind"] == "llm"]
        tools = [s for s in spans if s["kind"] == "tool"]
        k = lambda w: redact.known(w, self.vocab)                             # noqa: E731
        by_tool: dict[str, int] = {}
        err_tool: dict[str, int] = {}
        err_kind: dict[str, int] = {}
        rules: dict[str, int] = {}
        for s in tools:
            name = k(s["name"])
            by_tool[name] = by_tool.get(name, 0) + 1
            if not s["ok"]:
                err_tool[name] = err_tool.get(name, 0) + 1
                if s["error_kind"] == "safety_rejected":
                    try:
                        vio = (json.loads(s["output"] or "{}").get("detail") or {}).get("violations") or [{}]
                        rule = k(vio[0].get("rule"))
                    except (ValueError, AttributeError, IndexError):
                        rule = "other"
                    rules[rule] = rules.get(rule, 0) + 1
        for s in spans:
            if not s["ok"]:
                e = k(s["error_kind"] or "unknown")
                err_kind[e] = err_kind.get(e, 0) + 1
        meta = json.loads(run["meta"] or "{}")
        wall = ((run["ended"] or time.time()) - run["started"]) * 1000 if run["started"] else 0.0
        return self.add("run", {
            "agent": str(run["agent"] or "agent")[:60], "role": k(run["role"]), "status": k(run["status"]),
            "stop_reason": k(meta.get("stop_reason")), "model": self.meta["model"],
            "llm_calls": len(llm), "tool_calls": len(tools), "prompt_tokens": sum(s["prompt_tokens"] for s in llm),
            "completion_tokens": sum(s["completion_tokens"] for s in llm), "wall_ms": round(wall, 1),
            "llm_ms": redact.percentiles([s["latency_ms"] for s in llm]), "tool_ms": redact.percentiles([s["latency_ms"] for s in tools]),
            "tool_retries": sum(max(0, (s["attempts"] or 1) - 1) for s in tools),
            "tool_calls_by_tool": by_tool, "tool_errors_by_tool": err_tool, "errors_by_kind": err_kind,
            "safety_rejections_by_rule": rules})

    def record_trial(self, res: Any, failed_checks: list[dict[str, Any]] | None = None) -> int | None:
        st = res.state
        d = st.get("data") or {}
        k = lambda w: redact.known(w, self.vocab)                             # noqa: E731
        rejected: dict[str, int] = {}
        for r in (d.get("dry_run") or {}).get("rejections") or []:
            for v in r.get("violations") or [{}]:
                rule = k(v.get("rule"))
                rejected[rule] = rejected.get(rule, 0) + 1
        by_check: dict[str, int] = {}
        for f in failed_checks or []:
            key = redact.CHECK_KEYS.get(f.get("check", ""), "other")
            by_check[key] = by_check.get(key, 0) + 1
        usage = st.get("usage") or {}
        return self.add("trial", {
            "status": k(res.status), "outcome": k(d.get("outcome")), "machine_verdict": k((d.get("verdict") or {}).get("machine")),
            "operator_decision": k((d.get("operator_verdict") or {}).get("decision") or "none"),
            "motions": len(st.get("actions") or []), "recoveries": int(usage.get("recoveries", 0)),
            "rejections": int(usage.get("rejections", 0)) + sum(rejected.values()),
            "safety_rejections_by_rule": rejected, "preflight_failed_by_check": by_check})

    def record_health(self, health: dict[str, Any]) -> int | None:
        robot, perc = health.get("robot") or {}, health.get("perception") or {}
        model, retr = health.get("model_server") or {}, health.get("retrieval") or {}
        return self.add("health", {
            **self.meta, "robot_reachable": bool(robot.get("reachable")), "robot_ready": bool(robot.get("ready")),
            "robot_fault": bool(robot.get("fault")), "model_reachable": bool(model.get("reachable")),
            "models_installed": int(model.get("models_installed", 0) or 0),
            "retrieval_reachable": bool(retr.get("reachable", retr.get("configured", False))),
            "perception_reachable": bool(perc.get("reachable")),
            "perception_latency_ms": round(float(perc.get("latency_s", 0.0)) * 1000, 1),
            "disk_free_gb": float(health.get("disk_free_gb", 0.0)), "pending_approvals": int(health.get("pending_approvals", 0)),
            "outbox_backlog": self.backlog(), "uptime_s": float(health.get("uptime_s", 0.0)),
            "runs_needing_reverification": int(health.get("runs_needing_reverification", 0))})
