"""The fleet store: where the operational records of every site end up, and what the support
team can ask of them. One SQLite file.

Delivery is at least once, so the store makes it exactly once: a record is identified by its
site and its sequence number, a number it already has is a duplicate and is ignored, and a
number that would leave a gap is not accepted, so records are stored in the order they were
written at the site.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS fleet_records (
    site TEXT, seq INTEGER, kind TEXT, body TEXT, created REAL, received REAL, PRIMARY KEY (site, seq));
CREATE TABLE IF NOT EXISTS fleet_contact (site TEXT PRIMARY KEY, last_contact REAL, last_seq INTEGER);
CREATE TABLE IF NOT EXISTS fleet_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL, user_id TEXT, question TEXT, site TEXT);
"""
STALE_S = 900.0


class FleetStore:
    def __init__(self, path: Path | str = ":memory:"):
        self._lock = threading.RLock()
        self._con = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._con.row_factory = sqlite3.Row
        self._con.executescript(SCHEMA)
        self.duplicates = self.out_of_order = 0

    def _q(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._con.execute(sql, params).fetchall()]

    # ------------------------------------------------------------------ ingest

    def last_seq(self, site: str) -> int:
        rows = self._q("SELECT last_seq FROM fleet_contact WHERE site=?", (site,))
        return int(rows[0]["last_seq"] or 0) if rows else 0

    def ingest(self, site: str, batch: list[dict[str, Any]]) -> dict[str, int]:
        """Store a batch from one site. Returns what was accepted and the number to acknowledge."""
        with self._lock:
            last = self.last_seq(site)
            accepted = dup = gap = 0
            for rec in sorted(batch, key=lambda r: r["seq"]):
                seq = int(rec["seq"])
                if seq <= last:
                    dup += 1
                    continue
                if seq != last + 1:
                    gap += 1                           # a record is missing before this one: do not skip ahead
                    break
                self._con.execute("INSERT INTO fleet_records(site,seq,kind,body,created,received) VALUES (?,?,?,?,?,?)",
                                  (site, seq, rec["kind"], json.dumps(rec["body"]), rec.get("created"), time.time()))
                last, accepted = seq, accepted + 1
            self._con.execute("INSERT OR REPLACE INTO fleet_contact(site,last_contact,last_seq) VALUES (?,?,?)",
                              (site, time.time(), last))
            self.duplicates += dup
            self.out_of_order += gap
            return {"accepted": accepted, "duplicates": dup, "out_of_order": gap, "ack": last}

    def records(self, site: str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        rows = self._q("SELECT site, seq, kind, body, created FROM fleet_records ORDER BY site, seq")
        return [{"site": r["site"], "seq": r["seq"], "kind": r["kind"], "created": r["created"], **json.loads(r["body"])}
                for r in rows if (site is None or r["site"] == site) and (kind is None or r["kind"] == kind)]

    def audit(self, user_id: str, question: str, site: str = "") -> None:
        with self._lock:
            self._con.execute("INSERT INTO fleet_audit(at,user_id,question,site) VALUES (?,?,?,?)",
                              (time.time(), user_id, question[:120], site))

    def audit_rows(self) -> list[dict[str, Any]]:
        return self._q("SELECT * FROM fleet_audit ORDER BY id")

    # ------------------------------------------------------------------ what support may ask

    def sites(self) -> list[str]:
        return [r["site"] for r in self._q("SELECT site FROM fleet_contact ORDER BY site")]

    def _latest_health(self, site: str) -> dict[str, Any] | None:
        hs = self.records(site, "health")
        return hs[-1] if hs else None

    def overview(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        out = []
        for c in self._q("SELECT * FROM fleet_contact ORDER BY site"):
            h = self._latest_health(c["site"]) or {}
            runs = self.records(c["site"], "run")
            age = now - (c["last_contact"] or 0)
            healthy = bool(h) and h.get("robot_reachable") and h.get("model_reachable") and h.get("perception_reachable") \
                and not h.get("robot_fault")
            out.append({"site": c["site"], "last_contact_s_ago": round(age, 1), "in_contact": age <= STALE_S,
                        "healthy": bool(healthy), "software": h.get("software"), "config_version": h.get("config_version"),
                        "runs": len(runs), "trials": len(self.records(c["site"], "trial")), "records": c["last_seq"]})
        return {"sites": out, "count": len(out)}

    def site_health(self, site: str) -> dict[str, Any]:
        h = self._latest_health(site)
        if h is None:
            return {"site": site, "known": site in self.sites(), "health": None}
        keep = ("software", "config_version", "backend", "model", "robot_reachable", "robot_ready", "robot_fault",
                "model_reachable", "models_installed", "retrieval_reachable", "perception_reachable",
                "perception_latency_ms", "disk_free_gb", "pending_approvals", "outbox_backlog", "uptime_s",
                "runs_needing_reverification")
        return {"site": site, "known": True, "health": {k: h[k] for k in keep if k in h}, "health_records": len(self.records(site, "health"))}

    def run_stats(self, site: str | None = None) -> dict[str, Any]:
        out = {}
        for s in ([site] if site else self.sites()):
            runs = self.records(s, "run")
            trials = self.records(s, "trial")
            p50 = sorted(r["llm_ms"].get("p50", 0) for r in runs if r.get("llm_ms", {}).get("count"))
            rules: dict[str, int] = {}
            for r in runs + trials:
                for rule, n in (r.get("safety_rejections_by_rule") or {}).items():
                    rules[rule] = rules.get(rule, 0) + n
            checks: dict[str, int] = {}
            for t in trials:
                for c, n in (t.get("preflight_failed_by_check") or {}).items():
                    checks[c] = checks.get(c, 0) + n
            out[s] = {"runs": len(runs), "model_calls": sum(r["llm_calls"] for r in runs),
                      "tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in runs),
                      "median_model_call_ms": p50[len(p50) // 2] if p50 else None,
                      "tool_calls": sum(r["tool_calls"] for r in runs),
                      "tool_errors": sum(sum((r.get("tool_errors_by_tool") or {}).values()) for r in runs),
                      "trials": len(trials), "safety_rejections_by_rule": rules, "preflight_failures_by_check": checks}
        return {"by_site": out}

    def recent_failures(self, site: str | None = None) -> dict[str, Any]:
        out = {}
        for s in ([site] if site else self.sites()):
            kinds: dict[str, int] = {}
            tools: dict[str, int] = {}
            stops: dict[str, int] = {}
            for r in self.records(s, "run"):
                for k, n in (r.get("errors_by_kind") or {}).items():
                    kinds[k] = kinds.get(k, 0) + n
                for k, n in (r.get("tool_errors_by_tool") or {}).items():
                    tools[k] = tools.get(k, 0) + n
                if r.get("status") != "completed":
                    stops[r.get("stop_reason") or "unknown"] = stops.get(r.get("stop_reason") or "unknown", 0) + 1
            out[s] = {"errors_by_kind": kinds, "tool_errors_by_tool": tools, "runs_not_completed_by_stop_reason": stops}
        return {"by_site": out}

    def version_drift(self) -> dict[str, Any]:
        rows = {s: (self._latest_health(s) or {}) for s in self.sites()}
        out: dict[str, Any] = {"sites": {s: {k: h.get(k) for k in ("software", "config_version", "model", "backend")} for s, h in rows.items()}}
        differ = {}
        for key in ("software", "config_version", "model", "backend"):
            values = {s: h.get(key) for s, h in rows.items()}
            if len(set(values.values())) > 1:
                differ[key] = values
        out["differs"] = differ
        out["drift"] = bool(differ)
        return out
