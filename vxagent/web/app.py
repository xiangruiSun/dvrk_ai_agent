"""The console's API and the page it serves. One process.

Every route declares the roles it is for. A request is answered only after the session value
has been verified, the user has been read from the server-side table, and the role matches;
anything else is a 401 or a 403, and a 403 is audited. No route takes a site from the browser:
the site is the user's site.

    python -m uvicorn vxagent.web.app:app --port 8300

Built on Starlette, which is already on the machine with the MCP SDK. A lab network has no
package registry to pull from, so there is no front-end build step either: the page is plain
JavaScript in vxagent/web/static.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .. import access, chat
from ..memory import site as site_memory
from ..node import Node, robot_view
from ..service import changes, diagnosis, report, tickets
from ..store import Database
from ..structured import proposals
from ..structured.schemas import ConfigPatch
from ..tools.executor import decide_approval
from ..tools.registry import Principal
from ..tools.trialtools import trial_state, trials_of

STATIC = Path(__file__).parent / "static"
COOKIE = "vx_session"
BANNER = ("Research demonstration. The robot is a kinematic twin of a dVRK arm. It is not connected to a robot, "
          "and nothing here is a medical device.")
DISCLOSURE = {
    "recorded": "dVRK configuration files, dVRK documentation and support threads, ROSMA kinematic recordings, "
                "SurgicAI expert demonstrations",
    "declared_or_injected": "the needle and whether it is held, the camera and what it sees, calibration records, "
                            "every fault",
    "not_modelled": "contact, tissue, dynamics, a camera image, a real controller's timing",
    "accounts": "demonstration accounts; what you type is stored on this machine with the traces of the run",
}
PAGES = {
    "operator": ["Robot", "Trials", "Run a trial", "Copilot", "What the copilot can do"],
    "service": ["Tickets", "Diagnosis", "Calibration", "Propose a change", "Copilot"],
    "safety": ["Change queue", "Overridden verdicts", "Support grants", "Copilot"],
    "support": ["Fleet", "Site health", "Granted tickets", "Assistant"],
}
ALL = ("operator", "service", "safety", "support")
SITE = ("operator", "service", "safety")


class Ration:
    """How many copilot jobs one visitor may run: a few per hour, one at a time."""

    def __init__(self, per_hour: int = 30, concurrent: int = 1):
        self.per_hour, self.concurrent = per_hour, concurrent
        self._times: dict[str, list[float]] = {}
        self._running: dict[str, int] = {}
        self._lock = threading.Lock()

    def enter(self, key: str, now: float | None = None) -> str | None:
        now = time.time() if now is None else now
        with self._lock:
            ts = [t for t in self._times.get(key, []) if now - t < 3600.0]
            if self._running.get(key, 0) >= self.concurrent:
                return "a job of yours is still running"
            if len(ts) >= self.per_hour:
                return f"the limit of {self.per_hour} jobs per hour is reached"
            self._times[key] = ts + [now]
            self._running[key] = self._running.get(key, 0) + 1
            return None

    def leave(self, key: str) -> None:
        with self._lock:
            self._running[key] = max(0, self._running.get(key, 0) - 1)


class Console:
    """What the web process holds: the users, the edge nodes of the sites it serves, the fleet desk."""

    def __init__(self, db: Database, nodes: dict[str, Node], desk: Any = None, secret: str = "", public: bool = False,
                 ration: Ration | None = None):
        self.db, self.nodes, self.desk = db, nodes, desk
        self.secret = secret or "dev-secret-change-me"
        self.public = public
        self.ration = ration or Ration(per_hour=12 if public else 120, concurrent=1 if public else 4)
        self.jobs: dict[str, threading.Thread] = {}
        access.ready(db)

    def place(self, who: Principal) -> Any:
        if who.role == "support":
            return self.desk
        return self.nodes.get(who.site)


ROUTES: list[dict[str, Any]] = []


def route(path: str, methods: tuple[str, ...], roles: tuple[str, ...]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        ROUTES.append({"path": path, "methods": methods, "roles": roles, "fn": fn})
        return fn
    return deco


def _json(data: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(json.loads(json.dumps(data, default=str)), status_code=status)


async def _body(request: Request) -> dict[str, Any]:
    try:
        data = await request.json()
    except (ValueError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _join(thread: threading.Thread, wait_s: float) -> None:
    thread.join(timeout=wait_s)


# ------------------------------------------------------------------ robot (operator; read for the site's roles)

@route("/api/robot", ("GET",), SITE)
def get_robot(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    return robot_view(node)


@route("/api/robot/poses", ("GET",), ("operator", "service"))
def get_poses(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    start = node.site.poses["start"].T[:3, 3]
    return {"poses": [{"name": p.name, "description": p.description,
                       "offset_from_start_mm": [round(float(x) * 1000, 1) for x in (p.T[:3, 3] - start)],
                       "height_over_pad_mm": round(node.site.scene.height(p.T[:3, 3]) * 1000, 1)} for p in node.site.poses.values()]}


@route("/api/stop", ("POST",), ("operator", "service"))
def post_stop(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    return node.stop_all(who.user_id)


@route("/api/robot/event", ("POST",), ("operator", "service"))
def post_event(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    kind = str(body.get("kind", ""))
    if kind not in ("ecm_moved", "setup_joint_moved", "instrument_changed"):
        return _json({"error": "kind is ecm_moved, setup_joint_moved or instrument_changed"}, 400)
    out = site_memory.record_event(node.db, node.name, kind, who.user_id, str(body.get("detail", ""))[:200])
    out["approvals_ended"] = changes.expire_approvals(node.db, node.name, f"the event {kind}")
    return out


@route("/api/health", ("GET",), SITE)
def get_health(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    return node.health()


@route("/api/episodes", ("GET",), ("operator", "service"))
def get_episodes(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    from ..data import episodes as ep_store
    eps = ep_store.list_episodes(node.rosma_dir)
    return {"site": node.name, "count": len(eps), "episodes": [{"episode_id": e.episode_id, "task": e.task, "trial": e.trial} for e in eps]}


@route("/api/runbooks", ("GET",), ("operator", "service"))
def get_runbooks(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    return {"runbooks": node.library.index(who.role)}


# ------------------------------------------------------------------ trials

def _trial_view(node: Node, run_id: str) -> dict[str, Any] | None:
    st = trial_state(node.db, run_id)
    if st is None or (st.get("sheet") or {}).get("site") != node.name:
        return None
    d = st.get("data") or {}
    events = node.db.query("SELECT n, type, data, at, robot_seq FROM events WHERE run_id=? ORDER BY n", (run_id,))
    spans = node.db.query("SELECT kind, name, started, latency_ms, ok FROM spans WHERE run_id=? AND kind='llm' ORDER BY id", (run_id,))
    line = [{"at": e["at"], "what": e["type"], "robot_seq": e["robot_seq"],
             "data": {k: v for k, v in json.loads(e["data"] or "{}").items() if k in (
                 "node", "label", "status", "reason", "route", "check", "ok", "approval_id", "why", "steps", "path_mm", "source")}}
            for e in events]
    line += [{"at": s["started"], "what": "model_call", "robot_seq": None, "data": {"name": s["name"], "ms": round(s["latency_ms"], 1)}}
             for s in spans]
    line.sort(key=lambda r: r["at"] or 0)
    return {"run_id": run_id, "status": (st.get("task") or {}).get("status"), "outcome": d.get("outcome"), "waiting_for": st.get("waiting_for"),
            "stages_run": [h["node"] for h in st.get("history", [])], "preflight": d.get("preflight"), "rehearsal": d.get("dry_run"),
            "verdict": d.get("verdict"), "operator_verdict": d.get("operator_verdict"), "summary": d.get("summary"),
            "summary_source": d.get("summary_source"), "repairs": d.get("repairs"), "needle": "DECLARED",
            "timeline": line[-400:]}


@route("/api/trials", ("GET",), SITE)
def get_trials(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    return {"site": node.name, "trials": trials_of(node.db, node.name)[:50]}


@route("/api/trials", ("POST",), ("operator",))
def post_trial(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    declared = body.get("needle") if isinstance(body.get("needle"), dict) else None
    allowed = {"grasp_succeeds", "slips_on_lift", "slips_while_holding"}
    if declared is not None and (set(declared) - allowed or not all(isinstance(v, bool) for v in declared.values())):
        return _json({"error": f"needle may declare {sorted(allowed)} as true or false"}, 400)
    try:
        res = node.start_trial(who, int(body.get("demo", 0)), declared)
    except FileNotFoundError as exc:
        return _json({"error": f"the policy's demonstrations are not on this node: {exc}"}, 503)
    if res.status == "finished" and res.state["data"].get("outcome") == "preflight_failed" and body.get("open_ticket"):
        failed = res.state["data"]["preflight"]["failed"]
        tid = tickets.open_ticket(node.db, node.name, "Pre-flight failed: " + "; ".join(f["check"] for f in failed)[:400],
                                  "failed_preflight", who.user_id, {"failed_checks": failed, "run": res.run_id})
        return {**(_trial_view(node, res.run_id) or {}), "ticket_id": tid}
    return _trial_view(node, res.run_id)


@route("/api/trials/{run_id}", ("GET",), SITE)
def get_trial(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    v = _trial_view(node, request.path_params["run_id"])
    return v if v is not None else _json({"error": "no such trial at this site"}, 404)


def _resume(c: Console, node: Node, run_id: str, wait_s: float) -> None:
    def work() -> None:
        try:
            node.resume_trial(run_id)
        except Exception as exc:                                             # shown on the trial page, not lost in a thread
            node.bus.publish("trial_error", {"run_id": run_id, "error": type(exc).__name__})
    t = threading.Thread(target=work, daemon=True, name=f"trial-{run_id}")
    c.jobs[run_id] = t
    t.start()
    _join(t, wait_s)


@route("/api/trials/{run_id}/go", ("POST",), ("operator",))
def post_go(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    run_id = request.path_params["run_id"]
    st = trial_state(node.db, run_id)
    waiting = (st or {}).get("waiting_for") or {}
    if st is None or st["sheet"]["site"] != node.name or waiting.get("kind") != "approval" or run_id not in node.trials:
        return _json({"error": "that trial is not waiting for a go"}, 409)
    decide_approval(node.db, waiting["approval_id"], bool(body.get("approve")), who.user_id)
    node.db.audit(who.user_id, who.role, "trial:go", run_id, True, "approved" if body.get("approve") else "rejected")
    _resume(c, node, run_id, float(body.get("wait_s", 0)))
    return _trial_view(node, run_id)


@route("/api/trials/{run_id}/verdict", ("POST",), ("operator",))
def post_verdict(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    from ..pipelines import policy_trial as PT
    run_id = request.path_params["run_id"]
    st = trial_state(node.db, run_id)
    if st is None or st["sheet"]["site"] != node.name or run_id not in node.trials:
        return _json({"error": "that trial is not waiting for a verdict"}, 409)
    try:
        PT.decide_verdict(node.db, run_id, who.user_id, str(body.get("decision", "confirm")), body.get("to"), str(body.get("reason", "")))
    except ValueError as exc:
        return _json({"error": str(exc)}, 400)
    _resume(c, node, run_id, float(body.get("wait_s", 0)))
    return _trial_view(node, run_id)


# ------------------------------------------------------------------ approvals and the copilot

@route("/api/approvals", ("GET",), ("operator",))
def get_approvals(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    return {"pending": chat.pending(node)}


@route("/api/approvals/{approval_id}", ("POST",), ("operator",))
def post_approval(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    try:
        return chat.decide(node, who, int(request.path_params["approval_id"]), bool(body.get("approve")))
    except chat.NotAllowed as exc:
        return _json({"error": str(exc)}, 409)


@route("/api/chat", ("POST",), ALL)
def post_chat(c: Console, who: Principal, place: Any, request: Request, body: dict[str, Any]) -> Any:
    text = str(body.get("message", "")).strip()
    if not text:
        return _json({"error": "message is empty"}, 400)
    key = who.user_id if not c.public else hashlib.sha256((who.user_id + request.cookies.get(COOKIE, "")).encode()).hexdigest()[:16]
    no = c.ration.enter(key)
    if no:
        return _json({"error": no}, 429)
    try:
        return chat.start(place, who, text, str(body.get("session", "main"))[:40])
    finally:
        c.ration.leave(key)


@route("/api/runs/{run_id}", ("GET",), ALL)
def get_run(c: Console, who: Principal, place: Any, request: Request, body: dict[str, Any]) -> Any:
    run_id = request.path_params["run_id"]
    run = place.db.one("SELECT user_id FROM runs WHERE run_id=?", (run_id,))
    if run is None:
        return _json({"error": "no such run"}, 404)
    if run["user_id"] != who.user_id:
        place.db.audit(who.user_id, who.role, "trace:read", run_id, False, "not this person's run")
        return _json({"error": "not your run"}, 403)
    return chat.trace(place.db, run_id)


@route("/api/copilot", ("GET",), ALL)
def get_copilot(c: Console, who: Principal, place: Any, request: Request, body: dict[str, Any]) -> Any:
    cfg = chat.config_for(who)
    tools = place.registry.allowed_for(who, cfg.tool_names)
    return {"copilot": cfg.name, "tools": [{"name": t.name, "tier": t.tier, "needs_approval": bool(t.requires_approval),
                                             "what": t.description} for t in tools],
            "cannot": {"operator": "change a limit, a configuration file or a calibration; send coordinates; skip a check",
                       "service": "approve a change; activate a calibration; run a trial for the record; move the arm "
                                  "other than through the calibration poses with an operator's approval",
                       "safety": "anything but read", "support": "see a site's recordings, trials, poses or configuration; "
                                                                 "move a robot; change anything"}[who.role]}


# ------------------------------------------------------------------ tickets (service engineer; read for the safety lead)

@route("/api/tickets", ("GET",), ("service", "safety"))
def get_tickets(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    return {"site": node.name, "tickets": tickets.list_site(node.db, node.name)}


@route("/api/tickets", ("POST",), ("operator", "service"))
def post_ticket(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    symptom = str(body.get("symptom", "")).strip()
    if len(symptom) < 10:
        return _json({"error": "describe the symptom in at least 10 characters"}, 400)
    return {"ticket_id": tickets.open_ticket(node.db, node.name, symptom, "operator", who.user_id), "status": "open"}


def _site_ticket(node: Node, request: Request, ticket_id: Any = None) -> dict[str, Any] | None:
    try:
        t = tickets.get(node.db, int(request.path_params["ticket_id"] if ticket_id is None else ticket_id))
    except (TypeError, ValueError):
        return None
    return t if t is not None and t["site"] == node.name else None


@route("/api/tickets/{ticket_id}", ("GET",), ("service", "safety"))
def get_ticket(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    t = _site_ticket(node, request)
    if t is None:
        return _json({"error": "no such ticket at this site"}, 404)
    return {**t, "records": tickets.records(node.db, t["id"])}


@route("/api/tickets/{ticket_id}/diagnose", ("POST",), ("service",))
def post_diagnose(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    if _site_ticket(node, request) is None:
        return _json({"error": "no such ticket at this site"}, 404)
    if node.llm is None:
        return _json({"error": "the language model is not reachable; diagnosis needs it"}, 503)
    try:
        return diagnosis.diagnose_ticket(node, who, int(request.path_params["ticket_id"]), str(body.get("method", "eliminate")))
    except PermissionError as exc:
        return _json({"error": str(exc)}, 403)
    except ValueError as exc:
        return _json({"error": str(exc)}, 400)


@route("/api/tickets/{ticket_id}/close", ("POST",), ("service",))
def post_close(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    if _site_ticket(node, request) is None:
        return _json({"error": "no such ticket at this site"}, 404)
    try:
        return report.close_with_report(node, who, int(request.path_params["ticket_id"]))
    except PermissionError as exc:
        return _json({"error": str(exc)}, 403)


# ------------------------------------------------------------------ configuration, calibration, changes

@route("/api/config", ("GET",), ("service", "safety"))
def get_config(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    from ..robot import config as cfg
    from ..structured.rules import describe_paths
    if node.store is None:
        return _json({"error": "this node has no configuration store"}, 503)
    target = request.query_params.get("target") or node.site_target
    try:
        doc, h, version = node.store.current(target)
    except KeyError:
        return _json({"error": "no such file in this site's configuration"}, 404)
    return {"target": target, "version": version, "hash": h, "type": cfg.file_type(doc),
            "paths_a_change_may_touch": describe_paths(cfg.file_type(doc), doc)[:40]}


@route("/api/calibration", ("GET",), ("service", "safety"))
def get_calibration(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    from ..service import calibration as cal
    from ..tools.service import load_calibration_record
    rec, draft = load_calibration_record(node.db, node.name), cal.load_draft(node.db, node.name)
    return {"in_use": {"id": rec["id"], "method": rec.get("method"), "session": rec.get("session")} if rec else None,
            "validity": proposals.calibration(node.db, node.name),
            "draft": {"id": draft["id"], "status": draft["status"], "held_out_residual": draft["held_out_residual"],
                      "samples": draft["samples"]} if draft else None,
            "camera": "DECLARED: the twin has no camera"}


@route("/api/changes", ("GET",), ("service", "safety"))
def get_changes(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    status = request.query_params.get("status", "proposed")
    return {"site": node.name, "changes": changes.queue(node.db, node.name, None if status == "all" else status)}


@route("/api/changes", ("POST",), ("service",))
def post_change(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    from pydantic import ValidationError
    ticket = body.pop("ticket_id", None)
    try:
        if body.get("kind") == "calibration":
            return changes.propose_calibration(node, who, str(body.get("reason", "")), ticket)
        return changes.propose_patch(node, who, ConfigPatch.model_validate(body), ticket)
    except ValidationError as exc:
        return _json({"error": "not a valid patch", "problems": [e["msg"] for e in exc.errors()][:5]}, 400)
    except proposals.NotPermitted as exc:
        return _json({"error": str(exc), "kind": exc.kind}, 403 if exc.kind in ("role", "site") else 409)


@route("/api/changes/{proposal_id}/decision", ("POST",), ("safety",))
def post_decision(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    try:
        return changes.decide(node, who, int(request.path_params["proposal_id"]), bool(body.get("approve")), str(body.get("reason", "")))
    except KeyError:
        return _json({"error": "no such proposal"}, 404)
    except proposals.NotPermitted as exc:
        return _json({"error": str(exc), "kind": exc.kind}, 403 if exc.kind in ("role", "site", "self_approval") else 409)


@route("/api/overrides", ("GET",), ("safety",))
def get_overrides(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    return {"site": node.name, "overridden_verdicts": changes.overrides(node.db, node.name)}


# ------------------------------------------------------------------ support grants (safety lead)

@route("/api/grants", ("GET",), ("safety",))
def get_grants(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    from ..fleet import grants
    return {"site": node.name, "grants": grants.listing(node.db, node.name)}


@route("/api/grants", ("POST",), ("safety",))
def post_grant(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    from ..fleet import grants
    if _site_ticket(node, request, body.get("ticket_id", 0)) is None:
        return _json({"error": "no such ticket at this site"}, 404)
    try:
        return grants.give(node.db, who, int(body.get("ticket_id", 0)), str(body.get("grantee", "")), float(body.get("hours", 0)))
    except grants.NotGranted as exc:
        return _json({"error": str(exc), "kind": exc.kind}, 403 if exc.kind in ("role", "site") else 400)


@route("/api/grants/{grant_id}/revoke", ("POST",), ("safety",))
def post_revoke(c: Console, who: Principal, node: Node, request: Request, body: dict[str, Any]) -> Any:
    from ..fleet import grants
    if not any(g["grant_id"] == int(request.path_params["grant_id"]) for g in grants.listing(node.db, node.name)):
        return _json({"error": "no such grant at this site"}, 404)
    try:
        return grants.revoke(node.db, who, int(request.path_params["grant_id"]))
    except grants.NotGranted as exc:
        return _json({"error": str(exc)}, 403)


# ------------------------------------------------------------------ fleet (remote support)

@route("/api/fleet/overview", ("GET",), ("support",))
def fleet_overview(c: Console, who: Principal, desk: Any, request: Request, body: dict[str, Any]) -> Any:
    desk.store.audit(who.user_id, "overview (console)")
    return desk.store.overview()


@route("/api/fleet/sites/{site}", ("GET",), ("support",))
def fleet_site(c: Console, who: Principal, desk: Any, request: Request, body: dict[str, Any]) -> Any:
    site = request.path_params["site"]
    desk.store.audit(who.user_id, "site health (console)", site)
    return {**desk.store.site_health(site), "stats": desk.store.run_stats(site)["by_site"].get(site),
            "failures": desk.store.recent_failures(site)["by_site"].get(site)}


@route("/api/fleet/drift", ("GET",), ("support",))
def fleet_drift(c: Console, who: Principal, desk: Any, request: Request, body: dict[str, Any]) -> Any:
    desk.store.audit(who.user_id, "version drift (console)")
    return desk.store.version_drift()


@route("/api/fleet/tickets/{site}/{ticket_id}", ("GET",), ("support",))
def fleet_ticket(c: Console, who: Principal, desk: Any, request: Request, body: dict[str, Any]) -> Any:
    from ..fleet import grants
    site = request.path_params["site"]
    desk.store.audit(who.user_id, "granted ticket (console)", site)
    edge = desk.edges.get(site)
    if edge is None:
        return _json({"error": "that site's edge node cannot be reached"}, 503)
    try:
        return grants.read_ticket(edge, who, int(request.path_params["ticket_id"]))
    except grants.NotGranted as exc:
        return _json({"error": str(exc), "kind": exc.kind}, 403)


# ------------------------------------------------------------------ the application

def create_app(console: Console) -> Starlette:
    def session_user(request: Request) -> dict[str, Any] | None:
        token = request.cookies.get(COOKIE, "")
        uid = access.verify(token, console.secret) if token else None
        return access.user(console.db, uid) if uid else None        # role and site come from the table, every time

    def guarded(spec: dict[str, Any]) -> Callable[[Request], Any]:
        async def endpoint(request: Request) -> Response:
            row = session_user(request)
            if row is None:
                return _json({"error": "not logged in"}, 401)
            who = access.principal_of(row)
            if who.role not in spec["roles"]:
                console.db.audit(who.user_id, who.role, f"http:{request.method}", spec["path"], False, "role")
                return _json({"error": f"this page is not for the role {who.role}"}, 403)
            place = console.place(who)
            if place is None:
                console.db.audit(who.user_id, who.role, f"http:{request.method}", spec["path"], False, "no node for this site")
                return _json({"error": "no edge node for your site is served here"}, 403)
            body = await _body(request) if request.method == "POST" else {}
            out = await run_in_threadpool(spec["fn"], console, who, place, request, body)
            return out if isinstance(out, Response) else _json(out)
        return endpoint

    async def login(request: Request) -> Response:
        body = await _body(request)
        try:
            row = await run_in_threadpool(access.login, console.db, str(body.get("user_id", ""))[:80], str(body.get("password", ""))[:200])
        except access.LoginRefused as exc:
            return _json({"error": "the account is locked for a few minutes" if exc.kind == "locked" else "wrong user or password"},
                         423 if exc.kind == "locked" else 401)
        resp = _json({"user_id": row["user_id"], "role": row["role"], "site": row["site"], "display_name": row["display_name"],
                      "pages": PAGES[row["role"]]})
        resp.set_cookie(COOKIE, access.sign(row["user_id"], console.secret), httponly=True, samesite="strict",
                        secure=console.public, max_age=int(access.SESSION_S), path="/")
        return resp

    async def logout(request: Request) -> Response:
        resp = _json({"ok": True})
        resp.delete_cookie(COOKIE, path="/")
        return resp

    async def me(request: Request) -> Response:
        row = session_user(request)
        if row is None:
            return _json({"error": "not logged in"}, 401)
        return _json({**row, "pages": PAGES[row["role"]], "banner": BANNER})

    async def about(request: Request) -> Response:
        return _json({"banner": BANNER, **DISCLOSURE, "public": console.public})

    async def events(request: Request) -> Response:
        row = session_user(request)
        if row is None:
            return _json({"error": "not logged in"}, 401)
        who = access.principal_of(row)
        if who.role not in SITE or console.place(who) is None:
            console.db.audit(who.user_id, who.role, "http:GET", "/api/events", False, "role")
            return _json({"error": "the live view is for the roles of a site"}, 403)
        bus = console.place(who).bus
        try:
            last = int(request.headers.get("last-event-id") or request.query_params.get("after") or 0)
        except ValueError:
            last = 0
        once = request.query_params.get("once") == "1"

        async def stream() -> Any:
            nonlocal last
            while True:
                batch = await run_in_threadpool(bus.since, last, 0.0 if once else 10.0)
                for ev in batch:
                    last = ev["id"]
                    yield f"id: {ev['id']}\nevent: {ev['kind']}\ndata: {json.dumps({'at': ev['at'], **ev['data']}, default=str)}\n\n"
                if once:
                    return
                if not batch:
                    yield ": keep-alive\n\n"
                if await request.is_disconnected():
                    return
                await asyncio.sleep(0)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})

    async def index(request: Request) -> Response:
        return FileResponse(STATIC / "index.html")

    routes: list[Any] = [Route("/api/login", login, methods=["POST"]), Route("/api/logout", logout, methods=["POST"]),
                         Route("/api/me", me), Route("/api/about", about), Route("/api/events", events), Route("/", index)]
    by_path: dict[str, list[dict[str, Any]]] = {}
    for spec in ROUTES:
        by_path.setdefault(spec["path"], []).append(spec)
    for path, specs in by_path.items():
        for spec in specs:
            routes.append(Route(path, guarded(spec), methods=list(spec["methods"])))
    routes.append(Mount("/static", StaticFiles(directory=str(STATIC)), name="static"))
    class Headers(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next: Any) -> Response:
            resp = await call_next(request)
            resp.headers["X-Content-Type-Options"] = "nosniff"
            resp.headers["X-Frame-Options"] = "DENY"
            resp.headers["Referrer-Policy"] = "no-referrer"
            resp.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
            if request.url.path.startswith("/api/"):
                resp.headers["Cache-Control"] = "no-store"
            return resp

    app = Starlette(routes=routes, middleware=[Middleware(Headers)])
    app.state.console = console
    return app


def route_table() -> list[dict[str, Any]]:
    """Every guarded route with the roles it declares. The tests and the verification walk this."""
    return [{"path": r["path"], "method": m, "roles": list(r["roles"])} for r in ROUTES for m in r["methods"]]


def default_console() -> Console:
    """The console of the local demonstration: both sites, the fleet desk, demonstration accounts."""
    from .serve import build_console
    return build_console()


def __getattr__(name: str) -> Any:
    if name == "app":                                   # built on first use: `uvicorn vxagent.web.app:app`
        global _app
        try:
            return _app
        except NameError:
            _app = create_app(default_console())
            return _app
    raise AttributeError(name)
