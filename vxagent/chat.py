"""The chat path: one question from a logged-in person to the copilot of their role, and back.

A place is an edge node (operator, service engineer, safety lead) or the fleet desk (remote
support). The person's role picks the copilot; nothing the browser or the model sends picks the
site. A run that asks for a motion pauses with an approval card; the operator's decision is
recorded by the console and the run goes on. Every answer comes back with its trace.
"""
from __future__ import annotations

import json
import time
from typing import Any

from . import access
from .agent.loop import AgentState, run_agent
from .memory import site as site_memory
from .memory import stores
from .robot.motion import state_report
from .store import dumps
from .tools.executor import decide_approval
from .tools.registry import Principal, may_request

HISTORY_MESSAGES = 6
APPROVAL_WAIT_S = 120.0
UNAVAILABLE = ("The copilot is unavailable: the language model could not be reached. The robot view, the runbooks "
               "and the trial pipeline do not need it and still work from the console.")


class NotAllowed(Exception):
    pass


def config_for(who: Principal) -> Any:
    from .profiles import COPILOTS
    if who.role not in COPILOTS:
        raise NotAllowed(f"role {who.role!r} has no copilot")
    return COPILOTS[who.role]


def run_tools(place: Any, who: Principal, cfg: Any) -> list[str]:
    """The motion tools a run of this copilot may need: those on its list and those inside the
    runbooks its role may run. The run is registered with exactly these."""
    names = set(cfg.tool_names or [])
    library = getattr(place, "library", None)
    if "run_runbook" in names and library is not None:
        for rb in library.books.values():
            if who.role in rb.roles:
                names |= {s.tool for s in rb.steps}
    out = []
    for n in sorted(names):
        t = place.registry.get(n)
        if t is not None and t.tier == "actuate" and may_request(who, t):
            out.append(n)
    return out


def system_prompt(place: Any, who: Principal, cfg: Any) -> str:
    parts = [cfg.system_prompt]
    library = getattr(place, "library", None)
    if library is not None and "run_runbook" in (cfg.tool_names or []):
        idx = library.index(who.role)
        if idx:
            parts.append("Runbooks you can run with run_runbook (read one with load_runbook):\n"
                         + "\n".join(f"- {r['name']}: {r['when_to_use']}" for r in idx))
    site = getattr(place, "name", "")
    if who.site:
        parts.append(f"You are working at {who.site} for {who.display_name or who.user_id}. Every tool acts on this "
                     "site only; there is no way to name another one.")
        block = site_memory.context_block(place.db, who, site) if site else ""
        if block:
            parts.append(block)
    mem = stores.memories(place.db, who.user_id)
    if mem:
        parts.append("What this person asked you to remember:\n" + "\n".join(f"- {m}" for m in mem))
    return "\n\n".join(parts)


def start(place: Any, who: Principal, text: str, session: str = "main", on_step: Any = None) -> dict[str, Any]:
    """Run the copilot on one message until it answers, stops or waits for an approval."""
    from .node import RunBox
    cfg = config_for(who)
    text = text.strip()[:4000]
    if not text:
        raise ValueError("empty message")
    run_id = place.tracer.start_run(cfg.name, text, who.user_id, who.role, getattr(place, "name", ""), meta={"session": session})
    ctx = place.ctx(who, run_id)
    access.begin_run(place.db, run_id, who.user_id, getattr(place, "name", ""), run_tools(place, who, cfg))
    history = [m for m in stores.load_session(place.db, who.user_id, session) if m.get("role") in ("user", "assistant")]
    state = AgentState(run_id=run_id, task=text, messages=[{"role": "system", "content": system_prompt(place, who, cfg)},
                                                           *history[-HISTORY_MESSAGES:], {"role": "user", "content": text}])
    box = RunBox(run_id=run_id, who=who, agent=cfg.name, session=session, ctx=ctx, cfg=cfg, state=state,
                 extra={"on_step": on_step})
    place.boxes[run_id] = box
    return _advance(place, box)


def _advance(place: Any, box: Any) -> dict[str, Any]:
    llm = place.llm
    if llm is None:
        box.state.status, box.state.stop_reason, box.state.answer = "failed", "llm_error", UNAVAILABLE
    else:
        box.state = run_agent(llm, place.registry, box.ctx, config=box.cfg, state=box.state, on_step=box.extra.get("on_step"))
        if box.state.stop_reason == "llm_error":
            box.state.answer = UNAVAILABLE
    st = box.state
    if st.status == "awaiting_approval":
        place.db.execute("UPDATE runs SET status='awaiting_approval' WHERE run_id=?", (box.run_id,))
        place.bus.publish("approval", {"run_id": box.run_id, "approval_id": st.pending_approval["approval_id"]})
        return view(place, box)
    box.done = True
    access.end_run(place.db, box.run_id)                 # the run's permission to move ends with the run
    place.tracer.end_run(box.run_id, st.status, st.answer)
    place.db.execute("UPDATE runs SET meta=? WHERE run_id=?", (dumps({"session": box.session, "stop_reason": st.stop_reason}), box.run_id))
    if st.status == "completed":
        prior = stores.load_session(place.db, box.who.user_id, box.session)
        stores.save_session(place.db, box.who.user_id, box.session,
                            (prior + [{"role": "user", "content": st.task}, {"role": "assistant", "content": st.answer}])[-20:])
    fleet = getattr(place, "fleet", None)
    if fleet is not None:
        fleet.record_run(box.run_id)
    place.bus.publish("run", {"run_id": box.run_id, "status": st.status, "stop_reason": st.stop_reason})
    return view(place, box)


def view(place: Any, box: Any) -> dict[str, Any]:
    st = box.state
    out = {"run_id": box.run_id, "agent": box.agent, "status": st.status, "stop_reason": st.stop_reason, "answer": st.answer,
           "approval": None, "trace": trace(place.db, box.run_id)}
    if st.status == "awaiting_approval" and st.pending_approval:
        out["approval"] = card(place, st.pending_approval["approval_id"])
    return out


# ------------------------------------------------------------------ approvals

def card(place: Any, approval_id: int) -> dict[str, Any] | None:
    """What the operator is shown before a motion: what will move, from where to where, the
    profile in force, the rehearsal on the twin, and how long the card is valid."""
    a = place.db.one("SELECT * FROM approvals WHERE id=?", (approval_id,))
    if a is None:
        return None
    args = json.loads(a["arguments"] or "{}")
    robot, site = place.robot, place.site
    now = state_report(robot.state(), site.scene)
    out: dict[str, Any] = {
        "approval_id": a["id"], "run_id": a["run_id"], "status": a["status"], "requested_by": a["requested_by"],
        "tool": a["tool"], "arguments": {k: v for k, v in args.items() if k != "based_on_seq"},
        "what_moves": f"{site.arm} at {site.name} (kinematic twin)",
        "from": {k: now.get(k) for k in ("position_mm", "height_over_pad_mm", "jaw_deg", "state_seq") if k in now},
        "profile": f"{site.profile.name} v{site.profile.version}",
        "bound_to": {"robot_state_seq": a["robot_seq"], "profile_version": a["profile_version"]},
        "expires_at": a["created"] + APPROVAL_WAIT_S, "seconds_left": max(0.0, round(a["created"] + APPROVAL_WAIT_S - time.time(), 1))}
    pose = site.poses.get(args.get("named_pose", ""))
    if pose is not None:
        from .robot.kinematics import pose_distance
        d_m, d_rad = pose_distance(robot.state().measured_cp, pose.T)
        start = site.poses["start"].T[:3, 3]
        out["to"] = {"named_pose": pose.name, "offset_from_start_mm": [round(float(x) * 1000, 1) for x in (pose.T[:3, 3] - start)],
                     "height_over_pad_mm": round(site.scene.height(pose.T[:3, 3]) * 1000, 1),
                     "distance_mm": round(d_m * 1000, 1)}
        tool = place.registry.get("rehearse_move")
        try:
            sys_ctx = place.ctx(access.principal_of({"user_id": "system", "role": "operator", "site": site.name, "display_name": ""}),
                                f"card_{approval_id}")
            r = tool.fn(tool.args_model(named_pose=pose.name), sys_ctx)
            out["rehearsal"] = {k: r.get(k) for k in ("would_complete", "control_steps", "expected_duration_s", "stopped_by") if k in r}
        except Exception as exc:                                             # a card without a rehearsal says so
            out["rehearsal"] = {"would_complete": None, "error": type(exc).__name__}
    elif a["tool"] == "sweep_calibration_poses":
        from .service.calibration import calibration_poses
        out["to"] = {"calibration_poses": len(calibration_poses(site, place.spec)), "then": "start"}
        out["rehearsal"] = {"would_complete": True, "note": "every pose was checked against the safety filter before this card"}
    elif a["tool"] == "enable_and_home":
        out["to"] = {"state": "enabled and homed; the arm does not travel"}
    return out


def pending(place: Any) -> list[dict[str, Any]]:
    rows = place.db.query("SELECT id FROM approvals WHERE status='pending' ORDER BY id")
    return [c for c in (card(place, r["id"]) for r in rows) if c is not None]


def decide(place: Any, who: Principal, approval_id: int, approved: bool) -> dict[str, Any]:
    """The operator's decision on a card. Only an operator of this site decides, whoever asked."""
    a = place.db.one("SELECT * FROM approvals WHERE id=?", (approval_id,))
    resource = f"approval:{approval_id}"
    if who.role != "operator" or who.site != getattr(place, "name", ""):
        place.db.audit(who.user_id, who.role, "approval:decide", resource, False, "only an operator of this site decides a motion")
        raise NotAllowed("only an operator of this site decides a motion")
    if a is None or a["status"] != "pending":
        raise NotAllowed("that approval is not waiting for a decision")
    if time.time() - a["created"] > APPROVAL_WAIT_S:
        place.db.execute("UPDATE approvals SET status='expired' WHERE id=?", (approval_id,))
        place.db.audit(who.user_id, who.role, "approval:decide", resource, False, "the card has expired")
        approved, expired = False, True
    else:
        decide_approval(place.db, approval_id, approved, who.user_id)
        place.db.audit(who.user_id, who.role, "approval:decide", resource, True, "approved" if approved else "rejected")
        expired = False
    box = place.boxes.get(a["run_id"])
    if box is None or box.done:
        return {"approval_id": approval_id, "decision": "expired" if expired else ("approved" if approved else "rejected"), "run": None}
    out = _advance(place, box)
    out["decision"] = "expired" if expired else ("approved" if approved else "rejected")
    return out


# ------------------------------------------------------------------ trace

def trace(db: Any, run_id: str) -> dict[str, Any]:
    """Everything behind an answer: model calls, tool calls with inputs and outputs, their
    latency and tokens, and why the run stopped."""
    run = db.one("SELECT * FROM runs WHERE run_id=?", (run_id,)) or {}
    spans = db.query("SELECT * FROM spans WHERE run_id=? ORDER BY id", (run_id,))
    steps = []
    for s in spans:
        step = {"kind": s["kind"], "name": s["name"], "ok": bool(s["ok"]), "latency_ms": round(s["latency_ms"] or 0.0, 1)}
        if s["kind"] == "llm":
            step["tokens"] = {"prompt": s["prompt_tokens"], "completion": s["completion_tokens"]}
        if s["kind"] in ("tool", "guard"):
            step["input"], step["output"] = _load(s["input"]), _load(s["output"])
            if s["seq_start"] is not None and s["seq_end"] is not None and s["seq_end"] != s["seq_start"]:
                step["robot_state_seq"] = [s["seq_start"], s["seq_end"]]
        if not s["ok"]:
            step["error_kind"] = s["error_kind"]
        steps.append(step)
    llm = [s for s in spans if s["kind"] == "llm"]
    meta = _load(run.get("meta")) or {}
    return {"run_id": run_id, "status": run.get("status"), "stop_reason": meta.get("stop_reason"),
            "model_calls": len(llm), "tool_calls": sum(1 for s in spans if s["kind"] == "tool"),
            "tokens": sum(s["prompt_tokens"] + s["completion_tokens"] for s in llm),
            "model_ms": round(sum(s["latency_ms"] or 0 for s in llm), 1),
            "wall_ms": round(((run.get("ended") or time.time()) - run["started"]) * 1000, 1) if run.get("started") else None,
            "steps": steps}


def _load(text: Any) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return str(text)[:400]
