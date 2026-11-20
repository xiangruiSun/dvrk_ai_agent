"""Redaction at the edge. What may leave a lab is decided here, before the outbox.

A record for the fleet is built from counts, kinds and timings only. It is then checked against
a closed description of what such a record may contain: known keys, numbers, and strings from
fixed vocabularies (tool names, error kinds, rule names, check names, model names, version
strings). Anything else is refused. A record is never "cleaned": one that would need cleaning
is a bug in the code that built it.
"""
from __future__ import annotations

import re
from typing import Any

# Keys that would carry a lab's content. Their presence anywhere in a record is a failure.
PAYLOAD_KEYS = frozenset({
    "task", "answer", "input", "output", "arguments", "args", "content", "text", "message", "messages", "symptom",
    "prompt", "pose", "poses", "position", "position_mm", "measured", "setpoint", "joints", "q", "file", "files",
    "path", "filename", "episode", "episode_id", "detail", "error", "reason", "note", "user", "user_id", "operator",
    "transform", "T_est", "samples", "evidence", "finding", "summary", "name_of_person", "image"})
# Keys whose value is a free identifier chosen by code or configuration, not by a person at the rig.
ID_KEYS = frozenset({"site", "software", "config_version", "model", "backend", "node", "agent"})
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+\-]{0,63}$")
MAX_DEPTH = 4


class Leak(ValueError):
    pass


def vocabulary(registry: Any = None) -> frozenset[str]:
    """Strings a record may use as keys of its count tables."""
    from ..diagnostics.episode import CAUSES
    words = {"observe", "simulate", "actuate", "llm", "tool", "guard", "completed", "stopped", "failed", "awaiting_approval",
             "running", "finished", "waiting", "needs_reverification", "final_answer", "final_answer_unverified",
             "max_iterations", "token_budget", "wall_clock", "loop_detected", "tool_failures", "safety_rejections",
             "robot_fault", "operator_stop", "llm_error", "unknown", "p50", "p95", "max", "mean", "count",
             "invalid_arguments", "unknown_tool", "unauthorized", "timeout", "transient", "stale_observation",
             "safety_rejected", "robot_not_ready", "duplicate_command", "approval_expired", "rejected", "canceled",
             "result_too_large", "not_parallel", "tracking_error", "ungrounded_answer", "refused",
             "step_position", "step_rotation", "workspace", "clearance", "joint_margin", "joint_limit", "unreachable",
             "tracking_position", "tracking_rotation", "clipped", "preflight_failed", "rehearsal_rejected",
             "operator_rejected", "operator_timeout", "succeeded", "confirmed", "overridden", "reachable", "ready", "fault",
             "arm", "instrument", "calibration_record", "calibration_formed", "calibration_event", "calibration_residual",
             "scene_session", "policy_contract", "training_range", "perception", "other", "run", "trial", "health",
             "operator", "service", "safety", "support", "system", "success", "failed_grasp", "not_reached", "confirm",
             "override", "none"} | set(CAUSES)
    if registry is not None:
        words |= set(registry.names())
    return frozenset(words)


# Every key a record for the fleet is built with. A count table's keys come from the vocabulary.
STRUCT_KEYS = frozenset({
    "kind", "site", "node", "software", "config_version", "backend", "model", "agent", "role", "status", "stop_reason",
    "llm_calls", "tool_calls", "prompt_tokens", "completion_tokens", "wall_ms", "llm_ms", "tool_ms", "tool_calls_by_tool",
    "tool_errors_by_tool", "errors_by_kind", "safety_rejections_by_rule", "outcome", "machine_verdict",
    "operator_decision", "motions", "rejections", "recoveries", "preflight_failed_by_check", "robot_reachable",
    "robot_ready", "robot_fault", "model_reachable", "models_installed", "retrieval_reachable", "perception_reachable",
    "perception_latency_ms", "disk_free_gb", "pending_approvals", "outbox_backlog", "uptime_s", "control_period_ms",
    "perception_ms", "tool_retries", "runs_needing_reverification"})

CHECK_KEYS = {
    "arm is powered, enabled and homed, with no fault": "arm",
    "instrument is the one on the run sheet and has an instrument file": "instrument",
    "a calibration record exists": "calibration_record",
    "the calibration record is well formed": "calibration_formed",
    "the calibration has not been ended by an event": "calibration_event",
    "the calibration residual is under the site's limit": "calibration_residual",
    "pad-plane and site records are of the same session as the calibration": "scene_session",
    "the policy contract is compatible with the site": "policy_contract",
    "start and target lie in the policy's training range and inside the workspace": "training_range",
    "the perception source answered within its time limit": "perception",
}


def problems(record: Any, vocab: frozenset[str], _key: str = "", _depth: int = 0) -> list[str]:
    """Everything in a record that is not a count, a kind or a timing. Empty means it may leave."""
    out: list[str] = []
    if _depth > MAX_DEPTH:
        return [f"{_key}: nested deeper than {MAX_DEPTH}"]
    if isinstance(record, dict):
        for k, v in record.items():
            where = f"{_key}.{k}" if _key else str(k)
            if not isinstance(k, str):
                out.append(f"{_key}: a key that is not text")
            elif k in PAYLOAD_KEYS:
                out.append(f"{where}: a payload field")
            elif k not in STRUCT_KEYS and k not in vocab:
                out.append(f"{where}: not a known key or kind")
            else:
                out += problems(v, vocab, where, _depth + 1)
    elif isinstance(record, bool) or record is None or isinstance(record, (int, float)):
        pass
    elif isinstance(record, str):
        leaf = _key.rsplit(".", 1)[-1]
        if leaf in ID_KEYS:
            if not _ID.match(record):
                out.append(f"{_key}: not an identifier")
        elif record not in vocab:
            out.append(f"{_key}: free text")
    elif isinstance(record, list):
        out.append(f"{_key}: a list (lists can carry poses or text)")
    else:
        out.append(f"{_key}: a value of type {type(record).__name__}")
    return out


def check(record: dict[str, Any], vocab: frozenset[str]) -> dict[str, Any]:
    bad = problems(record, vocab)
    if bad:
        raise Leak("; ".join(bad[:5]))
    return record


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    xs = sorted(values)
    pick = lambda q: xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]      # noqa: E731
    return {"count": len(xs), "p50": round(pick(0.5), 1), "p95": round(pick(0.95), 1), "max": round(xs[-1], 1)}


def known(word: str | None, vocab: frozenset[str]) -> str:
    """A kind as the fleet may see it: itself if it is in the vocabulary, otherwise "other"."""
    return word if word in vocab else "other"
