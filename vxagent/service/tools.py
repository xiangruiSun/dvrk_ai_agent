"""Tools of the service engineer and, read-only, of the safety lead: configuration, the
calibration procedure, change proposals. Nothing here applies a change. A proposal is a record
for the safety lead; a calibration fit is a draft."""
from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..memory import site as site_memory
from ..robot import config as cfg
from ..robot.backend import BackendError
from ..robot.motion import MotionStopped, check_goal, execute_path, state_report
from ..structured import proposals
from ..structured.rules import describe_paths
from ..structured.schemas import ConfigPatch
from ..tools.actions import ActionAbort, ActionHandle, start_action
from ..tools.registry import Tool, ToolContext, ToolError, ToolRegistry
from ..tools.service import load_calibration_record
from . import calibration as cal
from . import changes, tickets

CONFIG_READ = frozenset({"config:read"})
CONFIG_PROPOSE = frozenset({"config:propose"})
CALIBRATE = frozenset({"calibration:collect"})
TICKET_WRITE = frozenset({"ticket:write"})
SITE_EVENT = frozenset({"site:event"})


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(Args):
    pass


def _node(ctx: ToolContext) -> Any:
    node = ctx.extras.get("node")
    if node is None:
        raise ToolError("failed", "no edge node is connected to this session")
    return node


# ------------------------------------------------------------------ configuration, read

class ReadConfigArgs(Args):
    target: str | None = Field(default=None, description="A file of this site's configuration, exactly as listed. "
                               "Leave out to list the files.")
    path: str | None = Field(default=None, description="A path inside the file, segments separated by '/', for "
                             "example safety_profile/near_step_mm. Leave out for the top-level keys.")


def read_config(a: ReadConfigArgs, ctx: ToolContext) -> dict[str, Any]:
    node = _node(ctx)
    if node.store is None:
        raise ToolError("failed", "this node has no configuration store")
    if a.target is None:
        mine = [t for t in sorted(node.store.sources) if t == node.site_target or any(
            node.spec.get(k, "").endswith(t.split("/", 1)[1]) for k in ("arm_file", "system_file"))]
        return {"site": node.name, "files": [{"target": t, "version": node.store.current(t)[2],
                                              "type": cfg.file_type(node.store.current(t)[0])} for t in mine],
                "note": "version 0 is the file as published; each applied change adds a version"}
    try:
        doc, _, version = node.store.current(a.target)
    except KeyError as exc:
        raise ToolError("invalid_arguments", str(exc), hint="call read_config without a target for the files")
    if a.path is None:
        return {"target": a.target, "version": version, "type": cfg.file_type(doc),
                "keys": list(doc)[:40] if isinstance(doc, dict) else f"a list of {len(doc)}"}
    try:
        value = cfg.get(doc, a.path)
    except cfg.PathError as exc:
        raise ToolError("invalid_arguments", str(exc), hint="call read_config with the target only to see its keys")
    if value is cfg.MISSING:
        return {"target": a.target, "version": version, "path": a.path, "set": False}
    text = cfg.canonical(value)
    return {"target": a.target, "version": version, "path": a.path, "set": True,
            "value": value if len(text) <= 1500 else {"shortened": text[:1500] + " ..."}}


class TargetArgs(Args):
    target: str = Field(description="A file of this site's configuration, exactly as read_config lists it.")


def describe_config_paths(a: TargetArgs, ctx: ToolContext) -> dict[str, Any]:
    node = _node(ctx)
    try:
        doc, _, version = node.store.current(a.target)
    except (KeyError, AttributeError) as exc:
        raise ToolError("invalid_arguments", str(exc), hint="call read_config without a target for the files")
    rows = describe_paths(cfg.file_type(doc), doc)
    return {"target": a.target, "version": version, "paths_a_change_may_touch": rows[:30], "count": len(rows),
            "note": "any other path is refused; a safety limit says which direction is tighter"}


# ------------------------------------------------------------------ the calibration procedure

class SweepArgs(Args):
    based_on_seq: int = Field(description="state_seq from your most recent get_robot_state.")


def sweep_calibration_poses(a: SweepArgs, ctx: ToolContext) -> ActionHandle:
    node, robot, site = _node(ctx), ctx.robot, ctx.site
    poses = cal.calibration_poses(site, node.spec)
    if len(poses) < 6:
        raise ToolError("failed", "this site declares fewer than 6 calibration poses", hint="escalate; do not improvise poses")
    s = robot.state()
    if not s.ready:
        raise ToolError("robot_fault" if s.fault else "robot_not_ready", "the arm is not enabled and homed",
                        hint="the operator has to bring the arm up first", detail=state_report(s, site.scene))
    for i, T in enumerate(poses):
        bad = check_goal(site.profile, site.scene, site.chain, T, s.q)
        if bad:
            raise ToolError("safety_rejected", f"calibration pose {i + 1} was rejected by rule {bad[0].rule}; nothing was moved",
                            hint="the target was not sent; do not try to get around the limit",
                            detail={"violations": [v.as_dict() for v in bad]})
    command_id = ctx.command_id
    robot.begin_motion(command_id)
    rng = np.random.default_rng(cal.world(node).get("seed", 20261120))

    def body(handle: ActionHandle) -> dict[str, Any]:
        samples: list[dict[str, Any]] = []
        try:
            for i, T in enumerate(poses):
                execute_path(robot, site.chain, site.profile, site.scene, T, command_id, tick=ctx.tick,
                             cancel=lambda: handle.cancel_requested)
                tip = robot.state().measured_cp[:3, 3]
                samples.append({"pose": i + 1, "touched_m": [float(x) for x in tip],
                                "seen_m": [float(x) for x in cal.camera_sees(node, tip, rng)]})
                handle.feedback(fraction=round((i + 1) / (len(poses) + 1), 3), sampled=i + 1)
            execute_path(robot, site.chain, site.profile, site.scene, site.poses["start"].T, command_id, tick=ctx.tick,
                         cancel=lambda: handle.cancel_requested)
        except MotionStopped as exc:
            if exc.kind == "canceled":
                return {"_canceled": True, "samples": len(samples)}
            raise ActionAbort(exc.kind, str(exc), {"violations": [v.as_dict() for v in exc.violations], "samples": len(samples)})
        except BackendError as exc:
            raise ActionAbort(exc.kind, str(exc), {"samples": len(samples)})
        sid = cal.store_samples(ctx.db, site.name, samples, ctx.principal.user_id)
        return {"sweep_id": sid, "samples": len(samples), "returned_to": "start",
                "camera": "DECLARED: what the camera sees is computed, the twin has no camera"}

    def on_end(handle: ActionHandle) -> dict[str, Any]:
        return state_report(robot.end_motion(command_id), site.scene)

    return start_action("sweep_calibration_poses", body, on_end=on_end)


def fit_calibration(a: NoArgs, ctx: ToolContext) -> dict[str, Any]:
    site = ctx.site.name
    sweep = cal.load_samples(ctx.db, site)
    if sweep is None:
        raise ToolError("failed", "no calibration samples have been collected at this site",
                        hint="run the runbook collect_calibration_samples first")
    old = load_calibration_record(ctx.db, site)
    try:
        res = cal.fit(sweep["samples"], np.array(old["T_est"], dtype=float) if old else None)
    except ValueError as exc:
        raise ToolError("failed", str(exc))
    draft = cal.store_draft(ctx.db, site, res, sweep["id"], ctx.principal.user_id)
    h = res["held_out_residual"]
    out = {"draft_id": draft["id"], "status": "draft", "sweep_id": sweep["id"], "samples": res["samples"],
           "used_for_the_fit": res["used_for_the_fit"], "held_out": res["held_out"],
           "held_out_rigid_mean_mm": h["rigid"]["mean_mm"], "held_out_rigid_max_mm": h["rigid"]["max_mm"],
           "held_out_with_affine_mean_mm": h["rigid_plus_affine"]["mean_mm"], "acceptance_mm": cal.ACCEPT_MM,
           "within_acceptance": h["rigid"]["mean_mm"] <= cal.ACCEPT_MM,
           "note": "a draft is not in use; it becomes active only by a change the safety lead approves"}
    if "record_in_use" in h:
        out["record_in_use_on_the_same_points_mean_mm"] = h["record_in_use"]["mean_mm"]
    return out


def compare_calibrations(a: NoArgs, ctx: ToolContext) -> dict[str, Any]:
    site = ctx.site.name
    old, draft = load_calibration_record(ctx.db, site), cal.load_draft(ctx.db, site)
    if old is None or draft is None:
        raise ToolError("failed", "there are not two calibration records to compare at this site",
                        hint="one is the record in use, the other the latest draft of the calibration procedure")
    d = cal.difference(np.array(old["T_est"], dtype=float), np.array(draft["T_est"], dtype=float), draft["seen_m"])
    h = draft["held_out_residual"]
    status = proposals.calibration(ctx.db, site)
    return {"record_in_use": {"calibration_id": old["id"], "valid": bool(status["valid"]), "validity_note": status["why"],
                              "mean_mm_on_the_new_held_out_points": (h.get("record_in_use") or {}).get("mean_mm")},
            "draft": {"calibration_id": draft["id"], "status": draft["status"], "mean_mm_on_the_new_held_out_points": h["rigid"]["mean_mm"],
                      "with_affine_mean_mm": h["rigid_plus_affine"]["mean_mm"]},
            "difference": {"shift_in_the_workspace_mm": d["shift_in_the_workspace_mm"], "rotation_deg": d["rotation_deg"]},
            "held_out_points": len(draft["touched_m"])}


# ------------------------------------------------------------------ proposals

class ProposeCalibrationArgs(Args):
    reason: str = Field(min_length=10, max_length=400)
    ticket_id: int | None = Field(default=None, description="The ticket this belongs to, if any.")


def propose_calibration(a: ProposeCalibrationArgs, ctx: ToolContext) -> dict[str, Any]:
    try:
        return changes.propose_calibration(_node(ctx), ctx.principal, a.reason, a.ticket_id)
    except proposals.NotPermitted as exc:
        raise ToolError("unauthorized" if exc.kind in ("role", "site") else "failed", str(exc), hint="do not retry")


class ProposePatchArgs(ConfigPatch):
    ticket_id: int | None = Field(default=None, description="The ticket this belongs to, if any.")


def propose_config_patch(a: ProposePatchArgs, ctx: ToolContext) -> dict[str, Any]:
    patch = ConfigPatch.model_validate(a.model_dump(exclude={"ticket_id"}))
    try:
        return changes.propose_patch(_node(ctx), ctx.principal, patch, a.ticket_id)
    except proposals.NotPermitted as exc:
        raise ToolError("unauthorized" if exc.kind in ("role", "site") else "failed", str(exc), hint="do not retry")


class ChangesArgs(Args):
    status: Literal["proposed", "applied", "rejected", "all"] = "proposed"


def list_changes(a: ChangesArgs, ctx: ToolContext) -> dict[str, Any]:
    rows = changes.queue(ctx.db, ctx.site.name if ctx.site is not None else ctx.principal.site, None if a.status == "all" else a.status)
    return {"count": len(rows), "changes": [{k: r[k] for k in ("proposal_id", "target", "kind", "status", "proposer", "reason",
                                                               "limit_effect")} for r in rows[:15]]}


class ChangeArgs(Args):
    proposal_id: int = Field(ge=1)


def read_change(a: ChangeArgs, ctx: ToolContext) -> dict[str, Any]:
    try:
        v = changes.view(ctx.db, a.proposal_id)
    except KeyError:
        raise ToolError("invalid_arguments", f"no proposal {a.proposal_id}", hint="call list_changes for the ids")
    if v["site"] != ctx.principal.site and "*" not in ctx.principal.scopes:
        raise ToolError("unauthorized", "that proposal belongs to another site", hint="do not retry this tool")
    eff = v["effect_on_the_twin"]
    v["effect_on_the_twin"] = {k: eff[k] for k in ("poses_that_move", "largest_move_mm", "largest_turn_deg", "note",
                                                   "newly_rejected_by_the_safety_filter", "newly_accepted_by_the_safety_filter",
                                                   "held_out_residual", "difference_to_the_record_in_use") if k in eff}
    return v


# ------------------------------------------------------------------ tickets and events

class NoteArgs(Args):
    ticket_id: int = Field(ge=1)
    note: str = Field(min_length=5, max_length=500)


def add_ticket_note(a: NoteArgs, ctx: ToolContext) -> dict[str, Any]:
    t = tickets.get(ctx.db, a.ticket_id)
    if t is None or not tickets.may_read(ctx.principal, t):
        raise ToolError("unauthorized", f"ticket {a.ticket_id} is not a ticket of your site", hint="do not retry this tool")
    rid = tickets.attach(ctx.db, a.ticket_id, "note", {"note": a.note}, ctx.principal.user_id)
    return {"record": f"ticket:{a.ticket_id}/note/{rid}"}


class EventArgs(Args):
    kind: Literal["ecm_moved", "setup_joint_moved", "instrument_changed"] = Field(
        description="What happened at the rig. Each of these ends the validity of the camera-to-robot calibration.")
    detail: str = Field(default="", max_length=200)


def record_site_event(a: EventArgs, ctx: ToolContext) -> dict[str, Any]:
    out = site_memory.record_event(ctx.db, ctx.site.name, a.kind, ctx.principal.user_id, a.detail)
    node = ctx.extras.get("node")
    ended = changes.expire_approvals(ctx.db, ctx.site.name, f"the event {a.kind}") if node is not None else 0
    return {"event": a.kind, "facts_expired": len(out["facts_expired"]), "approvals_ended": ended,
            "calibration_valid": bool(proposals.calibration(ctx.db, ctx.site.name)["valid"]),
            "note": "the calibration has to be redone by the service engineer before a trial can start"}


def register_service_work(reg: ToolRegistry) -> ToolRegistry:
    reg.register(Tool("read_config", "Read this site's configuration: the list of files, the keys of one file, or the "
                      "value at one path. Read-only.", ReadConfigArgs, read_config, scopes=CONFIG_READ, timeout_s=10))
    reg.register(Tool("describe_config_paths", "The paths of a configuration file that a change may touch, with unit, "
                      "current value and, for safety limits, which direction is tighter.", TargetArgs, describe_config_paths,
                      scopes=CONFIG_READ, timeout_s=10))
    reg.register(Tool("sweep_calibration_poses", "Move the arm through the site's declared calibration poses, take a "
                      "sample at each, and return to start. Needs the operator's approval. Needs based_on_seq from "
                      "your latest get_robot_state.", SweepArgs, sweep_calibration_poses, tier="actuate", kind="action",
                      scopes=CALIBRATE, wait_s=60))
    reg.register(Tool("fit_calibration", "Fit a camera-to-robot transform to the latest calibration samples and judge "
                      "it on held-out samples. The result is a draft, not the calibration in use.", NoArgs, fit_calibration,
                      scopes=CALIBRATE, side_effect=True, timeout_s=20))
    reg.register(Tool("compare_calibrations", "Compare the calibration record in use with the latest draft: residuals "
                      "on the same held-out points and the difference between the two transforms.", NoArgs,
                      compare_calibrations, scopes=CONFIG_READ, timeout_s=10))
    reg.register(Tool("propose_calibration", "File the activation of the calibration draft as a change for the safety "
                      "lead. It does not activate anything.", ProposeCalibrationArgs, propose_calibration,
                      scopes=CONFIG_PROPOSE, side_effect=True, timeout_s=10))
    reg.register(Tool("propose_config_patch", "File a small change to one configuration file as a proposal for the "
                      "safety lead. The patch is validated and its effect computed on the twin. It changes nothing.",
                      ProposePatchArgs, propose_config_patch, scopes=CONFIG_PROPOSE, side_effect=True, timeout_s=30))
    reg.register(Tool("list_changes", "Proposed changes of this site: target, proposer, reason, whether a limit is "
                      "loosened.", ChangesArgs, list_changes, scopes=CONFIG_READ, timeout_s=10))
    reg.register(Tool("read_change", "One proposed change in full: operations, validation, effect on the twin, "
                      "proposer and evidence.", ChangeArgs, read_change, scopes=CONFIG_READ, timeout_s=10))
    reg.register(Tool("add_ticket_note", "Attach a note to a ticket of this site.", NoteArgs, add_ticket_note,
                      scopes=TICKET_WRITE, side_effect=True, timeout_s=5))
    reg.register(Tool("record_site_event", "Record that something on the rig was moved or exchanged. The calibration "
                      "stops being valid and approvals to move end.", EventArgs, record_site_event, scopes=SITE_EVENT,
                      side_effect=True, timeout_s=5))
    return reg


__all__ = ["register_service_work", "math"]
