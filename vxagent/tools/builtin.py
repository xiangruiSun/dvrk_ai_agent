"""The tool catalogue.

No tool takes a coordinate, an angle, a gain or a limit from the model. A motion is requested by
the name of a pose the site has declared; the target is computed from the site's files, checked
by the safety filter, and rejected, never clipped, if it fails.
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..data import episodes as ep_store
from ..robot.backend import BackendError
from ..robot.kinematics import pose_distance
from ..robot.motion import MotionStopped, check_goal, execute_path, pose_report, state_report
from ..robot.safety import plan_steps
from .actions import ActionAbort, ActionHandle, start_action
from .executor import wait_for_action
from .registry import Tool, ToolContext, ToolError, ToolRegistry

ROBOT_READ = frozenset({"robot:read"})
ROBOT_MOVE = frozenset({"robot:move"})
EPISODE_READ = frozenset({"episodes:read"})


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------------------ observe: robot

class NoArgs(Args):
    pass


def _robot(ctx: ToolContext):
    if ctx.robot is None or ctx.site is None:
        raise ToolError("failed", "no robot is connected to this session")
    return ctx.robot, ctx.site


def get_robot_state(a: NoArgs, ctx: ToolContext) -> dict[str, Any]:
    robot, site = _robot(ctx)
    out = state_report(robot.state(), site.scene)
    out.update({"site": site.name, "arm": site.arm, "backend": robot.name, "safety_profile": site.profile.name})
    return out


def list_named_poses(a: NoArgs, ctx: ToolContext) -> dict[str, Any]:
    _, site = _robot(ctx)
    start = site.poses["start"].T[:3, 3]
    return {"site": site.name, "poses": [
        {"name": p.name, "description": p.description,
         "offset_from_start_mm": [round(float(x) * 1000, 1) for x in (p.T[:3, 3] - start)],
         "height_over_pad_mm": round(site.scene.height(p.T[:3, 3]) * 1000, 1)}
        for p in site.poses.values()]}


class PoseArgs(Args):
    named_pose: str = Field(description="Name of a pose declared for this site, exactly as list_named_poses returns it.")


def _pose(site: Any, name: str):
    if name not in site.poses:
        raise ToolError("invalid_arguments", f"no pose named {name!r} at this site",
                        hint=f"named poses: {', '.join(site.poses)}")
    return site.poses[name]


def check_named_target(a: PoseArgs, ctx: ToolContext) -> dict[str, Any]:
    """Would the safety filter accept this destination, and how far is it?"""
    robot, site = _robot(ctx)
    pose, s = _pose(site, a.named_pose), robot.state()
    bad = check_goal(site.profile, site.scene, site.chain, pose.T, s.q)
    d_m, d_rad = pose_distance(s.measured_cp, pose.T)
    return {"state_seq": s.seq, "named_pose": pose.name, "accepted": not bad,
            "violations": [v.as_dict() for v in bad],
            "distance_mm": round(d_m * 1000, 2), "rotation_deg": round(math.degrees(d_rad), 2),
            "control_steps": len(plan_steps(site.profile, site.scene, s.setpoint_cp, pose.T)) if not bad else 0}


# ------------------------------------------------------------------------------ simulate

def rehearse_move(a: PoseArgs, ctx: ToolContext) -> dict[str, Any]:
    """Run the motion on a copy of the twin. Nothing real moves."""
    robot, site = _robot(ctx)
    pose = _pose(site, a.named_pose)
    if not hasattr(robot, "clone"):
        raise ToolError("failed", "this backend cannot be rehearsed")
    copy = robot.clone()
    before = copy.state()
    if not before.ready:
        return {"state_seq": robot.state().seq, "named_pose": pose.name, "would_complete": False,
                "reason_kind": "robot_not_ready", "reason": "the arm is not enabled and homed"}
    bad = check_goal(site.profile, site.scene, site.chain, pose.T, before.q)
    if bad:
        return {"state_seq": robot.state().seq, "named_pose": pose.name, "would_complete": False,
                "reason_kind": "safety_rejected", "violations": [v.as_dict() for v in bad]}
    copy.begin_motion("rehearsal")
    try:
        info = execute_path(copy, site.chain, site.profile, site.scene, pose.T, "rehearsal", tick=lambda dt: None)
        ok, why, viol = True, None, []
    except MotionStopped as exc:
        info, ok, why, viol = {"steps": exc.steps_done}, False, exc.kind, [v.as_dict() for v in exc.violations]
    after = copy.end_motion("rehearsal")
    d_m, _ = pose_distance(after.measured_cp, pose.T)
    return {"state_seq": robot.state().seq, "named_pose": pose.name, "would_complete": ok, "reason_kind": why,
            "violations": viol, "control_steps": info["steps"],
            "expected_duration_s": round(info["steps"] / robot.rate_hz, 1),
            "final": pose_report(after.measured_cp, site.scene), "final_error_mm": round(d_m * 1000, 3)}


# ------------------------------------------------------------------------------ actuate

class EnableArgs(Args):
    based_on_seq: int = Field(description="state_seq from your most recent get_robot_state.")


def _end_report(ctx: ToolContext, command_id: str):
    def on_end(handle: ActionHandle) -> dict[str, Any]:
        robot, site = ctx.robot, ctx.site
        s = robot.end_motion(command_id) if command_id else robot.state()
        return state_report(s, site.scene)
    return on_end


def enable_and_home(a: EnableArgs, ctx: ToolContext) -> ActionHandle:
    robot, site = _robot(ctx)

    def body(handle: ActionHandle) -> dict[str, Any]:
        try:
            robot.enable()
            handle.feedback(fraction=0.5, stage="enabled")
            ctx.tick(0.5)
            if handle.cancel_requested:
                return {"_canceled": True}
            robot.home()
            handle.feedback(fraction=1.0, stage="homed")
        except BackendError as exc:
            raise ActionAbort(exc.kind, str(exc))
        return {}

    return start_action("enable_and_home", body, on_end=lambda h: state_report(robot.state(), site.scene))


class MoveArgs(Args):
    named_pose: str = Field(description="Name of a pose declared for this site, exactly as list_named_poses returns it.")
    based_on_seq: int = Field(description="state_seq from your most recent get_robot_state.")


def move_to_named_pose(a: MoveArgs, ctx: ToolContext) -> ActionHandle:
    robot, site = _robot(ctx)
    pose = _pose(site, a.named_pose)
    s = robot.state()
    if not s.ready:
        raise ToolError("robot_fault" if s.fault else "robot_not_ready",
                        f"the arm reports a fault: {s.fault}" if s.fault else "the arm is not enabled and homed",
                        hint="do not command the arm; report the fault" if s.fault else "run enable_and_home first",
                        detail=state_report(s, site.scene))
    bad = check_goal(site.profile, site.scene, site.chain, pose.T, s.q)
    if bad:                                            # rejected before anything moves, never clipped
        raise ToolError("safety_rejected", f"{pose.name} was rejected by rule {bad[0].rule}; nothing was moved",
                        hint="the target was not sent; do not try to get around the limit",
                        detail={"violations": [v.as_dict() for v in bad], **state_report(s, site.scene)})
    command_id = ctx.command_id
    robot.begin_motion(command_id)

    def body(handle: ActionHandle) -> dict[str, Any]:
        try:
            info = execute_path(robot, site.chain, site.profile, site.scene, pose.T, command_id, tick=ctx.tick,
                                cancel=lambda: handle.cancel_requested, feedback=handle.feedback)
        except MotionStopped as exc:
            if exc.kind == "canceled":
                return {"_canceled": True, "steps": exc.steps_done, "named_pose": pose.name}
            raise ActionAbort(exc.kind, str(exc), {"violations": [v.as_dict() for v in exc.violations],
                                                   "steps": exc.steps_done, "named_pose": pose.name})
        except BackendError as exc:
            raise ActionAbort(exc.kind, str(exc), {"named_pose": pose.name})
        d_m, d_rad = pose_distance(robot.state().measured_cp, pose.T)
        return {"named_pose": pose.name, "final_error_mm": round(d_m * 1000, 3),
                "final_error_deg": round(math.degrees(d_rad), 3), **info}

    return start_action("move_to_named_pose", body, on_end=_end_report(ctx, command_id))


class JawArgs(Args):
    jaw: Literal["open", "closed"] = Field(description="The jaw state to command.")
    based_on_seq: int = Field(description="state_seq from your most recent get_robot_state.")


def set_jaw(a: JawArgs, ctx: ToolContext) -> ActionHandle:
    robot, site = _robot(ctx)
    s = robot.state()
    if not s.ready:
        raise ToolError("robot_fault" if s.fault else "robot_not_ready", "the arm is not ready",
                        hint="run enable_and_home first", detail=state_report(s, site.scene))
    target = site.jaw_angles[a.jaw]
    command_id = ctx.command_id
    robot.begin_motion(command_id)

    def body(handle: ActionHandle) -> dict[str, Any]:
        start = robot.state().jaw
        try:
            for i in range(1, 6):
                if handle.cancel_requested:
                    return {"_canceled": True}
                robot.servo_jaw(start + (target - start) * i / 5, command_id)
                handle.feedback(fraction=i / 5)
                ctx.tick(1.0 / robot.rate_hz)
        except BackendError as exc:
            raise ActionAbort(exc.kind, str(exc))
        return {"jaw": a.jaw}

    return start_action("set_jaw", body, on_end=_end_report(ctx, command_id))


# ------------------------------------------------------------------------------ actions: wait, cancel

class ActionArgs(Args):
    action_id: str = Field(description="action_id returned by the tool that started the action.")


def _handle(a: ActionArgs, ctx: ToolContext) -> ActionHandle:
    h = ctx.actions.get(a.action_id)
    if h is None:
        raise ToolError("invalid_arguments", f"no action {a.action_id!r} in this run",
                        hint=f"actions in this run: {', '.join(ctx.actions) or 'none'}")
    return h


def wait_action(a: ActionArgs, ctx: ToolContext) -> dict[str, Any]:
    content, ok, err = wait_for_action(_handle(a, ctx), ctx, 20.0)
    if not ok and err is not None:
        raise ToolError(err.kind, str(err), hint=err.hint, detail=content)
    return content


def cancel_action(a: ActionArgs, ctx: ToolContext) -> dict[str, Any]:
    h = _handle(a, ctx)
    h.cancel()
    h.wait(5.0)
    return h.digest()


# ------------------------------------------------------------------------------ observe: recordings

def list_episodes(a: NoArgs, ctx: ToolContext) -> dict[str, Any]:
    eps = ep_store.list_episodes(ctx.extras.get("rosma_dir"))
    return {"source": "ROSMA (recorded dVRK kinematics, CC BY 4.0)", "count": len(eps),
            "episodes": [{"episode_id": e.episode_id, "operator": e.operator, "task": e.task, "trial": e.trial}
                         for e in eps]}


class EpisodeArgs(Args):
    episode_id: str = Field(description="Episode id exactly as list_episodes returns it, e.g. X01_Pea_on_a_Peg_01.")


def _episode(episode_id: str, ctx: ToolContext):
    hook = ctx.extras.get("archive_fault")
    if hook is not None:
        hook()                                         # verification injects archive failures here
    e = ep_store.get_episode(episode_id, ctx.extras.get("rosma_dir"))
    if e is None:
        raise ToolError("invalid_arguments", f"no episode {episode_id!r}", hint="call list_episodes for the ids")
    return e


def episode_summary(a: EpisodeArgs, ctx: ToolContext) -> dict[str, Any]:
    """Statistics of one recording, computed by code. The recording itself is never returned."""
    e = _episode(a.episode_id, ctx)
    out = dict(ep_store.stats(e))
    out.update(ep_store.scores(ctx.extras.get("rosma_dir")).get(e.episode_id, {}))
    return out


class WindowArgs(Args):
    episode_id: str = Field(description="Episode id exactly as list_episodes returns it.")
    arm: Literal["PSM1", "PSM2", "MTML", "MTMR"]
    signal: Literal["position", "velocity_linear", "joint_position"]
    start_s: float = Field(ge=0, description="Time from the start of the recording, in seconds.")
    count: int = Field(ge=1, le=40, description="Number of samples to return (at most 40).")


def read_signal_window(a: WindowArgs, ctx: ToolContext) -> dict[str, Any]:
    return ep_store.window(_episode(a.episode_id, ctx), a.arm, a.signal, a.start_s, a.count)


# ------------------------------------------------------------------------------ calculator

class CalcArgs(Args):
    expression: str = Field(max_length=200, description="Arithmetic with numbers, + - * / ** ( ), sqrt, abs, round, min, max.")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(x) for x in node.args])
    raise ValueError("only numbers, + - * / ** % ( ) and sqrt, abs, round, min, max are allowed")


def calculator(a: CalcArgs, ctx: ToolContext) -> dict[str, Any]:
    try:
        value = _eval(ast.parse(a.expression, mode="eval").body)
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError, OverflowError) as exc:
        raise ToolError("invalid_arguments", f"cannot evaluate: {exc}", hint="send plain arithmetic")
    return {"expression": a.expression, "value": round(float(value), 6)}


# ------------------------------------------------------------------------------ registration

def register_builtin(reg: ToolRegistry | None = None) -> ToolRegistry:
    reg = reg or ToolRegistry()
    add = reg.register
    add(Tool("get_robot_state", "Look at the robot now: operating state (powered, enabled, homed, busy, fault), "
             "measured tool-tip pose in mm and degrees, jaw angle, and state_seq. A motion request must quote the "
             "state_seq of your latest look.", NoArgs, get_robot_state, scopes=ROBOT_READ, timeout_s=5))
    add(Tool("list_named_poses", "The poses this site has declared by name, with their offset from the start pose "
             "in mm and their height over the pad. Motions can only go to these.", NoArgs, list_named_poses,
             scopes=ROBOT_READ, cacheable=True, timeout_s=5))
    add(Tool("check_named_target", "Ask the safety filter whether a named pose is an acceptable destination from "
             "where the arm is now, and how far away it is. Does not move anything.", PoseArgs, check_named_target,
             scopes=ROBOT_READ, timeout_s=10))
    add(Tool("rehearse_move", "Rehearse a move to a named pose on a copy of the kinematic twin. Returns whether it "
             "would complete, the number of control steps and the expected duration. Nothing real moves.",
             PoseArgs, rehearse_move, tier="simulate", scopes=ROBOT_READ, timeout_s=20))
    add(Tool("enable_and_home", "Enable the arm and home it. Needs the operator's approval. Quote the state_seq "
             "of your latest get_robot_state.", EnableArgs, enable_and_home, tier="actuate", kind="action",
             scopes=ROBOT_MOVE, wait_s=20))
    add(Tool("move_to_named_pose", "Move the tool tip to a named pose. The path is generated and checked by the "
             "safety filter; a destination outside the limits is rejected and nothing moves. Needs the operator's "
             "approval. Quote the state_seq of your latest get_robot_state. Never send two motions at once.",
             MoveArgs, move_to_named_pose, tier="actuate", kind="action", scopes=ROBOT_MOVE, wait_s=20))
    add(Tool("set_jaw", "Open or close the instrument jaw. Needs the operator's approval. Quote the state_seq of "
             "your latest get_robot_state.", JawArgs, set_jaw, tier="actuate", kind="action", scopes=ROBOT_MOVE,
             wait_s=10))
    add(Tool("wait_action", "Wait up to 20 s more for an action that was returned as still running.",
             ActionArgs, wait_action, scopes=ROBOT_READ, timeout_s=30, max_retries=0))
    add(Tool("cancel_action", "Stop a running action. Returns where the arm stopped.", ActionArgs, cancel_action,
             scopes=ROBOT_READ, timeout_s=10, max_retries=0))
    add(Tool("list_episodes", "List the recorded dVRK episodes available at this site (operator, task, trial).",
             NoArgs, list_episodes, scopes=EPISODE_READ, cacheable=True, timeout_s=10))
    add(Tool("episode_summary", "Statistics of one recorded episode, computed from the recording: samples, "
             "duration, sample rate, and for PSM1 and PSM2 the path length, peak and mean tip speed and workspace "
             "extent, plus the dataset's own completion time and error count.", EpisodeArgs, episode_summary,
             scopes=EPISODE_READ, timeout_s=30))
    add(Tool("read_signal_window", "Read at most 40 consecutive samples of one signal of one arm from a recorded "
             "episode, starting at a given time.", WindowArgs, read_signal_window, scopes=EPISODE_READ, timeout_s=30))
    add(Tool("calculator", "Evaluate arithmetic exactly. Use it for any sum, difference, ratio or unit conversion.",
             CalcArgs, calculator, cacheable=True, timeout_s=5))
    return reg
