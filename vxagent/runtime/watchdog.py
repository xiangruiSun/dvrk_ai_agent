"""Motions as supervised actions.

While a motion runs, the runtime expects feedback at the control rate. The motion itself runs
on its own thread; this thread only watches. It cancels the action and says why when

    the operator presses stop                      -> operator_stop
    the robot reports a fault                      -> robot_fault
    feedback has been missing for too long         -> watchdog
    the action has used its budget of steps        -> budget

The first three end the node from the outside (graph.Abort). Running out of budget is a result
the node decides about.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from ..robot.backend import BackendError
from ..robot.motion import MotionStopped, execute_path
from ..store import dumps
from ..tools.actions import ActionAbort, ActionHandle, start_action
from .graph import Abort, Runner


class Watchdog:
    def __init__(self, limit_s: float, clock: Callable[[], float] = time.monotonic):
        self.limit_s, self._clock = limit_s, clock
        self._last = clock()

    def feed(self) -> None:
        self._last = self._clock()

    def silent_for(self) -> float:
        return self._clock() - self._last

    def expired(self) -> bool:
        return self.silent_for() > self.limit_s


Work = Callable[[str, Callable[[], bool], Callable[..., None]], None]


def run_action(rt: Runner, label: str, work: Work, budget_steps: int) -> dict[str, Any]:
    """Run work(command_id, cancel_requested, feedback) as one motion with a one-use command id."""
    env, robot, db = rt.env, rt.env.robot, rt.env.db
    node = rt.state["task"]["stage"]
    command_id = f"{rt.run_id}:{len(rt.state['actions']) + 1}:{label}"
    seq0 = robot.state().seq
    db.execute("INSERT INTO commands(command_id,run_id,tool,arguments,user_id,seq_before,status,created) "
               "VALUES (?,?,?,?,?,?,?,?)", (command_id, rt.run_id, label, dumps({"node": node}), "trial runtime",
                                           seq0, "started", time.time()))
    try:
        robot.begin_motion(command_id)
    except BackendError as exc:
        db.execute("UPDATE commands SET status=?, ended=? WHERE command_id=?", (exc.kind, time.time(), command_id))
        if exc.kind == "robot_fault":
            raise Abort("robot_fault", {"fault": str(exc)})
        return {"command_id": command_id, "label": label, "status": "aborted", "reason": exc.kind, "steps": 0,
                "path_mm": 0.0}
    rt.emit("action_started", {"command_id": command_id, "node": node, "label": label})
    dog = Watchdog(env.watchdog_s)
    seen = {"steps": 0, "path": 0.0, "last": robot.state().measured_cp[:3, 3] * 1000.0}

    def feedback(**fields: Any) -> None:
        seen["steps"] += 1
        if "position_mm" in fields:
            p = np.array(fields["position_mm"], dtype=float)
            seen["path"] += float(np.linalg.norm(p - seen["last"]))
            seen["last"] = p
        if env.feedback_gate is None or env.feedback_gate(seen["steps"]):
            dog.feed()

    def body(handle: ActionHandle) -> dict[str, Any]:
        try:
            work(command_id, lambda: handle.cancel_requested, feedback)
        except MotionStopped as exc:
            if exc.kind == "canceled":
                return {"_canceled": True}
            raise ActionAbort(exc.kind, str(exc), {"violations": [v.as_dict() for v in exc.violations]})
        except BackendError as exc:
            raise ActionAbort(exc.kind, str(exc))
        return {}

    with env.tracer.span(rt.run_id, "action", label, input={"command_id": command_id, "node": node}) as sp:
        sp.seq_start = seq0
        handle = start_action(label, body, on_end=lambda h: (robot.end_motion(command_id), {})[1])
        why = None
        while not handle.wait(0.005):
            if env.stop_event.is_set():
                why = "operator_stop"
            elif robot.state().fault:
                why = "robot_fault"
            elif dog.expired():
                why = "watchdog"
            elif seen["steps"] > budget_steps:
                why = "budget"
            if why:
                silent = round(dog.silent_for(), 3)
                handle.cancel()
                handle.wait(5.0)
                break
        status = handle.status
        reason = why or handle.reason_kind
        result: dict[str, Any] = {"command_id": command_id, "label": label, "node": node, "status": status,
                                  "reason": reason, "steps": seen["steps"], "path_mm": round(seen["path"], 2)}
        if why == "watchdog":
            result["feedback_silent_s"], result["watchdog_limit_s"] = silent, env.watchdog_s
        if handle.result.get("violations"):
            result["violations"] = handle.result["violations"]
        end = robot.state()
        sp.seq_end, sp.output = end.seq, result
        if status != "succeeded":
            sp.fail(reason or "aborted", handle.reason or reason or "")
    db.execute("UPDATE commands SET status=?, seq_after=?, ended=? WHERE command_id=?",
               ("done" if status == "succeeded" else (reason or status), end.seq, time.time(), command_id))
    rt.emit("action_result", result)
    if reason in ("operator_stop", "robot_fault", "watchdog") or end.fault:
        raise Abort("robot_fault" if end.fault and reason not in ("operator_stop", "watchdog") else reason, result)
    return result


def move(rt: Runner, label: str, targets: list[np.ndarray], budget_steps: int) -> dict[str, Any]:
    """Follow Cartesian targets in order. Every target goes through the safety filter."""
    env = rt.env

    def work(command_id: str, cancel: Callable[[], bool], feedback: Callable[..., None]) -> None:
        for T in targets:
            execute_path(env.robot, env.site.chain, env.site.profile, env.site.scene, T, command_id,
                         tick=env.tick, cancel=cancel, feedback=feedback)
    return run_action(rt, label, work, budget_steps)


def jaw(rt: Runner, label: str, angle: float, steps: int = 5) -> dict[str, Any]:
    env, robot = rt.env, rt.env.robot

    def work(command_id: str, cancel: Callable[[], bool], feedback: Callable[..., None]) -> None:
        start = robot.state().jaw
        for i in range(1, steps + 1):
            if cancel():
                raise MotionStopped("canceled", "the jaw motion was canceled", [], i - 1)
            robot.servo_jaw(start + (angle - start) * i / steps, command_id)
            feedback(fraction=i / steps)
            env.tick(1.0 / robot.rate_hz)
    return run_action(rt, label, work, steps + 2)
