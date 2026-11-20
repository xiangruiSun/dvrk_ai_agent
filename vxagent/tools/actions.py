"""Long-running tools.

An action is started, reports progress while it runs, and ends in exactly one of three ways:
succeeded, aborted (with a reason and where the arm stopped) or canceled. This is the shape of a
ROS 2 action: goal, feedback, result, cancel. The agent loop waits for the result for a bounded
time and gives the model one digest of the feedback instead of every event.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from typing import Any, Callable


class ActionAbort(Exception):
    """Raised inside an action body to end it as aborted."""

    def __init__(self, kind: str, message: str, detail: dict[str, Any] | None = None):
        super().__init__(message)
        self.kind, self.detail = kind, detail or {}


class ActionHandle:
    def __init__(self, tool: str):
        self.id = f"act_{uuid.uuid4().hex[:8]}"
        self.tool = tool
        self.status = "running"            # running | succeeded | aborted | canceled
        self.reason_kind: str | None = None
        self.reason: str | None = None
        self.result: dict[str, Any] = {}
        self.started = time.time()
        self.ended: float | None = None
        self.events = 0
        self.last_feedback: dict[str, Any] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=50)
        self._cancel = threading.Event()
        self._done = threading.Event()
        self.thread: threading.Thread | None = None

    # ---- used by the action body
    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def feedback(self, **fields: Any) -> None:
        self.events += 1
        self.last_feedback = {"t_s": round(time.time() - self.started, 3), **fields}
        self._recent.append(self.last_feedback)

    # ---- used by the executor and the loop
    def cancel(self) -> None:
        self._cancel.set()

    def wait(self, timeout: float | None) -> bool:
        return self._done.wait(timeout)

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def digest(self) -> dict[str, Any]:
        """What the model reads about this action: one summary, not the event stream."""
        out: dict[str, Any] = {"action_id": self.id, "tool": self.tool, "status": self.status,
                               "feedback_events": self.events,
                               "elapsed_s": round((self.ended or time.time()) - self.started, 2)}
        if self.status == "running":
            out["progress"] = self.last_feedback
        if self.reason_kind:
            out["reason_kind"], out["reason"] = self.reason_kind, self.reason
        out.update(self.result)
        return out


def start_action(tool: str, body: Callable[[ActionHandle], dict[str, Any]],
                 on_end: Callable[[ActionHandle], dict[str, Any] | None] | None = None) -> ActionHandle:
    """Run body(handle) on its own thread. The body returns the result fields on success, raises
    ActionAbort to abort, and returns early after seeing handle.cancel_requested (the handle is
    then marked canceled). on_end runs in every case and may add fields (for example, where the
    arm is now)."""
    handle = ActionHandle(tool)

    def run() -> None:
        try:
            result = body(handle) or {}
            handle.status = "canceled" if handle.cancel_requested and result.get("_canceled") else "succeeded"
            result.pop("_canceled", None)
            handle.result = result
        except ActionAbort as exc:
            handle.status, handle.reason_kind, handle.reason = "aborted", exc.kind, str(exc)
            handle.result = dict(exc.detail)
        except Exception as exc:                       # a bug in the body must not hang the waiter
            handle.status, handle.reason_kind = "aborted", "failed"
            handle.reason = f"{type(exc).__name__}: {exc}"
        finally:
            if on_end is not None:
                try:
                    handle.result.update(on_end(handle) or {})
                except Exception as exc:
                    handle.result["end_error"] = f"{type(exc).__name__}: {exc}"
            handle.ended = time.time()
            handle._done.set()

    handle.thread = threading.Thread(target=run, name=f"action-{tool}", daemon=True)
    handle.thread.start()
    return handle
