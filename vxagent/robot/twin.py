"""Kinematic twin of a dVRK arm.

It does what the dVRK software's own kinematic simulation mode does: a commanded Cartesian set
point is turned into joint values by inverse kinematics, the joints are held inside their limits,
and the measured pose is the forward kinematics of those joints. There is no contact, no
dynamics, no needle and no camera. Faults are injected explicitly and are labelled as injected
wherever they are used.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .backend import BackendError, RobotState
from .kinematics import Chain


class KinematicTwin:
    name = "twin"

    def __init__(self, chain: Chain, home_q: Any, rate_hz: float = 10.0, jaw: float = 0.0,
                 persist: Path | str | None = None):
        self.chain = chain
        self.rate_hz = rate_hz
        # Where the arm is must outlive the process that commands it, as it does for a real arm.
        # With `persist` every change of state is written to that file (see restore()).
        self._persist = Path(persist) if persist else None
        self._lock = threading.RLock()
        self._home_q = chain.clamp(home_q)
        self._q = self._home_q.copy()
        self._jaw = float(jaw)
        self._setpoint = chain.fk(self._q)
        self._powered, self._enabled, self._homed = True, False, False
        self._fault: str | None = None
        self._seq, self._stamp = 1, time.time()
        self._active: str | None = None
        self._used: set[str] = set()
        # Injected faults (all off by default).
        self._tracking_offset = np.zeros(3)
        self._drop_commands = False

    # ------------------------------------------------------------------ state
    def _bump(self) -> None:
        self._seq += 1
        self._stamp = time.time()
        if self._persist is not None:
            body = json.dumps({"q": [float(x) for x in self._q], "jaw": self._jaw, "powered": self._powered,
                               "enabled": self._enabled, "homed": self._homed, "fault": self._fault,
                               "seq": self._seq, "used": sorted(self._used)})
            tmp = self._persist.with_suffix(".tmp")
            tmp.write_text(body, encoding="utf-8")
            os.replace(tmp, self._persist)

    @classmethod
    def restore(cls, chain: Chain, home_q: Any, persist: Path | str, rate_hz: float = 10.0, jaw: float = 0.0,
                controller_restarted: bool = False) -> "KinematicTwin":
        """The arm as the last process left it. A motion that was running is over: the
        controller stops when nothing commands it. If the controller itself was restarted
        (INJECTED when used), the arm comes back powered but neither enabled nor homed."""
        twin = cls(chain, home_q, rate_hz, jaw)
        path = Path(persist)
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            twin._q = chain.clamp(saved["q"])
            twin._jaw = float(saved["jaw"])
            twin._setpoint = chain.fk(twin._q)
            twin._powered, twin._fault = bool(saved["powered"]), saved["fault"]
            twin._enabled, twin._homed = bool(saved["enabled"]), bool(saved["homed"])
            twin._seq, twin._used = int(saved["seq"]), set(saved["used"])
        twin._persist = path
        if controller_restarted:
            twin._enabled, twin._homed = False, False
            twin._bump()
        # Otherwise the sequence number is the one the last process saw: a new process looking
        # at the arm is not a change of the arm.
        return twin

    def state(self) -> RobotState:
        with self._lock:
            measured = self.chain.fk(self._q)
            measured[:3, 3] += self._tracking_offset
            return RobotState(seq=self._seq, stamp=self._stamp, powered=self._powered, enabled=self._enabled,
                              homed=self._homed, busy=self._active is not None, fault=self._fault,
                              q=tuple(float(x) for x in self._q), jaw=self._jaw,
                              measured_cp=measured, setpoint_cp=self._setpoint.copy())

    def _require_ready(self) -> None:
        if self._fault:
            raise BackendError("robot_fault", f"the arm reports a fault: {self._fault}")
        if not self._powered:
            raise BackendError("robot_not_ready", "the arm is not powered")
        if not self._enabled:
            raise BackendError("robot_not_ready", "the arm is not enabled")
        if not self._homed:
            raise BackendError("robot_not_ready", "the arm is not homed")

    # ------------------------------------------------------------------ operating state
    def enable(self) -> RobotState:
        with self._lock:
            if self._fault:
                raise BackendError("robot_fault", f"the arm reports a fault: {self._fault}")
            if not self._powered:
                raise BackendError("robot_not_ready", "the arm is not powered")
            if not self._enabled:
                self._enabled = True
                self._bump()
            return self.state()

    def disable(self) -> RobotState:
        with self._lock:
            if self._enabled:
                self._enabled = False
                self._bump()
            return self.state()

    def home(self) -> RobotState:
        with self._lock:
            if self._fault:
                raise BackendError("robot_fault", f"the arm reports a fault: {self._fault}")
            if not self._enabled:
                raise BackendError("robot_not_ready", "the arm must be enabled before homing")
            # Homing does not move this arm. The dVRK software only drives a PSM to its zero
            # position during homing when no instrument is present, and this twin always has one.
            self._setpoint = self.chain.fk(self._q)
            self._homed = True
            self._bump()
            return self.state()

    # ------------------------------------------------------------------ motion
    def begin_motion(self, command_id: str) -> None:
        """A motion is identified by a one-use command id. The same id is never accepted twice,
        so a duplicated request cannot move the arm a second time."""
        with self._lock:
            self._require_ready()
            if command_id in self._used:
                raise BackendError("duplicate_command", f"command {command_id!r} was already executed")
            if self._active is not None:
                raise BackendError("bad_command", "another motion is running")
            self._used.add(command_id)
            self._active = command_id
            self._bump()

    def end_motion(self, command_id: str) -> RobotState:
        with self._lock:
            if self._active == command_id:
                self._active = None
                self._bump()
            return self.state()

    def _check_active(self, command_id: str) -> None:
        self._require_ready()
        if self._active != command_id:
            raise BackendError("bad_command", "no motion with this command id is running")

    def servo_cp(self, target: np.ndarray, command_id: str) -> RobotState:
        with self._lock:
            self._check_active(command_id)
            if not self._drop_commands:
                q, _ = self.chain.ik(np.asarray(target, dtype=float), self._q)
                self._q = q                              # limits were applied inside ik
                self._setpoint = self.chain.fk(q)        # differs from target when ik was clipped
            self._bump()
            return self.state()

    def servo_jaw(self, angle: float, command_id: str) -> RobotState:
        with self._lock:
            self._check_active(command_id)
            lo, hi = self.chain.jaw_limits
            self._jaw = float(min(max(angle, lo), hi))
            self._bump()
            return self.state()

    # ------------------------------------------------------------------ rehearsal
    def clone(self) -> "KinematicTwin":
        """An independent copy in the same state, without injected faults. Used to rehearse a
        motion before the real arm (or this twin) is asked to make it."""
        with self._lock:
            other = KinematicTwin(self.chain, self._home_q, self.rate_hz, self._jaw)
            other._q = self._q.copy()
            other._setpoint = self._setpoint.copy()
            other._powered, other._enabled, other._homed = self._powered, self._enabled, self._homed
            other._fault = self._fault
            return other

    # ------------------------------------------------------------------ injected faults
    def inject_tracking_offset(self, offset_m: Any) -> None:
        """INJECTED: the measured pose is displaced from the set point by this vector."""
        with self._lock:
            self._tracking_offset = np.asarray(offset_m, dtype=float)
            self._bump()

    def inject_dropped_commands(self, on: bool = True) -> None:
        """INJECTED: set points are accepted but the arm does not follow them."""
        with self._lock:
            self._drop_commands = on

    def inject_fault(self, message: str | None) -> None:
        """INJECTED: the arm reports a fault (None clears it)."""
        with self._lock:
            self._fault = message
            if message:
                self._enabled = False
                self._active = None
            self._bump()

    def perturb(self, dq: Any) -> None:
        """INJECTED: something other than this agent moved the arm (for example, a person
        using the teleoperation console)."""
        with self._lock:
            self._q = self.chain.clamp(self._q + np.asarray(dq, dtype=float))
            self._setpoint = self.chain.fk(self._q)
            self._bump()

    def set_joints(self, q: Any) -> None:
        """Test and set-up helper: place the arm at joint values q."""
        with self._lock:
            self._q = self.chain.clamp(q)
            self._setpoint = self.chain.fk(self._q)
            self._bump()
