"""The one interface between the agent stack and a robot.

The names follow the CRTK command set the dVRK software exposes: an operating state with
enable / disable / home, measured_cp and setpoint_cp for the Cartesian pose, measured_js for the
joints, servo_cp for a Cartesian set point at the control rate, and a jaw. Nothing above this
interface knows whether the kinematic twin or a ROS 2 node is behind it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class BackendError(Exception):
    """The robot refused a command. kind: robot_not_ready | robot_fault | duplicate_command | bad_command"""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class RobotState:
    seq: int                 # increases by one on every change of state
    stamp: float             # time of that change (time.time())
    powered: bool
    enabled: bool
    homed: bool
    busy: bool               # a motion is being executed
    fault: str | None
    q: tuple[float, ...]     # measured_js, SI
    jaw: float               # measured jaw angle, radians
    measured_cp: np.ndarray  # 4x4, base frame
    setpoint_cp: np.ndarray  # 4x4, base frame

    @property
    def ready(self) -> bool:
        return self.powered and self.enabled and self.homed and self.fault is None


class RobotBackend(Protocol):
    name: str
    rate_hz: float

    def state(self) -> RobotState: ...
    def enable(self) -> RobotState: ...
    def disable(self) -> RobotState: ...
    def home(self) -> RobotState: ...
    def begin_motion(self, command_id: str) -> None: ...
    def end_motion(self, command_id: str) -> RobotState: ...
    def servo_cp(self, target: np.ndarray, command_id: str) -> RobotState: ...
    def servo_jaw(self, angle: float, command_id: str) -> RobotState: ...
