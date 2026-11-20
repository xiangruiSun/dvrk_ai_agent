"""The same robot interface over the dVRK ROS 2 topics, for a lab machine.

NOT VERIFIED. Nothing run on the development PC exercises this file: there is no ROS 2, no
dVRK software and no arm there. It is written against the interface the dVRK ROS 2 bridge
publishes (the CRTK command set under the arm's namespace), and a run made with it is recorded
with the backend name "ros2", so it can never be mistaken for a run on the twin. Before it is
used, every topic name and message type below must be checked against `ros2 topic list -t` on
the lab machine.

    <arm>/operating_state     crtk_msgs/OperatingState     state, is_homed, is_busy
    <arm>/state_command       crtk_msgs/StringStamped      "enable" | "disable" | "home"
    <arm>/measured_cp         geometry_msgs/PoseStamped
    <arm>/setpoint_cp         geometry_msgs/PoseStamped
    <arm>/measured_js         sensor_msgs/JointState
    <arm>/servo_cp            geometry_msgs/PoseStamped    one set point per control tick
    <arm>/jaw/measured_js     sensor_msgs/JointState
    <arm>/jaw/servo_jp        sensor_msgs/JointState

The one-use command id is kept on this side: the dVRK controller has no such notion.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from .backend import BackendError, RobotState


class Ros2Backend:
    name = "ros2"

    def __init__(self, arm: str = "PSM1", rate_hz: float = 10.0, wait_s: float = 5.0):
        try:
            import rclpy
            from crtk_msgs.msg import OperatingState, StringStamped
            from geometry_msgs.msg import PoseStamped
            from sensor_msgs.msg import JointState
        except ImportError as exc:                       # the development PC ends here
            raise BackendError("robot_fault", "ROS 2 and the dVRK message packages are not installed on this machine") from exc
        self._rclpy, self._PoseStamped, self._JointState, self._StringStamped = rclpy, PoseStamped, JointState, StringStamped
        self.rate_hz = rate_hz
        self._lock = threading.RLock()
        self._seq, self._stamp = 0, time.time()
        self._op: Any = None
        self._measured = self._setpoint = None
        self._q: tuple[float, ...] = ()
        self._jaw = 0.0
        self._active: str | None = None
        self._used: set[str] = set()
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("vxagent_gateway")
        n = self._node
        n.create_subscription(OperatingState, f"{arm}/operating_state", self._on_op, 10)
        n.create_subscription(PoseStamped, f"{arm}/measured_cp", lambda m: self._on_pose(m, "_measured"), 10)
        n.create_subscription(PoseStamped, f"{arm}/setpoint_cp", lambda m: self._on_pose(m, "_setpoint"), 10)
        n.create_subscription(JointState, f"{arm}/measured_js", self._on_js, 10)
        n.create_subscription(JointState, f"{arm}/jaw/measured_js", self._on_jaw, 10)
        self._state_command = n.create_publisher(StringStamped, f"{arm}/state_command", 10)
        self._servo_cp = n.create_publisher(PoseStamped, f"{arm}/servo_cp", 10)
        self._servo_jaw = n.create_publisher(JointState, f"{arm}/jaw/servo_jp", 10)
        threading.Thread(target=lambda: rclpy.spin(n), daemon=True, name="ros2-spin").start()
        deadline = time.time() + wait_s
        while (self._measured is None or self._op is None) and time.time() < deadline:
            time.sleep(0.05)
        if self._measured is None or self._op is None:
            raise BackendError("robot_fault", f"no state from {arm} within {wait_s:g} s")

    # ------------------------------------------------------------------ subscriptions
    def _touch(self) -> None:
        self._seq += 1
        self._stamp = time.time()

    def _on_op(self, m: Any) -> None:
        with self._lock:
            self._op = m
            self._touch()

    def _on_pose(self, m: Any, field: str) -> None:
        p, o = m.pose.position, m.pose.orientation
        x, y, z, w = o.x, o.y, o.z, o.w
        T = np.eye(4)
        T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        T[:3, 3] = [p.x, p.y, p.z]
        with self._lock:
            setattr(self, field, T)
            self._touch()

    def _on_js(self, m: Any) -> None:
        with self._lock:
            self._q = tuple(float(x) for x in m.position)

    def _on_jaw(self, m: Any) -> None:
        with self._lock:
            self._jaw = float(m.position[0]) if m.position else 0.0

    # ------------------------------------------------------------------ the interface
    def state(self) -> RobotState:
        with self._lock:
            op = self._op
            st = getattr(op, "state", "")
            # The operating state has no separate "powered": while it is published the controller is up.
            return RobotState(seq=self._seq, stamp=self._stamp, powered=True, enabled=st == "ENABLED",
                              homed=bool(getattr(op, "is_homed", False)), busy=bool(getattr(op, "is_busy", False)) or self._active is not None,
                              fault="controller reports FAULT" if st == "FAULT" else None, q=self._q, jaw=self._jaw,
                              measured_cp=self._measured.copy(), setpoint_cp=(self._setpoint if self._setpoint is not None else self._measured).copy())

    def _command(self, text: str, until: Any, wait_s: float = 10.0) -> RobotState:
        msg = self._StringStamped()
        msg.string = text
        self._state_command.publish(msg)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            s = self.state()
            if until(s):
                return s
            time.sleep(0.05)
        raise BackendError("robot_not_ready", f"the controller did not reach the state asked for by {text!r}")

    def enable(self) -> RobotState:
        return self._command("enable", lambda s: s.enabled)

    def disable(self) -> RobotState:
        return self._command("disable", lambda s: not s.enabled)

    def home(self) -> RobotState:
        return self._command("home", lambda s: s.homed, wait_s=60.0)

    def begin_motion(self, command_id: str) -> None:
        with self._lock:
            if command_id in self._used:
                raise BackendError("duplicate_command", "this command id was already used")
            if not self.state().ready:
                raise BackendError("robot_not_ready", "the arm is not enabled and homed")
            self._used.add(command_id)
            self._active = command_id

    def end_motion(self, command_id: str) -> RobotState:
        with self._lock:
            if self._active == command_id:
                self._active = None
        return self.state()

    def servo_cp(self, target: np.ndarray, command_id: str) -> RobotState:
        if self._active != command_id:
            raise BackendError("bad_command", "no motion with this command id is active")
        m = self._PoseStamped()
        m.header.stamp = self._node.get_clock().now().to_msg()
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(x) for x in target[:3, 3])
        R = target[:3, :3]
        w = float(np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0)
        if w > 1e-6:
            x, y, z = (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)
        else:                                            # a half turn: take the axis from the diagonal
            x, y, z = (float(np.sqrt(max(0.0, (R[i, i] + 1) / 2))) for i in range(3))
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = float(x), float(y), float(z), w
        self._servo_cp.publish(m)
        time.sleep(1.0 / self.rate_hz)
        return self.state()

    def servo_jaw(self, angle: float, command_id: str) -> RobotState:
        if self._active != command_id:
            raise BackendError("bad_command", "no motion with this command id is active")
        m = self._JointState()
        m.position = [float(angle)]
        self._servo_jaw.publish(m)
        time.sleep(1.0 / self.rate_hz)
        return self.state()
