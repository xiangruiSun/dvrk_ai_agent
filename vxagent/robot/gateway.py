"""The robot gateway: the one process that talks to the robot backend, and the narrow interface
everything else uses to reach it.

    python -m vxagent.robot.gateway --backend twin --site site_a --port 8402

Two allowlists, checked against the address the connection really comes from:
    command hosts   may send state commands and set points; by default this machine only, and
                    only with the agent process's token
    state hosts     may read the state
A request from any other address is refused and audited, whatever it carries.

`GatewayBackend` is the client: the same RobotBackend interface as the twin, over HTTP. No TLS
between the processes; that is stated in the task's limits.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from .backend import BackendError, RobotState

LOCAL = ("127.0.0.1", "::1")


def state_to_json(s: RobotState) -> dict[str, Any]:
    return {"seq": s.seq, "stamp": s.stamp, "powered": s.powered, "enabled": s.enabled, "homed": s.homed, "busy": s.busy,
            "fault": s.fault, "measured_js": list(s.q), "jaw_measured_js": s.jaw,
            "measured_cp": np.asarray(s.measured_cp).tolist(), "setpoint_cp": np.asarray(s.setpoint_cp).tolist()}


def state_from_json(d: dict[str, Any]) -> RobotState:
    return RobotState(seq=d["seq"], stamp=d["stamp"], powered=d["powered"], enabled=d["enabled"], homed=d["homed"],
                      busy=d["busy"], fault=d["fault"], q=tuple(d["measured_js"]), jaw=d["jaw_measured_js"],
                      measured_cp=np.array(d["measured_cp"], dtype=float), setpoint_cp=np.array(d["setpoint_cp"], dtype=float))


class Gateway:
    COMMANDS = ("enable", "disable", "home", "begin_motion", "end_motion", "servo_cp", "servo_jaw")

    def __init__(self, backend: Any, token: str, command_hosts: tuple[str, ...] = LOCAL, state_hosts: tuple[str, ...] = LOCAL,
                 host: str = "127.0.0.1", port: int = 0):
        self.backend, self._token = backend, token
        self.command_hosts, self.state_hosts = set(command_hosts), set(state_hosts)
        self.refused: list[dict[str, Any]] = []
        self.httpd = ThreadingHTTPServer((host, port), self._handler())
        self.host, self.port = host, self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True, name="robot-gateway")

    def _refuse(self, peer: str, what: str, why: str) -> None:
        self.refused.append({"at": time.time(), "from": peer, "what": what, "why": why})

    def command(self, op: str, args: dict[str, Any]) -> RobotState:
        b = self.backend
        if op == "servo_cp":
            return b.servo_cp(np.array(args["target"], dtype=float), str(args["command_id"]))
        if op == "servo_jaw":
            return b.servo_jaw(float(args["angle"]), str(args["command_id"]))
        if op == "begin_motion":
            b.begin_motion(str(args["command_id"]))
            return b.state()
        if op == "end_motion":
            return b.end_motion(str(args["command_id"]))
        return getattr(b, op)()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        gw = self

        class Handler(BaseHTTPRequestHandler):
            disable_nagle_algorithm = True        # headers and body are two writes; neither should wait for the other

            def log_message(self, *a: Any) -> None:
                pass

            def _send(self, code: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:                     # noqa: N802
                peer = self.client_address[0]
                if self.path != "/state":
                    return self._send(404, {"error": "not found"})
                if peer not in gw.state_hosts:
                    gw._refuse(peer, "state", "host not on the state allowlist")
                    return self._send(403, {"error": "this host may not read the robot's state"})
                self._send(200, {"name": gw.backend.name, "rate_hz": gw.backend.rate_hz, **state_to_json(gw.backend.state())})

            def do_POST(self) -> None:                    # noqa: N802
                peer = self.client_address[0]
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    return self._send(400, {"error": "not JSON"})
                op = str(body.get("op", ""))
                if self.path != "/command" or op not in gw.COMMANDS:
                    return self._send(404, {"error": "not a command"})
                if peer not in gw.command_hosts:
                    gw._refuse(peer, op, "host not on the command allowlist")
                    return self._send(403, {"error": "this host may not command the robot"})
                if self.headers.get("Authorization", "") != f"Bearer {gw._token}":
                    gw._refuse(peer, op, "not the agent process's token")
                    return self._send(401, {"error": "not the agent process"})
                try:
                    s = gw.command(op, body.get("args") or {})
                except BackendError as exc:
                    return self._send(409, {"error": str(exc), "kind": exc.kind})
                except (KeyError, TypeError, ValueError) as exc:
                    return self._send(400, {"error": f"bad arguments: {type(exc).__name__}"})
                self._send(200, state_to_json(s))
        return Handler

    def start(self) -> "Gateway":
        self._thread.start()
        return self

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class GatewayBackend:
    """RobotBackend over the gateway. A rehearsal copy is always a local twin built from the
    site's kinematic files and set to the joints the gateway reports."""

    def __init__(self, url: str, token: str, site: Any = None, timeout_s: float = 2.0):
        import httpx

        from .. import net
        self._http, self._client = httpx, net                    # one client kept open: a set point per control tick goes through it
        self.url, self._token, self.site, self.timeout_s = url.rstrip("/"), token, site, timeout_s
        first = self._get()
        self.name, self.rate_hz = f"{first['name']}@gateway", float(first["rate_hz"])
        self.listeners: list[Any] = []

    def _get(self) -> dict[str, Any]:
        try:
            r = self._client.get(f"{self.url}/state", timeout=self.timeout_s)
        except self._http.HTTPError as exc:
            raise BackendError("robot_fault", f"the robot gateway cannot be reached: {type(exc).__name__}") from exc
        if r.status_code != 200:
            raise BackendError("robot_fault", f"the robot gateway refused the state request ({r.status_code})")
        return r.json()

    def _cmd(self, op: str, **args: Any) -> RobotState:
        try:
            r = self._client.post(f"{self.url}/command", json={"op": op, "args": args},
                                headers={"Authorization": f"Bearer {self._token}"}, timeout=self.timeout_s)
        except self._http.HTTPError as exc:
            raise BackendError("robot_fault", f"the robot gateway cannot be reached: {type(exc).__name__}") from exc
        if r.status_code == 409:
            raise BackendError(r.json().get("kind", "bad_command"), r.json().get("error", "refused"))
        if r.status_code != 200:
            raise BackendError("bad_command", f"the gateway refused {op} ({r.status_code}): {r.json().get('error', '')}")
        s = state_from_json(r.json())
        for fn in self.listeners:
            fn(s)
        return s

    def state(self) -> RobotState:
        return state_from_json(self._get())

    def enable(self) -> RobotState:
        return self._cmd("enable")

    def disable(self) -> RobotState:
        return self._cmd("disable")

    def home(self) -> RobotState:
        return self._cmd("home")

    def begin_motion(self, command_id: str) -> None:
        self._cmd("begin_motion", command_id=command_id)

    def end_motion(self, command_id: str) -> RobotState:
        return self._cmd("end_motion", command_id=command_id)

    def servo_cp(self, target: np.ndarray, command_id: str) -> RobotState:
        return self._cmd("servo_cp", target=np.asarray(target).tolist(), command_id=command_id)

    def servo_jaw(self, angle: float, command_id: str) -> RobotState:
        return self._cmd("servo_jaw", angle=float(angle), command_id=command_id)

    def clone(self) -> Any:
        if self.site is None:
            raise BackendError("bad_command", "no site description to build a rehearsal copy from")
        s = self.state()
        twin = self.site.twin()
        twin.set_joints(np.array(s.q, dtype=float))
        if s.enabled:
            twin.enable()
        if s.homed:
            twin._homed = True
        twin._jaw = s.jaw
        return twin


def main() -> int:
    import secrets

    from .site import load_site
    ap = argparse.ArgumentParser(description="Robot gateway.")
    ap.add_argument("--backend", default="twin", choices=["twin", "ros2"])
    ap.add_argument("--site", default="site_a")
    ap.add_argument("--port", type=int, default=8402)
    ap.add_argument("--state-hosts", default="127.0.0.1", help="comma-separated addresses that may read the state")
    a = ap.parse_args()
    site = load_site(a.site)
    if a.backend == "twin":
        backend: Any = site.twin()
    else:
        from .ros2_backend import Ros2Backend
        backend = Ros2Backend(site.arm)
    token = secrets.token_urlsafe(24)
    gw = Gateway(backend, token, state_hosts=tuple(h.strip() for h in a.state_hosts.split(",") if h.strip()), port=a.port)
    print(f"robot gateway on {gw.url}, backend {backend.name}, site {site.name}", flush=True)
    print("the agent process reads its token from the VXAGENT_GATEWAY_TOKEN variable of this session", flush=True)
    import os
    os.environ["VXAGENT_GATEWAY_TOKEN"] = token
    try:
        gw.httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
