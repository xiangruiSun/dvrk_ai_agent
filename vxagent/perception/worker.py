"""The perception worker: its own process, as the published system runs its stereo model on a
second machine.

    python -m vxagent.perception.worker --port 8401

One request is worked on at a time and the others wait in a queue, like requests to one GPU.
What it answers with is NOT computed here: it serves needle poses that were recorded as ground
truth in the AMBF scenes of the SurgicAI demonstrations (the goal pose stored with each
demonstration), or a declared pose if those are not on the machine. The process boundary, the
queue and the timing are real; the estimate is not.

Uses only the standard library, so the worker has nothing to install on the second machine.
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

WORK_MS = 40.0                      # declared time one estimate takes


def load_scenes() -> tuple[list[list[float]], str]:
    try:
        from ..data import quality
        demos = quality.from_surgicai("Approach")
        return [[float(x) for x in e.notes["goal"]] for e in demos], "recorded ground truth (SurgicAI Approach demonstrations, AMBF)"
    except Exception:                                    # the data is not on this machine
        return [[0.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0]], "declared (no recorded scenes on this machine)"


class Worker:
    def __init__(self, scenes: list[list[float]] | None = None, source: str = "", work_ms: float = WORK_MS):
        if scenes is None:
            scenes, source = load_scenes()
        self.scenes, self.source, self.work_ms = scenes, source or "declared", work_ms
        self.delay_ms = 0.0                               # INJECTED extra time per estimate
        self.jobs: queue.Queue[tuple[dict[str, Any], threading.Event, dict[str, Any]]] = queue.Queue()
        self.done = 0
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="perception-work").start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job, ready, out = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            time.sleep((self.work_ms + self.delay_ms) / 1000.0)
            scene = int(job.get("scene", 0)) % len(self.scenes)
            out.update({"visible": True, "needle_pose": self.scenes[scene], "scene": scene, "source": self.source,
                        "computed": False, "work_ms": round((time.perf_counter() - t0) * 1000, 1)})
            self.done += 1
            ready.set()

    def estimate(self, job: dict[str, Any], wait_s: float = 5.0) -> dict[str, Any] | None:
        ready, out = threading.Event(), {"queued_at": time.perf_counter()}
        self.jobs.put((job, ready, out))
        if not ready.wait(wait_s):
            return None
        out["total_ms"] = round((time.perf_counter() - out.pop("queued_at")) * 1000, 1)
        out["queue_ms"] = round(max(0.0, out["total_ms"] - out["work_ms"]), 1)
        return out

    def stop(self) -> None:
        self._stop.set()


def make_handler(worker: Worker) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:           # quiet
            pass

        def _send(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:                         # noqa: N802
            if self.path == "/health":
                self._send(200, {"ok": True, "scenes": len(worker.scenes), "queue": worker.jobs.qsize(), "done": worker.done,
                                 "source": worker.source, "computed": False})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:                        # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._send(400, {"error": "not JSON"})
            if self.path == "/estimate":
                out = worker.estimate(body)
                return self._send(200, out) if out is not None else self._send(503, {"error": "busy"})
            if self.path == "/control" and self.client_address[0] in ("127.0.0.1", "::1"):
                worker.delay_ms = float(body.get("delay_ms", 0.0))        # INJECTED slowness, for verification
                return self._send(200, {"delay_ms": worker.delay_ms})
            self._send(404, {"error": "not found"})
    return Handler


class WorkerServer:
    """The worker behind HTTP in this process, for tests. The verification starts it as a process."""

    def __init__(self, worker: Worker | None = None, host: str = "127.0.0.1", port: int = 0):
        self.worker = worker or Worker()
        self.httpd = ThreadingHTTPServer((host, port), make_handler(self.worker))
        self.host, self.port = host, self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True, name="perception-http")

    def start(self) -> "WorkerServer":
        self._thread.start()
        return self

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.worker.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description="Perception worker (serves recorded needle poses; computes nothing).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8401)
    a = ap.parse_args()
    server = WorkerServer(host=a.host, port=a.port)
    print(f"perception worker on {server.url}: {len(server.worker.scenes)} scenes, {server.worker.source}", flush=True)
    try:
        server.httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
