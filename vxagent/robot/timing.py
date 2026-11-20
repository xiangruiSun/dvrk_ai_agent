"""Measuring the control loop, and checking that the language model stays out of it.

The control rate is kept by the code that sends set points: one target, one tick. `TickMeter`
wraps that tick and records when each one was taken, so the period and its jitter can be
reported as measured. `model_calls_inside_actions` reads the trace of a run and counts model
calls that started while an action was executing; the design says that number is zero.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np


class TickMeter:
    def __init__(self, tick: Callable[[float], None] = time.sleep):
        self._tick = tick
        self.stamps: list[float] = []
        self.asked: list[float] = []

    def __call__(self, dt: float) -> None:
        self._tick(dt)
        self.stamps.append(time.perf_counter())
        self.asked.append(dt)

    def periods_ms(self, nominal_s: float = 0.1, gap_factor: float = 2.5) -> np.ndarray:
        """Time between consecutive control ticks, leaving out the pauses between actions
        (a gap longer than gap_factor periods is not a period of the loop)."""
        if len(self.stamps) < 2:
            return np.array([])
        asked = np.array(self.asked[1:])
        d = np.diff(np.array(self.stamps))
        keep = (np.abs(asked - nominal_s) < 1e-9) & (d < gap_factor * nominal_s)
        return d[keep] * 1000.0

    def report(self, nominal_s: float = 0.1) -> dict[str, Any]:
        p = self.periods_ms(nominal_s)
        if not len(p):
            return {"ticks": len(self.stamps), "periods": 0}
        return {"ticks": len(self.stamps), "periods": int(len(p)), "nominal_ms": nominal_s * 1000.0,
                "mean_ms": round(float(p.mean()), 2), "p50_ms": round(float(np.percentile(p, 50)), 2),
                "p95_ms": round(float(np.percentile(p, 95)), 2), "max_ms": round(float(p.max()), 2),
                "jitter_std_ms": round(float(p.std()), 2), "worst_late_ms": round(float(p.max() - nominal_s * 1000.0), 2)}


def model_calls_inside_actions(db: Any, run_id: str) -> dict[str, Any]:
    """Model calls of a run that started between the first command of an action and its result."""
    spans = db.query("SELECT kind, name, started, latency_ms FROM spans WHERE run_id=? ORDER BY id", (run_id,))
    actions = [(s["started"], s["started"] + (s["latency_ms"] or 0.0) / 1000.0, s["name"]) for s in spans if s["kind"] == "action"]
    llm = [(s["started"], s["name"]) for s in spans if s["kind"] == "llm"]
    inside = [{"model_call": n, "during": a} for t, n in llm for t0, t1, a in actions if t0 <= t <= t1]
    return {"actions": len(actions), "model_calls": len(llm), "model_calls_inside_an_action": len(inside), "which": inside[:5]}
