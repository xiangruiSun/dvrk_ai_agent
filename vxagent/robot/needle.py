"""A declared needle for the twin.

The twin has no contact and no needle. Procedures that are about the needle (is it held, did it
come up with the jaw, is it still there after two seconds) need something to observe, so the
outcome is DECLARED before the run and this model plays it out against what the arm does:

    grasp_succeeds        closing the jaw at the grasp pose takes hold of the needle
    slips_on_lift         a held needle drops once the tip has risen by 5 mm
    slips_while_holding   a held needle drops after it has been held still for one second

Everything it reports is a consequence of those declarations and of the arm's recorded motion.
It is not a measurement and is labelled as declared wherever it is used.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


class DeclaredNeedle:
    def __init__(self, site: Any, grasp_pose: str = "pregrasp", grasp_succeeds: bool = True,
                 slips_on_lift: bool = False, slips_while_holding: bool = False):
        self.site = site
        self.grasp_p = site.poses[grasp_pose].T[:3, 3].copy()      # where the jaw has to close to take the needle
        self.released_at: dict[str, float] | None = None
        self.grasp_succeeds, self.slips_on_lift, self.slips_while_holding = grasp_succeeds, slips_on_lift, slips_while_holding
        self.held = False
        self.height_mm = 0.0                       # of the needle over the pad
        self._jaw_closed = False
        self._grasp_height = 0.0
        self._held_still_s = 0.0
        self.log: list[str] = []

    def declared(self) -> dict[str, bool]:
        return {"grasp_succeeds": self.grasp_succeeds, "slips_on_lift": self.slips_on_lift,
                "slips_while_holding": self.slips_while_holding}

    def attach(self, twin: Any) -> "DeclaredNeedle":
        twin.listeners.append(self.on_state)
        self.on_state(twin.state())
        return self

    def on_state(self, s: Any) -> None:
        tip = s.measured_cp[:3, 3]
        tip_h = self.site.scene.height(tip) * 1000.0
        closed = s.jaw <= self.site.jaw_angles["closed"] + math.radians(1.0)
        if closed and not self._jaw_closed and not self.held:
            near = float(np.linalg.norm(tip - self.grasp_p)) * 1000.0 <= 2.0
            if near and self.grasp_succeeds:
                self.held, self._grasp_height = True, tip_h
                self.log.append("the jaw closed on the needle")
            elif near:
                self.log.append("the jaw closed and missed the needle")
        if self.held and not closed:
            self.held = False
            self.released_at = {"height_over_pad_mm": round(tip_h, 2), "x_m": float(tip[0]), "y_m": float(tip[1]), "z_m": float(tip[2])}
            self.log.append(f"the jaw opened {tip_h:.1f} mm over the pad")
        if self.held and self.slips_on_lift and tip_h - self._grasp_height >= 5.0:
            self.held = False
            self.log.append("the needle slipped out during the lift")
        self._jaw_closed = closed
        if self.held:
            self.height_mm = tip_h
        elif self.height_mm > 0.0:
            self.height_mm = 0.0                   # a needle that is not held lies on the pad

    def held_still(self, seconds: float) -> None:
        if self.held and self.slips_while_holding:
            self._held_still_s += seconds
            if self._held_still_s >= 1.0:
                self.held, self.height_mm = False, 0.0
                self.log.append("the needle slipped out while it was held still")

    def observe(self, ctx: Any = None) -> dict[str, Any]:
        return {"visible": True, "held": self.held, "height_over_pad_mm": round(self.height_mm, 2), "source": "declared"}

    __call__ = observe
