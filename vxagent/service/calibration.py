"""The camera-to-robot calibration procedure, in code.

The arm is moved through the site's declared calibration poses. At each pose a sample is taken:
where the arm's tip is (from the robot) and where the camera sees it. A rigid transform is
fitted to one part of the samples, a local affine correction on top of it, and both are judged
on the samples that were held out. The result is a draft. A draft does nothing until a change
that activates it has been approved by the site's safety lead.

On the twin there is no camera. What the camera "sees" is computed from a DECLARED true
transform kept by the node (and changed by an INJECTED camera move in the verification), plus
declared noise. So the fit has a known answer, and nothing here says anything about accuracy on
hardware.
"""
from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np

from ..diagnostics.episode import T_BC, _apply, _inv
from ..store import Database, dumps

NOISE_M = 0.0002               # declared camera noise, 1 sigma per axis
HOLD_OUT_EVERY = 3             # every third sample is kept out of the fit
ACCEPT_MM = 1.0


def world(node: Any) -> dict[str, Any]:
    """The truth of the scenario: where the camera really is. DECLARED; changed only by an
    injected camera move."""
    if not hasattr(node, "world"):
        node.world = {"T_base_camera": T_BC.copy(), "injected": None, "camera_scale": [1.0, 1.0, 1.0]}
    return node.world


def inject_camera_move(node: Any, offset_mm: list[float], scale: list[float] | None = None) -> dict[str, Any]:
    """INJECTED: the camera has been bumped. The stored calibration no longer describes it."""
    w = world(node)
    T = w["T_base_camera"].copy()
    T[:3, 3] += np.array(offset_mm, dtype=float) / 1000.0
    w.update({"T_base_camera": T, "injected": {"offset_mm": list(offset_mm), "scale": scale or [1.0, 1.0, 1.0]}})
    if scale:
        w["camera_scale"] = list(scale)
    return w["injected"]


def camera_sees(node: Any, tip_m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    w = world(node)
    p = _apply(_inv(w["T_base_camera"]), np.asarray(tip_m, dtype=float))
    return p * np.array(w["camera_scale"], dtype=float) + rng.normal(0.0, NOISE_M, size=3)


def calibration_poses(site: Any, spec: dict[str, Any]) -> list[np.ndarray]:
    """The declared calibration poses of the site, as tip targets in the arm's base frame."""
    start = site.poses["start"].T
    out = []
    for off in spec.get("calibration_poses_mm", []):
        T = start.copy()
        T[:3, 3] = start[:3, 3] + np.array(off, dtype=float) / 1000.0
        out.append(T)
    return out


# ------------------------------------------------------------------ samples

def store_samples(db: Database, site: str, samples: list[dict[str, Any]], by: str) -> str:
    sid = f"sweep-{int(time.time() * 1000) % 10 ** 9:09d}"
    db.execute("INSERT OR REPLACE INTO kv(k,v) VALUES (?,?)", (f"calsamples:{site}", dumps({"id": sid, "by": by, "at": time.time(),
                                                                                         "samples": samples})))
    return sid


def load_samples(db: Database, site: str) -> dict[str, Any] | None:
    row = db.one("SELECT v FROM kv WHERE k=?", (f"calsamples:{site}",))
    return json.loads(row["v"]) if row else None


# ------------------------------------------------------------------ the fit

def rigid_fit(seen: np.ndarray, touched: np.ndarray) -> np.ndarray:
    """The rotation and translation that best take the seen points onto the touched ones (Kabsch)."""
    ca, cb = seen.mean(axis=0), touched.mean(axis=0)
    U, _, Vt = np.linalg.svd((seen - ca).T @ (touched - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cb - R @ ca
    return T


def affine_fit(pred: np.ndarray, touched: np.ndarray) -> np.ndarray:
    """A 3 x 4 correction applied after the rigid transform, by least squares. It absorbs what a
    rigid transform cannot: a scale error or a shear of the camera's depth."""
    A = np.hstack([pred, np.ones((len(pred), 1))])
    X, *_ = np.linalg.lstsq(A, touched, rcond=None)
    return X.T


def _apply_rigid(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def _residual(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    n = np.linalg.norm(pred - truth, axis=1) * 1000.0
    return {"mean_mm": round(float(n.mean()), 3), "max_mm": round(float(n.max()), 3), "rms_mm": round(float(np.sqrt((n ** 2).mean())), 3)}


def fit(samples: list[dict[str, Any]], old_T: np.ndarray | None = None) -> dict[str, Any]:
    """Fit on the samples that are not held out, judge on those that are."""
    seen = np.array([s["seen_m"] for s in samples], dtype=float)
    touched = np.array([s["touched_m"] for s in samples], dtype=float)
    held = np.array([i % HOLD_OUT_EVERY == HOLD_OUT_EVERY - 1 for i in range(len(samples))])
    if (~held).sum() < 4 or held.sum() < 2:
        raise ValueError(f"{len(samples)} samples are too few: the fit needs 4 and the judgement needs 2 held out")
    T = rigid_fit(seen[~held], touched[~held])
    A = affine_fit(_apply_rigid(T, seen[~held]), touched[~held])
    pred_h = _apply_rigid(T, seen[held])
    corrected = np.hstack([pred_h, np.ones((int(held.sum()), 1))]) @ A.T
    out: dict[str, Any] = {
        "samples": len(samples), "used_for_the_fit": int((~held).sum()), "held_out": int(held.sum()),
        "T_est": T.tolist(), "affine": A.tolist(),
        "held_out_residual": {"rigid": _residual(pred_h, touched[held]), "rigid_plus_affine": _residual(corrected, touched[held])},
        "fit_residual": {"rigid": _residual(_apply_rigid(T, seen[~held]), touched[~held])},
        "held_out_seen_m": seen[held].tolist(), "held_out_touched_m": touched[held].tolist()}
    if old_T is not None:
        out["held_out_residual"]["record_in_use"] = _residual(_apply_rigid(old_T, seen[held]), touched[held])
    return out


def difference(Ta: np.ndarray, Tb: np.ndarray, seen: Any = None) -> dict[str, Any]:
    """How far two transforms are apart. The translation parts are compared as stored; what
    matters at the rig is `shift_in_the_workspace_mm`: how far apart the two put the same
    camera-seen points, which is given when such points are passed."""
    R = Ta[:3, :3].T @ Tb[:3, :3]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))))
    d = (Tb[:3, 3] - Ta[:3, 3]) * 1000.0
    out: dict[str, Any] = {"translation_mm": round(float(np.linalg.norm(d)), 3),
                           "translation_vector_mm": [round(float(x), 3) for x in d], "rotation_deg": round(ang, 3)}
    if seen is not None and len(seen):
        pts = np.asarray(seen, dtype=float)
        shift = (_apply_rigid(Tb, pts) - _apply_rigid(Ta, pts)) * 1000.0
        out["shift_in_the_workspace_mm"] = round(float(np.linalg.norm(shift, axis=1).mean()), 3)
        out["shift_vector_mm"] = [round(float(x), 3) for x in shift.mean(axis=0)]
    return out


# ------------------------------------------------------------------ drafts

def store_draft(db: Database, site: str, result: dict[str, Any], sweep_id: str, by: str) -> dict[str, Any]:
    draft = {"id": "cal-" + sweep_id[-6:], "T_est": result["T_est"], "affine": result["affine"],
             "seen_m": result["held_out_seen_m"], "touched_m": result["held_out_touched_m"],
             "method": "rigid fit on arm-touched points with a local affine correction", "session": sweep_id,
             "samples": result["samples"], "held_out_residual": result["held_out_residual"], "fitted_by": by,
             "status": "draft", "at": time.time()}
    db.execute("INSERT OR REPLACE INTO kv(k,v) VALUES (?,?)", (f"caldraft:{site}", dumps(draft)))
    return draft


def load_draft(db: Database, site: str) -> dict[str, Any] | None:
    row = db.one("SELECT v FROM kv WHERE k=?", (f"caldraft:{site}",))
    return json.loads(row["v"]) if row else None
