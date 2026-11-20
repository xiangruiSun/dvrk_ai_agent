"""SurgicAI expert demonstrations (github.com/surgical-robotics-ai/SurgicAI, MIT licence).

The demonstrations were recorded in the Surgical Robotics Challenge simulator with a dVRK PSM
and are published as pickle files. A pickle can run code when it is loaded, so the standard
loader is never used on them. `safe_load` rebuilds only what such a file needs to hold (arrays,
numbers, strings and plain containers) and refuses everything else.
"""
from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_settings

TASKS = ("Approach", "Place", "Regrasp", "Insert", "Pullout")
_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"), ("collections", "OrderedDict"),
    ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
}


class UnsafePickle(Exception):
    pass


class _Restricted(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _ALLOWED:
            if module.startswith("numpy.core"):                 # the same objects under their current name
                module = module.replace("numpy.core", "numpy._core")
            mod = __import__(module, fromlist=[name])
            return getattr(mod, name)
        if module == "numpy" and name in ("float64", "float32", "int64", "int32", "bool_", "bool"):
            return getattr(np, name)
        raise UnsafePickle(f"the file asks for {module}.{name}, which is not data; it was not loaded")


def safe_load(path: Path | str) -> Any:
    return _Restricted(io.BytesIO(Path(path).read_bytes())).load()


def folder(base: Path | None = None) -> Path:
    return (base or get_settings().data_dir) / "surgicai"


def load_task(task: str, base: Path | None = None) -> list[dict[str, np.ndarray]]:
    """The demonstrations of one sub-task as a list of episodes, each a dict of arrays."""
    path = folder(base) / "RL" / "Expert_traj" / task / "all_episodes_merged.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run: python scripts/fetch_surgicai.py")
    raw = safe_load(path)
    episodes = raw if isinstance(raw, list) else [raw]
    out = []
    for ep in episodes:
        if isinstance(ep, dict):
            out.append({k: np.asarray(v) if isinstance(v, (list, np.ndarray)) else v for k, v in ep.items()})
        else:
            out.append({"steps": ep})
    return out
