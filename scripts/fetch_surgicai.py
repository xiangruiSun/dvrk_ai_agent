"""Download the SurgicAI expert demonstrations and the files that describe their environments.

    python scripts/fetch_surgicai.py

From github.com/surgical-robotics-ai/SurgicAI (MIT licence) at a pinned commit: the merged
expert trajectories of the five sub-tasks (RL/Expert_traj/*/all_episodes_merged.pkl, 9.9 MB)
and the environment sources that say what an observation and an action mean. Written to
data/surgicai/ with a manifest. Nothing is committed to this repository.

The trajectories are pickle files. They are never unpickled with the standard loader: see
vxagent/data/surgicai.py, which only reconstructs arrays and plain containers. The Python
sources are read as text and never imported or run.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "surgical-robotics-ai/SurgicAI"
COMMIT = "b7ce5465ec0be897beeecb24791c0eaf7045abae"       # tree of main on 2026-11-20
TASKS = ("Approach", "Place", "Regrasp", "Insert", "Pullout")
FILES = [f"RL/Expert_traj/{t}/all_episodes_merged.pkl" for t in TASKS] + [
    "RL/Approach_env.py", "RL/Place_env.py", "RL/subtask_env.py", "RL/Low_level_env_complete.py", "README.md", "LICENSE"]
UA = {"User-Agent": "generic-client"}


def main() -> int:
    out = ROOT / "data" / "surgicai"
    manifest = {"repository": REPO, "tree": COMMIT, "licence": "MIT", "fetched": time.strftime("%Y-%m-%d %H:%M:%S"), "files": []}
    for path in FILES:
        url = f"https://raw.githubusercontent.com/{REPO}/main/{path}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
            data = r.read()
        dest = out / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        manifest["files"].append({"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        print(f"{len(data):>9}  {path}")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(FILES)} files, {sum(f['bytes'] for f in manifest['files']) / 1e6:.2f} MB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
