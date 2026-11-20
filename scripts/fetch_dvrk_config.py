"""Download dVRK configuration files from the dVRK project at a pinned commit.

    python scripts/fetch_dvrk_config.py

Fetches share/kinematic, share/tool and share/schemas of jhu-dvrk/sawIntuitiveResearchKit into
data/dvrk/share/ and writes data/dvrk/MANIFEST.json (path, size, SHA-256, commit). The files are
used locally and are not committed to this repository: see data/README.md for the licence note.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "jhu-dvrk/sawIntuitiveResearchKit"
COMMIT = "7435458dc847b95b9ff77723b33e4a177cfa4f43"      # main on 2026-11-19
FOLDERS = ("share/kinematic/", "share/tool/", "share/schemas/")
SUFFIXES = (".json", ".md")
UA = {"User-Agent": "generic-client"}


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def main() -> int:
    out = ROOT / "data" / "dvrk"
    tree = json.loads(get(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"))
    if tree.get("truncated"):
        print("the repository tree listing was truncated; refusing to continue", file=sys.stderr)
        return 1
    wanted = [e for e in tree["tree"] if e["type"] == "blob" and e["path"].startswith(FOLDERS)
              and e["path"].endswith(SUFFIXES)]
    manifest = {"repository": REPO, "commit": COMMIT, "fetched": time.strftime("%Y-%m-%d %H:%M:%S"), "files": []}
    total = 0
    for e in sorted(wanted, key=lambda e: e["path"]):
        data = get(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{e['path']}")
        dest = out / e["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        total += len(data)
        manifest["files"].append({"path": e["path"], "bytes": len(data),
                                  "sha256": hashlib.sha256(data).hexdigest()})
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    by = {f: sum(1 for x in manifest["files"] if x["path"].startswith(f)) for f in FOLDERS}
    print(f"{len(manifest['files'])} files, {total / 1e6:.2f} MB -> {out}")
    print(by)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
