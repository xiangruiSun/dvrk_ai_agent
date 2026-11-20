"""Download recorded dVRK kinematics from the ROSMA dataset (Zenodo record 3932964, CC BY 4.0).

    python scripts/fetch_rosma.py                       # default subset: 12 recordings
    python scripts/fetch_rosma.py --operators X01 X02 --trials 1 2
    python scripts/fetch_rosma.py --all                 # all 207 recordings, about 1.6 GB

Only the kinematics CSV files and scores.csv are fetched. The videos (20.7 GB) and the operator
questionnaire are never fetched. Each file is checked against the MD5 published by Zenodo.
Nothing downloaded here is committed to this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = "3932964"
UA = {"User-Agent": "generic-client"}
NAME = re.compile(r"^(X\d{2})_(.+)_(\d{2})\.csv$")


def get(url: str, timeout: float = 300) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--operators", nargs="*", default=["X01", "X02", "X03", "X04"])
    ap.add_argument("--trials", nargs="*", type=int, default=[1])
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    out = ROOT / "data" / "rosma"
    out.mkdir(parents=True, exist_ok=True)
    record = json.loads(get(f"https://zenodo.org/api/records/{RECORD}", 60))
    licence = (record["metadata"].get("license") or {}).get("id")
    wanted = []
    for f in record["files"]:
        m = NAME.match(f["key"])
        if f["key"] == "scores.csv":
            wanted.append(f)
        elif m and (a.all or (m.group(1) in a.operators and int(m.group(3)) in a.trials)):
            wanted.append(f)
    manifest = {"record": RECORD, "doi": record.get("doi"), "licence": licence,
                "title": record["metadata"].get("title"), "fetched": time.strftime("%Y-%m-%d %H:%M:%S"), "files": []}
    total = 0
    for f in sorted(wanted, key=lambda f: f["key"]):
        dest = out / f["key"]
        expect = (f.get("checksum") or "").removeprefix("md5:")
        if dest.exists() and hashlib.md5(dest.read_bytes()).hexdigest() == expect:
            data = dest.read_bytes()
        else:
            data = get(f["links"]["self"])
            if expect and hashlib.md5(data).hexdigest() != expect:
                print(f"checksum mismatch for {f['key']}; not saved", file=sys.stderr)
                return 1
            dest.write_bytes(data)
        total += len(data)
        manifest["files"].append({"name": f["key"], "bytes": len(data), "md5": expect})
        print(f"  {f['key']}  {len(data) / 1e6:.1f} MB")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(manifest['files'])} files, {total / 1e6:.1f} MB, licence {licence} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
