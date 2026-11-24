"""Fetch the public catalog of robot-learning datasets on the Hugging Face Hub.

    python scripts/fetch_hub_catalog.py                 two snapshots of the card dump and today's listing
    python scripts/fetch_hub_catalog.py --only listing  only the listing (about 100 API calls)

What is fetched, all of it public metadata, none of it committed to this repository:

  1. The dataset-card dump `librarian-bots/dataset_cards_with_metadata`: one row per dataset on
     the Hub with its card text and metadata, published daily as five parquet files (about 1 GB).
     Two revisions are taken: the newest one, and the one closest to --lag-days earlier. The
     older one is what the search index is built from; the newer one is what "the live catalog"
     looks like. The difference between them is real index staleness.
  2. A listing of every dataset tagged `LeRobot` through the Hub API: id, owner, created,
     last modified, revision, gated or not, downloads, likes, tags.

Every file is checked against the size and SHA-256 the Hub publishes for it, and
data/hub/MANIFEST.json records the revisions, dates, sizes and hashes. No file of any dataset
itself (no episode, no video, no parquet of robot data) is downloaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "hub"
DUMP = "librarian-bots/dataset_cards_with_metadata"
API = "https://huggingface.co/api"
UA = {"User-Agent": "generic-client/1.0"}
TAG = "LeRobot"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def get(url: str, tries: int = 6) -> tuple[bytes, dict[str, str]]:
    """GET with patience: the Hub answers 429 when its five-minute window is used up."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read(), {k.lower(): v for k, v in r.headers.items()}
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt + 1 < tries:
                wait = int(exc.headers.get("Retry-After") or 0) or min(300, 20 * 2 ** attempt)
                log(f"HTTP {exc.code}, waiting {wait} s")
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt + 1 == tries:
                raise
            log(f"{type(exc).__name__}, retrying")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("unreachable")


def commits() -> list[dict[str, str]]:
    """Every commit of the dump, newest first."""
    out: list[dict[str, str]] = []
    url: str | None = f"{API}/datasets/{DUMP}/commits/main?limit=100"
    while url:
        body, headers = get(url)
        out += [{"sha": c["id"], "date": c["date"]} for c in json.loads(body)]
        m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("link", ""))
        url = m.group(1) if m else None
    return out


def pick(history: list[dict[str, str]], lag_days: int) -> tuple[dict[str, str], dict[str, str]]:
    new = history[0]
    want = datetime.fromisoformat(new["date"].replace("Z", "+00:00")) - timedelta(days=lag_days)
    old = min(history, key=lambda c: abs(datetime.fromisoformat(c["date"].replace("Z", "+00:00")) - want))
    return new, old


def download(url: str, path: Path, size: int, sha256: str) -> None:
    if path.exists() and path.stat().st_size == size:
        return
    part = path.with_suffix(path.suffix + ".part")
    h = hashlib.sha256()
    done = 0
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, part.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
    if done != size or (sha256 and h.hexdigest() != sha256):
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{path.name}: got {done} bytes with hash {h.hexdigest()[:12]}, expected {size} / {sha256[:12]}")
    part.replace(path)


def fetch_revision(commit: dict[str, str]) -> dict[str, object]:
    sha = commit["sha"]
    body, _ = get(f"{API}/datasets/{DUMP}/revision/{sha}?blobs=true")
    files = [s for s in json.loads(body)["siblings"] if s["rfilename"].endswith(".parquet")]
    folder = OUT / "dump" / sha[:12]
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in files:
        name = Path(s["rfilename"]).name
        want = (s.get("lfs") or {}).get("sha256", "")
        log(f"  {sha[:12]} {name} ({s['size'] / 1e6:.0f} MB)")
        download(f"https://huggingface.co/datasets/{DUMP}/resolve/{sha}/{s['rfilename']}", folder / name, s["size"], want)
        rows.append({"file": name, "bytes": s["size"], "sha256": want})
    return {"revision": sha, "published": commit["date"], "files": rows, "bytes": sum(r["bytes"] for r in rows)}


def fetch_listing() -> dict[str, object]:
    """Every dataset with the tag, as the API lists it now. One JSON object per line."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = OUT / "listing" / f"{TAG.lower()}_{stamp}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = ("id", "author", "createdAt", "lastModified", "sha", "gated", "private", "disabled", "downloads", "likes", "tags")
    url: str | None = f"{API}/datasets?filter={TAG}&limit=1000&full=true"
    n = pages = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        while url:
            body, headers = get(url)
            for row in json.loads(body):
                f.write(json.dumps({k: row.get(k) for k in keep}, ensure_ascii=False) + "\n")
                n += 1
            pages += 1
            m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("link", ""))
            url = m.group(1) if m else None
            if pages % 10 == 0:
                log(f"  listing: {n} datasets")
            time.sleep(0.7)                                   # the API allows 500 calls in five minutes
    return {"file": str(path.relative_to(OUT)).replace("\\", "/"), "taken": stamp, "tag": TAG, "datasets": n, "api_calls": pages}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=["dump", "listing"])
    ap.add_argument("--lag-days", type=int, default=30, help="how much older the second snapshot of the dump is")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update({"source": "https://huggingface.co", "dump": DUMP, "fetched": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "note": "public metadata and dataset cards only; no file of any dataset itself"})
    if a.only != "listing":
        history = commits()
        new, old = pick(history, a.lag_days)
        log(f"dump: {len(history)} daily revisions from {history[-1]['date'][:10]} to {history[0]['date'][:10]}")
        log(f"newer snapshot {new['sha'][:12]} of {new['date'][:10]}, older snapshot {old['sha'][:12]} of {old['date'][:10]}")
        manifest["snapshots"] = {"old": fetch_revision(old), "new": fetch_revision(new)}
        manifest["dump_revisions_available"] = len(history)
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if a.only != "dump":
        log("listing")
        manifest.setdefault("listings", []).append(fetch_listing())
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
