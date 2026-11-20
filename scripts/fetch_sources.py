"""Download the text sources the retrieval system is built on.

    python scripts/fetch_sources.py            # documentation, change log, support threads
    python scripts/fetch_sources.py --skip-issues

  documentation    jhu-dvrk/dvrk-readthedocs at a pinned commit, reStructuredText only. The
                   repository is about 0.9 GB of images, PDFs and slides; a sparse, blob-filtered
                   checkout fetches only the .rst files (under 1 MB).
  change log       CHANGELOG.md of jhu-dvrk/sawIntuitiveResearchKit at the pinned commit.
  support threads  issues and their comments from the dVRK repositories listed below, through
                   the public GitHub API (no account; about 20 requests of the 60 per hour an
                   anonymous client is allowed). Pull requests are skipped.

Author names are not stored. Each post keeps only the author's association with the repository
(OWNER, MEMBER, COLLABORATOR, CONTRIBUTOR, NONE) and whether the author opened the thread.

Everything goes under data/sources/ and nothing there is committed. The documentation and the
source repositories state no licence, so they are used locally and not redistributed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sources"
UA = {"User-Agent": "generic-client", "Accept": "application/vnd.github+json"}

DOCS_REPO = "jhu-dvrk/dvrk-readthedocs"
DOCS_COMMIT = "3b231b2a59d223d3b73cd7f55c9e90b04be404bb"          # main on 2026-11-15
SAW_REPO = "jhu-dvrk/sawIntuitiveResearchKit"
SAW_COMMIT = "7435458dc847b95b9ff77723b33e4a177cfa4f43"           # main on 2026-11-19
ISSUE_REPOS = ["jhu-dvrk/sawIntuitiveResearchKit", "jhu-dvrk/dvrk-ros"]


def get(url: str) -> tuple[bytes, dict[str, str]]:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read(), dict(r.headers)


def git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _remove(path: Path) -> None:
    def writable(func, target, _exc):
        os.chmod(target, stat.S_IWRITE)          # git marks its object files read-only on Windows
        func(target)
    shutil.rmtree(path, onerror=writable)


def fetch_docs() -> dict:
    work = OUT / "_docs_checkout"
    if work.exists():
        _remove(work)
    work.mkdir(parents=True)
    git("init", "-q", cwd=work)
    git("remote", "add", "origin", f"https://github.com/{DOCS_REPO}.git", cwd=work)
    git("sparse-checkout", "set", "--no-cone", "*.rst", cwd=work)
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", DOCS_COMMIT, cwd=work)
    git("-c", "core.autocrlf=false", "checkout", "-q", "FETCH_HEAD", cwd=work)
    dest = OUT / "docs"
    if dest.exists():
        shutil.rmtree(dest)
    files = []
    for p in sorted((work / "docs").rglob("*.rst")):
        rel = p.relative_to(work / "docs")
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        data = p.read_bytes()
        (dest / rel).write_bytes(data)
        files.append({"path": rel.as_posix(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    _remove(work)
    return {"repository": DOCS_REPO, "commit": DOCS_COMMIT, "files": len(files),
            "bytes": sum(f["bytes"] for f in files), "list": files}


def fetch_changelog() -> dict:
    data, _ = get(f"https://raw.githubusercontent.com/{SAW_REPO}/{SAW_COMMIT}/CHANGELOG.md")
    (OUT / "changelog").mkdir(parents=True, exist_ok=True)
    (OUT / "changelog" / "CHANGELOG.md").write_bytes(data)
    return {"repository": SAW_REPO, "commit": SAW_COMMIT, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def paged(url: str) -> list[dict]:
    items, page = [], 1
    while True:
        body, headers = get(f"{url}{'&' if '?' in url else '?'}per_page=100&page={page}")
        batch = json.loads(body)
        items.extend(batch)
        if len(batch) < 100:
            return items
        if int(headers.get("X-RateLimit-Remaining", "1")) < 2:
            raise RuntimeError("GitHub's anonymous rate limit is nearly used up; try again in an hour")
        page += 1


def fetch_issues() -> dict:
    dest = OUT / "issues"
    dest.mkdir(parents=True, exist_ok=True)
    summary = {}
    for repo in ISSUE_REPOS:
        raw = paged(f"https://api.github.com/repos/{repo}/issues?state=all&sort=created&direction=asc")
        issues = {i["number"]: i for i in raw if "pull_request" not in i}
        comments = paged(f"https://api.github.com/repos/{repo}/issues/comments?sort=created&direction=asc")
        threads = []
        by_issue: dict[int, list[dict]] = {}
        for c in comments:
            n = int(c["issue_url"].rsplit("/", 1)[1])
            if n in issues:
                by_issue.setdefault(n, []).append(c)
        for n, i in sorted(issues.items()):
            opener = (i.get("user") or {}).get("id")
            threads.append({
                "repo": repo, "number": n, "title": i["title"], "state": i["state"],
                "state_reason": i.get("state_reason"), "created_at": i["created_at"], "closed_at": i.get("closed_at"),
                "updated_at": i["updated_at"], "labels": [lb["name"] for lb in i.get("labels", [])],
                "author_association": i.get("author_association"), "body": i.get("body") or "",
                "comments": [{"id": c["id"], "created_at": c["created_at"],
                              "author_association": c.get("author_association"),
                              "by_opener": (c.get("user") or {}).get("id") == opener,
                              "body": c.get("body") or ""} for c in by_issue.get(n, [])]})
        name = repo.split("/")[1]
        (dest / f"{name}.json").write_text(json.dumps(threads, ensure_ascii=False), encoding="utf-8")
        summary[repo] = {"threads": len(threads), "closed": sum(t["state"] == "closed" for t in threads),
                         "comments": sum(len(t["comments"]) for t in threads),
                         "latest_update": max((t["updated_at"] for t in threads), default=None)}
        print(f"  {repo}: {summary[repo]}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-issues", action="store_true")
    ap.add_argument("--skip-docs", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["fetched"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        if not a.skip_docs:
            manifest["docs"] = fetch_docs()
            print(f"documentation: {manifest['docs']['files']} files, {manifest['docs']['bytes'] / 1e6:.2f} MB")
            manifest["changelog"] = fetch_changelog()
            print(f"change log: {manifest['changelog']['bytes'] / 1e3:.1f} kB")
        if not a.skip_issues:
            manifest["issues"] = {"fetched": manifest["fetched"], "repositories": fetch_issues()}
    except (urllib.error.URLError, subprocess.CalledProcessError, RuntimeError) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        print(f"failed: {exc} {detail.decode(errors='replace')[:300]}", file=sys.stderr)
        return 1
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
