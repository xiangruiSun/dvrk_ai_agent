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
# An earlier state of the documentation, used to measure incremental sync (requirements/archive/rag/R09).
DOCS_PREVIOUS_COMMIT = "6f0167fa75"                               # main on 2026-09-13, resolved by git
# Sites that publish notes of their own beside their configuration files (the github.com/dvrk-config
# organization). In this project each is a private collection that only that site may read (R07).
SITE_REPOS = {
    "jhu": ("dvrk-config/dvrk_config_jhu", "7fe6edb3e4286e797431c25150303e56bb2ae052"),
    "ucl": ("dvrk-config/dvrk_config_ucl", "162f4d35dc036b7565c05f616189daf5bf97b309"),
    "cuhk": ("dvrk-config/dvrk_config_cuhk", "3d7b755299aa554ace53a177164cc0380f921f94"),
    "wpi": ("dvrk-config/dvrk_config_wpi", "32a8758ce241989ec8dc6d18d375d853ff727d8d"),
}


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


def sparse_checkout(repo: str, commit: str, patterns: list[str], work: Path) -> str:
    """Check out only the files matching `patterns` at `commit`, without downloading the rest of
    the repository. Returns the full commit hash."""
    if work.exists():
        _remove(work)
    work.mkdir(parents=True)
    git("init", "-q", cwd=work)
    git("remote", "add", "origin", f"https://github.com/{repo}.git", cwd=work)
    git("sparse-checkout", "set", "--no-cone", *patterns, cwd=work)
    if len(commit) == 40:
        git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit, cwd=work)
        git("-c", "core.autocrlf=false", "checkout", "-q", "FETCH_HEAD", cwd=work)
    else:                                   # an abbreviated hash cannot be fetched by itself
        git("fetch", "-q", "--filter=blob:none", "origin", cwd=work)
        git("-c", "core.autocrlf=false", "checkout", "-q", commit, cwd=work)
    done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=work, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def fetch_docs(commit: str = DOCS_COMMIT, folder: str = "docs") -> dict:
    work = OUT / "_docs_checkout"
    full = sparse_checkout(DOCS_REPO, commit, ["*.rst"], work)
    dest = OUT / folder
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
    return {"repository": DOCS_REPO, "commit": full, "files": len(files),
            "bytes": sum(f["bytes"] for f in files), "list": files}


def fetch_site_notes() -> dict:
    """Notes that dVRK sites keep beside their configuration files: README and note files only.
    The release each site's files are written for is read from the repository's CMakeLists.txt."""
    import re
    summary = {}
    for site, (repo, commit) in SITE_REPOS.items():
        work = OUT / "_site_checkout"
        full = sparse_checkout(repo, commit, ["*.md", "note*.txt", "/CMakeLists.txt"], work)
        dest = OUT / "sites" / site
        if dest.exists():
            shutil.rmtree(dest)
        files, release = [], ""
        cmake = work / "CMakeLists.txt"
        if cmake.exists():
            m = re.search(r"project\s*\([^)]*VERSION\s+(\d+\.\d+\.\d+)", cmake.read_text(encoding="utf-8", errors="replace"))
            release = m.group(1) if m else ""
        for p in sorted(work.rglob("*")):
            if ".git" in p.parts or not p.is_file() or p.name == "CMakeLists.txt":
                continue
            rel = p.relative_to(work)
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            data = p.read_bytes()
            (dest / rel).write_bytes(data)
            files.append({"path": rel.as_posix(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        _remove(work)
        summary[site] = {"repository": repo, "commit": full, "release": release, "files": len(files),
                         "bytes": sum(f["bytes"] for f in files), "list": files}
        print(f"  site {site}: {len(files)} files, {summary[site]['bytes']} bytes, written for release {release or '?'}")
    return summary


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


def thread_record(repo: str, i: dict, comments: list[dict], opener) -> dict:
    """What is kept of a thread. No author names: only the author's association with the repository."""
    return {
        "repo": repo, "number": i["number"], "title": i["title"], "state": i["state"],
        "state_reason": i.get("state_reason"), "created_at": i["created_at"], "closed_at": i.get("closed_at"),
        "updated_at": i["updated_at"], "labels": [lb["name"] for lb in i.get("labels", [])],
        "author_association": i.get("author_association"), "body": i.get("body") or "",
        "comments": [{"id": c["id"], "created_at": c["created_at"],
                      "author_association": c.get("author_association"),
                      "by_opener": (c.get("user") or {}).get("id") == opener,
                      "body": c.get("body") or ""} for c in comments]}


def fetch_issues(since: dict[str, str] | None = None) -> dict:
    """All threads, or, with `since` (repository -> time of the newest update already stored),
    only the threads updated after that time, merged into what is stored."""
    dest = OUT / "issues"
    dest.mkdir(parents=True, exist_ok=True)
    summary = {}
    for repo in ISSUE_REPOS:
        name = repo.split("/")[1]
        stored_path = dest / f"{name}.json"
        after = (since or {}).get(repo)
        if after and stored_path.exists():
            raw = paged(f"https://api.github.com/repos/{repo}/issues?state=all&since={after}")
            updated = [i for i in raw if "pull_request" not in i and i["updated_at"] > after]
            stored = {t["number"]: t for t in json.loads(stored_path.read_text(encoding="utf-8"))}
            for i in updated:
                opener = (i.get("user") or {}).get("id")
                cs = paged(f"https://api.github.com/repos/{repo}/issues/{i['number']}/comments")
                stored[i["number"]] = thread_record(repo, i, cs, opener)
            threads = [stored[n] for n in sorted(stored)]
            stored_path.write_text(json.dumps(threads, ensure_ascii=False), encoding="utf-8")
            summary[repo] = {"threads": len(threads), "closed": sum(t["state"] == "closed" for t in threads),
                             "comments": sum(len(t["comments"]) for t in threads),
                             "latest_update": max((t["updated_at"] for t in threads), default=None),
                             "updated_since": after, "threads_updated": sorted(i["number"] for i in updated)}
            print(f"  {repo}: {len(updated)} threads updated since {after}")
            continue
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
            threads.append(thread_record(repo, i, by_issue.get(n, []), opener))
        stored_path.write_text(json.dumps(threads, ensure_ascii=False), encoding="utf-8")
        summary[repo] = {"threads": len(threads), "closed": sum(t["state"] == "closed" for t in threads),
                         "comments": sum(len(t["comments"]) for t in threads),
                         "latest_update": max((t["updated_at"] for t in threads), default=None)}
        print(f"  {repo}: {summary[repo]}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-issues", action="store_true")
    ap.add_argument("--skip-docs", action="store_true")
    ap.add_argument("--only", choices=["sites", "docs-previous", "issues-since-last"],
                    help="fetch one thing: the sites' notes, the earlier documentation commit used to measure "
                         "incremental sync, or only the threads updated since the last fetch")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["fetched"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        if a.only == "sites":
            manifest["sites"] = fetch_site_notes()
        elif a.only == "docs-previous":
            manifest["docs_previous"] = fetch_docs(DOCS_PREVIOUS_COMMIT, "docs_previous")
            print(f"earlier documentation: {manifest['docs_previous']['files']} files at "
                  f"{manifest['docs_previous']['commit'][:7]}")
        elif a.only == "issues-since-last":
            last = {r: v["latest_update"] for r, v in manifest.get("issues", {}).get("repositories", {}).items()}
            manifest["issues"] = {"fetched": manifest["fetched"], "repositories": fetch_issues(since=last)}
        if a.only:
            manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            return 0
        if not a.skip_docs:
            manifest["docs"] = fetch_docs()
            print(f"documentation: {manifest['docs']['files']} files, {manifest['docs']['bytes'] / 1e6:.2f} MB")
            manifest["changelog"] = fetch_changelog()
            print(f"change log: {manifest['changelog']['bytes'] / 1e3:.1f} kB")
            manifest["sites"] = fetch_site_notes()
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
