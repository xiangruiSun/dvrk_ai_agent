"""A local single-node Elasticsearch for this project, without Docker.

    python scripts/elastic_local.py install     download the official archive, check its SHA-512, unpack, configure
    python scripts/elastic_local.py run         run it in the foreground on 127.0.0.1:9200 (Ctrl-C stops it)
    python scripts/elastic_local.py status      is it answering, which version, which licence

The archive comes from artifacts.elastic.co and is checked against the SHA-512 Elastic publishes
beside it. It is unpacked under data/tools/, which is not committed. The node listens on the
loopback address only, with security switched off: it holds public metadata and nothing else,
and nothing outside this machine can reach it. Do not reuse this configuration anywhere a
network can reach.

Elasticsearch runs on the JDK it ships with. The basic licence it starts with is free; this
project uses nothing that needs another licence (reciprocal rank fusion is done by the caller).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "9.2.1"
NAME = f"elasticsearch-{VERSION}"
ARCHIVE = f"{NAME}-windows-x86_64.zip" if os.name == "nt" else f"{NAME}-linux-x86_64.tar.gz"
BASE = "https://artifacts.elastic.co/downloads/elasticsearch/"
TOOLS = ROOT / "data" / "tools"
HOME = TOOLS / NAME
URL = "http://127.0.0.1:9200"
UA = {"User-Agent": "generic-client/1.0"}
CONFIG = """# Written by scripts/elastic_local.py. Loopback only; see the script's header.
cluster.name: vxfind-local
node.name: local-1
network.host: 127.0.0.1
http.port: 9200
discovery.type: single-node
xpack.security.enabled: false
xpack.security.enrollment.enabled: false
xpack.ml.enabled: false
ingest.geoip.downloader.enabled: false
path.data: {data}
path.logs: {logs}
"""


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def install() -> int:
    TOOLS.mkdir(parents=True, exist_ok=True)
    archive = TOOLS / ARCHIVE
    want = fetch(BASE + ARCHIVE + ".sha512").decode().split()[0]
    if not archive.exists():
        log(f"downloading {ARCHIVE}")
        part = archive.with_suffix(".part")
        with urllib.request.urlopen(urllib.request.Request(BASE + ARCHIVE, headers=UA), timeout=600) as r, part.open("wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        part.replace(archive)
    h = hashlib.sha512()
    with archive.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    if h.hexdigest() != want:
        archive.unlink()
        raise SystemExit(f"{ARCHIVE}: SHA-512 does not match the published one; the file was deleted")
    log(f"{ARCHIVE}: {archive.stat().st_size / 1e6:.0f} MB, SHA-512 matches the published value")
    if not HOME.exists():
        log("unpacking")
        if ARCHIVE.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(TOOLS)
        else:
            import tarfile
            with tarfile.open(archive) as t:
                t.extractall(TOOLS, filter="data")
    data, logs = ROOT / "data" / "es" / "data", ROOT / "data" / "es" / "logs"
    data.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    (HOME / "config" / "elasticsearch.yml").write_text(
        CONFIG.format(data=str(data).replace("\\", "/"), logs=str(logs).replace("\\", "/")), encoding="utf-8")
    opts = HOME / "config" / "jvm.options.d" / "heap.options"
    opts.write_text("-Xms4g\n-Xmx4g\n", encoding="utf-8")
    log(f"installed at {HOME}")
    return 0


def run() -> int:
    exe = HOME / "bin" / ("elasticsearch.bat" if os.name == "nt" else "elasticsearch")
    if not exe.exists():
        raise SystemExit("not installed: run `python scripts/elastic_local.py install` first")
    env = dict(os.environ)
    env.pop("JAVA_HOME", None)                      # use the JDK that ships with Elasticsearch
    env.pop("ES_JAVA_HOME", None)
    return subprocess.call([str(exe)], env=env, cwd=str(HOME))


def status() -> int:
    try:
        info = json.loads(fetch(URL))
        lic = json.loads(fetch(URL + "/_license"))["license"]
        health = json.loads(fetch(URL + "/_cluster/health"))
        print(json.dumps({"version": info["version"]["number"], "lucene": info["version"]["lucene_version"],
                          "licence": lic["type"], "status": health["status"]}))
        return 0
    except Exception as exc:                        # not running is an answer, not an error to trace
        print(json.dumps({"running": False, "why": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    sys.exit({"install": install, "run": run, "status": status}[cmd]())
