"""The write path, stage 2: one vector per table summary, per embedding model.

    python -m vxfind.ingest.embed --snapshot old --model nomic-embed-text

Vectors are cached on disk by the hash of the text they were computed from, so a snapshot that
shares most of its tables with the previous one only pays for what changed. The models are
served locally (Ollama); nothing is sent off this machine.

Cache layout: data/vxfind/emb/<model>/keys.json (text hash -> row) and vectors.npy (float32).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "vxfind"
OLLAMA = "http://127.0.0.1:11434"
# Some models are trained with a task prefix on each side; leaving it out costs accuracy.
PREFIX = {"nomic-embed-text": ("search_document: ", "search_query: ")}
DIMS = {"nomic-embed-text": 768, "bge-m3": 1024}
MAX_CHARS = 2000


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def text_key(model: str, text: str, side: str = "doc") -> str:
    return hashlib.sha256(f"{model}|{side}|{text}".encode("utf-8", "replace")).hexdigest()[:24]


class Embedder:
    def __init__(self, model: str, folder: Path | None = None, batch: int = 128, url: str = OLLAMA, load: bool = True):
        self.model, self.batch, self.url = model, batch, url
        self.folder = (folder or OUT / "emb") / model.replace(":", "_")
        self.folder.mkdir(parents=True, exist_ok=True)
        keys = self.folder / "keys.json"
        # load=False: for query vectors only; the stored document vectors are not read.
        self.keys: dict[str, int] = json.loads(keys.read_text(encoding="utf-8")) if load and keys.exists() else {}
        vec = self.folder / "vectors.npy"
        self.vectors = np.load(vec) if load and vec.exists() else np.zeros((0, DIMS.get(model, 0)), dtype=np.float32)
        self._client = httpx.Client(timeout=600)
        self.computed = 0
        self.seconds = 0.0

    def _call(self, texts: list[str]) -> np.ndarray:
        t0 = time.perf_counter()
        r = self._client.post(f"{self.url}/api/embed", json={"model": self.model, "input": texts, "truncate": True})
        r.raise_for_status()
        self.seconds += time.perf_counter() - t0
        out = np.asarray(r.json()["embeddings"], dtype=np.float32)
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm == 0, 1.0, norm)

    def query(self, text: str) -> np.ndarray:
        """A query vector. Not cached on disk: queries are short and rarely repeat."""
        return self._call([PREFIX.get(self.model, ("", ""))[1] + text[:MAX_CHARS]])[0]

    def documents(self, texts: list[str], save_every: int = 20000) -> np.ndarray:
        """Vectors for the texts, in order; only the ones not in the cache are computed."""
        prefix = PREFIX.get(self.model, ("", ""))[0]
        keys = [text_key(self.model, t[:MAX_CHARS]) for t in texts]
        todo: dict[str, str] = {}
        for k, t in zip(keys, texts):
            if k not in self.keys and k not in todo:
                todo[k] = prefix + t[:MAX_CHARS]
        items = list(todo.items())
        new: list[np.ndarray] = []
        for i in range(0, len(items), self.batch):
            chunk = items[i:i + self.batch]
            new.append(self._call([t for _, t in chunk]))
            self.computed += len(chunk)
            if (i // self.batch) % 50 == 49:
                log(f"  {self.model}: {self.computed}/{len(items)} new vectors, {self.computed / max(self.seconds, 1e-9):.0f} per second")
        if new:
            add = np.concatenate(new)
            base = len(self.vectors)
            self.vectors = np.concatenate([self.vectors, add]) if base else add
            for j, (k, _) in enumerate(items):
                self.keys[k] = base + j
            self.save()
        return self.vectors[[self.keys[k] for k in keys]]

    def save(self) -> None:
        np.save(self.folder / "vectors.npy", self.vectors)
        (self.folder / "keys.json").write_text(json.dumps(self.keys), encoding="utf-8")


def embed_snapshot(snapshot: str, model: str) -> dict[str, Any]:
    import pyarrow.parquet as pq
    t = pq.read_table(OUT / snapshot / "tables.parquet", columns=["table_id", "summary"])
    ids, texts = t.column("table_id").to_pylist(), t.column("summary").to_pylist()
    emb = Embedder(model)
    cached_before = len(emb.keys)
    t0 = time.time()
    vec = emb.documents(texts)
    out = OUT / snapshot / f"emb_{model.replace(':', '_')}.npy"
    np.save(out, vec)
    (OUT / snapshot / "emb_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    stats = {"snapshot": snapshot, "model": model, "tables": len(ids), "dims": int(vec.shape[1]), "computed": emb.computed,
             "reused_from_cache": len(ids) - emb.computed, "cache_before": cached_before,
             "seconds": round(time.time() - t0, 1), "model_seconds": round(emb.seconds, 1),
             "per_second": round(emb.computed / emb.seconds, 1) if emb.seconds else None,
             "file_mb": round(out.stat().st_size / 1e6, 1)}
    p = OUT / snapshot / "embed_stats.json"
    prior = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    prior[model] = stats
    p.write_text(json.dumps(prior, indent=1), encoding="utf-8")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Embed the table summaries of a snapshot.")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    log(json.dumps(embed_snapshot(a.snapshot, a.model)))
