"""Dense retrieval with local embedding models served by Ollama, and a persistent cache.

Nothing leaves the machine: the only endpoint called is the local model server. Each vector is
cached under a hash of the model name and the exact text, so re-indexing after a small change
embeds only what changed, and an experiment that reuses a model costs nothing.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

import httpx
import numpy as np

from .config import get_settings

# Some models were trained with a task prefix on each input.
PREFIX = {"nomic-embed-text": ("search_document: ", "search_query: ")}
MAX_CHARS = 6000          # inputs longer than this are embedded on their first 6,000 characters


class EmbeddingError(RuntimeError):
    pass


class Embedder:
    def __init__(self, model: str, cache: Path | str | None = None, url: str | None = None, batch: int = 32):
        s = get_settings()
        self.model, self.batch = model, batch
        self.url = (url or s.ollama_url).rstrip("/")
        self._con = sqlite3.connect(str(cache or s.embedding_cache), check_same_thread=False)
        self._con.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, model TEXT, dim INTEGER, v BLOB)")
        self.stats = {"cached": 0, "embedded": 0, "seconds": 0.0}
        self.read_cache = True          # switched off to time a query that has not been seen before

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\x1f{text}".encode("utf-8")).hexdigest()

    def _call(self, texts: list[str]) -> list[list[float]]:
        host = self.url.split("//", 1)[-1].split("/", 1)[0].split(":")[0]
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise EmbeddingError(f"embedding endpoint {host!r} is not local; refusing to send text to it")
        # The model server loads and unloads models as they are asked for; a request that arrives
        # while it is swapping can fail once. Retry a few times, then fail loudly.
        last = ""
        for attempt in range(4):
            try:
                r = httpx.post(f"{self.url}/api/embed", json={"model": self.model, "input": texts, "truncate": True},
                               timeout=300)
                if r.status_code == 200:
                    break
                last = f"model server returned {r.status_code}: {r.text[:200]}"
            except httpx.HTTPError as exc:
                last = f"cannot reach the model server at {self.url}: {exc}"
            self.stats["retries"] = self.stats.get("retries", 0) + 1
            time.sleep(2.0 * (attempt + 1))
        else:
            raise EmbeddingError(last)
        vectors = r.json().get("embeddings") or []
        if len(vectors) != len(texts):
            raise EmbeddingError(f"asked for {len(texts)} vectors, got {len(vectors)}")
        return vectors

    def embed(self, texts: list[str], query: bool = False) -> np.ndarray:
        doc_prefix, query_prefix = PREFIX.get(self.model, ("", ""))
        prepared = [(query_prefix if query else doc_prefix) + t[:MAX_CHARS] for t in texts]
        keys = [self._key(t) for t in prepared]
        found: dict[str, np.ndarray] = {}
        for i in range(0, len(keys) if self.read_cache else 0, 500):
            part = keys[i:i + 500]
            for k, v in self._con.execute(f"SELECT key, v FROM vectors WHERE key IN ({','.join('?' * len(part))})", part):
                found[k] = np.frombuffer(v, dtype=np.float32)
        todo = [(k, t) for k, t in dict(zip(keys, prepared)).items() if k not in found]
        self.stats["cached"] += len(keys) - len(todo)
        t0 = time.perf_counter()
        for i in range(0, len(todo), self.batch):
            part = todo[i:i + self.batch]
            vectors = self._call([t for _, t in part])
            rows = []
            for (k, _), v in zip(part, vectors):
                arr = np.asarray(v, dtype=np.float32)
                if not np.all(np.isfinite(arr)) or arr.size == 0:
                    raise EmbeddingError("the model returned a vector that is empty or not finite")
                found[k] = arr
                rows.append((k, self.model, int(arr.size), arr.tobytes()))
            with self._con:
                self._con.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)", rows)
        self.stats["embedded"] += len(todo)
        self.stats["seconds"] += time.perf_counter() - t0
        m = np.stack([found[k] for k in keys]) if keys else np.zeros((0, 1), dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1.0, norms)
