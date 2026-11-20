"""Settings of the retrieval system. Every value can be overridden with a VXRAG_* environment
variable. This package does not import the agent package: it runs as its own service and the
agents call it over HTTP, so it can be moved to its own repository unchanged."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"VXRAG_{name}", default)


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://127.0.0.1:11434"))
    docs_base_url: str = "https://dvrk.readthedocs.io/main/"

    @property
    def sources_dir(self) -> Path:
        """Text downloaded by scripts/fetch_sources.py."""
        return self.data_dir / "sources"

    @property
    def schemas_dir(self) -> Path:
        """dVRK JSON Schemas downloaded by scripts/fetch_dvrk_config.py."""
        return self.data_dir / "dvrk" / "share" / "schemas"

    @property
    def rag_dir(self) -> Path:
        p = self.data_dir / "rag"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        return self.rag_dir / "corpus.sqlite"

    @property
    def embedding_cache(self) -> Path:
        return self.rag_dir / "embeddings.sqlite"


def get_settings() -> Settings:
    return Settings()
