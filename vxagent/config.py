"""Central settings. Every value can be overridden with a VXAGENT_* environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"VXAGENT_{name}", default)


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    sites_dir: Path = ROOT / "sites"

    # Local models served by Ollama. The large one reasons, the small one classifies (task 14).
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://127.0.0.1:11434"))
    model: str = field(default_factory=lambda: _env("MODEL", "qwen3:14b"))
    small_model: str = field(default_factory=lambda: _env("SMALL_MODEL", "qwen3:4b"))
    num_ctx: int = field(default_factory=lambda: int(_env("NUM_CTX", "8192")))
    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "180")))

    # Deployment profile. "onprem" blocks every outbound call that is not to an allowed host.
    profile: str = field(default_factory=lambda: _env("PROFILE", "onprem"))

    @property
    def dvrk_dir(self) -> Path:
        """Configuration files downloaded from the dVRK project (scripts/fetch_dvrk_config.py)."""
        return self.data_dir / "dvrk"

    @property
    def rosma_dir(self) -> Path:
        """Recorded dVRK kinematics from the ROSMA dataset (scripts/fetch_rosma.py)."""
        return self.data_dir / "rosma"

    @property
    def runtime_dir(self) -> Path:
        p = self.data_dir / "runtime"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        return self.runtime_dir / "vxagent.sqlite"


def get_settings() -> Settings:
    return Settings()


class EgressBlocked(RuntimeError):
    pass


def assert_egress_allowed(url: str) -> None:
    """In the on-prem profile every outbound call must stay on this host or on a lab-network name
    listed in VXAGENT_ALLOWED_HOSTS. A lab's recordings never leave the site by accident."""
    from urllib.parse import urlparse
    if get_settings().profile != "onprem":
        return
    host = (urlparse(url).hostname or "").lower()
    allowed = {"127.0.0.1", "localhost", "::1"} | {
        h.strip().lower() for h in _env("ALLOWED_HOSTS", "").split(",") if h.strip()}
    if host not in allowed:
        raise EgressBlocked(f"on-prem profile: outbound call to {host!r} is not allowed")
