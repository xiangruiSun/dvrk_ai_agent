"""Dataset tools. They need the dataset scope: a site's episodes are the site's, and remote
support has no access to them without a grant (task 13)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..data import episodes as rosma
from ..data import quality
from .registry import Tool, ToolContext, ToolError, ToolRegistry

DATASET = frozenset({"dataset:read"})


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EpisodeArgs(Args):
    episode_id: str = Field(description="Episode id exactly as list_episodes returns it.")


def episode_quality(a: EpisodeArgs, ctx: ToolContext) -> dict[str, Any]:
    ep = rosma.get_episode(a.episode_id, ctx.extras.get("rosma_dir"))
    if ep is None:
        raise ToolError("invalid_arguments", f"no episode {a.episode_id!r}", hint="call list_episodes for the ids")
    limits = (ctx.site.chain.qmin, ctx.site.chain.qmax) if ctx.site is not None else None
    row = quality.check(quality.from_rosma(ep), limits)
    return {k: row[k] for k in ("episode_id", "task", "operator", "samples", "verdict", "flags", "absent")}


def register_dataset(reg: ToolRegistry) -> ToolRegistry:
    reg.register(Tool("episode_quality", "Quality checks of one recorded episode: time stamps, gaps, joint limits, unit, "
                      "jumps, frozen signals. Returns accepted, accepted with flags, or excluded, with the reasons.",
                      EpisodeArgs, episode_quality, scopes=DATASET, timeout_s=60))
    return reg
