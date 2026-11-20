"""Memory tools: remember (the only way a user memory is written), recall_site_memory, and
search_similar_episodes. Site facts are written by code after a tool call (see fact_sink), not
by any of these.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..data.similar import EpisodeIndex
from ..memory import site as site_memory
from ..memory import stores
from .registry import Tool, ToolContext, ToolError, ToolRegistry

ROBOT_READ = frozenset({"robot:read"})
EPISODE_READ = frozenset({"episodes:read"})


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RememberArgs(Args):
    text: str = Field(min_length=3, max_length=240, description="One durable preference of this user, as a short statement.")


class RecallArgs(Args):
    subject: str | None = Field(default=None, description="Limit to one subject, for example 'calibration'. Omit for all.")


class SimilarArgs(Args):
    episode_id: str = Field(description="Episode id exactly as list_episodes returns it.")
    k: int = Field(default=3, ge=1, le=5)


def remember(a: RememberArgs, ctx: ToolContext) -> dict[str, Any]:
    try:
        return stores.remember(ctx.db, ctx.principal.user_id, a.text)
    except stores.Refused as exc:
        raise ToolError("refused", str(exc), hint="do not retry; tell the user it was not stored")


def recall_site_memory(a: RecallArgs, ctx: ToolContext) -> dict[str, Any]:
    site = getattr(ctx.site, "name", "") or ctx.principal.site
    try:
        return {"site": site, **site_memory.facts(ctx.db, ctx.principal, site, a.subject)}
    except site_memory.NotVisible as exc:
        raise ToolError("unauthorized", str(exc), hint="do not retry this tool")


def search_similar_episodes(a: SimilarArgs, ctx: ToolContext) -> dict[str, Any]:
    index = ctx.extras.get("episode_index")
    if index is None:
        index = ctx.extras["episode_index"] = EpisodeIndex(ctx.extras.get("rosma_dir"))
    if a.episode_id not in index.ids:
        raise ToolError("invalid_arguments", f"no episode {a.episode_id!r}", hint="call list_episodes for the ids")
    return {"episode_id": a.episode_id, "task": index.task(a.episode_id), "compared_on": "statistics computed from the recordings",
            "neighbours": index.nearest(a.episode_id, a.k)}


def fact_sink(ctx: ToolContext):
    """Returns the hook the executor calls after every successful tool call of this run."""
    def sink(tool: str, result: Any, span: str) -> None:
        site = getattr(ctx.site, "name", "")
        if site:
            site_memory.record_from_result(ctx.db, site, tool, result, source=f"{tool} in run {ctx.run_id}")
    return sink


def register_memory(reg: ToolRegistry) -> ToolRegistry:
    reg.register(Tool("remember", "Store one durable preference of the user you are talking to (how they like answers, "
                      "units, defaults). Not for facts about the robot, and never for passwords or keys.", RememberArgs,
                      remember, side_effect=True, timeout_s=5))
    reg.register(Tool("recall_site_memory", "Facts recorded about this site from earlier measurements, each with its "
                      "source. Expired facts are returned separately with the event that ended them.", RecallArgs,
                      recall_site_memory, scopes=ROBOT_READ, timeout_s=5))
    reg.register(Tool("search_similar_episodes", "Recorded episodes most similar to a given one, by statistics computed "
                      "from the recordings, with the features in which they are closest.", SimilarArgs,
                      search_similar_episodes, scopes=EPISODE_READ, timeout_s=120))
    return reg
