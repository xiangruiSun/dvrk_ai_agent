"""Tools of the remote-support assistant. They read the fleet store, which holds counts, kinds
and timings, and nothing of a lab's work. The one way to a lab's material is read_granted_ticket,
which asks the edge node of that site, where the grant is checked."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..tools.registry import Tool, ToolContext, ToolError, ToolRegistry
from . import grants

FLEET = frozenset({"fleet:read"})


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(Args):
    pass


class SiteArgs(Args):
    site: str = Field(description="A site name exactly as fleet_overview lists it.")


class MaybeSiteArgs(Args):
    site: str | None = Field(default=None, description="A site name as fleet_overview lists it; leave out for all sites.")


class GrantedArgs(Args):
    site: str = Field(description="The site the ticket belongs to.")
    ticket_id: int = Field(ge=1)


def _store(ctx: ToolContext, question: str, site: str = "") -> Any:
    store = ctx.extras.get("fleet_store")
    if store is None:
        raise ToolError("failed", "the fleet store is not connected to this session")
    store.audit(ctx.principal.user_id, question, site)                    # the fleet end of the audit: who asked
    return store


def _known(store: Any, site: str | None) -> None:
    if site is not None and site not in store.sites():
        raise ToolError("invalid_arguments", f"no site {site!r}", hint=f"sites: {', '.join(store.sites()) or 'none'}")


def fleet_overview(a: NoArgs, ctx: ToolContext) -> dict[str, Any]:
    return _store(ctx, "fleet_overview").overview()


def site_health(a: SiteArgs, ctx: ToolContext) -> dict[str, Any]:
    store = _store(ctx, "site_health", a.site)
    _known(store, a.site)
    return store.site_health(a.site)


def ops_run_stats(a: MaybeSiteArgs, ctx: ToolContext) -> dict[str, Any]:
    store = _store(ctx, "ops_run_stats", a.site or "")
    _known(store, a.site)
    return store.run_stats(a.site)


def ops_recent_failures(a: MaybeSiteArgs, ctx: ToolContext) -> dict[str, Any]:
    store = _store(ctx, "ops_recent_failures", a.site or "")
    _known(store, a.site)
    return store.recent_failures(a.site)


def version_drift(a: NoArgs, ctx: ToolContext) -> dict[str, Any]:
    return _store(ctx, "version_drift").version_drift()


def read_granted_ticket(a: GrantedArgs, ctx: ToolContext) -> dict[str, Any]:
    _store(ctx, "read_granted_ticket", a.site)
    edge = (ctx.extras.get("edges") or {}).get(a.site)
    if edge is None:
        raise ToolError("failed", f"the edge node of {a.site} cannot be reached", hint="say so; do not guess the ticket")
    try:
        return grants.read_ticket(edge, ctx.principal, a.ticket_id)
    except grants.NotGranted as exc:
        raise ToolError("unauthorized", str(exc), hint="do not retry; a grant has to be asked for from that site's safety lead")


def register_fleet(reg: ToolRegistry) -> ToolRegistry:
    reg.register(Tool("fleet_overview", "Every site: last contact, whether it is healthy, software and configuration "
                      "version, and how many runs and trials it has reported.", NoArgs, fleet_overview, scopes=FLEET, timeout_s=10))
    reg.register(Tool("site_health", "Health of one site's edge node: robot, model server, retrieval, perception, disk, "
                      "pending approvals, outbox backlog.", SiteArgs, site_health, scopes=FLEET, timeout_s=10))
    reg.register(Tool("ops_run_stats", "Per site: runs, model calls, tokens, median model latency, tool errors, trials, "
                      "safety rejections by rule and pre-flight failures by check. Counts only.", MaybeSiteArgs, ops_run_stats,
                      scopes=FLEET, timeout_s=10))
    reg.register(Tool("ops_recent_failures", "Per site: errors by kind, tool errors by tool, and runs that did not "
                      "complete by stop reason. Counts only.", MaybeSiteArgs, ops_recent_failures, scopes=FLEET, timeout_s=10))
    reg.register(Tool("version_drift", "Software, configuration, model and backend version of each site, and where "
                      "they differ.", NoArgs, version_drift, scopes=FLEET, timeout_s=10))
    reg.register(Tool("read_granted_ticket", "Read one ticket of a site, if that site's safety lead has given you a "
                      "grant for it that has not expired or been revoked. Read-only.", GrantedArgs, read_granted_ticket,
                      scopes=FLEET, timeout_s=10))
    return reg
