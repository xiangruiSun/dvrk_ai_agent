"""Shared fixtures. The tests are hermetic: a made-up arm in the dVRK file format, a scripted
model, an in-memory database. No GPU, no network, no downloaded data."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vxagent.robot.kinematics import load_chain
from vxagent.robot.site import build_site
from vxagent.runtime.trace import Tracer
from vxagent.store import Database
from vxagent.tools.builtin import register_builtin
from vxagent.tools.registry import Principal, ToolContext

FIX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def chain():
    return load_chain(FIX / "test_arm.json", FIX / "test_tool.json")


@pytest.fixture()
def site(chain):
    spec = json.loads((ROOT / "sites" / "site_a.json").read_text(encoding="utf-8"))
    spec["named_poses"]["wrist_limit"]["joints_si"] = [0.0, 0.0, 0.12, 0.0, 1.36, 0.0]
    return build_site(spec, chain)


@pytest.fixture()
def operator():
    return Principal("op.test", "operator", frozenset({"robot:read", "robot:move", "episodes:read"}),
                     site="site-a", tiers=frozenset({"observe", "simulate", "actuate"}))


@pytest.fixture()
def env(site, operator):
    """(registry, ctx, twin) with the arm powered but not yet enabled or homed. Control ticks do
    not sleep, so motions finish immediately."""
    db = Database(":memory:")
    twin = site.twin()
    ctx = ToolContext(principal=operator, db=db, tracer=Tracer(db), run_id="run_test", robot=twin, site=site,
                      tick=lambda dt: None)
    return register_builtin(), ctx, twin


@pytest.fixture()
def ready(env):
    """The same, with the arm enabled and homed."""
    reg, ctx, twin = env
    twin.enable()
    twin.home()
    return reg, ctx, twin
