"""A whole console on made-up data, for the tests of tasks 11 to 15: two edge nodes with a
made-up arm, a made-up policy rollout, a configuration store on a temporary folder, the fleet
desk, demonstration accounts with known test passwords, and a scripted model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from runbook_bench import make_site
from test_policy_data import contract
from test_policy_trial import DISPLACEMENT_CM, rollout
from vxagent import access
from vxagent.fleet.desk import FleetDesk
from vxagent.fleet.outbox import Outbox
from vxagent.fleet.store import FleetStore
from vxagent.node import Node
from vxagent.store import Database

ROOT = Path(__file__).resolve().parents[1]
CONF = Path(__file__).parent / "fixtures" / "config"
PASSWORDS = {u: f"test-{u}-pw" for u, *_ in access.DEMO_ACCOUNTS}
CAL_POSES = [[20, 0, 0], [-20, 0, 0], [0, 20, 0], [0, -20, 0], [0, 0, 20], [20, 20, 10], [-20, 20, -10], [20, -20, -10],
             [-20, -20, 10], [10, 0, -15], [-10, 10, 15], [0, -10, -20]]


def site_spec(name: str = "site-a") -> dict[str, Any]:
    spec = json.loads((ROOT / "sites" / "site_a.json").read_text(encoding="utf-8"))
    spec["named_poses"]["wrist_limit"]["joints_si"] = [0.0, 0.0, 0.12, 0.0, 1.36, 0.0]
    spec["name"] = name
    spec["calibration_poses_mm"] = CAL_POSES
    return spec


def make_node(tmp: Path, name: str = "site-a", llm: Any = None, db: Database | None = None, fleet: bool = True, **kw: Any) -> Node:
    site = make_site()
    site.name = name
    spec = site_spec(name)
    folder = tmp / name
    folder.mkdir(parents=True, exist_ok=True)
    site_file = folder / "site.json"
    site_file.write_text(json.dumps(spec), encoding="utf-8")
    episodes = folder / "episodes"
    episodes.mkdir(exist_ok=True)
    node = Node(site, spec, db or Database(":memory:"), llm=llm, tick=lambda dt: None, rosma_dir=episodes, **kw)
    # A configuration store on the made-up files, without the dVRK folder layout.
    from vxagent.robot import config as cfg
    from vxagent.structured.rules import InstrumentSet
    node.site_file = site_file
    node.store = cfg.ConfigStore(folder / "config", {node.site_target: site_file, "system/system.json": CONF / "system.json"})
    node.schemas = cfg.SchemaSet(CONF / "schemas")
    node.instruments = InstrumentSet(CONF / "tool")
    c = contract(start_to_goal_min=list(DISPLACEMENT_CM - 0.05), start_to_goal_max=list(DISPLACEMENT_CM + 0.05))
    node.policy = {"contract": c, "source": "made up for the tests", "demos": [rollout(site, c)], "seen": None}
    if fleet:
        node.fleet = Outbox(node.db, name, node.registry, software=spec["software_release"], config_version="0", model="scripted")
    return node


def ready(node: Node) -> Node:
    node.robot.enable()
    node.robot.home()
    return node


def make_console(tmp: Path, llm: Any = None, **kw: Any) -> Any:
    from vxagent.web.app import Console
    users = Database(":memory:")
    access.seed_demo(users, PASSWORDS)
    nodes = {n: make_node(tmp, n, llm=llm) for n in ("site-a", "site-b")}
    desk = FleetDesk(FleetStore(), Database(":memory:"), llm=llm, edges=nodes)
    return Console(users, nodes, desk, secret="test-secret-not-a-default", **kw)


def client_for(console: Any, user_id: str | None = None) -> Any:
    from starlette.testclient import TestClient

    from vxagent.web.app import create_app
    client = TestClient(create_app(console))
    if user_id is not None:
        r = client.post("/api/login", json={"user_id": user_id, "password": PASSWORDS[user_id]})
        assert r.status_code == 200, r.text
    return client


def who(user_id: str) -> Any:
    role, site = next((r, s) for u, r, s, _ in access.DEMO_ACCOUNTS if u == user_id)
    return access.principal_of({"user_id": user_id, "role": role, "site": site, "display_name": user_id})


__all__ = ["CAL_POSES", "PASSWORDS", "client_for", "make_console", "make_node", "np", "ready", "site_spec", "who"]
