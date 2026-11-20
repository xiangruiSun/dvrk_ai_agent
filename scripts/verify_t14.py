"""Verification of task 14 (edge deployment beside the robot) on this PC.

    python scripts/verify.py t14     ->  verification/t14_edge_deployment.json

The perception worker is started as a separate process. Timing on a Windows desktop is not
real-time: period and jitter are reported as measured and say nothing about a controller. The
ROS 2 backend is not exercised by anything here.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_bench import Bench, ask, log, strip, who                           # noqa: E402
from vxagent import chat                                                   # noqa: E402
from vxagent.config import EgressBlocked, assert_egress_allowed, get_settings   # noqa: E402
from vxagent.fleet.sync import HttpLink, sync                              # noqa: E402
from vxagent.llm.base import LLMError, ToolCall                            # noqa: E402
from vxagent.llm.ollama import OllamaClient, get_llm                       # noqa: E402
from vxagent.llm.router import ROUTE, choose_runbook, label_ticket, route_request   # noqa: E402
from vxagent.perception.client import TIME_LIMIT_S, PerceptionClient, PerceptionUnavailable   # noqa: E402
from vxagent.pipelines import policy_trial as PT                           # noqa: E402
from vxagent.robot.gateway import Gateway, GatewayBackend                  # noqa: E402
from vxagent.robot.timing import TickMeter, model_calls_inside_actions     # noqa: E402
from vxagent.tools.executor import AwaitingApproval, decide_approval, execute_tool   # noqa: E402
from vxagent.tools.knowledge import KnowledgeClient                        # noqa: E402
from vxagent.tools.registry import SYSTEM, ToolContext                     # noqa: E402

SMOKE = bool(os.environ.get("VX_SMOKE"))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_worker(port: int) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-m", "vxagent.perception.worker", "--port", str(port)], cwd=str(ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.5).status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("the perception worker did not start")


def run_trial(node: Any, declared: dict[str, bool] | None = None, use_llm: bool = True) -> Any:
    op = who("op.lee")
    res = node.start_trial(op, 0, declared, use_llm=use_llm)
    if res.status == "waiting":
        decide_approval(node.db, res.waiting_for["approval_id"], True, op.user_id)
        res = node.resume_trial(res.run_id)
        if res.status == "waiting":
            PT.decide_verdict(node.db, res.run_id, op.user_id, "confirm")
            res = node.resume_trial(res.run_id)
    return res


def reset(node: Any) -> None:
    """Back to the state a trial starts from, by code: the arm at start with the jaw open."""
    from vxagent.robot.needle import DeclaredNeedle
    node.robot.set_joints(np.array(node.site.home_q, dtype=float))
    node.robot._jaw = node.site.jaw_angles["open"]
    node.needle = DeclaredNeedle(node.site).attach(node.robot)


def percentiles(xs: list[float]) -> dict[str, float]:
    a = np.array(xs)
    return {"n": int(len(a)), "p50_ms": round(float(np.percentile(a, 50)), 1), "p95_ms": round(float(np.percentile(a, 95)), 1),
            "max_ms": round(float(a.max()), 1)}


def measure(models: dict[str, Any], rows: list[tuple[str, str]], fn: Any) -> dict[str, Any]:
    out = {}
    for name, llm in models.items():
        res = [(fn(llm, text), want) for text, want in rows]
        wrong = [{"text": rows[i][0][:80], "expected": w, "chose": r["choice"]} for i, (r, w) in enumerate(res) if r["choice"] != w]
        out[name] = {"cases": len(res), "correct": len(res) - len(wrong), "no_answer": sum(r["choice"] is None for r, _ in res),
                     "median_ms": round(float(np.median([r["ms"] for r, _ in res])), 1),
                     "median_tokens": float(np.median([r["tokens"] for r, _ in res])), "wrong": wrong}
    return out


def t14() -> dict[str, Any]:
    s = get_settings()
    out: dict[str, Any] = {"task": "t14_edge_deployment" + ("_smoke" if SMOKE else ""), "ran": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "models": {"large": s.model, "small": s.small_model}, "backend": "twin",
                           "not_verified_here": "the ros2 backend (vxagent/robot/ros2_backend.py): no ROS 2, no dVRK software and no arm on this PC",
                           "timing_note": "a Windows desktop is not a real-time system; period and jitter are as measured"}
    port = free_port()
    worker = start_worker(port)
    meter = TickMeter(time.sleep)
    bench = Bench("t14", sites=("site_a",), tick=meter)
    node = bench.nodes["site-a"]
    node.robot.enable()
    node.robot.home()
    client = PerceptionClient(f"http://127.0.0.1:{port}")
    node.perception = client
    try:
        # ---------------------------------------------------------------- a. the control loop
        rows = []
        plan = [{}, {"slips_on_lift": True}] if SMOKE else [{}, {}, {"slips_on_lift": True}, {}, {"slips_while_holding": True}, {}]
        for i, declared in enumerate(plan):
            reset(node)
            t0 = time.time()
            res = run_trial(node, declared)
            inside = model_calls_inside_actions(node.db, res.run_id)
            llm_ms = sum(x["latency_ms"] for x in node.db.query("SELECT latency_ms FROM spans WHERE run_id=? AND kind='llm'", (res.run_id,)))
            tok = node.db.one("SELECT COALESCE(SUM(prompt_tokens+completion_tokens),0) AS n FROM spans WHERE run_id=? AND kind='llm'", (res.run_id,))["n"]
            rows.append({"trial": i + 1, "needle_declared": declared or "holds", "outcome": res.state["data"].get("outcome"),
                         "actions": inside["actions"], "model_calls": inside["model_calls"],
                         "model_calls_inside_an_action": inside["model_calls_inside_an_action"],
                         "model_ms": round(llm_ms, 1), "tokens": tok, "seconds": round(time.time() - t0, 1)})
            log(f"a trial {i + 1}: {rows[-1]['outcome']}, {inside['actions']} actions, model calls inside an action "
                f"{inside['model_calls_inside_an_action']} of {inside['model_calls']}")
        out["a_control_loop"] = {"trials": len(rows), "control_rate_hz": node.robot.rate_hz, "period": meter.report(1.0 / node.robot.rate_hz),
                                 "model_calls_inside_an_action_in_all_trials": sum(r["model_calls_inside_an_action"] for r in rows),
                                 "model_calls_in_all_trials": sum(r["model_calls"] for r in rows),
                                 "perception_requests": percentiles(client.latencies_ms) if client.latencies_ms else None, "rows": rows}

        # ---------------------------------------------------------------- b. the perception worker
        lat = PerceptionClient(f"http://127.0.0.1:{port}")
        for _ in range(20 if SMOKE else 200):
            lat()
        first = lat()
        httpx.post(f"http://127.0.0.1:{port}/control", json={"delay_ms": 700})          # INJECTED slowness
        t0 = time.perf_counter()
        try:
            lat()
            slow = "answered"
        except PerceptionUnavailable as exc:
            slow = f"unavailable: {exc}"
        slow_s = round(time.perf_counter() - t0, 2)
        reset(node)
        pre_slow = node.start_trial(who("op.lee"), 0, use_llm=False)
        httpx.post(f"http://127.0.0.1:{port}/control", json={"delay_ms": 0})
        time.sleep(1.0)

        # the worker is stopped while a trial is in the place stage, and (separately) in the grasp stage
        during = {}
        for stage in ("place", "grasp"):
            if worker.poll() is not None:
                worker = start_worker(port)
            reset(node)
            calls = {"n": 0}
            order = {1: "preflight", 2: "grasp", 3: "place"}

            def look(stage: str = stage, calls: dict[str, int] = calls) -> dict[str, Any]:
                calls["n"] += 1
                if order.get(calls["n"]) == stage:
                    worker.kill()                              # the process is terminated before this stage's request
                    worker.wait(timeout=10)
                return client()
            node.perception = look
            res = run_trial(node, use_llm=False)
            d = res.state["data"]
            during[stage] = {"worker_alive_after": worker.poll() is None, "outcome": d.get("outcome"),
                             "stages_run": [h["node"] for h in res.state["history"]],
                             "grasp_stopped_for": (d.get("grasp") or {}).get("reason"),
                             "needle_pose_in_place": (d.get("place") or {}).get("needle_pose_source"),
                             "machine_verdict": (d.get("verdict") or {}).get("machine")}
            log(f"b worker stopped during {stage}: {during[stage]}")
        node.perception = client
        reset(node)
        health_down = client.health()
        pre_down = node.start_trial(who("op.lee"), 0, use_llm=False)                     # the worker is still stopped
        failed = (pre_down.state["data"].get("preflight") or {}).get("failed") or []
        out["b_perception_worker"] = {
            "process": "a separate Python process reached over HTTP on this machine; it serves recorded poses and computes nothing",
            "source": first.get("source"), "time_limit_s": TIME_LIMIT_S, "latency": percentiles(lat.latencies_ms),
            "a_request_slower_than_the_limit_INJECTED": {"extra_ms": 700, "result": slow[:120], "gave_up_after_s": slow_s,
                                                          "pre_flight_then": pre_slow.state["data"].get("outcome"),
                                                          "motions_commanded": len(node.db.query("SELECT 1 FROM commands WHERE run_id=?", (pre_slow.run_id,)))},
            "worker_stopped_before_pre_flight": {"health": health_down, "outcome": pre_down.state["data"].get("outcome"),
                                                  "failed_checks": [f["check"] for f in failed], "repair": sorted({f["repair"] for f in failed}),
                                                  "motions_commanded": len(node.db.query("SELECT 1 FROM commands WHERE run_id=?", (pre_down.run_id,)))},
            "worker_stopped_during_a_stage": during}
        worker = start_worker(port)
        node.perception = client

        # ---------------------------------------------------------------- c. degraded modes
        deg: dict[str, Any] = {}
        good_llm = node.llm
        dead = OllamaClient(base_url=f"http://127.0.0.1:{free_port()}")                  # nothing listens there
        node.llm = dead
        reset(node)
        res = run_trial(node)
        rb = execute_tool(node.registry, ToolCall("r", "run_runbook", {"name": "rig_check", "parameters": {}}), node.ctx(who("op.lee"), node.new_run_id()))
        cop = ask(node, "op.lee", "Is the arm ready?")
        deg["model_server_down"] = {"model_health": dead.health(), "trial_outcome": res.state["data"].get("outcome"),
                                    "trial_summary_written_by": res.state["data"].get("summary_source"),
                                    "runbook_rig_check": rb.content["status"] if rb.ok else rb.error_kind,
                                    "copilot_answer": cop["answer"], "copilot_says_it_is_unavailable": cop["answer"] == chat.UNAVAILABLE}
        node.llm = good_llm
        log(f"c model down: trial {deg['model_server_down']['trial_outcome']}, summary {deg['model_server_down']['trial_summary_written_by']}")

        deg["perception_worker_down"] = {"see": "b_perception_worker.worker_stopped_before_pre_flight",
                                         "pre_flight_fails_closed": out["b_perception_worker"]["worker_stopped_before_pre_flight"]["outcome"] == "preflight_failed"}

        real_knowledge = node.knowledge
        node.knowledge = KnowledgeClient(base_url=f"http://127.0.0.1:{free_port()}", api_key="none", timeout_s=2.0)
        robot_q = ask(node, "op.lee", "Is the arm enabled and homed?")
        doc_q = ask(node, "op.lee", "How are the potentiometers of a PSM calibrated? Look it up in the documentation.")
        deg["retrieval_missing"] = {"robot_question": strip(robot_q), "documentation_question": strip(doc_q),
                                    "robot_question_answered_from_a_look": "get_robot_state" in robot_q["tools_called"],
                                    "search_failed_as_a_tool_error": any(t["tool"] == "search_docs" and not t["ok"] for t in doc_q["tools"]),
                                    "answer_says_the_search_is_unavailable": any(w in doc_q["answer"].lower() for w in (
                                        "could not be searched", "unavailable", "not available", "could not", "unable", "cannot", "failed"))}
        node.knowledge = real_knowledge
        log(f"c retrieval missing: {doc_q['tools_called']} -> {deg['retrieval_missing']['answer_says_the_search_is_unavailable']}")

        backlog0 = node.fleet.backlog()
        link = HttpLink(f"http://127.0.0.1:{free_port()}", "no-fleet", timeout_s=1.0)
        reset(node)
        res = run_trial(node, use_llm=False)
        node.fleet.record_health(node.health())
        down = sync(node.fleet, link)
        deg["fleet_link_down"] = {"trial_outcome": res.state["data"].get("outcome"), "sync": down, "outbox_before": backlog0,
                                  "outbox_after": node.fleet.backlog(), "outbox_grew": node.fleet.backlog() > backlog0}
        out["c_degraded_modes"] = deg

        # ---------------------------------------------------------------- d. model routing
        models = {"large": get_llm(s.model), "small": get_llm(s.small_model)}
        spec = json.loads((ROOT / "evals" / "t04_thread_labels.json").read_text(encoding="utf-8"))
        threads = {}
        for f in (s.data_dir / "sources" / "issues").glob("*.json"):
            for t in json.loads(f.read_text(encoding="utf-8")):
                threads[(t["repo"].split("/")[1], int(t["number"]))] = t
        labelled = [(f"{threads[(r, n)]['title']}\n{(threads[(r, n)]['body'] or '')[:1500]}", lab) for r, n, lab, _ in spec["threads"] if (r, n) in threads]
        routes = [tuple(x) for x in json.loads((ROOT / "evals" / "t14_route_requests.json").read_text(encoding="utf-8"))["requests"]]
        books = [tuple(x) for x in json.loads((ROOT / "evals" / "t14_runbook_requests.json").read_text(encoding="utf-8"))["requests"]]
        index = node.library.index()
        if SMOKE:
            labelled, routes, books = labelled[:3], routes[:4], books[:3]
        for m in models.values():                                 # load each model once, outside the timing
            route_request(m, "warm up")
        routing = {"label_ticket": {"data": "the 20 labelled dVRK issue threads of task 04 (title and opening post)", "classes": len(spec["classes"]),
                                    **measure(models, labelled, lambda m, t: label_ticket(m, t, spec["classes"]))},
                   "choose_runbook": {"data": "evals/t14_runbook_requests.json, over the index of all runbooks", "classes": len(index),
                                      **measure(models, books, lambda m, t: choose_runbook(m, t, index))},
                   "route_request": {"data": "evals/t14_route_requests.json, hand-labelled", "classes": 4,
                                     **measure(models, routes, route_request)}}
        for step, r in routing.items():
            r["route_in_code"] = ROUTE[step]
            log(f"d {step}: large {r['large']['correct']}/{r['large']['cases']} at {r['large']['median_ms']} ms, "
                f"small {r['small']['correct']}/{r['small']['cases']} at {r['small']['median_ms']} ms")
        out["d_model_routing"] = routing

        # ---------------------------------------------------------------- e. egress guard
        os.environ["VXAGENT_PROFILE"] = "onprem"
        os.environ["VXAGENT_ALLOWED_HOSTS"] = "perception.lab,fleet.lab"
        table = []
        for url, want in (("http://127.0.0.1:11434/api/chat", "allowed"), ("http://localhost:8300/api/robot", "allowed"),
                          ("http://perception.lab:8401/estimate", "allowed"), ("https://fleet.lab/v1/ingest", "allowed"),
                          ("https://api.example.com/v1/chat", "blocked"), ("http://perception.lab.example.org/estimate", "blocked"),
                          ("http://192.168.1.20:11434/api/chat", "blocked")):
            try:
                assert_egress_allowed(url)
                got = "allowed"
            except EgressBlocked:
                got = "blocked"
            table.append({"url": url, "expected": want, "got": got, "match": want == got})
        from vxagent import net
        sent_before = net.sent()                              # the process's one HTTP client counts what it sends
        try:
            try:
                OllamaClient(base_url="https://llm.example.com").chat([{"role": "user", "content": "a lab's unpublished result"}])
                external = "sent"
            except LLMError as exc:
                external = f"failed before sending: {exc.kind}"
            try:
                PerceptionClient("http://gpu.example.com:8401")()
                ext_perc = "sent"
            except PerceptionUnavailable:
                ext_perc = "unavailable, nothing sent"
            ext_fleet = sync(node.fleet, HttpLink("https://fleet.example.com", "t"))["why"]
        finally:
            sent = net.sent() - sent_before
        os.environ.pop("VXAGENT_ALLOWED_HOSTS", None)
        out["e_egress_guard"] = {"profile": "onprem", "allowlist": ["perception.lab", "fleet.lab"], "cases": len(table),
                                 "match": sum(t["match"] for t in table), "rows": table,
                                 "model_client_pointed_at_an_external_host": external, "perception_client_at_an_external_host": ext_perc,
                                 "fleet_link_at_an_external_host": ext_fleet[:80], "requests_that_reached_the_network_layer": sent}
        log(f"e egress: {out['e_egress_guard']['match']}/{len(table)}; external model: {external}; requests sent {sent}")

        # ---------------------------------------------------------------- f. the gateway's allowlists
        twin = node.site.twin()
        gw = Gateway(twin, "agent-process-token").start()
        closed = Gateway(node.site.twin(), "agent-process-token", command_hosts=("10.0.0.5",), state_hosts=("10.0.0.5",)).start()
        try:
            remote = GatewayBackend(gw.url, "agent-process-token", node.site)
            remote.enable()
            remote.home()
            ctx = ToolContext(principal=SYSTEM, db=node.db, tracer=node.tracer, run_id=node.new_run_id(), robot=remote, site=node.site)
            seq = execute_tool(node.registry, ToolCall("l", "get_robot_state", {}), ctx).content["state_seq"]
            call = ToolCall("m", "move_to_named_pose", {"named_pose": "hover_needle", "based_on_seq": seq})
            try:
                execute_tool(node.registry, call, ctx)
            except AwaitingApproval as wait:
                decide_approval(node.db, wait.approval_id, True, "op.lee")
            moved = execute_tool(node.registry, call, ctx)
            wrong_token = httpx.post(gw.url + "/command", json={"op": "disable"}, headers={"Authorization": "Bearer someone-else"}).status_code
            state_unlisted = httpx.get(closed.url + "/state").status_code
            cmd_unlisted = httpx.post(closed.url + "/command", json={"op": "enable"}, headers={"Authorization": "Bearer agent-process-token"}).status_code
            out["f_gateway"] = {
                "through_the_gateway": {"backend_name": remote.name, "motion": "succeeded" if moved.ok else moved.error_kind,
                                        "final_error_mm": (moved.content or {}).get("final_error_mm") if moved.ok else None,
                                        "arm_behind_the_gateway_moved": twin.state().seq > 3},
                "command_with_another_token": {"status": wrong_token, "arm_still_enabled": twin.state().enabled},
                "from_a_host_not_on_the_allowlists": {"peer_address": "127.0.0.1", "allowlist": ["10.0.0.5"], "state_request": state_unlisted,
                                                      "command": cmd_unlisted, "arm_enabled_by_it": closed.backend.state().enabled},
                "refusals_recorded_by_the_gateways": [r["why"] for r in gw.refused + closed.refused]}
            log(f"f gateway: motion {out['f_gateway']['through_the_gateway']['motion']}, other token {wrong_token}, unlisted host {state_unlisted}/{cmd_unlisted}")
        finally:
            gw.stop()
            closed.stop()

        # ---------------------------------------------------------------- g. sizing
        chat_run = ask(node, "op.lee", "Is the arm ready, and how high is the tip over the pad?")
        full = [r for r in out["a_control_loop"]["rows"] if r["outcome"] == "success"]
        out["g_sizing"] = {"per_chat_run": {"model_calls": chat_run["model_calls"], "tokens": chat_run["tokens"], "seconds": chat_run["seconds"]},
                           "per_trial_that_ran_to_the_end": {"trials": len(full), "median_seconds": round(float(np.median([r["seconds"] for r in full])), 1) if full else None,
                                                             "median_model_ms": round(float(np.median([r["model_ms"] for r in full])), 1) if full else None,
                                                             "median_tokens": round(float(np.median([r["tokens"] for r in full])), 1) if full else None},
                           "perception_request_ms": out["b_perception_worker"]["latency"],
                           "control_period_ms": {k: out["a_control_loop"]["period"].get(k) for k in ("nominal_ms", "mean_ms", "p95_ms", "max_ms", "jitter_std_ms")}}
    finally:
        try:
            worker.kill()
        except Exception:
            pass
        bench.close()
    return out
