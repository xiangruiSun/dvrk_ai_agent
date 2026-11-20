"""Verification of task 06 (memory and technical retrieval, the parts that are the agents').

    python scripts/verify.py t06     ->  verification/t06_memory_and_retrieval.json

Retrieval itself is measured in the retrieval package (verification/rag_*.json). Here: the
same index through the agents' tools for two kinds of site, refusal of questions about other
systems, similar recorded episodes, site memory that expires, and a long run under a small
context budget.
"""
from __future__ import annotations

import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knowledge_service import start_knowledge                              # noqa: E402
from vxagent.agent.guards import _CITE, grounding_check, nearly            # noqa: E402
from vxagent.agent.loop import AgentConfig, run_agent, tool_messages       # noqa: E402
from vxagent.config import get_settings                                    # noqa: E402
from vxagent.data import episodes as ep_store                              # noqa: E402
from vxagent.data.similar import FEATURES, EpisodeIndex                    # noqa: E402
from vxagent.diagnostics.episode import calibration                        # noqa: E402
from vxagent.llm.base import ToolCall, messages_tokens                     # noqa: E402
from vxagent.llm.ollama import get_llm                                     # noqa: E402
from vxagent.memory import site as site_memory                             # noqa: E402
from vxagent.profiles import DEMO, KNOWLEDGE, OPERATOR, principal          # noqa: E402
from vxagent.robot.site import load_site                                   # noqa: E402
from vxagent.runtime.trace import Tracer                                   # noqa: E402
from vxagent.store import Database                                         # noqa: E402
from vxagent.structured import proposals                                   # noqa: E402
from vxagent.tools.builtin import register_builtin                         # noqa: E402
from vxagent.tools.executor import execute_tool                            # noqa: E402
from vxagent.tools.knowledge import register_knowledge                     # noqa: E402
from vxagent.tools.memory import fact_sink, register_memory                # noqa: E402
from vxagent.tools.registry import ToolContext                             # noqa: E402
from vxagent.tools.service import register_service, store_calibration_record   # noqa: E402

SMOKE = bool(os.environ.get("VX_SMOKE"))
ANSWER = re.compile(r"ANSWER:\s*(-?\d+(?:[.,]\d+)?)", re.I)
OTHER_SYSTEMS = [
    "How do I calibrate the tool centre point on a UR5 teach pendant?",
    "What is the payload limit of a KUKA LBR iiwa 14?",
    "How do I home a Franka Emika Panda after an emergency stop?",
    "Which firmware does the Kinova Gen3 arm need for torque control?",
    "How do I reset the collision detection on an ABB YuMi?",
    "What is the baud rate of the Dynamixel MX-64 bus?",
    "How do I enable freedrive mode on a UR10e?",
    "Where is the brake release on a Fanuc CRX-10iA?",
    "How do I update the controller of a Sawyer robot from Rethink Robotics?",
    "What is the repeatability of a Stäubli TX2-60?",
]
DVRK_CONTROLS = [
    "How are the potentiometers of a PSM calibrated?",
    "What does the emergency stop chain of the dVRK controllers look like?",
    "How is the base frame of an arm set in the system configuration file?",
    "What is the tool detection setting of a PSM?",
]
_REFUSAL = re.compile(r"do(es)? not answer|cannot find|can't find|unable to find|nothing (relevant )?(was )?found|no (relevant )?(information|passage|result|documentation)|not (covered|found|"
                      r"available|related)|could not find|couldn't find|do(es)? not (cover|contain|mention|address)|unrelated|"
                      r"no mention", re.I)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def t06() -> dict[str, Any]:
    s = get_settings()
    site = load_site("site_a")
    llm = get_llm()
    db = Database()
    tracer = Tracer(db)
    out: dict[str, Any] = {"task": "t06_memory_and_retrieval" + ("_smoke" if SMOKE else ""),
                           "ran": time.strftime("%Y-%m-%d %H:%M:%S"), "model": s.model, "site": site.name}
    registry = register_memory(register_knowledge(register_service(register_builtin())))

    def ctx_for(who: Any, robot: bool = True, **extras: Any) -> ToolContext:
        twin = None
        if robot:
            twin = site.twin()
            twin.enable()
            twin.home()
        ctx = ToolContext(principal=who, db=db, tracer=tracer, robot=twin, site=site, run_id=f"run_{uuid.uuid4().hex[:12]}")
        ctx.extras.update(extras)
        return ctx

    def ask(cfg: AgentConfig, task: str, ctx: ToolContext) -> Any:
        tracer.start_run(cfg.name, task, ctx.principal.user_id, ctx.principal.role, site.name, run_id=ctx.run_id)
        state = run_agent(llm, registry, ctx, task, cfg)
        tracer.end_run(ctx.run_id, state.status, state.answer)
        return state

    # ---------------------------------------------------------------- b. the site decides what a search returns
    from vxrag.evaluation.labels import build_queries
    from vxrag.store import PUBLIC_SOURCES, CorpusStore
    units = {u.unit_id: u for u in CorpusStore().units(PUBLIC_SOURCES)}
    dev = [q for q in build_queries(list(units.values())) if q.split == "dev"]
    dev = dev[::max(1, len(dev) // (12 if SMOKE else 80))][:(12 if SMOKE else 80)]

    class Other:                                         # the same lab, had it an Si system
        name, generation, software_release = site.name, "Si", site.software_release

    who = principal("op.verify", "operator", site.name)
    per_site = {}
    returned: dict[str, list[set[str]]] = {}
    for label, as_site in (("Classic", site), ("Si", Other)):
        client, server = start_knowledge(as_site)
        ctx = ctx_for(who, robot=False, knowledge=client)
        n, flagged, sets, wrong_gen = 0, 0, [], 0
        other_gen = "Si" if label == "Classic" else "Classic"
        for q in dev:
            res = execute_tool(registry, ToolCall(f"q{n}", "search_docs", {"query": q.text[:300], "k": 5}), ctx)
            ids = set()
            for p in (res.content or {}).get("results", []) if res.ok else []:
                n += 1
                ids.add(p["id"])
                flagged += int(p.get("applies") is False)
                wrong_gen += int(units[p["id"]].generation == other_gen) if p["id"] in units else 0
            sets.append(ids)
        server.stop()
        returned[label] = sets
        per_site[label] = {"questions": len(dev), "passages_returned": n,
                           "passages_written_for_the_other_generation": wrong_gen,
                           "passages_marked_as_not_applying": flagged}
    differ = sum(a != b for a, b in zip(returned["Classic"], returned["Si"]))
    out["b_site_filter_through_the_tools"] = {
        "tool": "search_docs, called through the agents' tool executor; no model",
        "questions": "development queries of the retrieval evaluation (the locked test set is not touched)",
        "by_site": per_site, "questions_whose_results_differ_between_the_two_sites": differ}
    log(f"b site filter: {per_site}, differ {differ}")

    # ---------------------------------------------------------------- c. questions about other systems
    client, server = start_knowledge(site)
    rows = []
    for kind, questions in (("another system", OTHER_SYSTEMS[:2] if SMOKE else OTHER_SYSTEMS),
                            ("dVRK (control)", DVRK_CONTROLS[:1] if SMOKE else DVRK_CONTROLS)):
        for q in questions:
            ctx = ctx_for(who, robot=False, knowledge=client)
            state = ask(KNOWLEDGE, q, ctx)
            passages = sum(len(m.get("result", {}).get("results", [])) for m in tool_messages(state) if m.get("ok"))
            cited = bool(_CITE.findall(state.answer or ""))
            says_nothing = bool(_REFUSAL.search(state.answer or ""))
            rows.append({"kind": kind, "question": q, "passages_returned_by_the_search": passages, "cites_a_passage": cited,
                         "says_nothing_was_found": says_nothing, "refused": says_nothing and not cited,
                         "stop_reason": state.stop_reason, "answer": state.answer})
            log(f"c {kind}: refused {rows[-1]['refused']} cited {cited} passages {passages} | {q[:50]}")
    other = [r for r in rows if r["kind"] == "another system"]
    ctl = [r for r in rows if r["kind"] != "another system"]
    out["c_questions_about_other_systems"] = {
        "note": "the search has no score threshold: it returns its best passages for any wording, so refusing is "
                "left to the agent and its guard",
        "other_systems": len(other), "refused": sum(r["refused"] for r in other),
        "answered_with_a_citation": sum(r["cites_a_passage"] for r in other),
        "searches_that_returned_passages_anyway": sum(r["passages_returned_by_the_search"] > 0 for r in other),
        "dvrk_controls": len(ctl), "dvrk_controls_answered_with_a_citation": sum(r["cites_a_passage"] for r in ctl),
        "dvrk_controls_wrongly_refused": sum(r["refused"] for r in ctl), "runs": rows}
    server.stop()

    # ---------------------------------------------------------------- d. similar episodes
    index = EpisodeIndex()
    hits, detail = 0, []
    for eid in index.ids:
        nn = index.nearest(eid, 1)[0]
        same = index.task(eid) == nn["task"]
        hits += int(same)
        detail.append({"episode": eid, "nearest": nn["episode_id"], "same_task": same, "distance": nn["distance"],
                       "closest_in": list(nn["closest_in"])})
    tasks = [index.task(e) for e in index.ids]
    chance = sum((tasks.count(t) - 1) / (len(tasks) - 1) for t in tasks) / len(tasks)
    out["d_similar_episodes"] = {"episodes": len(index.ids), "features": list(FEATURES),
                                 "nearest_neighbour_is_the_same_task": f"{hits}/{len(index.ids)}",
                                 "expected_by_chance": round(chance, 3),
                                 "tasks": {t: tasks.count(t) for t in sorted(set(tasks))}, "detail": detail}
    log(f"d similar: {hits}/{len(index.ids)} same task, chance {chance:.2f}")

    # ---------------------------------------------------------------- e. site memory
    import numpy as np
    site_name = f"{site.name}"
    for table in ("site_facts", "site_events"):
        site_memory._ready(db)
        db.execute(f"DELETE FROM {table} WHERE site=?", (site_name,))
    cal = calibration(None, site.scene.workspace_center, np.random.default_rng(31))
    store_calibration_record(db, site.name, {"id": cal["id"], "T_est": cal["T_est"].tolist(), "seen_m": cal["seen_m"].tolist(),
                                             "touched_m": cal["touched_m"].tolist(), "method": "point pairs, camera to robot base"})
    proposals.set_calibration(db, site.name, True, "calibrated (verification)")
    memory_cfg = AgentConfig(name="operator_with_memory", max_iterations=8, token_budget=40000, answer_check=grounding_check,
                             tool_names=["get_robot_state", "read_calibration_record", "calibration_residual",
                                         "recall_site_memory", "calculator"],
                             system_prompt="You assist the operator of a da Vinci Research Kit (dVRK) arm. " + DEMO + "\n"
                             "Rules:\n1. Say only what a tool result or the site memory below supports.\n"
                             "2. A fact marked EXPIRED is not current. If asked about it, say that it has expired, since "
                             "when and because of which event, and that it has to be measured again. Do not give its "
                             "value as the current value.\n3. Use recall_site_memory when asked what is known about the site.\n")

    def with_memory(who_: Any) -> AgentConfig:
        block = site_memory.context_block(db, who_, site.name)
        return AgentConfig(**{**memory_cfg.__dict__, "system_prompt": memory_cfg.system_prompt + ("\n" + block if block else "")})

    eng = principal("eng.kim", "service", site.name)
    ctx = ctx_for(eng)
    ctx.extras["on_result"] = fact_sink(ctx)
    s1 = ask(with_memory(eng), "Measure the residual of the site's calibration record and tell me the mean.", ctx)
    stored = site_memory.facts(db, eng, site.name)
    truth = stored["active"][0]["value"]["mean_mm"] if stored["active"] else None
    ctx2 = ctx_for(who)                                   # another person, another session, nothing to measure with
    recall_only = lambda p: AgentConfig(**{**with_memory(p).__dict__, "tool_names": ["recall_site_memory", "read_calibration_record"]})   # noqa: E731
    q = "What was the mean residual of the camera-to-robot calibration when it was last measured? End with 'ANSWER: <mm>'."
    s2 = ask(recall_only(who), q, ctx2)
    said = ANSWER.findall(s2.answer or "")
    event = site_memory.record_event(db, site.name, "ecm_moved", eng.user_id, "camera arm repositioned (verification)")
    ctx3 = ctx_for(who)
    s3 = ask(recall_only(who), "Is the camera-to-robot calibration residual we have on record still current?", ctx3)
    record_after = execute_tool(registry, ToolCall("x0", "read_calibration_record", {}), ctx3)
    other_op = principal("op.other", "operator", "site-b")
    ctx4 = ctx_for(other_op)
    ctx4.site = type("S", (), {"name": site.name, "scene": site.scene, "profile": site.profile})()
    recall = execute_tool(registry, ToolCall("x1", "recall_site_memory", {}), ctx4)
    support = principal("sup.wong", "support", "")
    out["e_site_memory"] = {
        "written": {"by": "the executor's hook after calibration_residual returned; no model text is stored",
                    "tools_called_by_the_agent": [t["tool"] for t in s1.tool_log], "facts_active_after": len(stored["active"]),
                    "fact": stored["active"][0] if stored["active"] else None},
        "recalled_in_a_new_session": {"asked_by": who.user_id, "tools_called": [t["tool"] for t in s2.tool_log],
                                      "measured_again": "calibration_residual" in [t["tool"] for t in s2.tool_log],
                                      "value_in_memory_mm": truth, "value_stated_mm": float(said[-1]) if said else None,
                                      "correct": bool(said) and truth is not None and nearly(float(said[-1].replace(",", ".")), truth, abs_tol=0.006),
                                      "answer": s2.answer},
        "after_the_event": {"event": event, "calibration_record_reported_valid": (record_after.content or {}).get("valid"), "facts_active": len(site_memory.facts(db, who, site.name)["active"]),
                            "facts_expired": len(site_memory.facts(db, who, site.name)["expired"]),
                            "answer_says_expired": bool(re.search(r"expired|no longer (valid|current)|not current|out of date|outdated", s3.answer or "", re.I)),
                            "answer_names_the_event": bool(re.search(r"ecm[_ ]moved|camera arm|ECM", s3.answer or "")),
                            "answer": s3.answer},
        "other_site_operator": {"recall_tool": recall.error_kind or "returned", "context_block_chars": len(site_memory.context_block(db, other_op, site.name))},
        "remote_support": {"context_block_chars": len(site_memory.context_block(db, support, site.name))}}
    log(f"e memory: recalled correct {out['e_site_memory']['recalled_in_a_new_session']['correct']}, "
        f"expired said {out['e_site_memory']['after_the_event']['answer_says_expired']}, other site {recall.error_kind}")

    # ---------------------------------------------------------------- f. a long run under a small context budget
    eps = ep_store.list_episodes()[:(3 if SMOKE else 8)]
    durations = [ep_store.stats(e)["duration_s"] for e in eps]
    total = round(sum(durations), 3)
    task = ("Find the duration in seconds of each of these recordings, one at a time, then add them up with the "
            "calculator: " + ", ".join(e.episode_id for e in eps) + ". End with one line 'ANSWER: <sum> s'.")
    rows = []
    for label, budget in (("small budget", 1500), ("large budget", 12000)):
        cfg = AgentConfig(**{**OPERATOR.__dict__, "context_tokens": budget, "max_iterations": 24, "token_budget": 150000,
                             "inject_robot_state": True})
        ctx = ctx_for(who)
        state = ask(cfg, task, ctx)
        got = ANSWER.findall(state.answer or "")
        rows.append({"run": label, "context_budget_tokens": budget, "history_tokens_at_the_end": messages_tokens(state.messages),
                     "largest_context_sent_tokens": state.context_tokens_sent, "model_calls": state.iterations,
                     "tool_calls": len(state.tool_log), "stop_reason": state.stop_reason,
                     "sum_computed_by_code_s": total, "sum_stated_s": float(got[-1].replace(",", ".")) if got else None,
                     "correct": bool(got) and nearly(float(got[-1].replace(",", ".")), total, rel=0.001, abs_tol=0.06),
                     "answer_tail": (state.answer or "")[-300:],
                     "prompt_tokens_used": state.prompt_tokens})
        log(f"f {label}: history {rows[-1]['history_tokens_at_the_end']} sent {rows[-1]['largest_context_sent_tokens']} "
            f"correct {rows[-1]['correct']} ({rows[-1]['stop_reason']})")
    out["f_working_context"] = {"recordings": len(eps), "runs": rows}
    return out
