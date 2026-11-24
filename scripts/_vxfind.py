"""Shared set-up for the vxfind evaluation scripts."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VER = ROOT / "verification"
EVALS = ROOT / "evals" / "vxfind"
for noisy in ("httpx", "elastic_transport", "mcp", "uvicorn", "httpx2"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from vxfind.eval import queries as Q                                   # noqa: E402
from vxfind.ingest.index import ALIAS, client                          # noqa: E402
from vxfind.search.engine import Engine, SearchConfig, describe        # noqa: E402

JUDGE_MODEL = "qwen3:14b"
SMALL_MODEL = "qwen3:4b"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def evidence(name: str, data: dict[str, Any] | None = None, **sections: Any) -> dict[str, Any]:
    """Write (or update sections of) verification/vxfind_<name>.json."""
    p = VER / f"vxfind_{name}.json"
    cur = json.loads(p.read_text(encoding="utf-8")) if p.exists() and data is None else {}
    cur.update(data or {})
    cur.update(sections)
    cur["ran"] = time.strftime("%Y-%m-%d %H:%M")
    p.write_text(json.dumps(cur, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return cur


def read(name: str) -> dict[str, Any]:
    p = VER / f"vxfind_{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def llm(model: str) -> Any:
    from vxagent.llm.ollama import get_llm
    return get_llm(model)


def engine(with_models: bool = False) -> Engine:
    return Engine(client(), llm_small=llm(SMALL_MODEL) if with_models else None, llm_large=llm(JUDGE_MODEL) if with_models else None)


def index_info(es: Any) -> dict[str, Any]:
    name = sorted(es.indices.get_alias(name=ALIAS))[0]
    return {"alias": ALIAS, "index": name, "tables": es.count(index=name)["count"], "engine": es.info()["version"]["number"]}


def records() -> dict[str, dict[str, Any]]:
    return {r["table_id"]: r for r in Q.records("old")}


def serving() -> SearchConfig:
    """The configuration chosen on the development queries (written by the retrieval stage)."""
    p = EVALS / "serving.json"
    if not p.exists():
        return SearchConfig(name="rrf")
    d = json.loads(p.read_text(encoding="utf-8"))["config"]
    d["fields"] = tuple((k, v) for k, v in d["fields"].items())
    return SearchConfig(**d)


def save_serving(cfg: SearchConfig, why: dict[str, Any]) -> None:
    EVALS.mkdir(parents=True, exist_ok=True)
    (EVALS / "serving.json").write_text(json.dumps({"config": describe(cfg), "chosen": why}, indent=1), encoding="utf-8")


def judge_missing(judge: Any, eng: Engine, queries: list[dict[str, Any]], run_: dict[str, dict[str, Any]]) -> int:
    """Judge the tables in a run's top 10 (written and no-answer needs) that no judgment covers yet."""
    from vxfind.eval.judge import sources
    qrels = judge.qrels(queries)
    by = {q["qid"]: q for q in queries if q["kind"] in "CD"}
    pairs = [(qid, t) for qid in by if qid in run_ for t in run_[qid]["ids"][:10] if t not in qrels.get(qid, {})]
    srcs = sources(eng.es, eng.index, sorted({t for _, t in pairs}))
    judge.grade_many([(qid, by[qid]["need"], t, srcs[t]) for qid, t in pairs if t in srcs])
    return len(pairs)
