"""The read path: from a stated need to at most ten tables.

    need (free text)
      -> understand   optional: a structured reading of the need (robot, fps, size, cameras)
      -> retrieve     BM25 over names, descriptions and schemas; kNN over summary vectors
      -> fuse         reciprocal rank fusion of the two lists, by the caller
      -> collapse     one table per family of near-identical tables
      -> rerank       optional: a language model orders the head of the list
      -> select       optional: a language model picks what to show and says why

Every stage is a named setting in `SearchConfig`, so an experiment is a pair of configurations
and nothing else. The model never writes a table id or a URL that reaches the user: the select
stage may only choose among the candidates it was given, and ids and URLs are copied from the
index.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from ..ingest.embed import Embedder
from ..ingest.index import ALIAS, MODELS, client

FIELDS = {"name": 3.0, "description": 2.0, "summary": 1.0, "schema_text": 1.0, "robot_type.text": 2.0}
# Fields that hold identifiers (table names, column names, robot types). A question's function
# words must not be matched against them: found by running the first search, where "a Franka arm
# opening a drawer" returned tables whose whole name is "a". In running text such words are
# common and score next to nothing; in names they are rare, so they scored highly.
IDENT_FIELDS = {"name", "schema_text", "robot_type.text"}
STOP = frozenset("""a an and are as at be but by for if in into is it no not of on or such that the their then there these they
this to was will with i me my we need want looking look find which any anything something some have has can where what do does
show give please data dataset datasets table tables""".split())


def content_words(text: str) -> str:
    """The question without function words and request words, for matching against identifiers."""
    import re
    return " ".join(w for w in re.findall(r"[\w.\-]+", text) if w.lower() not in STOP)


SOURCE = ["table_id", "owner", "name", "url", "summary", "robot_type", "fps", "total_episodes", "total_frames", "n_cameras",
          "cameras", "codebase_version", "gated", "has_description", "is_scratch", "family", "family_size", "last_modified",
          "content_hash", "schema_hash", "licence", "downloads"]


@dataclass(frozen=True)
class SearchConfig:
    name: str = "default"
    retriever: str = "rrf"                 # bm25 | knn | rrf
    embedding: str = "nomic-embed-text"    # which vector field kNN uses
    k: int = 10                            # how many tables are returned
    window: int = 100                      # how deep each retriever goes before fusion
    rank_constant: int = 60                # the k of reciprocal rank fusion
    num_candidates: int = 400              # kNN candidates per shard
    collapse: bool = True                  # one table per family
    demote_scratch: float = 1.0            # multiply the score of scratch-named tables (1.0 = off)
    boost_described: float = 1.0           # multiply the score of tables a person described (1.0 = off)
    fields: tuple[tuple[str, float], ...] = tuple(FIELDS.items())
    ident_stop: bool = True                # keep function words out of the match against names and columns
    understand: str = "off"                # off | soft (everything read boosts) | hard (numbers and format filter, the robot boosts)
    understand_model: str = "small"        # which model reads the need: small | large
    understand_version: str = "u2"         # which reading prompt
    rerank: str = "off"                    # off | small | large
    rerank_depth: int = 20
    select: bool = False

    def but(self, **kw: Any) -> "SearchConfig":
        return replace(self, **kw)


@dataclass
class Hit:
    table_id: str
    rank: int
    score: float
    source: dict[str, Any]
    ranks: dict[str, int] = field(default_factory=dict)       # rank in each retriever's own list
    reason: str = ""


@dataclass
class SearchResult:
    query: str
    config: str
    hits: list[Hit]
    timings_ms: dict[str, float]
    reading: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    pool: list[str] = field(default_factory=list)             # ids seen before the cut, for pooling

    def ids(self) -> list[str]:
        return [h.table_id for h in self.hits]


READING_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["keywords", "robot", "min_episodes", "fps", "cameras", "format"],
    "properties": {
        "keywords": {"type": "string", "description": "The task or content the person wants, as a few search words."},
        "robot": {"type": ["string", "null"], "description": "The robot or arm named in the need, as written, or null."},
        "min_episodes": {"type": ["integer", "null"], "description": "Smallest number of episodes asked for, or null."},
        "fps": {"type": ["number", "null"], "description": "Frame rate asked for, or null."},
        "cameras": {"type": ["integer", "null"], "description": "Smallest number of camera views asked for, or null."},
        "format": {"type": ["string", "null"], "enum": ["v2.0", "v2.1", "v3.0", None],
                   "description": "Dataset format version asked for, or null."},
    },
}
# u1 is the first version. MEASURED (verification/vxfind_llm_stages_first_run.json): told "do not guess
# a robot", both models left the robot empty for almost every request that named one, and the small
# model missed most stated numbers. u2 says per field what counts as stated and gives the forms a
# robot name takes.
READ_PROMPT_U1 = ("You read a request for robot-learning training data and fill a form. Copy only what the request states. "
                  "Use null for anything it does not state. Do not guess a robot, a size or a frame rate.")
READ_PROMPT = """You read a request for robot-learning training data and fill a form. Fill a field only from words that are in the request.

keywords: the task or content asked for, as a few search words. Leave out the robot's name and the numbers.
robot: the robot or arm the request names, in the request's own words, for example "SO-101", "Franka Panda", "UR5", "Piper", "ALOHA", "Koch". null only when the request names no robot at all.
min_episodes: a number, when the request states a minimum number of episodes ("at least 200 episodes", "500 episodes or more").
fps: a number, when the request states a frame rate ("30 fps", "50 Hz", "20 frames per second").
cameras: a number, when the request states how many cameras or camera views it needs ("at least 3 camera views").
format: when the request states a dataset format version ("v2.1", "format v3.0").

Use null for every field the request does not state. Never fill a field from what is usual for such data."""
READ_PROMPTS = {"u1": READ_PROMPT_U1, "u2": READ_PROMPT}
GENERIC = frozenset("arm arms robot robots robotic the a an any or and manipulator data dataset".split())
RERANK_PROMPT = ("You order training datasets by how well each one meets a request. You get the request and a numbered "
                 "list of dataset summaries. Return the numbers, best first. A dataset that names the robot, the task and "
                 "the size the request asks for comes first. A dataset whose summary says nothing about the task comes "
                 "after one that does. Datasets named like tests or scratch work come last. Use every number once.")
SELECT_PROMPT = ("You choose which training datasets to show to a new team member for their request. You get the request "
                 "and a numbered list of candidates. Choose at most {k}. Leave out candidates that do not fit; fewer good "
                 "ones are better than ten weak ones. For each one you choose give one short reason that states only what "
                 "its summary says. If none fits, choose none and say what is missing.")


class Engine:
    def __init__(self, es: Any = None, index: str = ALIAS, llm_small: Any = None, llm_large: Any = None):
        self.es = es or client()
        self.index = index
        self.embedders: dict[str, Embedder] = {}
        self._qcache: dict[tuple[str, str], list[float]] = {}
        self.llm_small, self.llm_large = llm_small, llm_large or llm_small

    # ------------------------------------------------------------------ stages

    def embed(self, model: str, text: str) -> list[float]:
        if model not in self.embedders:
            self.embedders[model] = Embedder(model, load=False)
        key = (model, text)
        if key not in self._qcache:                       # the same need is searched by several configurations
            if len(self._qcache) > 5000:
                self._qcache.clear()
            self._qcache[key] = self.embedders[model].query(text).tolist()
        return self._qcache[key]

    def understand(self, need: str, cfg: SearchConfig | None = None) -> dict[str, Any] | None:
        cfg = cfg or SearchConfig()
        llm = self.llm_large if cfg.understand_model == "large" else self.llm_small
        if llm is None:
            return None
        try:
            r = llm.chat([{"role": "system", "content": READ_PROMPTS[cfg.understand_version]}, {"role": "user", "content": need[:1500]}],
                         json_schema=READING_SCHEMA, temperature=0.0, max_tokens=200)
            out = json.loads(r.content)
            return out if isinstance(out, dict) else None
        except Exception:                                   # a need that cannot be read is searched as plain text
            return None

    @staticmethod
    def _constraints(reading: dict[str, Any] | None, hard: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """(filters, boosts) from a reading. With `hard` every stated constraint is a filter."""
        if not reading:
            return [], []
        clauses: list[dict[str, Any]] = []
        # "Piper arm" must not match every robot whose type contains the word "arm".
        robot = " ".join(w for w in str(reading.get("robot") or "").replace(",", " ").split() if w.lower().strip("()") not in GENERIC)
        # The robot is matched as text, and text is not enough to make it a filter: "Universal Robots"
        # shares no word with "ur5e". A number is a fact and can filter; the robot's name only boosts.
        name = [{"match": {"robot_type.text": {"query": robot, "operator": "or"}}}] if robot else []
        if reading.get("min_episodes"):
            clauses.append({"range": {"total_episodes": {"gte": int(reading["min_episodes"])}}})
        if reading.get("fps"):
            clauses.append({"range": {"fps": {"gte": float(reading["fps"]) - 0.5, "lte": float(reading["fps"]) + 0.5}}})
        if reading.get("cameras"):
            clauses.append({"range": {"n_cameras": {"gte": int(reading["cameras"])}}})
        if reading.get("format"):
            clauses.append({"term": {"codebase_version": reading["format"]}})
        return (clauses, name) if hard else ([], name + clauses)

    def _text_query(self, text: str, cfg: SearchConfig, filters: list[dict[str, Any]], boosts: list[dict[str, Any]]) -> dict[str, Any]:
        ident = content_words(text) if cfg.ident_stop else text
        per_field = [{"match": {f: {"query": ident if f in IDENT_FIELDS else text, "boost": b}}} for f, b in cfg.fields
                     if not (f in IDENT_FIELDS and not ident)]
        q: dict[str, Any] = {"bool": {"must": [{"dis_max": {"queries": per_field, "tie_breaker": 0.3}}], "filter": filters,
                                      "should": [{"constant_score": {"filter": b, "boost": 2.0}} for b in boosts]}}
        functions = []
        if cfg.demote_scratch != 1.0:
            functions.append({"filter": {"term": {"is_scratch": True}}, "weight": cfg.demote_scratch})
        if cfg.boost_described != 1.0:
            functions.append({"filter": {"term": {"has_description": True}}, "weight": cfg.boost_described})
        if functions:
            q = {"function_score": {"query": q, "functions": functions, "score_mode": "multiply", "boost_mode": "multiply"}}
        return q

    def bm25(self, text: str, cfg: SearchConfig, filters: list[dict[str, Any]], boosts: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
        r = self.es.search(index=self.index, query=self._text_query(text, cfg, filters, boosts), size=size, source=SOURCE)
        return r["hits"]["hits"]

    def knn(self, text: str, cfg: SearchConfig, filters: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
        knn = {"field": MODELS[cfg.embedding], "query_vector": self.embed(cfg.embedding, text), "k": size,
               "num_candidates": max(cfg.num_candidates, size)}
        if filters:
            knn["filter"] = filters
        r = self.es.search(index=self.index, knn=knn, size=size, source=SOURCE)
        return r["hits"]["hits"]

    @staticmethod
    def fuse(lists: dict[str, list[dict[str, Any]]], rank_constant: int) -> list[Hit]:
        """Reciprocal rank fusion: a table's score is the sum over lists of 1 / (constant + rank)."""
        score: dict[str, float] = {}
        ranks: dict[str, dict[str, int]] = {}
        src: dict[str, dict[str, Any]] = {}
        for name, hits in lists.items():
            for i, h in enumerate(hits, 1):
                tid = h["_id"]
                score[tid] = score.get(tid, 0.0) + 1.0 / (rank_constant + i)
                ranks.setdefault(tid, {})[name] = i
                src[tid] = h["_source"]
        order = sorted(score, key=lambda t: (-score[t], t))
        return [Hit(t, i, score[t], src[t], ranks[t]) for i, t in enumerate(order, 1)]

    @staticmethod
    def collapse(hits: list[Hit]) -> list[Hit]:
        seen: set[str] = set()
        out = []
        for h in hits:
            fam = h.source.get("family") or h.table_id
            if fam in seen:
                continue
            seen.add(fam)
            out.append(h)
        return out

    def rerank(self, need: str, hits: list[Hit], cfg: SearchConfig) -> tuple[list[Hit], str]:
        llm = self.llm_small if cfg.rerank == "small" else self.llm_large
        head, tail = hits[:cfg.rerank_depth], hits[cfg.rerank_depth:]
        if llm is None or len(head) < 2:
            return hits, "no model for reranking"
        listing = "\n".join(f"{i}. {h.source.get('summary', '')[:420]}" for i, h in enumerate(head, 1))
        schema = {"type": "object", "additionalProperties": False, "required": ["order"],
                  "properties": {"order": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": len(head)}}}}
        try:
            r = llm.chat([{"role": "system", "content": RERANK_PROMPT},
                          {"role": "user", "content": f"Request: {need[:800]}\n\nDatasets:\n{listing}"}],
                         json_schema=schema, temperature=0.0, max_tokens=300)
            order = [int(x) for x in json.loads(r.content)["order"]]
        except Exception as exc:
            return hits, f"rerank failed ({type(exc).__name__}); order unchanged"
        seen: set[int] = set()
        picked = [i for i in order if 1 <= i <= len(head) and not (i in seen or seen.add(i))]
        picked += [i for i in range(1, len(head) + 1) if i not in seen]      # whatever the model left out keeps its place after
        return [head[i - 1] for i in picked] + tail, ""

    def select(self, need: str, hits: list[Hit], cfg: SearchConfig) -> tuple[list[Hit], str, str]:
        llm = self.llm_large
        cands = hits[:max(cfg.k * 2, 12)]
        if llm is None or not cands:
            return hits[:cfg.k], "", "no model for selection"
        listing = "\n".join(f"{i}. {h.source.get('summary', '')[:420]}" for i, h in enumerate(cands, 1))
        schema = {"type": "object", "additionalProperties": False, "required": ["chosen", "missing"],
                  "properties": {"chosen": {"type": "array", "maxItems": cfg.k, "items": {
                      "type": "object", "additionalProperties": False, "required": ["number", "reason"],
                      "properties": {"number": {"type": "integer", "minimum": 1, "maximum": len(cands)},
                                     "reason": {"type": "string"}}}},
                      "missing": {"type": "string", "description": "What the request asks for that no candidate offers; empty if nothing."}}}
        try:
            r = llm.chat([{"role": "system", "content": SELECT_PROMPT.format(k=cfg.k)},
                          {"role": "user", "content": f"Request: {need[:800]}\n\nCandidates:\n{listing}"}],
                         json_schema=schema, temperature=0.0, max_tokens=900)
            out = json.loads(r.content)
        except Exception as exc:
            return hits[:cfg.k], "", f"selection failed ({type(exc).__name__}); the ranked list is shown as it is"
        chosen, seen = [], set()
        for c in out.get("chosen", []):
            n = c.get("number")
            if isinstance(n, int) and 1 <= n <= len(cands) and n not in seen:      # a number outside the list is dropped, not looked up
                seen.add(n)
                h = cands[n - 1]
                h.reason = str(c.get("reason", ""))[:300]
                chosen.append(h)
        return chosen[:cfg.k], str(out.get("missing", ""))[:300], ""

    # ------------------------------------------------------------------ the whole path

    def search(self, need: str, cfg: SearchConfig | None = None, visible: dict[str, Any] | None = None) -> SearchResult:
        cfg = cfg or SearchConfig()
        t: dict[str, float] = {}
        notes: list[str] = []
        t0 = time.perf_counter()
        reading = self.understand(need, cfg) if cfg.understand != "off" else None
        t["understand"] = (time.perf_counter() - t0) * 1000
        filters, boosts = self._constraints(reading, cfg.understand == "hard")
        if visible and visible.get("hidden_owners"):
            filters = filters + [{"bool": {"must_not": {"terms": {"owner": visible["hidden_owners"]}}}}]
        text = (reading or {}).get("keywords") or need if cfg.understand == "hard" else need
        depth = cfg.window * (3 if cfg.collapse else 1)
        lists: dict[str, list[dict[str, Any]]] = {}
        if cfg.retriever in ("bm25", "rrf"):
            t1 = time.perf_counter()
            lists["bm25"] = self.bm25(text, cfg, filters, boosts, depth)
            t["bm25"] = (time.perf_counter() - t1) * 1000
        if cfg.retriever in ("knn", "rrf"):
            t1 = time.perf_counter()
            lists["knn"] = self.knn(need, cfg, filters, depth)
            t["knn"] = (time.perf_counter() - t1) * 1000
        hits = self.fuse(lists, cfg.rank_constant)
        if cfg.collapse:
            hits = self.collapse(hits)
        pool = [h.table_id for h in hits[:cfg.window]]
        if cfg.rerank != "off":
            t1 = time.perf_counter()
            hits, note = self.rerank(need, hits, cfg)
            t["rerank"] = (time.perf_counter() - t1) * 1000
            if note:
                notes.append(note)
        missing = ""
        if cfg.select:
            t1 = time.perf_counter()
            hits, missing, note = self.select(need, hits, cfg)
            t["select"] = (time.perf_counter() - t1) * 1000
            if note:
                notes.append(note)
        hits = hits[:cfg.k]
        for i, h in enumerate(hits, 1):
            h.rank = i
        if missing:
            notes.append("missing: " + missing)
        t["total"] = (time.perf_counter() - t0) * 1000
        return SearchResult(need, cfg.name, hits, {k: round(v, 1) for k, v in t.items()}, reading, notes, pool)


def describe(cfg: SearchConfig) -> dict[str, Any]:
    d = asdict(cfg)
    d["fields"] = dict(cfg.fields)
    return d
