"""Graded relevance by a language model, with every judgment kept.

A judge reads one need and one table (its generated summary and what a person wrote about it)
and returns a grade from 0 to 3 with a short reason. One table per call: a judge shown a list
grades the list's order as much as its items.

The judge is a measuring instrument and is treated as one: its prompt has a version, every
judgment is stored with the model, the prompt version and the hash of the text that was judged,
and its error is measured against truth that does not come from a model (meta.py). A judgment
is reused only when all of those match; a table whose text changed is judged again.

Judgments live in evals/vxfind/qrels/<judge>.jsonl, one per line.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
QRELS = ROOT / "evals" / "vxfind" / "qrels"
PROMPT_VERSION = "g4"
# g2 (kept in the judgment file and in verification/vxfind_judge_first_run.json) listed the four
# grades and left the order of the checks to the model. Measured against computed truth it called
# "usable" tables whose text CONTRADICTED the request, while writing the contradiction into its
# own reason ("Only 10 episodes, not ten million" -> grade 2). g3 made the checks a procedure and
# put the contradiction check before the grade; false positives fell, and the remaining
# disagreements were read: the judge refused "ur5e" for a request naming UR5 and "bi_piper_follower"
# for a Piper arm. g4 adds the line about spellings of robot names, and says that an empty dataset
# meets no request. Every version's measured error: verification/vxfind_judge_iterations.json.
RUBRIC_G2 = """You grade whether a dataset meets a request for robot-learning training data.

Grades:
3 = Meets the request. The robot, the task and every stated requirement (size, cameras, frame rate, format) match what the dataset says.
2 = Usable for the request. The task or the robot matches and nothing the dataset says contradicts the request, but one stated requirement is not confirmed.
1 = Related only. Same robot or the same kind of task, but not what was asked for; or a stated requirement is contradicted.
0 = Not relevant, or the dataset says too little to tell what it holds.

Judge only from the dataset text given. A dataset that does not say what task it shows cannot get more than 1 for a request about a task. Give a reason of at most 20 words, then the grade."""
RUBRIC = """You grade whether a dataset meets a request for robot-learning training data. Judge only from the dataset text given.

Robot names are spelled in many ways. A robot name in the dataset that contains the requested robot's name, or is a variant or a two-arm version of it (so101_follower for SO-101, ur5e for a UR arm, bi_piper_follower for a Piper arm, panda for Franka), IS the requested robot.

Decide in this order and stop at the first step that applies:

Step 1. Is the dataset the kind of data the request asks for: the same robot activity or the same content? If it is something else, or the text is too thin to tell what it holds, or the dataset has 0 episodes: grade 0. If it only shares a word, a sensor or a robot with the request but shows something else: grade 1.

Step 2. Does the dataset text state a value that CONTRADICTS a requirement stated in the request? Examples: fewer episodes than the minimum asked for; a different frame rate; fewer cameras than asked for; a different robot; a different format. If yes: grade 1, however well everything else matches.

Step 3. If the request is about a task and the dataset text does not say what task it shows: grade 1.

Step 4. If the robot, the task and every stated requirement are confirmed by the text: grade 3.

Step 5. Otherwise (the task or the robot matches, nothing is contradicted, and a stated requirement is simply not mentioned): grade 2.

Give a reason of at most 20 words that names the step you stopped at, then the grade."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reason", "grade"],
          "properties": {"reason": {"type": "string"}, "grade": {"type": "integer", "enum": [0, 1, 2, 3]}}}


def doc_text(src: dict[str, Any]) -> str:
    d = (src.get("description") or "").strip()
    return (src.get("summary") or "")[:900] + (f"\n\nWritten by its owner:\n{d[:700]}" if d else "")


def key(model: str, need: str, content_hash: str, version: str = "") -> str:
    return hashlib.sha256(f"{model}|{version or PROMPT_VERSION}|{need}|{content_hash}".encode()).hexdigest()[:20]


class Judge:
    def __init__(self, llm: Any, name: str | None = None, folder: Path = QRELS, version: str = "", rubric: str = ""):
        """`version` and `rubric` default to the ones in use. An earlier rubric can be passed with its
        version to measure it again on new pairs; its judgments are stored under that version."""
        self.llm = llm
        self.version, self.rubric = version or PROMPT_VERSION, rubric or RUBRIC
        self.name = (name or llm.model).replace(":", "_")
        self.path = folder / f"{self.name}.jsonl"
        folder.mkdir(parents=True, exist_ok=True)
        self.store: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    j = json.loads(line)
                    self.store[j["key"]] = j
        self._lock = threading.Lock()
        self.calls = 0
        self.seconds = 0.0
        self.failures = 0

    def grade(self, qid: str, need: str, table_id: str, src: dict[str, Any]) -> dict[str, Any] | None:
        k = key(self.llm.model, need, src.get("content_hash") or doc_text(src), self.version)
        if k in self.store:
            return self.store[k]
        t0 = time.perf_counter()
        try:
            r = self.llm.chat([{"role": "system", "content": self.rubric},
                               {"role": "user", "content": f"Request:\n{need[:700]}\n\nDataset:\n{doc_text(src)}"}],
                              json_schema=SCHEMA, temperature=0.0, max_tokens=160)
            out = json.loads(r.content)
            g = int(out["grade"])
            assert 0 <= g <= 3
        except Exception:
            with self._lock:
                self.failures += 1
            return None
        j = {"key": k, "qid": qid, "table_id": table_id, "grade": g, "reason": str(out.get("reason", ""))[:160],
             "judge": self.llm.model, "prompt": self.version, "content_hash": src.get("content_hash")}
        with self._lock:
            self.calls += 1
            self.seconds += time.perf_counter() - t0
            self.store[k] = j
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")
        return j

    def grade_many(self, pairs: list[tuple[str, str, str, dict[str, Any]]], workers: int = 4, log: Any = None) -> list[dict[str, Any] | None]:
        """pairs: (qid, need, table_id, source). Results in the same order."""
        out: list[dict[str, Any] | None] = [None] * len(pairs)
        done = 0
        with cf.ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(self.grade, *p): i for i, p in enumerate(pairs)}
            for f in cf.as_completed(futs):
                out[futs[f]] = f.result()
                done += 1
                if log and done % 200 == 0:
                    log(f"  judged {done}/{len(pairs)} ({self.calls} new, {self.failures} failed)")
        return out

    def qrels(self, queries: list[dict[str, Any]] | None = None) -> dict[str, dict[str, int]]:
        """qid -> table_id -> grade, from everything this judge has graded with the current prompt
        and model. With `queries`, a judgment made for an older wording of a need is left out."""
        needs = {q["qid"]: q["need"] for q in queries} if queries else None
        out: dict[str, dict[str, int]] = {}
        for j in self.store.values():
            if j["prompt"] != PROMPT_VERSION or j["judge"] != self.llm.model:
                continue
            if needs is not None and (j["qid"] not in needs or key(self.llm.model, needs[j["qid"]], j["content_hash"] or "") != j["key"]):
                continue
            out.setdefault(j["qid"], {})[j["table_id"]] = j["grade"]
        return out


def sources(es: Any, index: str, ids: list[str]) -> dict[str, dict[str, Any]]:
    """The text the judge reads, from the index, for many tables."""
    out: dict[str, dict[str, Any]] = {}
    fields = ["table_id", "summary", "description", "content_hash", "robot_type", "fps", "total_episodes", "n_cameras",
              "cameras", "codebase_version", "family", "is_scratch", "has_description"]
    for i in range(0, len(ids), 500):
        r = es.mget(index=index, ids=ids[i:i + 500], source=fields)
        for d in r["docs"]:
            if d.get("found"):
                out[d["_id"]] = d["_source"]
    return out
