"""Getting a valid patch out of a language model, or nothing.

Two modes: schema-constrained decoding (the model server only lets the model emit JSON of the
right shape) and free text from which a JSON object is extracted. Either way the reply goes
through the same stages (json, schema, target, allowlist, file_schema, physical), and after a
failure the model gets its own reply back with a numbered list of the exact errors. Three
attempts at most. There is no fallback: if no attempt passes, nothing is proposed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ..llm.base import LLMClient, LLMError
from ..robot import config as cfg
from .rules import RuleContext, Validation, describe_paths, validate_patch
from .schemas import ConfigPatch

MAX_ATTEMPTS = 3
FILE_CHARS = 2500

SYSTEM = (
    "You write one configuration patch for a da Vinci Research Kit (dVRK) site, as a single JSON object. You never "
    "write a configuration file. A patch is a short list of operations on one existing file.\n"
    "Fields of the object:\n"
    '  "target"        the file named in the request, exactly\n'
    '  "operations"    a list of {"op", "path", "value", "unit"}\n'
    '  "limit_change"  "tighten", "loosen" or "none"\n'
    '  "reason"        one or two sentences\n'
    '  "evidence"      ids of the records given below that support the change\n'
    "Rules:\n"
    "1. Use only paths from the list of allowed paths, written exactly as listed.\n"
    "2. op is \"replace\" for a path that has a value now and \"add\" for a path marked (not set).\n"
    "3. Give each value as the request states it and name the unit it was stated in. Never convert a number from "
    "one unit to another: the software converts. A pure number has unit \"ratio\"; text, true/false and rotation "
    "matrices have unit \"none\".\n"
    "4. limit_change is \"tighten\" if the patch makes a safety limit stricter, \"loosen\" if it relaxes one, and "
    "\"none\" if no safety limit is touched. The list says for each safety limit which direction is tighter.\n"
    "5. Change only what the request asks for. Do not touch other values.\n"
    "6. Cite only record ids that are given to you.\n"
    "Reply with the JSON object and nothing else.")


@dataclass
class Attempt:
    n: int
    stage: str                       # json | schema | target | allowlist | file_schema | physical | ok | llm_error
    errors: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    reply: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"attempt": self.n, "stage": self.stage, "errors": self.errors, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "latency_ms": round(self.latency_ms, 1),
                "reply": self.reply[:1500]}


@dataclass
class Generation:
    mode: str
    patch: ConfigPatch | None = None
    validation: Validation | None = None
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.patch is not None


def extract_json(text: str) -> Any:
    """The first complete JSON object in a reply that may have prose or a code fence around it."""
    start = text.find("{")
    while start >= 0:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("the reply contains no complete JSON object")


def request_message(request: str, ctx: RuleContext, records: dict[str, str]) -> str:
    paths = describe_paths(ctx.file_type, ctx.doc)
    body = cfg.canonical(ctx.doc)
    parts = [f"Request: {request}", f"File to change (target): {ctx.target}", f"Kind of file: {ctx.file_type}",
             "Allowed paths in this file:\n" + "\n".join(json.dumps(p, ensure_ascii=False) for p in paths)]
    if ctx.file_type == "dvrk-system":
        parts.append("Frames of this site: " + ", ".join(sorted(ctx.frames_declared)))
    parts.append("Records of this run:\n" + ("\n".join(f"{k}: {v}" for k, v in records.items()) or "(none)"))
    if len(body) <= FILE_CHARS:
        parts.append("The file as it is now:\n" + body)
    return "\n\n".join(parts)


def generate_patch(llm: LLMClient, request: str, ctx: RuleContext, records: dict[str, str],
                   mode: str = "constrained", max_attempts: int = MAX_ATTEMPTS) -> Generation:
    if mode not in ("constrained", "free"):
        raise ValueError(f"unknown mode {mode!r}")
    out = Generation(mode=mode)
    schema = ConfigPatch.model_json_schema() if mode == "constrained" else None
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": request_message(request, ctx, records)}]
    for n in range(1, max_attempts + 1):
        att = Attempt(n=n, stage="llm_error")
        out.attempts.append(att)
        try:
            resp = llm.chat(messages, json_schema=schema, temperature=0.0, max_tokens=900)
        except LLMError as exc:
            att.errors = [f"{exc.kind}: {exc}"]
            break                                     # the model cannot be reached: retrying here would not help
        att.prompt_tokens, att.completion_tokens = resp.usage.prompt_tokens, resp.usage.completion_tokens
        att.latency_ms, att.reply = resp.latency_ms, resp.content
        patch, validation = None, None
        try:
            raw = json.loads(resp.content) if mode == "constrained" else extract_json(resp.content)
        except ValueError as exc:
            att.stage, att.errors = "json", [str(exc)]
        else:
            try:
                patch = ConfigPatch.model_validate(raw)
            except ValidationError as exc:
                att.stage = "schema"
                att.errors = [f"{'.'.join(map(str, e['loc'])) or 'object'}: {e['msg']}" for e in exc.errors()][:12]
            else:
                validation = validate_patch(patch, ctx)
                att.stage, att.errors = validation.stage, list(validation.errors)
        if att.stage == "ok":
            out.patch, out.validation = patch, validation
            return out
        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content":
                         "That patch was not accepted. Problems:\n"
                         + "\n".join(f"{i}. {e}" for i, e in enumerate(att.errors, 1))
                         + "\nSend the corrected patch as one JSON object. Keep everything that was not a problem."})
    return out                                        # no valid patch: nothing is proposed
