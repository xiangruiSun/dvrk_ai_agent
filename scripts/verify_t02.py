"""Verification of task 02 (structured change proposals) on the real dVRK configuration files.

    python scripts/verify.py t02     ->  verification/t02_structured_output.json
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vxagent.config import get_settings                                    # noqa: E402
from vxagent.llm.ollama import get_llm                                     # noqa: E402
from vxagent.profiles import principal                                     # noqa: E402
from vxagent.robot import config as cfg                                    # noqa: E402
from vxagent.robot import jsonc                                            # noqa: E402
from vxagent.store import Database                                         # noqa: E402
from vxagent.structured import proposals                                   # noqa: E402
from vxagent.structured.dryrun import dry_apply                            # noqa: E402
from vxagent.structured.generate import generate_patch                     # noqa: E402
from vxagent.structured.rules import InstrumentSet, RuleContext, validate_patch   # noqa: E402
from vxagent.structured.schemas import ConfigPatch                         # noqa: E402
from vxagent.tools.registry import Principal                               # noqa: E402

REQUESTS = ROOT / "evals" / "t02_change_requests.json"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2] if len(xs) % 2 else round((xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2, 1)


def t02() -> dict[str, Any]:
    s = get_settings()
    dvrk = s.dvrk_dir
    site_path = s.sites_dir / "site_a.json"
    site_spec = json.loads(site_path.read_text(encoding="utf-8"))
    schemas = cfg.SchemaSet(dvrk / "share" / "schemas")
    instruments = InstrumentSet(dvrk / "share" / "tool")
    sources = cfg.site_sources(site_path, dvrk)
    work = s.runtime_dir / "t02_config"
    shutil.rmtree(work, ignore_errors=True)
    store = cfg.ConfigStore(work, sources)
    spec = json.loads(REQUESTS.read_text(encoding="utf-8"))
    llm = get_llm()

    def context(target: str, records: dict[str, str] | list[str]) -> RuleContext:
        doc, _, _ = store.current(target)
        return RuleContext(target=target, file_type=cfg.file_type(doc), doc=doc, schemas=schemas,
                           instruments=instruments, frames_declared=frozenset(site_spec.get("frames", [])),
                           evidence=frozenset(records))

    out: dict[str, Any] = {"task": "t02_structured_output", "ran": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "model": s.model, "site": site_spec["name"],
                           "dvrk_commit": json.loads((dvrk / "MANIFEST.json").read_text(encoding="utf-8"))["commit"][:10],
                           "requests_file": str(REQUESTS.relative_to(ROOT)).replace("\\", "/")}

    # a. Every published arm and system file against the schema it names, as it is.
    rows = []
    for folder in ("arm", "system"):
        for p in sorted((dvrk / "share" / folder).glob("*.json")):
            doc = jsonc.load(p)
            errors = schemas.errors(doc)
            rows.append({"file": f"{folder}/{p.name}", "names_schema": doc.get("$id") if isinstance(doc, dict) else None,
                         "validated": errors is not None, "errors": errors or []})
    checked = [r for r in rows if r["validated"]]
    out["a_baseline"] = {"files": len(rows), "name_a_schema_of_the_set": len(checked),
                         "pass": sum(not r["errors"] for r in checked),
                         "fail": [{"file": r["file"], "errors": r["errors"]} for r in checked if r["errors"]],
                         "name_no_schema_of_the_set": [r["file"] for r in rows if not r["validated"]],
                         "schemas": len(schemas.names())}
    log(f"a baseline: {out['a_baseline']['pass']}/{len(checked)} pass, {len(rows) - len(checked)} name no schema")

    # The reference patches must themselves be valid, or (b) would compare against nothing.
    refs: dict[str, Any] = {}
    for r in spec["requests"]:
        ctx = context(r["target"], r["records"])
        patch = ConfigPatch.model_validate({"target": r["target"], "reason": "reference patch for " + r["id"],
                                            "evidence": list(r["records"]), **r["reference"]})
        v = validate_patch(patch, ctx)
        refs[r["id"]] = (ctx, patch, v)
        if not v.ok:
            raise SystemExit(f"reference patch {r['id']} is not valid: {v.stage} {v.errors}")

    # b. The change requests, in both generation modes.
    runs = []
    for mode in ("constrained", "free"):
        for r in spec["requests"]:
            ctx, _, ref_v = refs[r["id"]]
            t0 = time.time()
            g = generate_patch(llm, r["request"], ctx, r["records"], mode=mode)
            row: dict[str, Any] = {
                "id": r["id"], "mode": mode, "target": r["target"], "accepted": g.ok,
                "attempts": len(g.attempts), "valid_on_first_attempt": g.attempts[0].stage == "ok",
                "stages": [a.stage for a in g.attempts],
                "tokens": sum(a.prompt_tokens + a.completion_tokens for a in g.attempts),
                "seconds": round(time.time() - t0, 1),
                "attempt_log": [a.as_dict() for a in g.attempts]}
            if g.ok:
                assert g.patch is not None and g.validation is not None
                row["same_file_as_reference"] = cfg.canonical(g.validation.patched) == cfg.canonical(ref_v.patched)
                row["same_limit_declaration"] = g.patch.limit_change == r["reference"]["limit_change"]
                row["equals_reference"] = row["same_file_as_reference"] and row["same_limit_declaration"]
                row["units_given"] = [o.unit for o in g.patch.operations]
                row["patch"] = g.patch.model_dump()
            runs.append(row)
            log(f"b {mode} {r['id']}: {row['stages']} equals reference {row.get('equals_reference')} ({row['seconds']} s)")
    by_mode = {}
    for mode in ("constrained", "free"):
        mine = [x for x in runs if x["mode"] == mode]
        errs: dict[str, int] = {}
        for x in mine:
            for st in x["stages"]:
                if st != "ok":
                    errs[st] = errs.get(st, 0) + 1
        by_mode[mode] = {"requests": len(mine), "accepted": sum(x["accepted"] for x in mine),
                         "valid_on_first_attempt": sum(x["valid_on_first_attempt"] for x in mine),
                         "accepted_after_repair": sum(x["accepted"] and not x["valid_on_first_attempt"] for x in mine),
                         "nothing_proposed": sum(not x["accepted"] for x in mine),
                         "equals_reference": sum(bool(x.get("equals_reference")) for x in mine),
                         "accepted_but_different_from_reference": [x["id"] for x in mine if x["accepted"]
                                                                   and not x["equals_reference"]],
                         "failed_attempts_by_stage": errs,
                         "median_tokens": median([x["tokens"] for x in mine]),
                         "median_seconds": median([x["seconds"] for x in mine])}
    out["b_change_requests"] = {"requests": len(spec["requests"]), "max_attempts": 3, "by_mode": by_mode, "runs": runs}

    # c. Patches the schema accepts and the rules must refuse. Hand-written.
    neg = []
    for n in spec["negative_controls"]:
        records = ["rec:1"]
        ctx = context(n["target"], records)
        patch = ConfigPatch.model_validate({"target": n["target"], "reason": "negative control: " + n["id"],
                                            "evidence": n.get("evidence", records), **n["patch"]})
        v = validate_patch(patch, ctx)
        neg.append({"id": n["id"], "target": n["target"], "passes_pydantic_schema": True,
                    "refused": not v.ok, "refused_at": v.stage, "expected_stage": n["expect"],
                    "as_expected": (not v.ok) and v.stage == n["expect"] and any(n["says"] in e for e in v.errors),
                    "errors": v.errors})
        log(f"c {n['id']}: refused at {v.stage} as expected {neg[-1]['as_expected']}")
    phys = [x for x in neg if x["expected_stage"] == "physical"]
    allow = [x for x in neg if x["expected_stage"] == "allowlist"]
    out["c_negative_controls"] = {
        "accepted_by_both_schemas_refused_by_a_physical_rule": f"{sum(x['as_expected'] for x in phys)}/{len(phys)}",
        "refused_by_the_allowlist": f"{sum(x['as_expected'] for x in allow)}/{len(allow)}",
        "any_accepted": [x["id"] for x in neg if not x["refused"]], "controls": neg}

    # d. Authorization: the same valid patch, different people.
    db = Database(":memory:")
    site = site_spec["name"]
    people = {"operator": principal("op.lee", "operator", site), "engineer": principal("eng.kim", "service", site),
              "engineer of another site": principal("eng.ruiz", "service", "site-b"),
              "second engineer": principal("eng.oyelaran", "service", site),
              "remote support": principal("sup.wong", "support", ""),
              "safety lead": principal("saf.adler", "safety", site),
              "safety lead of another site": principal("saf.novak", "safety", "site-b")}
    ctx, patch, v = refs["site-clearance"]
    ctx_l, patch_l, v_l = refs["site-workspace-wider"]
    matrix = []

    def attempt(label: str, fn) -> dict[str, Any]:
        try:
            result = fn()
            row = {"case": label, "allowed": True, "refusal": None}
        except proposals.NotPermitted as exc:
            result, row = None, {"case": label, "allowed": False, "refusal": exc.kind}
        matrix.append(row)
        return {"result": result, **row}

    for who in ("operator", "engineer of another site", "remote support", "safety lead"):
        attempt(f"proposed by {who}", lambda w=who: proposals.propose(db, people[w], site, patch, v, ctx.doc, ctx.file_type))
    pid = attempt("proposed by engineer", lambda: proposals.propose(db, people["engineer"], site, patch, v, ctx.doc,
                                                                    ctx.file_type))["result"]
    for who in ("engineer", "second engineer", "operator", "remote support", "safety lead of another site"):
        label = "approved by the proposer" if who == "engineer" else f"approved by {who}"
        attempt(label, lambda w=who: proposals.decide(db, people[w], pid, True))
    attempt("approved by safety lead", lambda: proposals.decide(db, people["safety lead"], pid, True))
    attempt("approved a second time", lambda: proposals.decide(db, people["safety lead"], pid, True))
    # The rule "the approver is not the proposer" only bites for someone who holds both roles.
    both = Principal("lead.both", "safety", people["engineer"].scopes | people["safety lead"].scopes, site=site,
                     tiers=people["safety lead"].tiers)
    pid_b = attempt("proposed by a person holding both roles",
                    lambda: proposals.propose(db, both, site, patch, v, ctx.doc, ctx.file_type))["result"]
    attempt("approved by that same person", lambda: proposals.decide(db, both, pid_b, True))
    pid_l = proposals.propose(db, people["engineer"], site, patch_l, v_l, ctx_l.doc, ctx_l.file_type)
    attempt("loosened limit approved without a reason", lambda: proposals.decide(db, people["safety lead"], pid_l, True))
    attempt("loosened limit approved with a written reason",
            lambda: proposals.decide(db, people["safety lead"], pid_l, True,
                                     "Phantom B is 86 mm wide; reviewed with the lab on site."))
    expected = {"proposed by operator": "role", "proposed by engineer of another site": "site",
                "proposed by remote support": "role", "proposed by safety lead": "role", "proposed by engineer": None,
                "approved by the proposer": "role", "approved by second engineer": "role",
                "approved by operator": "role", "approved by remote support": "role",
                "approved by safety lead of another site": "site", "approved by safety lead": None,
                "approved a second time": "state", "proposed by a person holding both roles": None,
                "approved by that same person": "self_approval",
                "loosened limit approved without a reason": "reason_required",
                "loosened limit approved with a written reason": None}
    for row in matrix:
        row["expected_refusal"] = expected[row["case"]]
        row["as_expected"] = row["refusal"] == row["expected_refusal"]
    out["d_authorization"] = {"cases": len(matrix), "as_expected": sum(r["as_expected"] for r in matrix),
                              "refusals_written_to_the_audit_log": len(db.query("SELECT 1 FROM audit WHERE allowed=0")),
                              "matrix": matrix}
    log(f"d authorization: {out['d_authorization']['as_expected']}/{len(matrix)} as expected")

    # e. Dry application of every reference patch, then one approved change applied as a version.
    effects = []
    for r in spec["requests"]:
        ctx, patch, v = refs[r["id"]]
        e = dry_apply(ctx.file_type, ctx.doc, v.patched, site_spec, dvrk, v.changes)
        effects.append({"id": r["id"], "target": r["target"], "poses_that_move": len(e.get("poses_that_move", [])),
                        "largest_move_mm": e.get("largest_move_mm", 0.0), "largest_turn_deg": e.get("largest_turn_deg", 0.0),
                        "largest_change_of_height_over_pad_mm": e.get("largest_change_of_height_over_pad_mm", 0.0),
                        "newly_rejected_by_the_safety_filter": e.get("newly_rejected_by_the_safety_filter", []),
                        "newly_accepted_by_the_safety_filter": e.get("newly_accepted_by_the_safety_filter", []),
                        "expressed_in": e.get("poses_expressed_in"), "note": e.get("note"),
                        "invalidates_calibration": v.invalidates_calibration, "limit_effect": v.limit_effect,
                        "file_values": [{"path": c["path"], "before": c["before"], "after": c["after"],
                                         "stored_in": c["unit"], "given": c["given"], "given_in": c["given_unit"]}
                                        for c in v.changes]})
    applied = []
    before_hash = {t: sha(p) for t, p in sources.items()}
    for rid in ("site-clearance", "system-base-translation"):
        ctx, patch, v = refs[rid]
        eff = next(x for x in effects if x["id"] == rid)
        p = proposals.propose(db, people["engineer"], site, patch, v, ctx.doc, ctx.file_type, eff)
        proposals.decide(db, people["safety lead"], p, True)
        res = proposals.apply(db, store, p)
        doc, _, ver = store.current(patch.target)
        res.update({"id": rid, "version_now": ver,
                    "value_in_new_version": [cfg.get(doc, c["path"].replace("/translation", "")) for c in v.changes],
                    "calibration_now": proposals.calibration(db, site)})
        applied.append(res)
    # A proposal checked against a file that has changed since must not be applied.
    ctx, patch, v = refs["site-free-step"]                 # checked against version 0 of the site file
    p = proposals.propose(db, people["engineer"], site, patch, v, ctx.doc, ctx.file_type)
    proposals.decide(db, people["safety lead"], p, True)
    try:
        proposals.apply(db, store, p)
        stale = "applied"
    except proposals.NotPermitted as exc:
        stale = exc.kind
    out["e_dry_application_and_apply"] = {
        "effects": effects, "applied": applied,
        "source_files_unchanged": all(sha(p) == before_hash[t] for t, p in sources.items()),
        "source_files_checked": len(sources),
        "proposal_checked_against_an_older_version": stale}
    log(f"e applied {[a['id'] for a in applied]}; sources unchanged "
        f"{out['e_dry_application_and_apply']['source_files_unchanged']}; stale proposal: {stale}")
    return out
