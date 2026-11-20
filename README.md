# dVRK Agent Tools

Role-scoped AI agent tools for operating and servicing a da Vinci Research Kit (dVRK): an operator
who runs trials, a service engineer who calibrates and repairs, a safety lead who approves anything
that moves or changes a limit, and a remote support team that sees fleet health without seeing the
lab's data.

**Research demonstration on public data. Not connected to a robot, not a medical device.** The robot
is a kinematic twin built from the published dVRK kinematic files. It has no contact, no needle and
no camera, and nothing measured here is a statement about a physical robot.

The repository holds two packages, in the same relation as the earlier pair
[deepwise_rag](https://github.com/xiangruiSun/deepwise_rag) and
[ai_agent_tools_deepwise](https://github.com/xiangruiSun/ai_agent_tools_deepwise):

- **`vxrag`**, a retrieval system over the dVRK documentation, issue threads, change log,
  configuration schemas and sites' own notes, run as an HTTP service. It imports nothing from the
  agents and can be moved to its own repository unchanged. Requirements:
  [`requirements/rag/`](requirements/rag/R00_overview.txt).
- **`vxagent`**, the agent tools that call it over HTTP. Requirements:
  [`requirements/`](requirements/00_overview.txt).

**Status.** Built and measured on real data: the retrieval system (ten tasks) and the fifteen agent
tasks. Each requirement file starts with an "As built" block that says what differs from its plan, and
every measured result is in [`verification/REPORT.md`](verification/REPORT.md), written by the scripts.
What is not verified is said where it applies: the ROS 2 backend has never run against a robot, the
needle and the camera are declared, and the public mode has not been put behind a tunnel.

## Central rule

The language model never produces a number that reaches the robot. It chooses a pose the site has
declared by name, or a validated procedure with typed parameters. Targets are computed by code and
checked by a safety filter that rejects, and never clips, anything outside the envelope.

## What is built

| Layer | What it does | Code |
|---|---|---|
| Kinematics | Forward and inverse kinematics from dVRK arm and instrument files (modified DH) | `vxagent/robot/kinematics.py` |
| Kinematic twin | CRTK-style backend: operating state, measured and set-point pose, servo commands, one-use command ids, injectable faults | `vxagent/robot/twin.py`, `backend.py` |
| Safety filter | Step limits in free space and near the needle, local workspace, joint margins, clearance, tracking stop; reject without clipping | `vxagent/robot/safety.py`, `motion.py` |
| Tool execution | Authorization by scope and tier, validation, freshness of the robot state, state-bound approval, retry by tier, never-twice motions, trace | `vxagent/tools/executor.py` |
| Actions | Long-running tools with progress, result and cancel | `vxagent/tools/actions.py` |
| Agent loop | model → tool → observation → model, thirteen explicit stop reasons; nothing keeps moving after a stop | `vxagent/agent/loop.py` |
| Grounding guard | Rejects answers whose numbers, robot state or citations no tool returned | `vxagent/agent/guards.py` |
| Recordings | Statistics and bounded windows of recorded dVRK episodes; raw recordings never reach the model | `vxagent/data/episodes.py` |
| Change proposals | The model proposes a typed patch to one configuration file and never writes a file. Units converted by code; allowlist of paths; the dVRK project's own JSON Schemas; physical rules (range, step, rotation, frames, direction of a limit change, instruments, evidence); three attempts and no fallback | `vxagent/structured/`, `vxagent/robot/config.py`, `frames.py` |
| Approval of changes | Engineer of the site proposes, safety lead of the site approves, never their own; a loosened limit needs a written reason; applied as a new version with the geometric effect on named poses shown first | `vxagent/structured/proposals.py`, `dryrun.py` |
| Trial runtime | A graph of nodes whose requirements are checked by looking at the robot on every entry; event log from which the state is rebuilt; checkpoints; watchdog; recovery; approvals bound to robot state and envelope version; nothing resumes by itself after a restart | `vxagent/runtime/events.py`, `graph.py`, `watchdog.py` |
| Trial pipeline | preflight, dry run on a copy, operator's go, reach along a recorded path, grasp, hold check, place, recover, verdict, summary | `vxagent/pipelines/trial.py`, `paths.py`, `trial_worker.py` |

## What is built: the job tasks on top

| Task | What it does | Code |
|---|---|---|
| 04 Diagnosis and planning | Five diagnosis strategies compared on faults injected into a deployment pipeline that replays recorded dVRK motion, and on labelled issue threads; plans checked against observed facts, rehearsed on a copy of the twin, re-planned once | `vxagent/agent/strategies.py`, `planner.py`, `facts.py`, `vxagent/diagnostics/` |
| 05 Runbooks | Procedures as YAML files with checks and sources, run by code through the tool executor; reference traces; a searchable catalogue | `runbooks/`, `vxagent/skills/` |
| 06 Memory | Site facts written by code from tool results and ended by site events; per-user memory and sessions; similar recorded episodes | `vxagent/memory/`, `vxagent/data/similar.py` |
| 07 MCP | Three servers on the official SDK over Streamable HTTP: tokens, subscriptions, progress, cancellation, and a confirmation the model cannot give | `vxagent/mcp_layer/` |
| 08 Incident triage | One agent, hand-off, fan-out and debate over an append-only evidence ledger; safety review by rule; scored without a model | `vxagent/multi/` |
| 09 Policy trial | Pre-flight with repair runbooks, rehearsal, staged activation, reach by replaying SurgicAI expert rollouts through the policy contract's adapter and the safety filter, grasp, retention, place, machine verdict, operator's confirmation or override | `vxagent/pipelines/policy_trial.py`, `vxagent/robot/preflight.py` |
| 10 Policy and data | Contract drafted from sources with quotations, compatibility with a site, adapter with round trip, episode checks, split by operator, dataset card | `vxagent/policy/`, `vxagent/data/quality.py`, `surgicai.py` |
| 11 Operator | Logins bound to a site; permission to move held per run; copilot with approval cards and a trace behind every answer; no configuration tool at all | `vxagent/access.py`, `vxagent/chat.py`, `vxagent/node.py` |
| 12 Service engineer, safety lead | Tickets; diagnosis on a ticket; calibration sweep and fit as a draft; change proposals under the two-person rule; applying a change ends what was known and approved before it; grounded service report | `vxagent/service/` |
| 13 Remote support | Outbox of counts, kinds and timings with redaction at the edge; store and forward; fleet view; one ticket readable under a time-boxed grant checked at the site | `vxagent/fleet/` |
| 14 Edge deployment | Perception worker as its own process with a time limit; robot gateway with sender allowlists; ROS 2 backend (not verified); model routing set by measurement; degraded modes; egress guard; control-loop timing | `vxagent/perception/`, `vxagent/robot/gateway.py`, `ros2_backend.py`, `timing.py`, `vxagent/llm/router.py` |
| 15 Console and gate | One process for the API and a plain-JavaScript page with four roles; golden traces, replay, trajectory assertions and a gate that refuses a change on any safety failure; public mode that refuses unsafe settings | `vxagent/web/`, `vxagent/evals/`, `scripts/gate.py`, `evals/` |

## When to use what

Use a workflow when the steps and their order are known and a mistake moves metal: here the trial
pipeline and every runbook, where the model writes two pieces of text and cannot change the route. Use
a single agent when the question is open and reading is all it needs to do: the copilots, whose
reliability comes from typed tools, tiers, freshness, the grounding guard and runbooks. Use several
agents when the evidence is of different kinds and too large for one context: incident triage, and
whether that paid off is the measured result of task 08.

| Trade-off | Lever in this system | Measured in |
|---|---|---|
| Reliability | no model-authored numbers; reject without clipping; requirements checked on every resume; rehearsal on the twin; two-person rule; the evaluation gate | T01, T03, T09, T12, T15 |
| Latency | model outside the control loop; small model only where it measured as accurate; runbooks cut model round trips | T14, T05 |
| Cost | token budgets; context fitting; index-then-load for runbooks; local models | T05, T06, T14 |
| Security | role, site and tier in the executor; state-bound approvals; redaction at the edge; time-boxed grants; description pinning; tool text as data; egress guard | T07, T11, T13, T14 |

## What is built: retrieval (`vxrag`)

| Layer | What it does | Code |
|---|---|---|
| Sources | A reStructuredText reader (sections, labels, cross-references, code blocks kept whole), and adapters for issue threads, the change log and JSON Schemas | `vxrag/rst.py`, `vxrag/ingest.py` |
| Store | Authoritative SQLite store of citable units with content hashes; idempotent re-ingestion that removes what disappeared upstream | `vxrag/store.py` |
| Chunking | Fixed windows, sections, sections with breadcrumb; chunks belong to units | `vxrag/chunking.py` |
| Indexing | BM25 with a plain and an identifier-aware tokenizer; two local embedding models with a persistent cache | `vxrag/lexical.py`, `vxrag/dense.py` |
| Retrieval | Lexical, dense and reciprocal-rank fusion as named configurations; filters inside the search | `vxrag/retrieval.py` |
| Evaluation | Labels built by code from the documentation's own links and from thread replies; paired bootstrap; a written selection policy; the test set scored once and locked | `vxrag/evaluation/`, `scripts/rag_experiments.py` |
| Applicability | A site's hardware generation and release filter and annotate results; renames are read from the change log, so a result says which name applies at the site | `vxrag/applicability.py` |
| Grounded answers | Ordered steps with citations; every command, file name, key and number must occur word for word in a cited unit; abstention with a named cause | `vxrag/answer.py`, `vxrag/llm.py` |
| Trust and access | Standing of each source on every answer step; a site's notes are a private collection filtered inside the search and the cache; text addressed to an automated reader is kept from the model | `vxrag/access.py`, `vxrag/untrusted.py` |
| Operations | Index versions (build, validate, activate, roll back), release gates, sync that reports exactly what changed | `vxrag/versions.py`, `vxrag/gates.py` |
| Service | HTTP API with keys, validation, typed errors, a per-caller cache and metrics | `vxrag/service.py`, `vxrag/engine.py` |
| Agent tools | `search_docs`, `search_threads`, `search_by_log_line` over HTTP; the grounding guard checks every citation | `vxagent/tools/knowledge.py` |

```bash
.venv/Scripts/python -m vxrag.cli search "where is console.json" --release 2.3.0 --generation Classic
.venv/Scripts/python -m vxrag.cli answer "How do I calibrate the potentiometers of a PSM?" --generation Classic
.venv/Scripts/python -m vxrag.cli index list
.venv/Scripts/python -m vxrag.cli serve          # http://127.0.0.1:8310, needs a key: vxrag.cli keys add
```

## Run it

Needs Python 3.13 and [Ollama](https://ollama.com) with `qwen3:14b` (about 12 GB of GPU memory).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Linux/macOS: .venv/bin/python
ollama pull qwen3:14b
.venv/Scripts/python scripts/fetch_dvrk_config.py           # dVRK kinematic, instrument, arm and system files and schemas
.venv/Scripts/python scripts/fetch_rosma.py                 # 12 recorded dVRK episodes, 114 MB, CC BY 4.0
.venv/Scripts/python scripts/fetch_sources.py               # documentation text, change log, sites' notes, issue threads, 2 MB
.venv/Scripts/python scripts/fetch_sources.py --only docs-previous   # an earlier documentation commit, for the sync measurement
ollama pull nomic-embed-text && ollama pull bge-m3          # embedding models for dense retrieval, 1.5 GB
.venv/Scripts/python -m vxrag.cli sync                      # all sources into the store; prints what changed
.venv/Scripts/python scripts/fetch_surgicai.py              # SurgicAI expert demonstrations, 10 MB, MIT
ollama pull qwen3:4b                                        # the small model of task 14
.venv/Scripts/python -m vxagent.web.serve                   # the console on http://127.0.0.1:8300
```

The console creates demonstration accounts on first start and writes their generated passwords to
`data/runtime/demo_accounts.json`, which is not in the repository. Other processes:
`python -m vxagent.perception.worker` (perception worker), `python -m vxagent.robot.gateway` (robot
gateway), `python -m vxagent.web.serve --public` (refuses to start with unsafe settings).

## Verify it

```bash
.venv/Scripts/python -m pytest tests -q          # hermetic: made-up arm, scripted model, no downloads needed
.venv/Scripts/python scripts/verify.py t01       # real recordings, real dVRK kinematics, real model
.venv/Scripts/python scripts/verify.py t02       # change requests against the real dVRK files and schemas, 2 minutes
.venv/Scripts/python scripts/verify.py t03       # trials, crashes and faults on the twin in real time, 4 minutes
.venv/Scripts/python scripts/verify.py t04 t05 t06 t07 t08 t09 t10    # about 50 minutes with the model
.venv/Scripts/python scripts/verify.py t11 t12 t13 t14                # the roles, the fleet, the edge: about 10 minutes
.venv/Scripts/python scripts/record_cases.py     # only when the baseline is meant to change: records evals/cases
.venv/Scripts/python scripts/gate.py --all       # the baseline and three planted regressions through the gate
.venv/Scripts/python scripts/gate.py --candidate evals/configs/<name>.json    # a change of your own; exit code 0 if accepted
.venv/Scripts/python scripts/rag_experiments.py  # retrieval experiments on the real corpus, then the test set once
.venv/Scripts/python scripts/rag_verify.py       # R05, R06, R07, R09, R10: about 25 minutes with the model
.venv/Scripts/python scripts/build_report.py     # verification/REPORT.md
```

Measured results are in [`verification/REPORT.md`](verification/REPORT.md). They are written by the
scripts, never by hand. Eleven of the twelve golden cases of the gate are in `evals/cases`; the
twelfth holds passages of the dVRK documentation, which is not redistributed here, and is recorded
again by `scripts/record_cases.py`.

## Data and licences

See [`data/README.md`](data/README.md). ROSMA is CC BY 4.0. The dVRK configuration files, the
documentation text, the issue threads and the sites' notes are used locally and not redistributed
here, because their repositories state no licence (one site repository is MIT). Author names of
issue posts are not stored. No data is committed to this repository. The sites' notes are public;
this project treats each as that site's private collection to exercise access control. The SurgicAI
demonstrations are MIT. `sites/site_a.json` (Classic PSM) and `sites/site_b.json` (Si PSM) are sites
declared for this project, not real labs' set-ups.

## Related work

The safety limits and the trial stages follow the deployment framework in
[On Transferring Surgical Policies from Simulation to Reality](https://github.com/xiangruiSun/On-Transferring-Surgical-Policies-from-Simulation-to-Reality)
(under review). That paper's physical results are not reproduced or claimed here.

The same agent concepts were applied to a chest CT service in
[ai_agent_tools_deepwise](https://github.com/xiangruiSun/ai_agent_tools_deepwise); the job tasks
here are different in every slot.
