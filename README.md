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

- **`vxrag`**, a retrieval system over the dVRK documentation, issue threads, change log and
  configuration schemas. It imports nothing from the agents and can be moved to its own repository
  unchanged. Requirements: [`requirements/rag/`](requirements/rag/R00_overview.txt).
- **`vxagent`**, the agent tools that will call it. Requirements:
  [`requirements/`](requirements/00_overview.txt).

**Status: in progress.** Built and measured so far: the agents' task 01 (tool calling against a live
system, with the kinematic twin and the safety filter), and the retrieval system's ingestion,
chunking, lexical and dense indexing, retrieval strategies and evaluation protocol. Not built yet:
agent tasks 02 to 15; grounded answers, applicability by hardware generation and release, trust and
access, the service, and the agents' search tools.

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
| Agent loop | model → tool → observation → model, twelve explicit stop reasons; nothing keeps moving after a stop | `vxagent/agent/loop.py` |
| Grounding guard | Rejects answers whose numbers, robot state or citations no tool returned | `vxagent/agent/guards.py` |
| Recordings | Statistics and bounded windows of recorded dVRK episodes; raw recordings never reach the model | `vxagent/data/episodes.py` |

## What is built: retrieval (`vxrag`)

| Layer | What it does | Code |
|---|---|---|
| Sources | A reStructuredText reader (sections, labels, cross-references, code blocks kept whole), and adapters for issue threads, the change log and JSON Schemas | `vxrag/rst.py`, `vxrag/ingest.py` |
| Store | Authoritative SQLite store of citable units with content hashes; idempotent re-ingestion that removes what disappeared upstream | `vxrag/store.py` |
| Chunking | Fixed windows, sections, sections with breadcrumb; chunks belong to units | `vxrag/chunking.py` |
| Indexing | BM25 with a plain and an identifier-aware tokenizer; two local embedding models with a persistent cache | `vxrag/lexical.py`, `vxrag/dense.py` |
| Retrieval | Lexical, dense and reciprocal-rank fusion as named configurations; filters inside the search | `vxrag/retrieval.py` |
| Evaluation | Labels built by code from the documentation's own links and from thread replies; paired bootstrap; a written selection policy; the test set scored once and locked | `vxrag/evaluation/`, `scripts/rag_experiments.py` |

```bash
.venv/Scripts/python -m vxrag.cli search "firewire nodes not found after power on"
```

## What is planned

| Task | Job | Requirement file |
|---|---|---|
| 02 | Configuration changes as typed, validated patches | `02_structured_change_proposals.txt` |
| 03 | Event-driven trial runtime with recovery and re-verified resume | `03_event_driven_runtime.txt` |
| 04 | Fault diagnosis strategies; plans checked against robot state | `04_diagnostic_reasoning_and_planning.txt` |
| 05 | Runbooks as versioned skills, loaded on demand | `05_runbooks_as_skills.txt` |
| 06 | Technical retrieval and site memory that expires | `06_memory_and_technical_retrieval.txt` |
| 07 | MCP servers at the robot boundary | `07_mcp_robot_boundary.txt` |
| 08 | Multi-agent incident triage | `08_multi_agent_incident_triage.txt` |
| 09 | Trial pipeline: pre-flight, rehearsal, run, verdict | `09_trial_pipeline.txt` |
| 10 | Policy interface contracts and episode data checks | `10_policy_onboarding_and_episode_data.txt` |
| 11–13 | Operator, service engineer and safety lead, remote support and fleet | `11_…`, `12_…`, `13_…` |
| 14 | Edge deployment beside the robot | `14_edge_deployment.txt` |
| 15 | Console, evaluation gate and trade-offs | `15_console_evaluation_and_tradeoffs.txt` |

## Run it

Needs Python 3.13 and [Ollama](https://ollama.com) with `qwen3:14b` (about 12 GB of GPU memory).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Linux/macOS: .venv/bin/python
ollama pull qwen3:14b
.venv/Scripts/python scripts/fetch_dvrk_config.py           # dVRK kinematic files and schemas, 0.28 MB
.venv/Scripts/python scripts/fetch_rosma.py                 # 12 recorded dVRK episodes, 114 MB, CC BY 4.0
.venv/Scripts/python scripts/fetch_sources.py               # documentation text, change log, issue threads, 2 MB
ollama pull nomic-embed-text && ollama pull bge-m3          # embedding models for dense retrieval, 1.5 GB
```

## Verify it

```bash
.venv/Scripts/python -m pytest tests -q          # hermetic: made-up arm, scripted model, no downloads needed
.venv/Scripts/python scripts/verify.py t01       # real recordings, real dVRK kinematics, real model
.venv/Scripts/python scripts/rag_experiments.py  # retrieval experiments on the real corpus, then the test set once
.venv/Scripts/python scripts/build_report.py     # verification/REPORT.md
```

Measured results are in [`verification/REPORT.md`](verification/REPORT.md). They are written by the
scripts, never by hand.

## Data and licences

See [`data/README.md`](data/README.md). ROSMA is CC BY 4.0. The dVRK configuration files, the
documentation text and the issue threads are used locally and not redistributed here, because
their repositories state no licence. Author names of issue posts are not stored. No data is
committed to this repository. `sites/site_a.json` is a site declared for this project, not a real
lab's set-up.

## Related work

The safety limits and the trial stages follow the deployment framework in
[On Transferring Surgical Policies from Simulation to Reality](https://github.com/xiangruiSun/On-Transferring-Surgical-Policies-from-Simulation-to-Reality)
(under review). That paper's physical results are not reproduced or claimed here.

The same agent concepts were applied to a chest CT service in
[ai_agent_tools_deepwise](https://github.com/xiangruiSun/ai_agent_tools_deepwise); the job tasks
here are different in every slot.
