# dVRK Agent Tools

Role-scoped AI agent tools for operating and servicing a da Vinci Research Kit (dVRK): an operator
who runs trials, a service engineer who calibrates and repairs, a safety lead who approves anything
that moves or changes a limit, and a remote support team that sees fleet health without seeing the
lab's data.

**Research demonstration on public data. Not connected to a robot, not a medical device.** The robot
is a kinematic twin built from the published dVRK kinematic files. It has no contact, no needle and
no camera, and nothing measured here is a statement about a physical robot.

**Status: in progress.** Task 01 (tool calling against a live system) is built and verified, with
the kinematic twin and the safety filter it depends on. Tasks 02 to 15 are specified in
[`requirements/`](requirements/00_overview.txt) and not built yet.

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
```

## Verify it

```bash
.venv/Scripts/python -m pytest tests -q          # hermetic: made-up arm, scripted model, no downloads needed
.venv/Scripts/python scripts/verify.py t01       # real recordings, real dVRK kinematics, real model
.venv/Scripts/python scripts/build_report.py     # verification/REPORT.md
```

Measured results are in [`verification/REPORT.md`](verification/REPORT.md). They are written by the
scripts, never by hand.

## Data and licences

See [`data/README.md`](data/README.md). ROSMA is CC BY 4.0. The dVRK configuration files are used
locally and not redistributed here, because their repository states no licence. No data is
committed to this repository. `sites/site_a.json` is a site declared for this project, not a real
lab's set-up.

## Related work

The safety limits and the trial stages follow the deployment framework in
[On Transferring Surgical Policies from Simulation to Reality](https://github.com/xiangruiSun/On-Transferring-Surgical-Policies-from-Simulation-to-Reality)
(under review). That paper's physical results are not reproduced or claimed here.

The same agent concepts were applied to a chest CT service in
[ai_agent_tools_deepwise](https://github.com/xiangruiSun/ai_agent_tools_deepwise); the job tasks
here are different in every slot.
