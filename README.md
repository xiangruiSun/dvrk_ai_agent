# dVRK Agent Tools

Role-scoped AI agent tools for operating and servicing a da Vinci Research Kit (dVRK): an operator
who runs trials, a service engineer who calibrates and repairs, a safety lead who approves anything
that moves or changes a limit, and a remote support team that sees fleet health without seeing the
lab's data.

**Status: design. No code has been written and no result has been measured.** This repository
currently holds the requirements, one file per job task, in [`requirements/`](requirements/00_overview.txt).

**Research demonstration on public data. Not connected to a robot, not a medical device.** The
default robot backend will be a kinematic twin built from the published dVRK kinematic files. It
has no contact, no needle and no camera.

## What is planned

| Task | Job | Requirement file |
|---|---|---|
| 01 | Tool calling against a live system: actions with progress and cancel, stale-state checks | `01_tool_calling_live_system.txt` |
| 02 | Configuration changes as typed, validated patches | `02_structured_change_proposals.txt` |
| 03 | Event-driven trial runtime with recovery and re-verified resume | `03_event_driven_runtime.txt` |
| 04 | Fault diagnosis strategies; plans checked against robot state | `04_diagnostic_reasoning_and_planning.txt` |
| 05 | Runbooks as versioned skills, loaded on demand | `05_runbooks_as_skills.txt` |
| 06 | Technical retrieval and site memory that expires | `06_memory_and_technical_retrieval.txt` |
| 07 | MCP servers at the robot boundary | `07_mcp_robot_boundary.txt` |
| 08 | Multi-agent incident triage | `08_multi_agent_incident_triage.txt` |
| 09 | Trial pipeline: pre-flight, rehearsal, run, verdict | `09_trial_pipeline.txt` |
| 10 | Policy interface contracts and episode data checks | `10_policy_onboarding_and_episode_data.txt` |
| 11 | Operator access | `11_operator_access.txt` |
| 12 | Service engineer and safety lead | `12_service_engineer_and_safety_lead.txt` |
| 13 | Remote support and fleet view | `13_remote_support_and_fleet.txt` |
| 14 | Edge deployment beside the robot | `14_edge_deployment.txt` |
| 15 | Console, evaluation gate and trade-offs | `15_console_evaluation_and_tradeoffs.txt` |

## Central rule

The language model never produces a number that reaches the robot. It chooses a validated
procedure and its typed parameters. Targets are computed by code and checked by a safety filter
that rejects, and never clips, anything outside the envelope.

## Data

All public, downloaded by scripts, none committed: the dVRK documentation and support threads, the
dVRK configuration files and schemas, the ROSMA recordings (CC BY 4.0), and SurgicAI simulator
episodes (MIT, CC BY-SA 4.0). Sources, sizes and licences are listed in
[`requirements/00_overview.txt`](requirements/00_overview.txt).

## Related work

The trial stages and safety limits follow the deployment framework in
[On Transferring Surgical Policies from Simulation to Reality](https://github.com/xiangruiSun/On-Transferring-Surgical-Policies-from-Simulation-to-Reality)
(under review). That paper's physical results are not reproduced or claimed here.

The same agent concepts were applied to a chest CT service in
[ai_agent_tools_deepwise](https://github.com/xiangruiSun/ai_agent_tools_deepwise); the job tasks
here are different in every slot.
