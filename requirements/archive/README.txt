ARCHIVE: THE EARLIER JOB TASKS
==============================

These files are the job tasks this repository was first built around:

  agent/   fifteen tasks for an AI agent on a surgical-robot (dVRK) console: tool calling,
           structured proposals, runtime, diagnosis, runbooks, memory, MCP at the robot boundary,
           multi-agent triage, trials, policy onboarding, operator and service access, fleet
           support, edge deployment, console evaluation.
  rag/     eleven tasks for retrieval over the robot's documentation.

After a review with the team's mentor they were set aside as the main line of work: they
describe a product without a client, and their retrieval evaluation uses labels derived from
document structure instead of from what a user asks. They are kept unchanged, with their "As
built" blocks. Their code (vxagent/, vxrag/), tests and evidence (verification/t*.json,
rag_*.json and the corresponding sections of verification/REPORT.md) are also kept and still
run; the new work reuses the agent loop, the tool registry and the MCP client and host.

The current job tasks are in requirements/00_overview.txt, requirements/system/ and
requirements/eval/.
