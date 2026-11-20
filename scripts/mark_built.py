"""Write the "As built" block of a requirement file. Used once per task when it is finished:

    python scripts/mark_built.py

The block states what differs from the plan below it. It carries no measured numbers; those
are in verification/REPORT.md.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = "As built", "(end of as built)"

NOTES: dict[str, str] = {
    "04_diagnostic_reasoning_and_planning.txt": """\
Status: BUILT and verified. Results: verification/REPORT.md, section T04.
Code: vxagent/agent/strategies.py, planner.py, facts.py; vxagent/diagnostics/episode.py, tools.py;
      vxagent/tools/service.py. Labels: evals/t04_thread_labels.json. Tests: tests/test_reasoning.py.
Differences from the plan below:
- The injected faults live in a small deployment pipeline on the twin (camera, calibration,
  policy steps, action mapping, safety filter, arm) that replays stretches of recorded ROSMA
  motion. Seven causes in four of the five families; nothing is injected for task verification.
- The symptom text is written by code from what an operator could see (how the run ended and
  how far the tip was). Several causes end the same way, which is the point of the comparison.
- "eliminate" has a budget of three of the five checks. It is also run with all five allowed, to
  tell choosing a check apart from reading its result.
- The hypothesis state is typed and written by code only. The model reports what a result
  changes (drops, supports, ranking, next check). The first design, in which the model rewrote
  the whole state at every step, failed and is kept in verification/t04_first_run.json.
- The threads are labelled with their own seven classes (software defect, version mismatch,
  configuration, hardware, environment, documentation, unsupported), not with the five
  sim-to-real families, which do not fit software issue threads. Twenty threads, one labeller.
- A plan's motions are approved one by one through the tool executor; a plan is offered for that
  only after the symbolic check and the rehearsal on the twin.
Further limits: a candidate may be dropped citing a result that says nothing about it; code checks
that the result exists, not that it is relevant. The synthesis is checked for numbers only.""",
    "05_runbooks_as_skills.txt": """\
Status: BUILT and verified. Results: verification/REPORT.md, section T05.
Code: vxagent/skills/runbook.py, index.py; vxagent/robot/needle.py; runbooks/*.yaml;
      tests/runbooks/*.json (reference traces), scripts/record_runbook_traces.py. Tests: tests/test_skills.py.
Differences from the plan below:
- Five runbooks are written: bring_up, rig_check, calibration_validity, verify_grasp_retention,
  return_to_start. instrument_change and collect_calibration_samples belong to the service
  engineer and arrive with task 12.
- The twin has no needle. Procedures about the needle observe a DECLARED needle that follows the
  arm (it is taken when the jaw closes at the grasp pose; it can be declared to slip on the lift
  or while held). Everything it reports is a consequence of the declaration.
- A step cites a documentation unit id of the retrieval corpus where one exists, and the paper's
  framework otherwise. Only the first kind is checked by code.
- In the atomic design the procedures are rendered from the same runbook files into the prompt,
  so the three designs know the same procedure.
Further limits: eight service requests, one model, one run each.""",
    "06_memory_and_technical_retrieval.txt": """\
Status: BUILT and verified. Results: verification/REPORT.md, section T06 (and the RAG sections for Part A).
Code: vxagent/memory/site.py, stores.py, working.py; vxagent/tools/memory.py; vxagent/data/similar.py.
Tests: tests/test_memory.py.
Differences from the plan below:
- R6: the search has no score threshold. It returns its best passages for any wording, so
  refusing a question about another robot is left to the agent and its grounding guard, and the
  verification counts how often that works.
- R7: the recordings have no jaw signal, so there are no jaw events among the episode features.
  Twelve recordings are local; nearest-neighbour agreement is reported against chance.
- R12: recording an event that ends a calibration also marks the calibration record itself
  invalid. A first run showed the remembered fact expired while the record still read valid.
- R9: the robot's current state can be put in front of the model on every turn. It is placed
  beside the task, not at the end: placed last, the model answered it instead of the question.
Further limits: site facts are written for one kind of tool result so far (a calibration residual).""",
    "07_mcp_robot_boundary.txt": """\
Status: BUILT and verified. Results: verification/REPORT.md, section T07.
Code: vxagent/mcp_layer/servers.py (the three servers), client.py, host.py. Tests: tests/test_mcp.py.
Differences from the plan below:
- Built on the official SDK, version 2.3.0 (protocol 2026-07-28), which was inspected and probed
  before anything was written. In this protocol version a server cannot send a request of its
  own over Streamable HTTP, so a motion's confirmation is an input-required round: the server
  answers "input required", the client asks the connected person and repeats the call. The
  answer is recorded and enforced at the server; the model has no parameter for it.
- Subscriptions use subscriptions/listen. A change of the robot's state made outside the server
  is published as well as one made through it.
- The three servers are one module with three configurations, not three files.
- stdio is not used, also not in tests: the tests run the servers over HTTP on this machine.
Not built: the OAuth authorization flow (a local token table stands in), TLS.""",
    "08_multi_agent_incident_triage.txt": """\
Status: BUILT and verified. Results: verification/REPORT.md, section T08.
Code: vxagent/multi/ledger.py, team.py. Tests: tests/test_multi.py.
Differences from the plan below:
- The incidents are the injected faults of task 04 on recorded ROSMA motion. The AMBF scenes and
  the paper's failure names are not used.
- A specialist is one structured model call over its own check results; the ledger refuses a
  claim whose number is not in the data or that points at another role's data.
- The single agent gets every check result in one call and writes claims and finding together.
- Debate: after the fan-out, the specialists who own the two best-supported causes each get one
  call to test the other cause against their own data.
- The runbook for a cause is a fixed table given to the lead, so that check is a lookup.
Further limits: one run per incident and design; the specialists do not choose which checks to run.""",
    "09_trial_pipeline.txt": """\
Status: BUILT and verified on the twin. Results: verification/REPORT.md, section T09.
Code: vxagent/pipelines/policy_trial.py, vxagent/robot/preflight.py. Tests: tests/test_policy_trial.py.
Differences from the plan below:
- Reach replays real SurgicAI expert rollouts: each recorded action goes through the adapter
  generated from the policy contract (task 10) and through the safety filter. Every Approach
  demonstration starts from one pose and goes to one goal, so a trial first brings the arm to
  that start, anchored on the site's grasp pose.
- The jaw component of the recorded actions is not used: the jaw belongs to the grasp procedure.
- Place uses the published approach heights over the site's place pose; the Place
  demonstrations are not replayed.
- Staged activation: the first control step of each stage is sent alone and its tracking is
  checked before the rest of the stage.
- The needle is the declared needle of task 05. "recorded" outcomes do not exist here.
- The verdict record, the operator's confirmation and an override with its reason are kept side
  by side in the trial record.
- The twin's kinematics are checked against ROSMA: forward kinematics of the recorded joint
  values against the recorded positions, after one rigid transform per recording.
Further limits: perception is a callable with a declared latency until task 14 gives it a process.""",
    "10_policy_onboarding_and_episode_data.txt": """\
Status: BUILT and verified. Results: verification/REPORT.md, section T10; verification/t10_dataset_card.md.
Code: vxagent/policy/contract.py, compat.py, adapter.py; vxagent/data/surgicai.py, quality.py;
      vxagent/tools/dataset.py; scripts/fetch_surgicai.py. Reference: evals/t10_reference_contract.json.
Tests: tests/test_policy_data.py.
Differences from the plan below:
- The demonstrations are pickle files. They are read with a restricted unpickler that rebuilds
  arrays and plain containers and refuses anything else; the environment sources are read as
  text and never imported.
- The contract is drafted from two source files. A field is kept only with a quotation that
  occurs in the named file. What the files do not state (the step size, the control rate, the
  range of start poses) is left empty by rule and filled from the demonstrations by code, with
  that provenance written into the contract.
- split.py and card.py are functions of vxagent/data/quality.py.
- SurgicAI names no operator or session, so its episodes cannot be split by operator: they stay
  on the training side as one session.
Further limits: one policy (Approach). The checkpoint hash is not recorded because no checkpoint
is loaded; the rollouts are the recorded expert demonstrations.""",
}


def main() -> int:
    for name, note in NOTES.items():
        path = ROOT / "requirements" / name
        text = path.read_text(encoding="utf-8")
        if BEGIN in text and END in text:
            head, rest = text.split(BEGIN, 1)
            text = head.rstrip("\n") + "\n" + rest.split(END, 1)[1].lstrip("\n")
        lines = text.split("\n")
        block = [BEGIN, "-" * len(BEGIN), *note.split("\n"), END, ""]
        at = 2 if lines[1].startswith("=") else 1
        while at < len(lines) and lines[at].strip():       # keep the "Covers:" lines with the title
            at += 1
        new = lines[:at] + [""] + block + lines[at + 1:] if at < len(lines) else lines + [""] + block
        path.write_text("\n".join(new), encoding="utf-8", newline="\n")
        print("marked", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
