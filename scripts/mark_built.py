"""Write the "As built" block of a requirement file. Used once per task when it is finished:

    python scripts/mark_built.py

The block states what differs from the plan below it. It carries no measured numbers; those
are in verification/REPORT.md.
"""
from __future__ import annotations

from pathlib import Path

from built_notes_vxfind import NOTES as VXFIND_NOTES

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = "As built", "(end of as built)"
# Where a requirement file may be: the current job tasks, then the archived ones.
FOLDERS = ("system", "eval", "archive/agent", "archive", "")

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
    "11_operator_access.txt": """\
Status: BUILT and verified. Results: verification/REPORT.md, section T11.
Code: vxagent/access.py (users, sessions, lock-out, actuate per run), vxagent/chat.py, vxagent/node.py,
      vxagent/tools/trialtools.py, vxagent/profiles.py (OPERATOR_COPILOT), vxagent/web/app.py.
Tests: tests/test_operator_access.py.
Differences from the plan below:
- The web layer is Starlette, which is on the machine with the MCP SDK, not FastAPI: nothing
  further has to be installed on a lab network.
- Actuate per run applies to console sessions. An operator's console principal holds observe and
  simulate; a motion tool is accepted only inside an active run of that operator that was
  registered with that tool, and then needs the approval card. The verification harnesses of
  tasks 01 to 10 use a standing operator principal that holds actuate.
- Starting a trial, the go and the verdict are console buttons. The copilot can check whether a
  trial can start and rehearse it; it has no tool that starts one.
- calibration_validity may also be run by the operator: it only reads.
- The operator's copilot searches the documentation; thread search is offered to the service
  engineer and to support.
- An approval card is valid for 120 seconds and shows the rehearsal of a move to a named pose.
- First run kept in verification/t11_first_run.json: the pre-flight tool returned a failed check
  under its name, which states the requirement ("the calibration has not been ended by an
  event"), and the copilot repeated the name as the reason. The pre-flight now writes a sentence
  that says what is wrong for every failed check; the tool and the ticket carry that sentence.
Further limits: most probes were refused by the copilot in words, without a tool call; "held" is
judged on effects (nothing moved, no limit changed, no trial started), not on the wording.""",
    "12_service_engineer_and_safety_lead.txt": """\
Status: BUILT and verified. Results: verification/REPORT.md, section T12.
Code: vxagent/service/tickets.py, diagnosis.py, calibration.py, changes.py, report.py, tools.py;
      vxagent/profiles.py (SERVICE, SAFETY); runbooks/collect_calibration_samples.yaml,
      instrument_change.yaml; vxagent/web/app.py. Tests: tests/test_service_access.py.
Differences from the plan below:
- Diagnosis on a ticket is started from the console, not by a copilot tool. It works on a
  diagnostic run: recorded motion replayed through the deployment pipeline on a copy of the twin
  with whatever was injected into the node's declared world. Every check it runs becomes a
  ticket record, and the FaultReport is written by code around the cause the model names.
- The calibration sweep is one action over all declared calibration poses under one approval
  by an operator, not one approval per pose. The engineer holds the scope for this one tool and
  no scope to move the arm otherwise.
- The camera is DECLARED: what it sees is computed from a declared transform with declared
  noise, and the camera move is INJECTED.
- Activating a calibration goes through the same proposal record and the same two-person rule
  as a configuration patch, as a change of kind "calibration".
- Site events: limits_changed was added; a calibration change records calibration_redone; any
  other configuration change records software_updated.
- First run kept in verification/t12_first_run.json: the grounding guard refused two correct
  answers because the unit was not at the end of the key that carried the number, and refused
  values the engineer had stated in the request; and the copilot gave a ticket number where an
  evidence record id was needed. Tool results now carry the unit in the key, values stated in
  the request may be repeated, and the prompt says what an evidence id is.
Further limits: one diagnostic run per ticket; the affine correction is stored with the record
but the pre-flight residual check uses the rigid transform only.""",
    "13_remote_support_and_fleet.txt": """\
Status: BUILT and verified. Results: verification/REPORT.md, section T13.
Code: vxagent/fleet/redact.py, outbox.py, sync.py, store.py, grants.py, tools.py, desk.py;
      sites/site_b.json; vxagent/profiles.py (SUPPORT); vxagent/web/app.py. Tests: tests/test_fleet.py.
Differences from the plan below:
- Redaction is a closed description of what a record may contain (known keys, numbers, and
  strings from fixed vocabularies), checked at the outbox. A record that does not fit is refused
  and audited; nothing is cleaned. A kind outside the vocabulary is reported as "other".
- The fleet store is reached over HTTP with one token per site. It accepts a record only if its
  number is the next one for that site, so duplicates are ignored and nothing is stored out of
  order.
- The fleet desk and the edge nodes run in one process in this demonstration; a granted ticket
  is read by calling the edge node's own check directly. Between two labs that call would be an
  HTTP request to the node.
- Site B has no recordings: the ROSMA recordings are given to site A.
- The fleet view shows totals and the latest health of each site. The records are stored with
  their times, but no chart over time is drawn.
Further limits: as stated below, counts can still say how often and when a lab works.""",
    "14_edge_deployment.txt": """\
Status: BUILT and verified on this PC. Results: verification/REPORT.md, section T14.
Code: vxagent/robot/gateway.py, ros2_backend.py, timing.py; vxagent/perception/worker.py, client.py;
      vxagent/llm/router.py; vxagent/node.py; vxagent/config.py. Tests: tests/test_edge.py.
Differences from the plan below:
- R1: move_cp is not a backend command. A move is executed by the agent side as servo_cp steps,
  each checked by the safety filter, so no interpolation happens behind the interface.
- R3: the ros2 backend is written and NOT verified: there is no ROS 2, no dVRK software and no
  arm on the development PC. Every topic name in it has to be checked on a lab machine first.
- R4: the gateway checks the address a connection comes from against two allowlists and, for
  commands, the agent process's token. The trials of the verification command the twin in the
  node's own process; the gateway is exercised separately, with a motion through it.
- R6: the perception worker serves the goal poses recorded with the SurgicAI demonstrations
  (ground truth of the AMBF scenes). It serves no depth and computes nothing.
- R8: a step is routed to the small model only if it was at least as accurate there as the
  large one on the measurement. Two of the three classification steps therefore stay on the
  large model.
- Two earlier runs are kept (verification/t14_first_run.json, t14_second_run.json): a perception
  request took about 0.2 s for 0.04 s of declared work. The first explanation, the worker's two
  writes per reply, was wrong and changed nothing. The cause was that every request created a
  new HTTP client. The process now keeps one client open (vxagent/net.py); this also applies to
  model calls, retrieval calls and gateway commands, whose earlier timings include that cost.
- Keeping connections open exposed a fault in the retrieval service (vxrag/service.py): after
  an error sent before the request body was read, the next request on the same connection was
  answered with 501. The service now reads the unread body first and ends the connection.
Further limits: as stated below. The start-up rule is covered by task 03's verification and by
tests/test_edge.py, not measured again here.""",
    "15_console_evaluation_and_tradeoffs.txt": """\
Status: BUILT and verified. Results: verification/REPORT.md, section T15; verification/gate_*.json.
Code: vxagent/web/app.py, serve.py, public.py, static/; vxagent/evals/cases.py, replay.py, assertions.py,
      compare.py; scripts/record_cases.py, scripts/gate.py; evals/cases/, evals/configs/, evals/gate.json.
Tests: tests/test_gate.py, tests/test_operator_access.py (routes, sessions, event stream).
Differences from the plan below:
- R1: Starlette, not FastAPI (see task 11).
- R4: the time axis shows the trial's events, approvals and model calls; for a selected moment
  it shows the twin's pose as it was last observed on entering a stage, not a pose per control tick.
- R10: the golden cases are twelve runs of the operator's copilot recorded for this purpose by
  scripts/record_cases.py, not runs taken from the other tasks' verification.
- R11: a replay's prompt is the candidate's prompt followed by what the node added when the case
  was recorded (runbook index, site, memory), so only the candidate's own edits differ.
- R12: "a look before a motion" follows the executor's rule: a motion must be based on the state
  number the latest tool result reported, whichever tool that was. The first recording, kept in
  verification/t15_first_recording.json, used a narrower rule and also showed the model server
  returning an empty reply; see the report.
- R15: the cases marked live are run on the twin for every candidate.
- Part D: public mode is built and tested (vxagent/web/public.py, `serve --public`). No tunnel
  was opened for this verification.
- Browser run on 2026-11-20 on the local console with the four demonstration logins. Operator:
  robot view, a trial that failed pre-flight and opened a ticket, the copilot bringing the arm up
  through an approval card, a trial with go and confirmation and its time axis. Service engineer:
  ticket queue, calibration page, a change proposal filed. Safety lead: the proposal with its
  effect on the twin, approved and applied; a support grant given. Remote support: fleet view,
  version drift, the granted ticket read, every site page refused. The browser pane was hidden
  for the operator's last steps, so those were driven through the page's own requests, not by
  clicks. The run found two faults, both fixed and pinned in tests/test_found_by_verification.py:
  the page read the tip position from the wrong field, and after a restart a node ignored a
  change that had been applied to its site file.
Further limits: the gate was run on one model server with temperature 0; a baseline replayed
twice is not guaranteed to be identical on other hardware.""",
}


def main() -> int:
    for name, note in {**NOTES, **VXFIND_NOTES}.items():
        path = next(p for p in (ROOT / "requirements" / d / name for d in FOLDERS) if p.exists())
        text = path.read_text(encoding="utf-8")
        if BEGIN in text and END in text:
            head, rest = text.split(BEGIN, 1)
            # the blank line that ends the title block stays, or the next marking would land below "Job"
            text = head.rstrip("\n") + "\n\n" + rest.split(END, 1)[1].lstrip("\n")
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
