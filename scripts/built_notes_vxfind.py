"""The "As built" notes of the vxfind job tasks, written into the requirement files by
scripts/mark_built.py. They say what was built here and what differs from the plan. They carry
no measured numbers; those are in verification/REPORT.md, Part 1."""

NOTES: dict[str, str] = {
    "S1_sources_and_catalog_record.txt": """\
Status: BUILT and run on the real catalog, except what "Not built here" lists. Results: REPORT.md, "Write path".
Code: scripts/fetch_hub_catalog.py; vxfind/catalog/parse.py, store.py. Tests: tests/test_vxfind.py.
Differences from the plan below:
- Source 4 (pipeline documentation, PDFs) is not ingested: there are no company documents here and
  no download of stand-ins was asked for.
- The dump holds no card for most gated tables, so gating is nearly absent from the indexed catalog
  although the listing shows several hundred gated tables. Gating is therefore tested only as a field.
- R7 turned out to matter more than planned: the dump never drops a deleted table (measured in E08),
  so "only in the dump" is where the dead tables are, mixed with live ones the listing omits.
Further limits: the 50-row hand check of the parser (Acceptance) was done on far fewer rows, by the
builder, while writing it; the tests pin the cases that were seen.""",
    "S2_write_path_spark_and_embeddings.txt": """\
Status: BUILT and run on both snapshots. Results: REPORT.md, "Write path".
Code: vxfind/ingest/spark_job.py, embed.py, index.py; scripts/elastic_local.py.
Differences from the plan below:
- Spark runs in local mode on one Windows workstation. Reading a glob and writing through Spark
  failed without the Hadoop native helper (the first trial hung for ten minutes); files are passed by
  explicit URI and the result is written from the driver through Arrow. Noted in the code.
- The "deleted" branch of the diff and of the sync is coded and has never fired: the source reports
  no deletions.
- The incremental sync is measured on a second copy of the index that does not serve, because the
  evaluation measures an index that is behind on purpose. The synced copy is left in the engine as
  the next version; the alias was not switched.
- The engine's own rank fusion is refused under the free licence (verified against the running
  engine), which is why S3 fuses in the caller.
Further limits: one node, one shard, no replica; no scheduler; R9 (a build with failures must not
switch the alias) is not enforced in code, only reported.""",
    "S3_index_and_read_path.txt": """\
Status: BUILT and evaluated (E06, E07). Results: REPORT.md, "Retrieval" and "Model stages".
Code: vxfind/search/engine.py. Tests: tests/test_vxfind.py.
Differences from the plan below:
- R4 was not in the first design. It was added after the first search on the real index returned a
  table named "a" for a question containing "a". The old behaviour is the setting ident_stop=False
  and is one of the measured configurations.
- The reranker is a general language model reading summaries, not a cross-encoder.
- The serving configuration is chosen by E06's rule and stored in evals/vxfind/serving.json; the
  engine has no opinion about which settings are good.
- R6 changed after E07 measured it: the reading prompt was rewritten (the first one made the model
  leave the robot empty), the reader can be the small or the large model, and in "hard" only numbers
  and the format filter; the robot's name boosts.
Further limits: the robot is matched as text, not by family, so a need for "Universal Robots" is not
tied to "ur5e"; the family rules belong to the evaluation and are deliberately not used by search
(search using the rules that also define set A's truth would make that set circular).""",
    "S4_mcp_servers.txt": """\
Status: BUILT and tested against the live site and by the leak tests. Results: REPORT.md, "Freshness" and "Access".
Code: vxfind/mcp/servers.py, hub.py, access.py. Tests: tests/test_vxfind.py.
Differences from the plan below:
- R5 was violated by the first version and the leak test caught it: check_tables answered a hidden
  table itself and passed made-up ids to the site, so the two replies differed in their reason. Now
  any id the catalog does not hold, or the caller may not see, gets one reply and the site is asked
  about neither. First run: verification/vxfind_access_first_run.json. Pinned in the tests.
- A consequence of that fix: the new address of a moved table is not in the catalog database, so a
  second check of the NEW id answers "not in the catalog". The first check already gives the new
  address and its state; the agent's answer check accepts it.
- The three servers run in one process on loopback ports; tokens come from a local table.
Further limits: check_tables asks about its ids one after another (the evaluation measured what four
at a time would save); no notification reaches a mentor; approvals do not expire.""",
    "S5_agent_and_fixed_pipeline.txt": """\
Status: BUILT and evaluated on cases with known right behaviour (E10). Results: REPORT.md, "Agent".
Code: vxfind/agent.py; reuses vxagent/agent/loop.py, vxagent/tools/registry.py, vxagent/mcp_layer/.
Differences from the plan below:
- R2 failed in the first run of the agent cases: twenty candidates at full length were larger than
  the executor accepts, the tool failed, the model asked again and gave up. The tool now cuts the
  summaries until the reply fits (pinned by a test). First run: verification/vxfind_agent_first_run.json.
- R9: the pipeline showed one table twice when a moved table's new address was also in the list
  (found by reading a real answer). It now shows a table once.
- The agent does not stay silent for a need with no answer: it lists the nearest tables and says they
  do not fit. The selection stage, which does abstain (E07), is not inside the agent's search tool.
- There is no chat interface. The assistant is a function; the evaluation drives it.
- R6's budgets are those of the reused loop; nothing was tuned for this task.
- The degraded modes of S6 R5 that the code paths already give are present (site unreachable ->
  "could not be verified"; selection failing -> the ranked list as it is); the others are not coded.
Further limits: evaluated with two local models, one run per case.""",
    "S6_serving_capacity_and_rollout.txt": """\
Status: PARTLY BUILT. Measured: capacity, storage, growth, the feedback path (SCRIPTED sessions), the
release gate. Not built: a service, deployment, dashboards, real roll-out stages. Results: REPORT.md,
"Capacity" and "Roll-out and the release gate".
Code: vxfind/eval/capacity.py, feedback.py, gate.py; scripts/vxfind_ops.py.
Differences from the plan below:
- R2's first run measured the client, not the engine: with the search client's default of ten
  connections, callers queued inside the client and the queueing appeared as a tail of seconds. The
  pool is now sized for the callers; first run kept (verification/vxfind_capacity_first_run.json).
  Throughput stops growing between 4 and 16 callers, and it was not separated whether the Python
  load generator or the engine is the limit there.
- R7's decision changed character once E08 was run: the dead addresses are in the source, not a
  product of index age, so the missing piece is a deletion sweep in the write path, and the live
  check is what protects users until that exists.
- R8: no stage beyond the builder's own use has happened. There are no users.
- R1's latency targets are not set; the measured latencies are reported for the lead to set them.
Further limits: one machine; the load generator, the engine and the embedding model share it.""",
    "E00_evaluation_plan_zero_to_one.txt": """\
Status: the plan below was carried out in compressed form by the builder of the system, not over six
months by a newcomer to it. What exists: every deliverable of section 2 as code, evidence and report
sections. What does not: human relevance labels (E05), a second author of questions (E02), real
users (E12), repeated agent runs (E10). Each is stated in the report where its number would be.""",
    "E01_failure_analysis_before_metrics.txt": """\
Status: PARTLY DONE. The taxonomy (evals/vxfind/failure_taxonomy.md) was written by the builder from
the runs in this repository, with real examples; the generated part (scripts/vxfind_failures.py ->
verification/vxfind_failures.json, REPORT.md "Where the serving configuration fails") is built.
Differences from the plan below:
- R1 (100 rows read end to end with a tally sheet) and R2's count of 50 needs were not done as
  written: 40 needs were written before any result was seen, and the catalog was studied through
  its distributions and through the first searches, not through a row-by-row sheet.
- R7's sign-off by a team did not happen; the four grades are the builder's.
- The value of looking first showed immediately: the function-word fault (F1) and the
  contradiction-blind judge were both found by reading results, not by a metric.""",
    "E02_query_sets_and_sources_of_truth.txt": """\
Status: BUILT. Sets A, B, C, D are in evals/vxfind/queries/. Results: REPORT.md, "Query sets".
Code: vxfind/eval/families.py, queries.py; scripts/vxfind_queries.py.
Differences from the plan below:
- Set C has 40 needs by one author (the builder), not 50, and no second author.
- Set A was repaired in place after the judge calibration showed two faults of the "truth": a table
  with no episodes counted as a match, and one family's label listed three models and so excluded
  the others. Ids and specs are unchanged; the counts and six wordings changed (stage A_recount).
- R5 did its job: needs written as having no answer turned out to have one, and were moved out.
- Set B's needs often repeat exact figures from the description. They were kept; the bias this
  gives lexical search is visible in the results and discussed in E11.
- Set F (from feedback) holds candidates only; nobody has labelled them.""",
    "E03_metrics_and_statistics.txt": """\
Status: BUILT, with hand-worked tests. Code: vxfind/eval/metrics.py, harness.py. Tests: tests/test_vxfind.py.
Differences from the plan below:
- The paired bootstrap and the randomization test can disagree at the margin, and did (an interval
  excluding zero beside p above 0.05 in the first run). The selection rule and the gate use the
  randomization test; the report prints both.
Further limits: no correction for multiple comparisons is applied; the report states how many
comparisons a table holds.""",
    "E04_pooling_and_the_relevance_judge.txt": """\
Status: BUILT and run. Judgments: evals/vxfind/qrels/. Results: REPORT.md, "Retrieval" (the judging block).
Code: vxfind/eval/harness.py (run, pool), judge.py.
Differences from the plan below:
- Implementation step 4 (judge twenty pairs and READ the reasons before the full run) was skipped
  the first time, and it cost a full re-judging: the first rubric let contradicted requirements
  through. The rubric is now at its third revision (g2 -> g3 -> g4); every judgment of every
  version is still in the file, and the first full run is kept as evidence.
- Under the procedural rubric the judge almost never gives grade 2, so nDCG on set C is close to a
  binary measure. Stated in the report.
- Test pools come from fewer configurations than dev pools (the reference points only).""",
    "E05_judge_calibration_and_hand_labels.txt": """\
Status: (a)-(d) BUILT and run for three versions of the rubric, plus a held-out measurement on pairs
of the test needs. (e) NOT DONE: the sheet of 200 pairs and the guide exist; no person has filled
it. Results: REPORT.md, "The judge" and "The judge's rubric, version by version".
Code: vxfind/eval/meta.py; scripts/vxfind_eval.py (judge, calibrate).
Differences from the plan below:
- The calibration did more than measure: it drove two revisions of the rubric, and reading the
  judge's remaining disagreements found two faults in the computed truth (see E02). The pairs are
  pinned (evals/vxfind/labels/calibration_pairs.json) so versions are compared on the same pairs.
- Because the rubric was revised while reading the dev pairs, the dev figures flatter it; the
  held-out pairs are the figure to quote.
- R9 (a second person) not done. R10's correction with human labels not possible yet.
- The sheet in git carries a short excerpt per table; the text the judge read is written to data/
  beside it, because the tables' own text is not redistributed.""",
    "E06_retrieval_experiments_and_selection.txt": """\
Status: BUILT and run twice (first with the first rubric, kept as *_first_run; then with the revised one).
Results: REPORT.md, "Retrieval". The serving configuration: evals/vxfind/serving.json.
Code: vxfind/eval/harness.py; scripts/vxfind_eval.py (retrieval).
Differences from the plan below:
- The rule chose the same configuration under both rubrics, which says the choice does not hinge on
  the judge: it is driven by set B, whose truth needs no judge.
- What the rule chose is modest (scratch demotion and a boost for described tables); the larger
  findings are comparative: which retriever each query set favours, what removing a field costs,
  and that no retriever handles stated numbers.
- R10's list of hardest queries is produced by scripts/vxfind_failures.py.
Further limits: no learned fusion and no trained reranker (the stretch goals).""",
    "E07_model_stages_reasons_and_abstention.txt": """\
Status: BUILT and run on the fixed dev subset. Results: REPORT.md, "Model stages" and "Reasons and
'nothing fits'".
Code: vxfind/eval/answers.py; scripts/vxfind_eval.py (llm_stages, answers).
Differences from the plan below:
- R4 earned its place. Measured directly, the first reading prompt ("do not guess a robot") left the
  robot empty for almost every need that names one and missed most stated numbers; with it, hard
  filters helped set A a little and wrecked set B. The prompt was rewritten field by field, a second
  reader (the large model) was added as a setting, and the robot became a boost instead of a filter
  (a name matched as text cannot bridge "Universal Robots" and "ur5e"). The first run is kept
  (verification/vxfind_llm_stages_first_run.json, vxfind_answers_first_run.json) and quoted in the report.
- The largest gain of the whole evaluation is here: stated numbers, which every retriever ignores
  (set A), are honoured once the need is read and the numbers become filters. The configuration is
  put to the release gate on the test queries as a candidate (E12); it is not yet the serving one,
  because it adds a model call to every question and that is the lead's decision.
- Reranking and selection are judged, on set C, by the same model family that performs them; the
  report says so and leans on sets A and B for those comparisons.
- The select prompt was not revised after measurement: its weaknesses are reported as found.
- The check of reasons by code had a false alarm (digits inside a table's own name); fixed and pinned.
Further limits: one run per configuration; latencies are from one machine with a local model.""",
    "E08_freshness_and_the_live_check.txt": """\
Status: BUILT and run against the live site. Results: REPORT.md, "Freshness".
Code: vxfind/eval/fresh.py; scripts/vxfind_ops.py (fresh). Record: data/vxfind/eval/site_answers.json.
Differences from the plan below:
- R9 was added after the first run. The dead share was far larger than a month could explain; the
  check showed that every dead table found is still in the newer dump and that the newer dump
  dropped nothing. The number measures the source, not the index's age.
- The listing is not a safe cheap signal: it catches every dead table and also flags many live ones.
- "Gone" cannot be told from "made private" from outside, and is not.
Further limits: one day's measurement; the stretch (the same measurement after the sync) was not run.""",
    "E09_access_and_leak_tests.txt": """\
Status: BUILT and run. Results: REPORT.md, "Access". First run kept: verification/vxfind_access_first_run.json.
Code: scripts/vxfind_ops.py (access); tests in tests/test_vxfind.py.
Differences from the plan below:
- The test found a real leak on its first run (R4 and R5 failed for check_tables); see S4. The fix
  is pinned by a test.
- R9's agent case is in E10.
Further limits: no timing analysis; the policy is three owners and one group.""",
    "E10_agent_trajectories.txt": """\
Status: BUILT and run with two local models. Results: REPORT.md, "Agent". Cases: evals/vxfind/agent_cases.json.
Code: vxfind/eval/agentcases.py; scripts/vxfind_ops.py (agent).
Differences from the plan below:
- The first run taught the task's own lesson. Several runs ended with no answer at all and PASSED,
  because every property was about what an answer must not contain. The property listed_some was
  added, the cause (a tool reply too large for the executor) was fixed in S5, and the cases were run
  again. First run: verification/vxfind_agent_first_run.json.
- Scoring bug found on the way: a request filed in one case was counted against every later case,
  because the servers' call log runs across cases. Properties now read only the calls of their case.
- One run per case per model; no repetitions, no held-out cases.
- Site answers are replayed from E08's record with INJECTED faults on top.
Further limits: the properties see addresses and tool calls; a wrong sentence about a right table
passes them (E07 measures that separately).""",
    "E11_evaluating_the_evaluation.txt": """\
Status: BUILT and run on dev and test. Results: REPORT.md, "Can these numbers be trusted".
Code: vxfind/eval/meta.py; scripts/vxfind_eval.py (meta).
Differences from the plan below:
- R8's one-page trust statement is the report section itself, not a separate page.
- R1 (the judge) is answered in E05 with a held-out measurement and three rubric versions.
- What it found, in words: large planted faults are seen on every set, on dev and on test. A fault
  of one result in ten is at the edge of what the sets can see. Reordering the top ten does not move
  nDCG on the written needs significantly, because the judge's grades are nearly two-valued there:
  set C measures WHICH tables are shown far better than their order, and the order is measured by
  set B. The model-written set ranks configurations differently from the other two.
Further limits: no topic-set-size analysis and no judge-swap (the stretch goals).""",
    "E12_feedback_loop_and_release_gate.txt": """\
Status: BUILT; exercised with SCRIPTED sessions only. Results: REPORT.md, "Roll-out and the release gate".
Code: vxfind/eval/feedback.py, gate.py; scripts/vxfind_ops.py (rollout). Rules: evals/vxfind/gate.json.
Differences from the plan below:
- R12's "real alternative" turned out to be the instructive one: reading the need with a model and
  filtering by the stated numbers is the largest gain measured on dev, and on the locked test
  queries it fails the gate, because known-item needs lose significantly. On dev the loss was within
  noise. Returning to plain fusion also fails (it gives up the gain on set B).
- R8's max_drop is the same fixed value for every set; E11's smallest detectable differences are
  close to it for sets A and B and larger for set C on dev. Not yet set per set.
- There are no users, so R6 (the weekly review) has not happened and every online measure in the
  report comes from a script that opens what the evaluation's own labels call relevant.
- The gate is a command, not yet a step of a build pipeline.
- R14: no exception has been needed or written.
Further limits: candidates from scripted sessions are the test queries themselves; they show that
triage works, not what real feedback would add.""",
}
