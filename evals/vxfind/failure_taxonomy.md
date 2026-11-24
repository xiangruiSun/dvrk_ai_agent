# How this system fails: a taxonomy from real runs

Deliverable of E01. Written by the builder of the system from the runs in this repository, not from
a week of row-by-row reading by a newcomer to it (E01's "As built" block says what was skipped).
Every example is a real result on the real catalog. Counts are not typed here: the ones that code
can make are in `verification/vxfind_failures.json` and in `verification/REPORT.md` under "Where the
serving configuration fails"; the rest are in the sections named in the last column.

| | Failure | A real example | Seen by | Stage that addresses it |
|---|---|---|---|---|
| F1 | Function or request words matched against names | The first search ever run, "Are there any datasets of a Franka arm opening a drawer or a cabinet door?", returned a table whose whole name is `a`, and tables named `..._a` and `...make-a-sandwich`, at the top of the lexical list | Reading the first result list; then measured as the configurations `*_names_match_function_words` (E06) | S3 R4: function words are kept out of the match against identifier fields |
| F2 | Right robot, task unknown | For a need about a task, tables that say only "so101_follower, 30 fps, 50 episodes" | The judge's procedure, step 3 (E04); set C scores | Understanding and selection (E07); nothing fixes the missing text itself |
| F3 | Right words, wrong robot | An attribute need for one arm answered with tables of another arm that share camera names and format | Set A, computed (`robot_right@10`) | Hard filters from the understanding stage |
| F4 | Near-identical copies fill the list | One recording session uploaded as dozens of tables (family sizes: REPORT.md, "Write path") | `repeat@10` with collapse off (E06) | S3 R7: one table per family |
| F5 | Scratch tables outrank real ones | Tables named `test`, `record-tmp`, `debug_3` in the top ten | Count of scratch tables in the top ten (E01 script) | S3 R7: scratch multiplier; chosen by E06's rule |
| F6 | A stated number is ignored | "at least 200 episodes" answered with tables of 10 episodes. Every retriever does this: retrieval matches words, and "200" is not a word of the table | Set A, computed: `fit@10` is low for every retriever | Understanding with hard filters (E07) |
| F7 | Word collision with another meaning | "MRI brain scans with tumour segmentation masks" returned image-mask variants of a robot benchmark, and the selection stage showed six of them while saying that MRI data is missing | Set D with selection on (E07, abstention) | Selection prompt; not fixed here |
| F8 | Nothing fits and ten tables are shown | Any need of set D with every model stage off: retrieval always returns ten | Set D (E07) | Selection; abstention is measured together with false abstention |
| F9 | The table fits and its address is dead, moved or changed | A shown table answers with a redirect to another owner; others answer "not found". The source never drops a deleted table | The live check on every shown table (E08) | S4 `check_tables`; S5's answer check. A deletion sweep in the write path is not built |
| F10 | The table fits and the person must not see it | Tables of an owner restricted to one group (DECLARED policy) are among the most-returned | Leak tests (E09). The first run found a leak in `check_tables` | Engine filter (S3 R10) and one identical reply for hidden and missing ids (S4 R5) |
| F11 | The reason states something the table does not say | "This dataset shows a Franka robot opening a drawer" for a table whose text names no robot | Groundedness by code and by model (E07) | Not enforced; reported |
| F12 | The right table exists beyond the first ten | A known-item target at rank 30 | "found in the first k" curve (E06 R9) | Fusion settings, reranking, "return more" |

## Failures of the evaluation itself, found the same way

These are not failures of search. They are failures of the instrument, and they were found by
reading, not by a metric.

| | Failure | What happened | Seen by | Fix |
|---|---|---|---|---|
| M1 | The judge passed contradicted requirements | It wrote "Only 10 episodes, not ten million" and gave grade 2 | Calibration against computed truth; the no-answer set suddenly "having answers" | The rubric as an ordered procedure (g3, g4). First run kept as evidence |
| M2 | Computed truth counted empty tables | Tables with 0 episodes met "every stated constraint" | Reading the revised judge's disagreements: the judge was right | A table with no episodes is not a match |
| M3 | A query's wording excluded its own answers | "Universal Robots arm (UR3, UR5, UR10)" made the judge refuse UR5e | Same reading | The family label and six needs reworded |
| M4 | A need "with no answer" had one | Toy-excavator teleoperation and forklift pallet handling exist in the catalog | Judging the pool of set D instead of trusting its author | Those needs are moved out of the abstention scoring |
| M5 | The model-written set favours one kind of system | Needs written from a description repeat its words and even its frame counts; lexical search wins on that set and loses on the hand-written one | Comparing the ranking of retrievers across sets (E11) | Reported; the set is used as a regression check, not as the measure of quality |
| M6 | The judge barely uses grade 2 after the revision | The procedure sends almost every pair to 0, 1 or 3 | The grade distribution in the report | Stated; nDCG on the written needs is close to a binary measure |

## What has no measurement yet

- Whether a returned table is actually good training data (the recordings are never read).
- What real newcomers ask. Set C is one author's guess until the roll-out collects questions (E12).
- Agreement of the judge with a person. The sheet exists; nobody has filled it (E05).
