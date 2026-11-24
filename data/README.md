# Data

Nothing in this folder is committed except this file. Everything is downloaded by a script and can
be downloaded again.

| Folder | What | Source | Licence | Script |
|---|---|---|---|---|
| `dvrk/share/kinematic`, `tool`, `schemas`, `arm`, `system` | dVRK kinematic files, instrument files, JSON Schemas, and the arm and system files for simulated arms (129 files, 0.36 MB) | `jhu-dvrk/sawIntuitiveResearchKit` at commit `7435458` | see note | `scripts/fetch_dvrk_config.py` |
| `rosma/` | Recorded dVRK kinematics: 12 recordings by default (113.8 MB), 207 available (1.64 GB), plus `scores.csv` | ROSMA, Zenodo record 3932964 | CC BY 4.0 | `scripts/fetch_rosma.py` |
| `surgicai/` | Expert demonstrations of the SurgicAI sub-tasks (pickle files of arrays) and the two environment source files the policy contract is drafted from: 11 files, 9.9 MB. The pickles are read with a restricted unpickler; the Python sources are read as text and never imported | `surgical-robotics-ai/SurgicAI` at tree `b7ce5465` | MIT | `scripts/fetch_surgicai.py` |
| `sources/docs`, `sources/changelog` | dVRK documentation text (234 `.rst` files, 0.74 MB) and the change log | `jhu-dvrk/dvrk-readthedocs` at commit `3b231b2`; `CHANGELOG.md` of `sawIntuitiveResearchKit` at `7435458` | see note | `scripts/fetch_sources.py` |
| `sources/issues` | 161 issue threads with 205 replies, fetched 2026-11-19. No author names: only the author's association with the repository | `sawIntuitiveResearchKit` and `dvrk-ros`, public GitHub API | public posts, no licence stated | `scripts/fetch_sources.py` |
| `sources/sites` | Notes that four dVRK sites (JHU, UCL, CUHK, WPI) keep beside their configuration files: 8 README and note files, 8 kB. Public; treated here as each site's private collection | `github.com/dvrk-config`, four repositories at pinned commits | MIT for JHU, none stated for the others | `scripts/fetch_sources.py` |
| `sources/docs_previous` | The documentation at an earlier commit (`6f0167f`, 230 files), to measure incremental sync | `jhu-dvrk/dvrk-readthedocs` | see note | `scripts/fetch_sources.py --only docs-previous` |
| `rag/` | The retrieval store, the embedding cache, the index versions (`indexes/`) and the hashed API keys | produced here | | |
| `runtime/` | SQLite database with traces, approvals, commands, proposals, events and checkpoints; cached episode statistics; versions of changed configuration files (`t02_config/`); the database, the arm-state file and the configuration versions of the task 03 verification (`t03/`) ; one folder per edge node (`site-a/`, `site-b/`) with its database and configuration versions; `console.sqlite` (users with password hashes), `fleet.sqlite`, `demo_accounts.json` (generated passwords of the demonstration accounts) and `session_secret`; `verify/` (a fresh folder per verification of tasks 11 to 15) | produced here | | |

## Licence note for the dVRK files and text

On 2026-11-19 the repository `jhu-dvrk/sawIntuitiveResearchKit` had no licence file at its root and
GitHub reported no detected licence. The files are therefore used locally (to build the kinematic
twin and to check configuration changes) and are **not redistributed** in this repository. `data/dvrk/MANIFEST.json` records the commit
and a SHA-256 for every file fetched.

## ROSMA

Zenodo title: "Training dataset from the Da Vinci Research Kit". Twelve operators (X01 to X12), three
tasks (pea on a peg, post and sleeve, wire chaser). Each recording is a CSV at about 50 Hz with Cartesian pose, velocity, wrench and joint values for MTML,
MTMR, PSM1 and PSM2. There is no jaw signal. The time stamps write milliseconds without leading zeros
(`17:15:25.2` is 25.002 s), and in some samples the seconds field is one sample late at a roll-over;
the reader corrects both and counts the samples it moved. The fetch script takes the kinematics files and
`scores.csv` only: the videos (20.7 GB) and the operator questionnaire are never downloaded. Each
file is checked against the MD5 published by Zenodo.

## Site files

`sites/site_a.json` (Classic PSM) and `sites/site_b.json` (Si PSM, the same layout on the published Si files) are **declared for this project**. What follows describes site A. It is not a real lab's set-up: the start
pose, the pad and needle positions and the named poses were chosen so that every rule of the safety
filter has a pose on each side of it. The step, workspace and tracking limits in its safety profile
are the published ones (see `vxagent/robot/safety.py`); the joint-limit margin and the clearance
over the pad are site choices, because the paper states that they are enforced without giving
figures. The frames it lists (`world`, `ECM`, `pad`, `camera`) are the names a change to a base
frame may refer to, besides the arms of the system file. The arm and system files it names are
the published ones for a simulated Classic PSM1.

`evals/t02_change_requests.json` holds the change requests of task 02 with their reference patches
and the negative controls. They and the "records" attached to them were written for this project.

## The training-data catalog (vxfind)

| Folder | What | Source | Licence | Script |
|---|---|---|---|---|
| `hub/dump/<revision>/` | Two revisions, about a month apart, of the daily dump of dataset cards with metadata: one row per public dataset on the Hugging Face Hub (id, owner, dates, tags, counts, card text). About 1 GB each. `hub/MANIFEST.json` records revision, publication time, size and checksum of every file | `librarian-bots/dataset_cards_with_metadata` | each card carries its dataset's own licence (mostly Apache-2.0); the dump has none of its own stated | `scripts/fetch_hub_catalog.py` |
| `hub/listing/` | The Hub's listing of datasets tagged `LeRobot`: id, revision, gating, private, disabled | the Hub's public API | metadata | `scripts/fetch_hub_catalog.py` |
| `vxfind/old`, `vxfind/new` | Output of the Spark job per snapshot (records, changes, statistics), the vectors and their statistics | produced here | | `python -m vxfind.ingest.spark_job`, `vxfind.ingest.embed` |
| `vxfind/emb/` | The embedding cache, keyed by the hash of the text | produced here | | |
| `vxfind/catalog.db` | The catalog database (newer snapshot) | produced here | | `python -m vxfind.catalog.store build` |
| `vxfind/eval/` | Stored search runs, the recorded answers of the hosting site, the scripted feedback log | produced here | | `scripts/vxfind_eval.py`, `scripts/vxfind_ops.py` |
| `tools/elasticsearch-9.2.1`, `es/` | A local single-node Elasticsearch and its data | elastic.co, checked against the published SHA-512 | Elastic License 2.0 (basic) | `scripts/elastic_local.py` |

Only metadata is downloaded: never the recordings or the video the cards describe. The cards are
other people's text. They are read locally to build the index and are **not redistributed**: what is
in git is table ids (public), questions written here or by a local model, grades with one-line reasons
written by a local model, and in the hand-label sheet an excerpt of at most 300 characters of the
generated summary per table.

The live check (`vxfind/mcp/hub.py`) sends one request per table to the Hub's public metadata
endpoint with the table's public id and a generic client name. No user text and no identity is sent.
It stays under 420 requests per 300 seconds (the Hub allows 500).

The local Elasticsearch listens on 127.0.0.1 only, with its security features switched off because
nothing else can reach it. Do not use that configuration on a network.

The access policy (`vxfind/eval/access_policy.json`, under this folder and so not in git) is **declared
for this project**: the three owners that appear most often in results are treated as restricted to one
group so that the leak tests have something to catch. It says nothing about those owners or their
datasets, which is why they are not named in git or in the report.
