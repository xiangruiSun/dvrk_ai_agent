# Data

Nothing in this folder is committed except this file. Everything is downloaded by a script and can
be downloaded again.

| Folder | What | Source | Licence | Script |
|---|---|---|---|---|
| `dvrk/share/kinematic`, `tool`, `schemas` | dVRK kinematic files, instrument files and JSON Schemas (84 files, 0.28 MB) | `jhu-dvrk/sawIntuitiveResearchKit` at commit `7435458` | see note | `scripts/fetch_dvrk_config.py` |
| `rosma/` | Recorded dVRK kinematics: 12 recordings by default (113.8 MB), 207 available (1.64 GB), plus `scores.csv` | ROSMA, Zenodo record 3932964 | CC BY 4.0 | `scripts/fetch_rosma.py` |
| `sources/docs`, `sources/changelog` | dVRK documentation text (234 `.rst` files, 0.74 MB) and the change log | `jhu-dvrk/dvrk-readthedocs` at commit `3b231b2`; `CHANGELOG.md` of `sawIntuitiveResearchKit` at `7435458` | see note | `scripts/fetch_sources.py` |
| `sources/issues` | 161 issue threads with 205 replies, fetched 2026-11-19. No author names: only the author's association with the repository | `sawIntuitiveResearchKit` and `dvrk-ros`, public GitHub API | public posts, no licence stated | `scripts/fetch_sources.py` |
| `sources/sites` | Notes that four dVRK sites (JHU, UCL, CUHK, WPI) keep beside their configuration files: 8 README and note files, 8 kB. Public; treated here as each site's private collection | `github.com/dvrk-config`, four repositories at pinned commits | MIT for JHU, none stated for the others | `scripts/fetch_sources.py` |
| `sources/docs_previous` | The documentation at an earlier commit (`6f0167f`, 230 files), to measure incremental sync | `jhu-dvrk/dvrk-readthedocs` | see note | `scripts/fetch_sources.py --only docs-previous` |
| `rag/` | The retrieval store, the embedding cache, the index versions (`indexes/`) and the hashed API keys | produced here | | |
| `runtime/` | SQLite database with traces, approvals and commands; cached episode statistics | produced here | | |

## Licence note for the dVRK files and text

On 2026-11-19 the repository `jhu-dvrk/sawIntuitiveResearchKit` had no licence file at its root and
GitHub reported no detected licence. The files are therefore used locally to build the kinematic
twin and are **not redistributed** in this repository. `data/dvrk/MANIFEST.json` records the commit
and a SHA-256 for every file fetched.

## ROSMA

Zenodo title: "Training dataset from the Da Vinci Research Kit". Twelve operators (X01 to X12), three
tasks (pea on a peg, post and sleeve, wire chaser). Each recording is a CSV at about 50 Hz with Cartesian pose, velocity, wrench and joint values for MTML,
MTMR, PSM1 and PSM2. There is no jaw signal. The fetch script takes the kinematics files and
`scores.csv` only: the videos (20.7 GB) and the operator questionnaire are never downloaded. Each
file is checked against the MD5 published by Zenodo.

## Site files

`sites/site_a.json` is **declared for this project**. It is not a real lab's set-up: the start
pose, the pad and needle positions and the named poses were chosen so that every rule of the safety
filter has a pose on each side of it. The step, workspace and tracking limits in its safety profile
are the published ones (see `vxagent/robot/safety.py`); the joint-limit margin and the clearance
over the pad are site choices, because the paper states that they are enforced without giving
figures.
