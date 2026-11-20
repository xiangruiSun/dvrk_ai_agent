# Data

Nothing in this folder is committed except this file. Everything is downloaded by a script and can
be downloaded again.

| Folder | What | Source | Licence | Script |
|---|---|---|---|---|
| `dvrk/share/kinematic`, `tool`, `schemas` | dVRK kinematic files, instrument files and JSON Schemas (84 files, 0.28 MB) | `jhu-dvrk/sawIntuitiveResearchKit` at commit `7435458` | see note | `scripts/fetch_dvrk_config.py` |
| `rosma/` | Recorded dVRK kinematics: 12 recordings by default (113.8 MB), 207 available (1.64 GB), plus `scores.csv` | ROSMA, Zenodo record 3932964 | CC BY 4.0 | `scripts/fetch_rosma.py` |
| `runtime/` | SQLite database with traces, approvals and commands; cached episode statistics | produced here | | |

## Licence note for the dVRK files

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
