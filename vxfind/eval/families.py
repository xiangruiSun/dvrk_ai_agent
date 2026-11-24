"""Robot families: which catalog values of `robot_type` mean the same hardware.

The catalog has more than a thousand spellings of the robot. A person asks for "an SO-101 arm"
and the tables say so101_follower, so_follower, so-101, bi_so101_follower. To label by code
whether a table is the robot a need asks for, the spellings are grouped into families by the
rules below.

The rules were written by hand from the most frequent values in the 2026-10-23 snapshot (one
author, not a hardware vendor). They cover the frequent spellings, not all of them: a table
whose robot_type matches no rule belongs to no family and is never counted as a match, so the
labels built from these rules err on the side of "not relevant". Coverage is measured and
reported with the query set.
"""
from __future__ import annotations

import re

# family -> (what a person calls it, pattern over the lower-cased robot_type)
FAMILIES: dict[str, tuple[str, str]] = {
    "so_arm": ("SO-100 or SO-101 arm", r"(^|[^a-z])so[-_ ]?(100|101|arm)?([^a-z0-9]|$)|^so10[01]|so_follower|so10[01]_follower"),
    "franka": ("Franka Panda arm", r"panda|franka|(^|_)fr3"),
    "aloha": ("ALOHA bimanual set-up", r"aloha"),
    "koch": ("Koch arm", r"(^|_)koch"),
    "lekiwi": ("LeKiwi mobile manipulator", r"lekiwi"),
    "ur": ("Universal Robots arm (any UR model)", r"(^|[^a-z])ur(3|5|10|16)?e?([^a-z]|$)"),
    "unitree_g1": ("Unitree G1 humanoid", r"unitree[_-]?g1|(^|_)g1(_|$)"),
    "piper": ("AgileX Piper arm", r"piper"),
    "trossen": ("Trossen arm (WidowX, ViperX, Trossen AI)", r"trossen|widowx|viperx"),
    "openarm": ("OpenArm", r"openarms?"),
    "arx": ("ARX arm (X5, L5)", r"(^|_)arx"),
    "fanuc": ("Fanuc arm", r"fanuc"),
    "yam": ("YAM arm", r"(^|_)yam(_|$)|bimanual_yam"),
    "omx": ("OMX arm", r"(^|_)omx"),
    "xarm": ("UFACTORY xArm", r"xarm|(^|_)u850|lite6"),
    "rby1": ("Rainbow RB-Y1", r"rby1|rb-y1"),
}
_COMPILED = {k: re.compile(p) for k, (_, p) in FAMILIES.items()}
BIMANUAL = re.compile(r"(^|_)bi_|bimanual|dual|aloha|_two_|both_arms")


def family_of(robot_type: str | None) -> str | None:
    """The family a catalog value belongs to, or None. The first matching rule wins."""
    if not robot_type:
        return None
    s = robot_type.lower()
    for name, pat in _COMPILED.items():
        if pat.search(s):
            return name
    return None


def is_bimanual(robot_type: str | None) -> bool:
    return bool(robot_type and BIMANUAL.search(robot_type.lower()))


def label(family: str) -> str:
    return FAMILIES[family][0]
