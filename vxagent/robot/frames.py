"""The frame graph of a system: which frame each arm's base is expressed in.

In a dVRK system file an arm's `base_frame` either names a reference frame and gives a fixed
transform, or names a component (the setup joints, the camera arm) that provides the transform
at run time. Both make the arm a child of something. A change that names a frame nobody
declared, or that makes a frame its own ancestor, would be accepted by the JSON Schema and
break every pose computed afterwards.
"""
from __future__ import annotations

from typing import Any, Iterable


class FrameGraph:
    def __init__(self, frames: Iterable[str] = (), parents: dict[str, str] | None = None):
        self.frames: set[str] = set(frames)
        self.parents: dict[str, str] = dict(parents or {})
        self.frames |= set(self.parents)

    @classmethod
    def from_system(cls, doc: dict[str, Any], declared: Iterable[str] = ()) -> "FrameGraph":
        """Frames are the arms of the file and the frames the site has declared. `known` is what
        a new reference may name."""
        g = cls(declared)
        for arm in doc.get("arms") or []:
            name = arm.get("name")
            if not name:
                continue
            g.frames.add(name)
            base = arm.get("base_frame") or {}
            parent = base.get("reference_frame") or base.get("component")
            if parent:
                g.parents[name] = parent
        return g

    def cycle(self) -> list[str] | None:
        for start in self.parents:
            seen = [start]
            node = start
            while node in self.parents:
                node = self.parents[node]
                if node in seen:
                    return seen[seen.index(node):] + [node]
                seen.append(node)
        return None

    def undeclared(self) -> list[str]:
        return sorted(p for p in set(self.parents.values()) if p not in self.frames)

    def describe(self) -> dict[str, Any]:
        return {"frames": sorted(self.frames), "parents": dict(sorted(self.parents.items()))}
