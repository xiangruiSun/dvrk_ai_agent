"""Units. A closed list, and conversion done by code.

The model names the unit a value was given in. It never converts: a request that says "210 mm"
becomes value 210 with unit "mm", and this module turns it into the 0.21 the file stores.
"""
from __future__ import annotations

import math

UNITS = ("m", "mm", "rad", "deg", "s", "ms", "ratio", "none")

QUANTITY = {"m": "length", "mm": "length", "rad": "angle", "deg": "angle", "s": "time", "ms": "time",
            "ratio": "ratio", "none": "none"}
_TO_SI = {"m": 1.0, "mm": 1e-3, "rad": 1.0, "deg": math.pi / 180.0, "s": 1.0, "ms": 1e-3, "ratio": 1.0}


class UnitError(ValueError):
    pass


def convert(value: float, unit: str, to_unit: str) -> float:
    """value given in `unit`, expressed in `to_unit`. Both must measure the same quantity."""
    if unit not in UNITS or to_unit not in UNITS:
        raise UnitError(f"unknown unit {unit!r} or {to_unit!r}; units are {', '.join(UNITS)}")
    if QUANTITY[unit] != QUANTITY[to_unit]:
        raise UnitError(f"a value in {unit!r} ({QUANTITY[unit]}) cannot be written where the file expects "
                        f"{to_unit!r} ({QUANTITY[to_unit]})")
    if unit == to_unit or QUANTITY[unit] == "none":
        return float(value)
    return float(value) * _TO_SI[unit] / _TO_SI[to_unit]
