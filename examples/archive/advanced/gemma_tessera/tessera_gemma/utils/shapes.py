"""
tessera_gemma/utils/shapes.py — Symbolic shape checking utilities.

Changes vs v0.1:
  • `ShapeSpec` constructor now also accepts a tuple/list of ints or strings.
  • `check_shape` returns the filled `symbols` dict for chaining.
  • Added `assert_divisible` helper used by attention to validate head counts.
  • Added `TesseraShapeError` subclass for easier pytest filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Union


class TesseraShapeError(ValueError):
    """Raised when a tensor shape fails a ShapeSpec check."""


@dataclass
class ShapeSpec:
    """Symbolic shape descriptor.

    Each element of `dims` may be:
      - A digit string "64" → literal integer check.
      - A symbol string "B"  → bound on first encounter, checked thereafter.
      - "?"                  → any size, no binding.

    Examples::

        spec = ShapeSpec(["B", "T", "H", "D"])
        spec = ShapeSpec(["B", "T", "16", "64"])
    """

    dims: List[str]

    def __post_init__(self) -> None:
        # Accept a plain list/tuple of ints or strings
        self.dims = [str(d) for d in self.dims]

    @property
    def rank(self) -> int:
        return len(self.dims)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ShapeSpec({self.dims})"


def check_shape(
    name: str,
    actual: Tuple[int, ...],
    spec: ShapeSpec,
    symbols: Dict[str, int],
) -> Dict[str, int]:
    """Validate *actual* against *spec*, filling *symbols* in place.

    Returns *symbols* (the same dict, updated) so callers can chain calls::

        syms = {}
        check_shape("x",   x.shape,   ShapeSpec(["B","T","C"]), syms)
        check_shape("mask", m.shape,  ShapeSpec(["B","T"]),      syms)

    Raises:
        TesseraShapeError: on rank mismatch or dimension mismatch.
    """
    if len(actual) != len(spec.dims):
        raise TesseraShapeError(
            f"{name}: rank {len(actual)} != expected rank {len(spec.dims)} "
            f"(spec={spec.dims}, actual={list(actual)})"
        )

    for i, (a, s) in enumerate(zip(actual, spec.dims)):
        if s == "?":
            continue
        if s.lstrip("-").isdigit():
            expected = int(s)
            if a != expected:
                raise TesseraShapeError(
                    f"{name}: dim[{i}] = {a}, expected literal {expected}"
                )
        else:
            if s in symbols:
                if symbols[s] != a:
                    raise TesseraShapeError(
                        f"{name}: dim[{i}] symbolic '{s}' conflict: "
                        f"previously bound to {symbols[s]}, got {a}"
                    )
            else:
                symbols[s] = a

    return symbols


def assert_divisible(numerator: int, denominator: int, msg: str = "") -> None:
    """Assert that *numerator* is evenly divisible by *denominator*."""
    if denominator <= 0:
        raise TesseraShapeError(
            f"Divisor must be positive, got {denominator}. {msg}"
        )
    if numerator % denominator != 0:
        raise TesseraShapeError(
            f"{numerator} is not divisible by {denominator}. {msg}"
        )
