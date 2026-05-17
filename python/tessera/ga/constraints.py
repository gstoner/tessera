"""GA2 — Grade-aware constraint predicates.

Five new predicate classes that plug into the existing
``ConstraintSolver`` machinery in ``tessera.compiler.constraints``:

    GradeIn(name, grades)   — parameter `name` must have grade set ⊆ `grades`
    Even(name)              — parameter `name` must be even-grade only
    Odd(name)               — parameter `name` must be odd-grade only
    IsRotor(name)           — parameter `name` must be a rotor (even, unit)
    IsForm(name, k)         — parameter `name` must be a pure k-form

Each follows the existing ``Constraint`` protocol exactly — a frozen
dataclass with ``check(bindings)`` returning ``None`` or
``TesseraConstraintError``. Bindings are interpreted as
``name → MultivectorSpec`` for these constraints (existing
``Divisible``/``Range``/``Equal`` use ``name → int`` for dimensions).
The two binding modes coexist because ``check()`` is duck-typed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Union

from tessera.compiler.constraints import Constraint, TesseraConstraintError
from tessera.ga.multivector import Multivector, MultivectorSpec


_GradeArg = Union[int, FrozenSet[int], set, list, tuple]


def _coerce_grade_set(grades: _GradeArg) -> FrozenSet[int]:
    if isinstance(grades, int):
        return frozenset({grades})
    return frozenset(int(g) for g in grades)


def _resolve_spec(value: Any) -> Optional[MultivectorSpec]:
    """Map a binding value to a MultivectorSpec, or None if non-multivector."""
    if isinstance(value, MultivectorSpec):
        return value
    if isinstance(value, Multivector):
        return MultivectorSpec(algebra=value.algebra, grades=value.active_grades)
    return None


# ---------------------------------------------------------------------------
# GradeIn
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GradeIn(Constraint):
    """Require parameter ``name`` to be a multivector with grades ⊆ ``grades``.

    Args:
        name:   parameter / dimension binding name (string).
        grades: int or iterable of ints — the allowed grade set.

    Example::

        tessera.require(GradeIn("x", {0, 2}))
        # x must be a scalar-plus-bivector value in its algebra.
    """

    name: str
    grades: FrozenSet[int]

    def __init__(self, name: str, grades: _GradeArg) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"GradeIn.name must be a non-empty string; got {name!r}."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "grades", _coerce_grade_set(grades))

    def check(self, bindings: Dict[str, Any]) -> Optional[TesseraConstraintError]:
        if self.name not in bindings:
            return None  # symbolic — cannot check yet
        spec = _resolve_spec(bindings[self.name])
        if spec is None:
            return None  # binding is not a multivector — skip (treat as N/A)
        if spec.grades is None:
            return TesseraConstraintError(
                constraint=self,
                dim_name=self.name,
                actual=None,
                message=(
                    f"Parameter {self.name!r} is an unrestricted multivector but "
                    f"GradeIn requires grades ⊆ {sorted(self.grades)}. "
                    f"Annotate with Multivector[..., {sorted(self.grades)}] or "
                    f"a grade-pure alias (Rotor/DiffForm/VectorField)."
                ),
            )
        if not spec.grades.issubset(self.grades):
            offending = sorted(spec.grades - self.grades)
            return TesseraConstraintError(
                constraint=self,
                dim_name=self.name,
                actual=sorted(spec.grades),
                message=(
                    f"Parameter {self.name!r} has grades {sorted(spec.grades)} "
                    f"which include disallowed grade(s) {offending}; "
                    f"GradeIn requires grades ⊆ {sorted(self.grades)}."
                ),
            )
        return None

    def dim_names(self) -> List[str]:
        return [self.name]

    def __repr__(self) -> str:
        return f"GradeIn({self.name!r}, {sorted(self.grades)})"


# ---------------------------------------------------------------------------
# Even / Odd
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Even(Constraint):
    """Require parameter ``name`` to be even-grade only (grades ∈ {0, 2, 4, ...})."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"Even.name must be a non-empty string; got {self.name!r}."
            )

    def check(self, bindings: Dict[str, Any]) -> Optional[TesseraConstraintError]:
        if self.name not in bindings:
            return None
        spec = _resolve_spec(bindings[self.name])
        if spec is None:
            return None
        if spec.grades is None:
            return TesseraConstraintError(
                constraint=self,
                dim_name=self.name,
                actual=None,
                message=(
                    f"Parameter {self.name!r} is an unrestricted multivector but "
                    f"Even requires only even-grade components."
                ),
            )
        odd = sorted(g for g in spec.grades if g % 2 == 1)
        if odd:
            return TesseraConstraintError(
                constraint=self,
                dim_name=self.name,
                actual=sorted(spec.grades),
                message=(
                    f"Parameter {self.name!r} has odd-grade component(s) {odd}; "
                    f"Even requires only even grades."
                ),
            )
        return None

    def dim_names(self) -> List[str]:
        return [self.name]


@dataclass(frozen=True)
class Odd(Constraint):
    """Require parameter ``name`` to be odd-grade only (grades ∈ {1, 3, 5, ...})."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"Odd.name must be a non-empty string; got {self.name!r}."
            )

    def check(self, bindings: Dict[str, Any]) -> Optional[TesseraConstraintError]:
        if self.name not in bindings:
            return None
        spec = _resolve_spec(bindings[self.name])
        if spec is None:
            return None
        if spec.grades is None:
            return TesseraConstraintError(
                constraint=self,
                dim_name=self.name,
                actual=None,
                message=(
                    f"Parameter {self.name!r} is an unrestricted multivector but "
                    f"Odd requires only odd-grade components."
                ),
            )
        even = sorted(g for g in spec.grades if g % 2 == 0)
        if even:
            return TesseraConstraintError(
                constraint=self,
                dim_name=self.name,
                actual=sorted(spec.grades),
                message=(
                    f"Parameter {self.name!r} has even-grade component(s) {even}; "
                    f"Odd requires only odd grades."
                ),
            )
        return None

    def dim_names(self) -> List[str]:
        return [self.name]


# ---------------------------------------------------------------------------
# IsRotor / IsForm
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IsRotor(Constraint):
    """Require parameter ``name`` to be a rotor: even-grade with ``kind='rotor'``.

    Like ``Even`` plus a check on the annotation's ``kind`` field — so
    a value annotated as ``Rotor[Cl(3,0)]`` passes, but a generic
    even-grade ``Multivector[..., {0, 2}]`` does not. This is how the
    compiler distinguishes "happens to be even" from "is provably a
    rotor" (Decision GA-L4).
    """

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"IsRotor.name must be a non-empty string; got {self.name!r}."
            )

    def check(self, bindings: Dict[str, Any]) -> Optional[TesseraConstraintError]:
        if self.name not in bindings:
            return None
        spec = _resolve_spec(bindings[self.name])
        if spec is None:
            return None
        if spec.kind != "rotor":
            return TesseraConstraintError(
                constraint=self,
                dim_name=self.name,
                actual=spec.kind,
                message=(
                    f"Parameter {self.name!r} has kind={spec.kind!r}; "
                    f"IsRotor requires a Rotor[...]-annotated value (kind='rotor'). "
                    f"Even-grade is necessary but not sufficient — the rotor kind "
                    f"carries the equivariance proof obligation."
                ),
            )
        return None

    def dim_names(self) -> List[str]:
        return [self.name]


@dataclass(frozen=True)
class IsForm(Constraint):
    """Require parameter ``name`` to be a pure k-form (grade-pure, kind='diff_form')."""

    name: str
    k: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"IsForm.name must be a non-empty string; got {self.name!r}."
            )
        if not isinstance(self.k, int) or self.k < 0:
            raise ValueError(
                f"IsForm.k must be a non-negative int; got {self.k!r}."
            )

    def check(self, bindings: Dict[str, Any]) -> Optional[TesseraConstraintError]:
        if self.name not in bindings:
            return None
        spec = _resolve_spec(bindings[self.name])
        if spec is None:
            return None
        if spec.kind != "diff_form":
            return TesseraConstraintError(
                constraint=self,
                dim_name=self.name,
                actual=spec.kind,
                message=(
                    f"Parameter {self.name!r} has kind={spec.kind!r}; "
                    f"IsForm requires a DiffForm[..., k]-annotated value."
                ),
            )
        if not spec.is_grade_pure() or spec.grade_value() != self.k:
            actual = sorted(spec.grades) if spec.grades is not None else None
            return TesseraConstraintError(
                constraint=self,
                dim_name=self.name,
                actual=actual,
                message=(
                    f"Parameter {self.name!r} has grades {actual}; "
                    f"IsForm(k={self.k}) requires a pure grade-{self.k} value."
                ),
            )
        return None

    def dim_names(self) -> List[str]:
        return [self.name]


__all__ = [
    "Even",
    "GradeIn",
    "IsForm",
    "IsRotor",
    "Odd",
]
