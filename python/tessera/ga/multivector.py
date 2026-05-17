"""GA3 — Multivector value class + GA2 annotation surface.

A `Multivector` is a numpy-backed value carrying a coefficient vector of
length ``algebra.dim`` along its last axis. Higher-rank leading axes are
batch / spatial dimensions and are broadcast through every operation in
``tessera.ga.ops``.

The same class doubles as the **type-annotation surface** via
``__class_getitem__``:

    Multivector[Cl(3, 0)]              # any multivector in Cl(3,0)
    Multivector[Cl(3, 0), {0, 2}]      # restricted to scalar+bivector grades
    Multivector[Cl(1, 3), 2]           # restricted to grade-2 (bivectors)

The subscripted form returns a frozen ``MultivectorSpec`` consumed by
GA2 constraints and by the function-signature validator that
``@tessera.jit`` invokes at decoration time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Iterable, Optional, Tuple, Union

import numpy as np

from tessera.ga.signature import Basis, Cl, TesseraAlgebraError


_GradeArg = Union[int, Iterable[int]]
_FloatDType = Union[str, np.dtype]


def _coerce_grades(grades: Optional[_GradeArg]) -> Optional[FrozenSet[int]]:
    if grades is None:
        return None
    if isinstance(grades, int):
        return frozenset({grades})
    return frozenset(int(g) for g in grades)


@dataclass(frozen=True)
class MultivectorSpec:
    """Type-annotation specification for a multivector.

    Carries the algebra signature and an optional restricted grade set.
    Used by ``@tessera.jit`` and by the GA constraint predicates to
    enforce grade-purity contracts on function signatures.

    ``kind`` distinguishes annotation surfaces that look identical
    grade-wise but carry different equivariance claims:

        ``"multivector"``    — arbitrary multivector
        ``"rotor"``          — even-grade unit (preserves grade structure)
        ``"diff_form"``      — grade-pure k-form (antisymmetric by grade)
        ``"vector_field"``   — grade-1 multivector
    """

    algebra: Cl
    grades: Optional[FrozenSet[int]] = None
    kind: str = "multivector"

    def __post_init__(self) -> None:
        if not isinstance(self.algebra, Cl):
            raise TesseraAlgebraError(
                f"MultivectorSpec.algebra must be a Cl signature, got {type(self.algebra).__name__}."
            )
        if self.grades is not None:
            valid = set(self.algebra.grades)
            invalid = set(self.grades) - valid
            if invalid:
                raise TesseraAlgebraError(
                    f"grades {sorted(invalid)} are not valid for {self.algebra!r}; "
                    f"valid grades are {sorted(valid)}."
                )
        if self.kind not in {"multivector", "rotor", "diff_form", "vector_field"}:
            raise TesseraAlgebraError(
                f"MultivectorSpec.kind must be one of "
                f"'multivector'/'rotor'/'diff_form'/'vector_field'; got {self.kind!r}."
            )

    def is_grade_pure(self) -> bool:
        """True if the spec restricts to exactly one grade."""
        return self.grades is not None and len(self.grades) == 1

    def is_even(self) -> bool:
        """True if every allowed grade is even."""
        if self.grades is None:
            return False
        return all((g % 2 == 0) for g in self.grades)

    def is_odd(self) -> bool:
        if self.grades is None:
            return False
        return all((g % 2 == 1) for g in self.grades)

    def grade_value(self) -> int:
        """Return the single grade if grade-pure; otherwise raise."""
        if not self.is_grade_pure():
            raise TesseraAlgebraError(
                f"{self!r} is not grade-pure; cannot extract a single grade."
            )
        return next(iter(self.grades))

    def __repr__(self) -> str:
        if self.grades is None:
            grade_part = ""
        else:
            grade_part = f", grades={sorted(self.grades)}"
        if self.kind == "multivector":
            return f"Multivector[{self.algebra!r}{grade_part}]"
        return f"{self.kind.capitalize()}[{self.algebra!r}{grade_part}]"


class Multivector:
    """Numpy-backed multivector value + type-annotation surface.

    Coefficient layout: a real-valued ndarray whose **last axis** has
    length ``algebra.dim``. Higher axes are batch / spatial / etc. and
    are preserved through every GA operation. Coefficient index ``i``
    multiplies the basis blade with bitmask ``i`` (so index 0 is the
    scalar coefficient, ``algebra.dim - 1`` is the pseudoscalar).

    Construction is strict — passing coefficients with the wrong final
    dimension, or grades outside the algebra's range, raises
    ``TesseraAlgebraError``.
    """

    __slots__ = ("_coefficients", "_algebra", "_grades")

    def __init__(
        self,
        coefficients: Any,
        algebra: Cl,
        *,
        grades: Optional[_GradeArg] = None,
    ) -> None:
        if not isinstance(algebra, Cl):
            raise TesseraAlgebraError(
                f"Multivector.algebra must be a Cl signature, got {type(algebra).__name__}."
            )
        arr = np.asarray(coefficients)
        if arr.ndim == 0 or arr.shape[-1] != algebra.dim:
            raise TesseraAlgebraError(
                f"Multivector coefficient array must have last-axis length "
                f"{algebra.dim} for {algebra!r}; got shape {arr.shape}."
            )
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32, copy=False)
        grade_set = _coerce_grades(grades)
        if grade_set is not None:
            valid = set(algebra.grades)
            invalid = set(grade_set) - valid
            if invalid:
                raise TesseraAlgebraError(
                    f"grades {sorted(invalid)} are not valid for {algebra!r}; "
                    f"valid grades are {sorted(valid)}."
                )
            # Zero out coefficients outside the declared grade set.
            mask = np.array(
                [b.grade in grade_set for b in algebra.blades()],
                dtype=bool,
            )
            arr = np.where(mask, arr, np.zeros_like(arr))
        # frozen-ish: keep our own copy so external mutations don't leak in.
        object.__setattr__(self, "_coefficients", np.ascontiguousarray(arr))
        object.__setattr__(self, "_algebra", algebra)
        object.__setattr__(self, "_grades", grade_set)

    # -- Annotation surface ------------------------------------------------

    def __class_getitem__(cls, params: Any) -> MultivectorSpec:
        if not isinstance(params, tuple):
            params = (params,)
        if len(params) == 1:
            algebra = params[0]
            grades = None
        elif len(params) == 2:
            algebra, grades = params
        else:
            raise TesseraAlgebraError(
                f"Multivector[...] takes 1 or 2 arguments (algebra[, grades]); "
                f"got {len(params)}."
            )
        if not isinstance(algebra, Cl):
            raise TesseraAlgebraError(
                f"Multivector[...] first argument must be a Cl signature, "
                f"got {type(algebra).__name__}."
            )
        return MultivectorSpec(algebra=algebra, grades=_coerce_grades(grades))

    # -- Properties --------------------------------------------------------

    @property
    def coefficients(self) -> np.ndarray:
        """Read-only view of the underlying coefficient array."""
        view = self._coefficients.view()
        view.flags.writeable = False
        return view

    @property
    def algebra(self) -> Cl:
        return self._algebra

    @property
    def grades(self) -> Optional[FrozenSet[int]]:
        """Declared grade set, or ``None`` for unrestricted (mixed-grade) values."""
        return self._grades

    @property
    def active_grades(self) -> FrozenSet[int]:
        """The grades that are actually non-zero in this value."""
        if self._grades is not None:
            return self._grades
        active = set()
        for blade in self._algebra.blades():
            if np.any(self._coefficients[..., blade.mask] != 0):
                active.add(blade.grade)
        return frozenset(active)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Leading (batch) shape; the algebra axis is the last axis."""
        return self._coefficients.shape[:-1]

    @property
    def dtype(self) -> np.dtype:
        return self._coefficients.dtype

    # -- Construction helpers ---------------------------------------------

    @classmethod
    def zeros(
        cls, algebra: Cl, *, shape: Tuple[int, ...] = (), dtype: _FloatDType = np.float32
    ) -> "Multivector":
        return cls(np.zeros((*shape, algebra.dim), dtype=dtype), algebra)

    @classmethod
    def scalar(
        cls,
        value: float,
        algebra: Cl,
        *,
        shape: Tuple[int, ...] = (),
        dtype: _FloatDType = np.float32,
    ) -> "Multivector":
        coeffs = np.zeros((*shape, algebra.dim), dtype=dtype)
        coeffs[..., 0] = value
        return cls(coeffs, algebra, grades={0})

    @classmethod
    def from_blade(
        cls,
        blade: Basis,
        algebra: Cl,
        *,
        coefficient: float = 1.0,
        dtype: _FloatDType = np.float32,
    ) -> "Multivector":
        coeffs = np.zeros((algebra.dim,), dtype=dtype)
        coeffs[blade.mask] = coefficient
        return cls(coeffs, algebra, grades={blade.grade})

    @classmethod
    def from_vector(
        cls,
        components: Iterable[float],
        algebra: Cl,
        *,
        dtype: _FloatDType = np.float32,
    ) -> "Multivector":
        """Construct a grade-1 multivector from a length-n list of components."""
        comps = np.asarray(tuple(components), dtype=dtype)
        if comps.shape != (algebra.n,):
            raise TesseraAlgebraError(
                f"from_vector requires {algebra.n} components for {algebra!r}; "
                f"got shape {comps.shape}."
            )
        coeffs = np.zeros((algebra.dim,), dtype=dtype)
        for i, blade in enumerate(algebra.blades_of_grade(1)):
            coeffs[blade.mask] = comps[i]
        return cls(coeffs, algebra, grades={1})

    # -- Arithmetic --------------------------------------------------------

    def _check_same_algebra(self, other: "Multivector") -> None:
        if self._algebra != other._algebra:
            raise TesseraAlgebraError(
                f"Multivector algebra mismatch: {self._algebra!r} vs {other._algebra!r}."
            )

    def __add__(self, other: "Multivector") -> "Multivector":
        self._check_same_algebra(other)
        new_coeffs = self._coefficients + other._coefficients
        # Union of grade restrictions (None on either side ⇒ unrestricted).
        if self._grades is None or other._grades is None:
            new_grades: Optional[FrozenSet[int]] = None
        else:
            new_grades = self._grades | other._grades
        return Multivector(new_coeffs, self._algebra, grades=new_grades)

    def __sub__(self, other: "Multivector") -> "Multivector":
        self._check_same_algebra(other)
        new_coeffs = self._coefficients - other._coefficients
        if self._grades is None or other._grades is None:
            new_grades = None
        else:
            new_grades = self._grades | other._grades
        return Multivector(new_coeffs, self._algebra, grades=new_grades)

    def __neg__(self) -> "Multivector":
        return Multivector(-self._coefficients, self._algebra, grades=self._grades)

    def __mul__(self, scalar: float) -> "Multivector":
        if isinstance(scalar, Multivector):
            # Reserve `*` for scalar multiplication; geometric_product is explicit.
            raise TypeError(
                "use tessera.ga.geometric_product(a, b) for the geometric product; "
                "Multivector * Multivector is intentionally not provided."
            )
        return Multivector(
            self._coefficients * float(scalar), self._algebra, grades=self._grades
        )

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "Multivector":
        if isinstance(scalar, Multivector):
            raise TypeError(
                "Multivector division by a Multivector is not provided; "
                "compute the geometric inverse explicitly."
            )
        return Multivector(
            self._coefficients / float(scalar), self._algebra, grades=self._grades
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Multivector):
            return NotImplemented
        if self._algebra != other._algebra:
            return False
        return np.array_equal(self._coefficients, other._coefficients)

    def __hash__(self) -> int:  # pragma: no cover - intentionally unhashable
        raise TypeError("Multivector is mutable-view; not hashable.")

    # -- Convenience -------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        """Return a writable copy of the coefficient array."""
        return self._coefficients.copy()

    def scalar_part(self) -> np.ndarray:
        return self._coefficients[..., 0]

    def grade(self, k: int) -> "Multivector":
        """Project to the grade-k component (delegates to ``ops.grade_projection``)."""
        from tessera.ga.ops import grade_projection

        return grade_projection(self, k)

    def is_close(self, other: "Multivector", *, atol: float = 1e-5, rtol: float = 1e-5) -> bool:
        """Return True if coefficient arrays match within tolerance."""
        if self._algebra != other._algebra:
            return False
        return bool(
            np.allclose(self._coefficients, other._coefficients, atol=atol, rtol=rtol)
        )

    def __repr__(self) -> str:
        return (
            f"Multivector(algebra={self._algebra!r}, shape={self.shape}, "
            f"dtype={self.dtype}, grades={sorted(self.grades) if self.grades else None})"
        )


__all__ = [
    "Multivector",
    "MultivectorSpec",
]
