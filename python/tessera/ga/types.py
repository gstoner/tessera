"""GA2 — Grade-aware type annotation markers.

Each class here is **annotation-only** — you don't instantiate them.
You subscript them in type annotations, and the subscripted form
returns a frozen ``MultivectorSpec`` (from
``tessera.ga.multivector``) that ``@tessera.jit`` and the GA
constraint predicates consume at decoration time.

    @tessera.jit
    def rotate(R: Rotor[Cl(3, 0)],
               v: VectorField[Cl(3, 0)]) -> VectorField[Cl(3, 0)]:
        return tessera.ga.rotor_sandwich(R, v)

Decision GA-L4 (equivariance-from-algebra): a parameter annotated
``Rotor[Cl(p, q)]`` is provably ``Cl(p, q)``-equivariant — the compiler
takes the type as a proof obligation, not a runtime assertion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet

from tessera.ga.multivector import MultivectorSpec
from tessera.ga.signature import Cl, TesseraAlgebraError


def _require_cl(value: Any, name: str) -> Cl:
    if not isinstance(value, Cl):
        raise TesseraAlgebraError(
            f"{name}[...] first argument must be a Cl signature; "
            f"got {type(value).__name__}."
        )
    return value


class Rotor:
    """Even-grade unit multivector — annotation surface.

    ``Rotor[Cl(p, q)]`` is structurally a multivector restricted to the
    even-grade subalgebra ``Cl⁺(p, q)``. The "unit" property cannot be
    checked at decoration time (it's a value invariant), but the
    even-grade restriction is — and that alone is enough to
    guarantee grade-structure preservation under the rotor sandwich
    ``R x R†`` (Decision GA-L4).
    """

    def __class_getitem__(cls, algebra: Any) -> MultivectorSpec:
        cl = _require_cl(algebra, "Rotor")
        even_grades = frozenset(g for g in cl.grades if g % 2 == 0)
        return MultivectorSpec(algebra=cl, grades=even_grades, kind="rotor")

    def __init__(self) -> None:  # pragma: no cover - annotation-only
        raise TypeError(
            "Rotor is an annotation marker; construct concrete values via "
            "tessera.ga.rotor_from_axis() or tessera.ga.Multivector(...)."
        )


class DiffForm:
    """Grade-pure k-form — annotation surface.

    ``DiffForm[Cl(p, q), k]`` annotates a value as a pure grade-k
    multivector. Antisymmetry is automatic from the algebra; no runtime
    enforcement needed.
    """

    def __class_getitem__(cls, params: Any) -> MultivectorSpec:
        if not isinstance(params, tuple) or len(params) != 2:
            raise TesseraAlgebraError(
                "DiffForm[...] takes exactly 2 arguments (algebra, k); "
                f"got {params!r}."
            )
        algebra, k = params
        cl = _require_cl(algebra, "DiffForm")
        if not isinstance(k, int) or k < 0:
            raise TesseraAlgebraError(
                f"DiffForm[..., k]: k must be a non-negative int; got {k!r}."
            )
        if k > cl.n:
            raise TesseraAlgebraError(
                f"DiffForm[{cl!r}, {k}]: grade {k} exceeds algebra grades "
                f"{cl.grades}."
            )
        return MultivectorSpec(algebra=cl, grades=frozenset({k}), kind="diff_form")


class VectorField:
    """Grade-1 multivector — annotation surface.

    Convenience equivalent of ``DiffForm[Cl, 1]`` with a clearer name
    when the value semantically represents a vector field rather than
    a 1-form on a manifold.
    """

    def __class_getitem__(cls, algebra: Any) -> MultivectorSpec:
        cl = _require_cl(algebra, "VectorField")
        return MultivectorSpec(algebra=cl, grades=frozenset({1}), kind="vector_field")


@dataclass(frozen=True)
class MorphismSpec:
    """Annotation specification for a map ``Cl(p,q) → Cl(p',q')``."""

    source: Cl
    target: Cl


class Morphism:
    """Map between two Clifford algebras — annotation surface.

    ``Morphism[Cl(3, 0), Cl(1, 3)]`` annotates a callable that maps
    multivectors from one signature to another. Useful for embeddings
    (e.g. ℝ³ → spacetime).
    """

    def __class_getitem__(cls, params: Any) -> MorphismSpec:
        if not isinstance(params, tuple) or len(params) != 2:
            raise TesseraAlgebraError(
                "Morphism[...] takes exactly 2 arguments (source_algebra, target_algebra); "
                f"got {params!r}."
            )
        src, dst = params
        return MorphismSpec(
            source=_require_cl(src, "Morphism"),
            target=_require_cl(dst, "Morphism (target)"),
        )


__all__ = [
    "DiffForm",
    "Morphism",
    "MorphismSpec",
    "Rotor",
    "VectorField",
]
