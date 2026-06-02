"""GA3 acceptance (part 1): Multivector value class + annotation surface.

Sprint: GA3 (multivector Python reference) + GA2 (annotation surface).
Roadmap: docs/audit/domain/DOMAIN_AUDIT.md

Covers:
  - Multivector construction: rank-1 coefficient array, optional grade
    restriction (auto-zeros out-of-grade components).
  - Annotation surface: Multivector[Cl(3,0), {0,2}] returns a frozen
    MultivectorSpec usable by GA2 constraints.
  - Arithmetic: +, -, scalar *, scalar / — preserve grade-set union;
    Multivector * Multivector intentionally rejected (use geometric_product).
  - Algebra-mismatch raises TesseraAlgebraError on every binary op.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.ga import (
    Cl,
    Multivector,
    MultivectorSpec,
    TesseraAlgebraError,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_multivector_construction_zero_init() -> None:
    a = Cl(3, 0)
    mv = Multivector.zeros(a)
    assert mv.algebra is a or mv.algebra == a
    assert mv.coefficients.shape == (8,)
    assert np.all(mv.coefficients == 0)
    assert mv.dtype == np.float32


def test_multivector_scalar_construction() -> None:
    a = Cl(3, 0)
    mv = Multivector.scalar(2.5, a)
    assert mv.coefficients[0] == pytest.approx(2.5)
    assert np.all(mv.coefficients[1:] == 0)
    assert mv.grades == frozenset({0})


def test_multivector_from_vector() -> None:
    a = Cl(3, 0)
    mv = Multivector.from_vector([1.0, 2.0, 3.0], a)
    # In Cl(3,0) blade order: [1, e1, e2, e12, e3, e13, e23, e123].
    # Vector components map to grade-1 blades e1, e2, e3.
    assert mv.coefficients[a.blade("e1").mask] == pytest.approx(1.0)
    assert mv.coefficients[a.blade("e2").mask] == pytest.approx(2.0)
    assert mv.coefficients[a.blade("e3").mask] == pytest.approx(3.0)
    assert mv.grades == frozenset({1})


def test_multivector_from_blade() -> None:
    a = Cl(3, 0)
    e12 = a.blade("e12")
    mv = Multivector.from_blade(e12, a, coefficient=2.0)
    assert mv.coefficients[e12.mask] == pytest.approx(2.0)
    assert mv.grades == frozenset({2})


def test_multivector_grades_restriction_zeros_out_others() -> None:
    a = Cl(3, 0)
    raw = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    mv = Multivector(raw, a, grades={1})
    # Only grade-1 components (e1, e2, e3) should be kept.
    for blade in a.blades():
        if blade.grade == 1:
            assert mv.coefficients[blade.mask] == raw[blade.mask]
        else:
            assert mv.coefficients[blade.mask] == 0


def test_multivector_rejects_wrong_dimension() -> None:
    a = Cl(3, 0)
    with pytest.raises(TesseraAlgebraError, match="last-axis length 8"):
        Multivector(np.zeros(7), a)


def test_multivector_rejects_grades_outside_algebra() -> None:
    a = Cl(3, 0)
    with pytest.raises(TesseraAlgebraError, match=r"grades \[4\]"):
        Multivector(np.zeros(8), a, grades={4})


def test_multivector_batch_shape_preserved() -> None:
    a = Cl(3, 0)
    coeffs = np.random.RandomState(0).rand(2, 5, 8).astype(np.float32)
    mv = Multivector(coeffs, a)
    assert mv.shape == (2, 5)
    assert mv.coefficients.shape == (2, 5, 8)


# ---------------------------------------------------------------------------
# Annotation surface — Multivector[Cl, ...]
# ---------------------------------------------------------------------------

def test_annotation_single_arg_returns_spec_with_no_grades() -> None:
    spec = Multivector[Cl(3, 0)]
    assert isinstance(spec, MultivectorSpec)
    assert spec.algebra == Cl(3, 0)
    assert spec.grades is None
    assert spec.kind == "multivector"


def test_annotation_with_grade_set() -> None:
    spec = Multivector[Cl(3, 0), {0, 2}]
    assert spec.grades == frozenset({0, 2})
    assert spec.kind == "multivector"


def test_annotation_with_int_grade_shortcut() -> None:
    spec = Multivector[Cl(3, 0), 2]
    assert spec.grades == frozenset({2})
    assert spec.is_grade_pure()


def test_annotation_rejects_non_cl_first_arg() -> None:
    with pytest.raises(TesseraAlgebraError, match="must be a Cl signature"):
        Multivector["not a Cl"]  # type: ignore[index]


def test_annotation_rejects_invalid_grades() -> None:
    with pytest.raises(TesseraAlgebraError, match=r"grades \[4\]"):
        Multivector[Cl(3, 0), 4]


def test_spec_is_grade_pure_and_even_odd_predicates() -> None:
    pure_bivector = Multivector[Cl(3, 0), 2]
    assert pure_bivector.is_grade_pure()
    assert pure_bivector.is_even()
    assert not pure_bivector.is_odd()
    assert pure_bivector.grade_value() == 2

    odd_vec = Multivector[Cl(3, 0), 1]
    assert odd_vec.is_odd()
    assert not odd_vec.is_even()

    mixed = Multivector[Cl(3, 0), {0, 2}]
    assert mixed.is_even()
    assert not mixed.is_grade_pure()
    with pytest.raises(TesseraAlgebraError, match="not grade-pure"):
        mixed.grade_value()


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

def test_addition_unifies_grade_sets() -> None:
    a = Cl(3, 0)
    x = Multivector.from_vector([1.0, 0.0, 0.0], a)  # grade {1}
    y = Multivector.from_blade(a.blade("e12"), a)    # grade {2}
    s = x + y
    assert s.grades == frozenset({1, 2})
    assert s.coefficients[a.blade("e1").mask] == pytest.approx(1.0)
    assert s.coefficients[a.blade("e12").mask] == pytest.approx(1.0)


def test_subtraction_unifies_grade_sets() -> None:
    a = Cl(3, 0)
    x = Multivector.from_vector([1.0, 0.0, 0.0], a)
    y = Multivector.from_vector([0.0, 1.0, 0.0], a)
    d = x - y
    assert d.grades == frozenset({1})
    assert d.coefficients[a.blade("e1").mask] == pytest.approx(1.0)
    assert d.coefficients[a.blade("e2").mask] == pytest.approx(-1.0)


def test_scalar_multiplication() -> None:
    a = Cl(3, 0)
    x = Multivector.from_vector([1.0, 2.0, 3.0], a)
    y = 2.0 * x
    assert y.coefficients[a.blade("e1").mask] == pytest.approx(2.0)
    assert y.coefficients[a.blade("e3").mask] == pytest.approx(6.0)
    z = x * 0.5
    assert z.coefficients[a.blade("e2").mask] == pytest.approx(1.0)


def test_negation() -> None:
    a = Cl(3, 0)
    x = Multivector.from_vector([1.0, 2.0, 3.0], a)
    y = -x
    assert np.allclose(y.coefficients, -x.coefficients)


def test_scalar_division() -> None:
    a = Cl(3, 0)
    x = Multivector.from_vector([2.0, 4.0, 6.0], a)
    y = x / 2.0
    assert y.coefficients[a.blade("e1").mask] == pytest.approx(1.0)


def test_multiplication_by_multivector_is_rejected() -> None:
    """`*` is scalar-only; use geometric_product for Multivector × Multivector."""
    a = Cl(3, 0)
    x = Multivector.from_vector([1.0, 0.0, 0.0], a)
    y = Multivector.from_vector([0.0, 1.0, 0.0], a)
    with pytest.raises(TypeError, match="geometric_product"):
        _ = x * y


def test_algebra_mismatch_raises_on_addition() -> None:
    x = Multivector.from_vector([1.0, 0.0, 0.0], Cl(3, 0))
    y = Multivector.zeros(Cl(1, 3))
    with pytest.raises(TesseraAlgebraError, match="algebra mismatch"):
        _ = x + y


# ---------------------------------------------------------------------------
# Equality / hash / repr
# ---------------------------------------------------------------------------

def test_equality_uses_coefficient_arrays() -> None:
    a = Cl(3, 0)
    x = Multivector.from_vector([1.0, 2.0, 3.0], a)
    y = Multivector.from_vector([1.0, 2.0, 3.0], a)
    z = Multivector.from_vector([1.0, 2.0, 4.0], a)
    assert x == y
    assert x != z


def test_multivector_is_unhashable() -> None:
    a = Cl(3, 0)
    x = Multivector.zeros(a)
    with pytest.raises(TypeError, match="not hashable"):
        hash(x)


def test_to_numpy_returns_writable_copy() -> None:
    a = Cl(3, 0)
    x = Multivector.from_vector([1.0, 2.0, 3.0], a)
    arr = x.to_numpy()
    arr[0] = 99.0
    # Original should be unchanged.
    assert x.coefficients[0] == 0.0


def test_active_grades_walks_nonzero_components() -> None:
    a = Cl(3, 0)
    coeffs = np.zeros(8)
    coeffs[a.blade("e1").mask] = 1.0
    coeffs[a.blade("e12").mask] = 1.0
    mv = Multivector(coeffs, a)  # no declared grades
    assert mv.active_grades == frozenset({1, 2})
