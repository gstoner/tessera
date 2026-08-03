"""W1.4 — a declared grade set must reach `geometric_product`.

`MultivectorSpec.grades` / `Multivector.grades` were declared, validated on
construction, and read by nothing. The integrated compiler plan cites this as a
live Decision #29 violation ("a declaration must have a consumer"), and it had a
second edge: `geometric_product` skipped blades by asking ``np.any(coeff)`` —
scanning the DATA at runtime for a fact the TYPE already states, which is
Decision #30 inverted. A vector in Cl(3,0) is grade {1}; three of its eight
blades can be non-zero, and no amount of scanning makes that more certain than
the annotation does.

Two things follow, and both are checked here:

  * the declaration prunes which blades participate, and
  * the product's own grade set is DERIVED, so a chain keeps narrowing instead
    of falling back to "unrestricted" after one step.

The MLIR half (`input_grades` on `geo_product`, consumed by
`ExpandProductTable`) is covered by the lit fixtures
`input_grade_fusion.mlir` and `input_grade_fusion_prunes.mlir`.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.ga import Cl, Multivector
from tessera.ga.ops import geometric_product


def _cl30() -> Cl:
    return Cl(3, 0, 0)


def test_vector_times_vector_is_scalar_plus_bivector():
    """The textbook result, and the one the derivation has to reproduce.

    `uv = u·v + u∧v` — grade 0 and grade 2, never grades 1 or 3.
    """
    algebra = _cl30()
    u = Multivector.from_vector([1.0, 2.0, 3.0], algebra)
    v = Multivector.from_vector([4.0, 5.0, 6.0], algebra)
    product = geometric_product(u, v)
    assert product.grades == frozenset({0, 2}), product.grades
    # The scalar part is the dot product.
    assert product.coefficients[0] == pytest.approx(32.0)


def test_pruning_does_not_change_the_numbers():
    """The declared-grade path and the scan path must agree exactly.

    This is the assertion that makes the optimisation safe rather than merely
    fast: pruning by declaration is only sound if the blades it skips were
    genuinely zero.
    """
    algebra = _cl30()
    u = Multivector.from_vector([1.0, 2.0, 3.0], algebra)
    v = Multivector.from_vector([4.0, 5.0, 6.0], algebra)
    declared = geometric_product(u, v)

    # Same coefficients, but with no declared restriction — forces the runtime
    # scan path.
    bare_u = Multivector(u.coefficients.copy(), algebra)
    bare_v = Multivector(v.coefficients.copy(), algebra)
    scanned = geometric_product(bare_u, bare_v)

    assert scanned.grades is None, "unrestricted operands must stay unrestricted"
    np.testing.assert_allclose(declared.coefficients, scanned.coefficients)


def test_the_derived_grade_set_narrows_through_a_chain():
    """`(uv)u` is grades {1,3}: the restriction propagates rather than resetting.

    Without propagation the product would return `None` and every step after
    the first would fall back to scanning — the declaration would help once and
    then be lost.
    """
    algebra = _cl30()
    u = Multivector.from_vector([1.0, 2.0, 3.0], algebra)
    v = Multivector.from_vector([4.0, 5.0, 6.0], algebra)
    chained = geometric_product(geometric_product(u, v), u)
    assert chained.grades == frozenset({1, 3}), chained.grades


def test_an_unrestricted_operand_leaves_the_result_unrestricted():
    """Mixing declared and undeclared must NOT invent a restriction.

    An absent declaration means "unrestricted", not "empty". Getting that
    backwards is the failure mode with teeth: it would prune blades that can be
    non-zero and silently produce a wrong product.
    """
    algebra = _cl30()
    u = Multivector.from_vector([1.0, 2.0, 3.0], algebra)
    mixed = Multivector(np.ones(algebra.dim, dtype=np.float64), algebra)
    assert geometric_product(u, mixed).grades is None
    assert geometric_product(mixed, u).grades is None


def test_a_restricted_operand_still_multiplies_correctly_against_a_mixed_one():
    """The numbers, not just the grade bookkeeping."""
    algebra = _cl30()
    u = Multivector.from_vector([1.0, 0.0, 0.0], algebra)
    mixed = Multivector(np.arange(1.0, algebra.dim + 1.0), algebra)
    restricted = geometric_product(u, mixed)
    bare = geometric_product(Multivector(u.coefficients.copy(), algebra), mixed)
    np.testing.assert_allclose(restricted.coefficients, bare.coefficients)


def test_scalar_times_anything_keeps_the_other_operands_grades():
    """A grade-0 operand cannot change which grades are present."""
    algebra = _cl30()
    scalar = Multivector.scalar(2.0, algebra)
    u = Multivector.from_vector([1.0, 2.0, 3.0], algebra)
    product = geometric_product(scalar, u)
    assert product.grades == frozenset({1}), product.grades
    np.testing.assert_allclose(product.coefficients, 2.0 * u.coefficients)
