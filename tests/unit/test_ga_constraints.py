"""GA2 acceptance: grade-aware type annotations + constraint predicates.

Sprint: GA2 (grade-aware ConstraintSolver types).
Roadmap: docs/audit/domain/DOMAIN_AUDIT.md § GA2

Covers the GA2 acceptance criteria:
  - Annotation surface: Rotor[Cl(3,0)] / DiffForm[Cl(3,0), k=2] /
    VectorField[Cl(3,0)] / Morphism[Cl(3,0), Cl(3,0)] each produce a
    well-formed spec usable in @tessera.jit signatures.
  - Predicate types: GradeIn / Even / Odd / IsRotor / IsForm follow the
    existing Constraint protocol (check(bindings) → None | error,
    dim_names()).
  - Compatibility: adding a Multivector{Grade={1}} arg where the
    constraint requires Even (or grade ⊆ {0, 2}) raises
    TesseraConstraintError via ConstraintSolver.check().
  - Rotor type carries equivariance proof: IsRotor accepts the
    Rotor[Cl]-annotated value, rejects a "happens to be even-grade"
    Multivector.
"""

from __future__ import annotations

import pytest

from tessera.compiler.constraints import (
    ConstraintSolver,
    TesseraConstraintError,
)
from tessera.ga import (
    Cl,
    DiffForm,
    Even,
    GradeIn,
    IsForm,
    IsRotor,
    Morphism,
    MorphismSpec,
    Multivector,
    MultivectorSpec,
    Odd,
    Rotor,
    TesseraAlgebraError,
    VectorField,
)


# ---------------------------------------------------------------------------
# Annotation markers — each must produce a well-formed MultivectorSpec
# ---------------------------------------------------------------------------

def test_rotor_subscript_returns_even_grade_spec() -> None:
    spec = Rotor[Cl(3, 0)]
    assert isinstance(spec, MultivectorSpec)
    assert spec.kind == "rotor"
    assert spec.grades == frozenset({0, 2})


def test_diff_form_subscript_returns_pure_grade_spec() -> None:
    spec = DiffForm[Cl(3, 0), 2]
    assert isinstance(spec, MultivectorSpec)
    assert spec.kind == "diff_form"
    assert spec.grades == frozenset({2})
    assert spec.is_grade_pure()


def test_vector_field_subscript_returns_grade1_spec() -> None:
    spec = VectorField[Cl(3, 0)]
    assert spec.kind == "vector_field"
    assert spec.grades == frozenset({1})


def test_morphism_subscript_carries_both_algebras() -> None:
    spec = Morphism[Cl(3, 0), Cl(1, 3)]
    assert isinstance(spec, MorphismSpec)
    assert spec.source == Cl(3, 0)
    assert spec.target == Cl(1, 3)


def test_rotor_is_annotation_only_not_constructable() -> None:
    with pytest.raises(TypeError, match="annotation marker"):
        Rotor()


def test_diff_form_rejects_grade_outside_algebra() -> None:
    with pytest.raises(TesseraAlgebraError, match="exceeds algebra grades"):
        DiffForm[Cl(3, 0), 4]


def test_diff_form_rejects_negative_grade() -> None:
    with pytest.raises(TesseraAlgebraError, match="non-negative int"):
        DiffForm[Cl(3, 0), -1]


def test_annotation_markers_require_cl_signature() -> None:
    with pytest.raises(TesseraAlgebraError, match="must be a Cl signature"):
        Rotor["not a Cl"]  # type: ignore[index]
    with pytest.raises(TesseraAlgebraError):
        VectorField["not a Cl"]  # type: ignore[index]


# ---------------------------------------------------------------------------
# GradeIn — grade-set subset predicate
# ---------------------------------------------------------------------------

def test_grade_in_accepts_compatible_grade_set() -> None:
    c = GradeIn("x", {0, 2})
    spec = Multivector[Cl(3, 0), {0, 2}]
    err = c.check({"x": spec})
    assert err is None


def test_grade_in_rejects_extra_grades() -> None:
    c = GradeIn("x", {0, 2})
    spec = Multivector[Cl(3, 0), {0, 2, 3}]
    err = c.check({"x": spec})
    assert err is not None
    assert "disallowed grade" in str(err)
    assert "3" in str(err)


def test_grade_in_rejects_unrestricted_multivector_spec() -> None:
    c = GradeIn("x", {0, 2})
    spec = Multivector[Cl(3, 0)]  # unrestricted
    err = c.check({"x": spec})
    assert err is not None
    assert "unrestricted multivector" in str(err)


def test_grade_in_returns_none_when_binding_missing_or_non_mv() -> None:
    c = GradeIn("x", {1})
    assert c.check({}) is None
    # If the binding is something else (an int dimension, for example),
    # this constraint is N/A — skip cleanly.
    assert c.check({"x": 128}) is None


def test_grade_in_validates_actual_multivector_value() -> None:
    """Pass an actual Multivector (not a Spec); active grades are inferred."""
    c = GradeIn("x", {1})
    a = Cl(3, 0)
    vec = Multivector.from_vector([1.0, 2.0, 3.0], a)  # grades = {1}
    assert c.check({"x": vec}) is None

    # A bivector value violates GradeIn({1}).
    bi = Multivector.from_blade(a.blade("e12"), a)
    err = c.check({"x": bi})
    assert err is not None
    assert "disallowed grade" in str(err)


def test_grade_in_dim_names_reports_parameter() -> None:
    c = GradeIn("x", {0, 2})
    assert c.dim_names() == ["x"]


def test_grade_in_constructor_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        GradeIn("", {0, 2})


# ---------------------------------------------------------------------------
# Even / Odd
# ---------------------------------------------------------------------------

def test_even_accepts_even_grade_spec() -> None:
    c = Even("x")
    assert c.check({"x": Multivector[Cl(3, 0), {0, 2}]}) is None
    assert c.check({"x": Rotor[Cl(3, 0)]}) is None


def test_even_rejects_odd_grade_component() -> None:
    c = Even("x")
    spec = Multivector[Cl(3, 0), {0, 1, 2}]
    err = c.check({"x": spec})
    assert err is not None
    assert "odd-grade component" in str(err)


def test_odd_accepts_pure_vector_field() -> None:
    c = Odd("x")
    assert c.check({"x": VectorField[Cl(3, 0)]}) is None


def test_odd_rejects_scalar_grade_zero() -> None:
    c = Odd("x")
    spec = Multivector[Cl(3, 0), {0, 1}]
    err = c.check({"x": spec})
    assert err is not None
    assert "even-grade component" in str(err)


# ---------------------------------------------------------------------------
# IsRotor — kind-aware: even alone is not enough
# ---------------------------------------------------------------------------

def test_is_rotor_accepts_rotor_kind_annotation() -> None:
    c = IsRotor("R")
    assert c.check({"R": Rotor[Cl(3, 0)]}) is None


def test_is_rotor_rejects_even_grade_multivector_without_rotor_kind() -> None:
    """Even-grade is necessary but not sufficient — Rotor carries the proof."""
    c = IsRotor("R")
    even_but_not_rotor = Multivector[Cl(3, 0), {0, 2}]
    err = c.check({"R": even_but_not_rotor})
    assert err is not None
    assert "kind='multivector'" in str(err) or "kind='multivector" in str(err)
    assert "equivariance proof" in str(err)


def test_is_rotor_rejects_diff_form() -> None:
    c = IsRotor("R")
    err = c.check({"R": DiffForm[Cl(3, 0), 2]})
    assert err is not None
    assert "kind='diff_form'" in str(err)


# ---------------------------------------------------------------------------
# IsForm — kind- and grade-pure check
# ---------------------------------------------------------------------------

def test_is_form_accepts_matching_kind_and_grade() -> None:
    c = IsForm("omega", 2)
    assert c.check({"omega": DiffForm[Cl(3, 0), 2]}) is None


def test_is_form_rejects_mismatched_grade() -> None:
    c = IsForm("omega", 2)
    err = c.check({"omega": DiffForm[Cl(3, 0), 1]})
    assert err is not None
    assert "grade-2" in str(err)


def test_is_form_rejects_non_diff_form_kind() -> None:
    c = IsForm("omega", 1)
    err = c.check({"omega": VectorField[Cl(3, 0)]})
    assert err is not None
    assert "kind='vector_field'" in str(err)


# ---------------------------------------------------------------------------
# Integration with ConstraintSolver
# ---------------------------------------------------------------------------

def test_constraint_solver_runs_ga_predicates_alongside_existing() -> None:
    from tessera.compiler.constraints import Divisible

    solver = ConstraintSolver()
    solver.add(GradeIn("x", {0, 2}))
    solver.add(Divisible("D", 64))

    # Passing bindings: both dimensions valid.
    solver.check({"x": Multivector[Cl(3, 0), {0, 2}], "D": 128})

    # Failing GA binding: detected even though Divisible would also be checkable.
    with pytest.raises(TesseraConstraintError, match="disallowed grade"):
        solver.check({"x": Multivector[Cl(3, 0), {0, 1, 2}], "D": 128})


def test_constraint_solver_check_all_collects_ga_violations() -> None:
    solver = ConstraintSolver()
    solver.add(GradeIn("x", {0, 2}))
    solver.add(Even("y"))

    errors = solver.check_all({
        "x": Multivector[Cl(3, 0), {1}],
        "y": Multivector[Cl(3, 0), {0, 1}],
    })
    assert len(errors) == 2


def test_mismatched_grade_addition_raises_via_constraint_check() -> None:
    """The acceptance scenario: a function expects grade-{1} args; the
    binding wires through a value with grade-{2} components and the
    decoration-time check catches it.
    """
    grade_one_only = GradeIn("a", {1})
    a = Cl(3, 0)
    bad_value = Multivector.from_blade(a.blade("e12"), a)  # grade {2}
    err = grade_one_only.check({"a": bad_value})
    assert err is not None
    assert isinstance(err, TesseraConstraintError)
    assert "disallowed grade" in str(err)
    # The error carries enough information to localize the violation.
    assert err.dim_name == "a"
