from fractions import Fraction

import pytest

from tessera.compiler.pde_operator import (
    PDEClassification,
    PDEOperator,
    PDETerm,
    certify_explicit_diagonal_diffusion,
)


def _operator(*coefficients: Fraction) -> PDEOperator:
    rank = len(coefficients)
    return PDEOperator(
        domain_rank=rank,
        max_order=2,
        element_dtype="f64",
        terms=tuple(
            PDETerm(tuple(2 if axis == i else 0 for axis in range(rank)), value)
            for i, value in enumerate(coefficients)
        ),
    )


def test_exact_classification_distinguishes_canonical_second_order_types() -> None:
    assert _operator(Fraction(1), Fraction(1)).classify() is PDEClassification.ELLIPTIC
    assert _operator(Fraction(1), Fraction(0)).classify() is PDEClassification.PARABOLIC
    assert _operator(Fraction(1), Fraction(-1)).classify() is PDEClassification.HYPERBOLIC
    mixed = PDEOperator(
        domain_rank=3,
        max_order=2,
        element_dtype="f64",
        terms=(PDETerm((2, 0, 0), Fraction(1)), PDETerm((0, 2, 0), Fraction(-1))),
    )
    assert mixed.classify() is PDEClassification.MIXED


def test_mixed_derivative_uses_symmetric_quadratic_form_convention() -> None:
    operator = PDEOperator(
        domain_rank=2,
        max_order=2,
        element_dtype="f64",
        terms=(PDETerm((1, 1), Fraction(6)),),
    )
    assert operator.principal_matrix() == ((0, 3), (3, 0))
    assert operator.classify() is PDEClassification.HYPERBOLIC


def test_exact_inertia_retains_coupling_after_a_two_by_two_pivot() -> None:
    operator = PDEOperator(
        domain_rank=3,
        max_order=2,
        element_dtype="f64",
        terms=(
            PDETerm((1, 1, 0), Fraction(2)),
            PDETerm((1, 0, 1), Fraction(2)),
            PDETerm((0, 1, 1), Fraction(2)),
        ),
    )
    # Eigenvalues are 2,-1,-1.  A defective implementation that drops the
    # third-axis coupling after its first 2x2 pivot reports (1,1,1).
    assert operator.classify() is PDEClassification.HYPERBOLIC


def test_exact_ftcs_certificate_keeps_the_boundary_and_rejects_one_step_over() -> None:
    laplacian = _operator(Fraction(1), Fraction(1))
    at_limit = certify_explicit_diagonal_diffusion(
        laplacian, spacing=(Fraction(1), Fraction(2)), dt=Fraction(2, 5)
    )
    assert at_limit.stable and at_limit.dt_max == Fraction(2, 5)
    assert not certify_explicit_diagonal_diffusion(
        laplacian, spacing=(Fraction(1), Fraction(2)), dt=Fraction(401, 1000)
    ).stable


def test_certificate_fails_closed_for_mixed_derivatives() -> None:
    operator = PDEOperator(
        domain_rank=2,
        max_order=2,
        element_dtype="f64",
        terms=(
            PDETerm((2, 0), Fraction(2)),
            PDETerm((1, 1), Fraction(1)),
            PDETerm((0, 2), Fraction(2)),
        ),
    )
    with pytest.raises(ValueError, match="mixed derivatives"):
        certify_explicit_diagonal_diffusion(
            operator, spacing=(Fraction(1), Fraction(1)), dt=Fraction(1, 10)
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"domain_rank": 0}, "rank"),
        ({"element_dtype": "f16"}, "f32/f64"),
        ({"terms": (PDETerm((3, 0), Fraction(1)),)}, "maximum order"),
    ],
)
def test_typed_operator_rejects_invalid_contracts(kwargs: dict, message: str) -> None:
    values = {
        "domain_rank": 2,
        "max_order": 2,
        "element_dtype": "f64",
        "terms": (PDETerm((2, 0), Fraction(1)),),
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        PDEOperator(**values)
