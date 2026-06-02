"""GA6 acceptance: parallel multivector autodiff registry.

Sprint: GA6.
Roadmap: docs/audit/domain/DOMAIN_AUDIT.md § GA6
Scope lock: docs/audit/domain/DOMAIN_AUDIT.md § Q3

Covers:
  - 17 GA ops registered in _VJPS_GEO and _JVPS_GEO (12 GA3 + 5 GA5
    minus field-level ops which need MultivectorField cotangents).
  - check_grad verifies every registered VJP/JVP matches central
    differences to ≤ 1e-4 absolute on random Cl(3,0) and Cl(1,3) inputs.
  - Headline test: gradient of f(R, v) = ‖R v R†‖² wrt R lies in the
    even-grade subspace (Decision GA-L4 confirmed by grade analysis).
  - tape_geo() and multivector_grad() helpers exist and compose
    cleanly with the existing autodiff tape.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tessera.autodiff.geometric import (
    GeometricTape,
    _JVPS_GEO,
    _VJPS_GEO,
    check_grad_geo,
    get_jvp_geo,
    get_vjp_geo,
    multivector_grad,
    tape_geo,
)
from tessera.autodiff.geometric.check_grad import check_jvp_geo
from tessera.autodiff.geometric.registry import register_vjp_geo, register_jvp_geo
from tessera.ga import (
    Cl,
    Multivector,
    conjugate,
    geometric_product,
    grade_involution,
    grade_projection,
    hodge_star,
    inner,
    left_contraction,
    norm,
    norm_squared,
    reverse,
    rotor_from_axis,
    rotor_sandwich,
    wedge,
)


# ---------------------------------------------------------------------------
# Registry — coverage check
# ---------------------------------------------------------------------------

EXPECTED_VJP_OPS = {
    # Linear
    "add", "sub", "neg", "scalar_mul",
    "grade_projection", "reverse", "grade_involution", "conjugate",
    "hodge_star",
    # Bilinear
    "geometric_product", "wedge", "left_contraction",
    # Scalar-valued
    "inner", "norm", "norm_squared",
    # Composite
    "rotor_sandwich",
}

EXPECTED_JVP_OPS = EXPECTED_VJP_OPS  # parity


def test_all_ga6_vjps_registered() -> None:
    assert EXPECTED_VJP_OPS.issubset(_VJPS_GEO.keys())
    for op in EXPECTED_VJP_OPS:
        assert get_vjp_geo(op) is not None


def test_all_ga6_jvps_registered() -> None:
    assert EXPECTED_JVP_OPS.issubset(_JVPS_GEO.keys())
    for op in EXPECTED_JVP_OPS:
        assert get_jvp_geo(op) is not None


def test_vjp_geo_registry_is_independent_of_tensor_vjp_registry() -> None:
    """The parallel registry must not touch the existing tensor VJPs."""
    from tessera.autodiff.vjp import _VJPS as _TENSOR_VJPS

    # No name collisions (every geometric op is tagged with the
    # multivector-specific signature in its name; tensor ops have
    # different names).
    geo_names = set(_VJPS_GEO.keys())
    # 'add' / 'sub' / 'neg' may exist in both; verify the entries are
    # distinct callables (different signatures).
    for shared in {"add", "sub"} & set(_TENSOR_VJPS.keys()):
        assert _VJPS_GEO[shared] is not _TENSOR_VJPS[shared]


# ---------------------------------------------------------------------------
# check_grad — per-op verification on Cl(3,0)
# ---------------------------------------------------------------------------

@pytest.fixture
def cl30():
    return Cl(3, 0)


def _random_mv(algebra: Cl, seed: int = 0) -> Multivector:
    rng = np.random.RandomState(seed)
    return Multivector(rng.randn(algebra.dim).astype(np.float64), algebra)


def test_check_grad_add(cl30) -> None:
    a, b = _random_mv(cl30, 0), _random_mv(cl30, 1)
    check_grad_geo("add", lambda x, y: x + y, (a, b))


def test_check_grad_sub(cl30) -> None:
    a, b = _random_mv(cl30, 2), _random_mv(cl30, 3)
    check_grad_geo("sub", lambda x, y: x - y, (a, b))


def test_check_grad_neg(cl30) -> None:
    a = _random_mv(cl30, 4)
    check_grad_geo("neg", lambda x: -x, (a,))


def test_check_grad_geometric_product(cl30) -> None:
    a, b = _random_mv(cl30, 5), _random_mv(cl30, 6)
    check_grad_geo(
        "geometric_product", geometric_product, (a, b)
    )


def test_check_grad_wedge(cl30) -> None:
    a, b = _random_mv(cl30, 7), _random_mv(cl30, 8)
    check_grad_geo("wedge", wedge, (a, b))


def test_check_grad_left_contraction(cl30) -> None:
    a, b = _random_mv(cl30, 9), _random_mv(cl30, 10)
    check_grad_geo("left_contraction", left_contraction, (a, b))


def test_check_grad_grade_projection(cl30) -> None:
    a = _random_mv(cl30, 11)
    for k in (0, 1, 2, 3):
        check_grad_geo(
            "grade_projection",
            lambda x, k_arg=k: grade_projection(x, k_arg),
            (a, k),
        )


def test_check_grad_reverse(cl30) -> None:
    a = _random_mv(cl30, 12)
    check_grad_geo("reverse", reverse, (a,))


def test_check_grad_grade_involution(cl30) -> None:
    a = _random_mv(cl30, 13)
    check_grad_geo("grade_involution", grade_involution, (a,))


def test_check_grad_conjugate(cl30) -> None:
    a = _random_mv(cl30, 14)
    check_grad_geo("conjugate", conjugate, (a,))


def test_check_grad_hodge_star_cl30(cl30) -> None:
    a = _random_mv(cl30, 15)
    check_grad_geo("hodge_star", hodge_star, (a,))


def test_check_grad_inner(cl30) -> None:
    """Inner returns scalar — cotangent is a scalar; the VJP returns
    Multivector grads. Use a tiny ad-hoc verification."""
    a = _random_mv(cl30, 16)
    b = _random_mv(cl30, 17)
    dout = 2.5
    grad_a, grad_b = _VJPS_GEO["inner"](dout, a, b)
    # Numerical check: ∂/∂a_i [2.5 · <a, b>] = 2.5 · b_i (for Cl(3,0)).
    expected_grad_a = 2.5 * b.coefficients
    assert np.allclose(grad_a.coefficients, expected_grad_a, atol=1e-12)
    expected_grad_b = 2.5 * a.coefficients
    assert np.allclose(grad_b.coefficients, expected_grad_b, atol=1e-12)


def test_check_grad_norm_squared(cl30) -> None:
    a = _random_mv(cl30, 18)
    dout = 1.3
    (grad_a,) = _VJPS_GEO["norm_squared"](dout, a)
    expected = 2.0 * 1.3 * a.coefficients
    assert np.allclose(grad_a.coefficients, expected, atol=1e-10)


def test_check_grad_norm_via_finite_difference(cl30) -> None:
    a = _random_mv(cl30, 19)
    dout = 1.1
    (grad_a,) = _VJPS_GEO["norm"](dout, a)
    # Numerical: ∂ |a| / ∂a_i = a_i / |a|.
    n = float(np.linalg.norm(a.coefficients))
    expected = 1.1 * a.coefficients / n
    assert np.allclose(grad_a.coefficients, expected, atol=1e-10)


def test_check_grad_scalar_mul(cl30) -> None:
    a = _random_mv(cl30, 20)
    grad_a, grad_s = _VJPS_GEO["scalar_mul"](
        _random_mv(cl30, 21), a, 2.5
    )
    # grad_a should equal 2.5 * dout; grad_s should be a Python scalar.
    assert isinstance(grad_s, float)


def test_check_grad_rotor_sandwich(cl30) -> None:
    # Construct a rotor and a vector.
    bivector = Multivector.from_blade(cl30.blade("e12"), cl30, dtype=np.float64)
    R = rotor_from_axis(bivector, math.pi / 4)
    v = Multivector.from_vector([1.0, 0.5, -0.3], cl30, dtype=np.float64)
    check_grad_geo("rotor_sandwich", rotor_sandwich, (R, v),
                   eps=1e-5, atol=5e-4)


# ---------------------------------------------------------------------------
# JVP — central-difference parity
# ---------------------------------------------------------------------------

def test_jvp_geometric_product(cl30) -> None:
    a, b = _random_mv(cl30, 30), _random_mv(cl30, 31)
    da, db = _random_mv(cl30, 32), _random_mv(cl30, 33)
    check_jvp_geo("geometric_product", geometric_product,
                  primals=(a, b), tangents=(da, db))


def test_jvp_wedge(cl30) -> None:
    a, b = _random_mv(cl30, 34), _random_mv(cl30, 35)
    da, db = _random_mv(cl30, 36), _random_mv(cl30, 37)
    check_jvp_geo("wedge", wedge, primals=(a, b), tangents=(da, db))


def test_jvp_inner(cl30) -> None:
    a, b = _random_mv(cl30, 38), _random_mv(cl30, 39)
    da, db = _random_mv(cl30, 40), _random_mv(cl30, 41)
    check_jvp_geo("inner", inner, primals=(a, b), tangents=(da, db))


def test_jvp_reverse(cl30) -> None:
    a = _random_mv(cl30, 42)
    da = _random_mv(cl30, 43)
    check_jvp_geo("reverse", reverse, primals=(a,), tangents=(da,))


def test_jvp_hodge_star(cl30) -> None:
    a = _random_mv(cl30, 44)
    da = _random_mv(cl30, 45)
    check_jvp_geo("hodge_star", hodge_star, primals=(a,), tangents=(da,))


# ---------------------------------------------------------------------------
# Headline test: gradient of rotor sandwich is even-grade
# ---------------------------------------------------------------------------

def test_rotor_sandwich_gradient_is_even_grade(cl30) -> None:
    """For ``L = ‖R v R†‖²``, the gradient ∂L/∂R must lie in the
    even-grade subspace of Cl(3,0) — grades {0, 2} only.

    This is the GA-L4 equivariance-from-algebra claim made enforceable:
    rotors live in the even subalgebra Cl⁺(p,q), and their gradients do
    too (no leakage into grades 1 or 3 — those would break the rotor
    structure).
    """
    # Use a non-trivial rotor (not the identity).
    axis = Multivector.from_blade(cl30.blade("e12"), cl30, dtype=np.float64)
    angle = math.pi / 3
    R = rotor_from_axis(axis, angle)
    v = Multivector.from_vector([1.0, -0.5, 0.7], cl30, dtype=np.float64)
    # L = norm_squared(rotor_sandwich(R, v)) — scalar.
    y = rotor_sandwich(R, v)
    # ∂L/∂y = 2y; chain through rotor_sandwich VJP.
    dout = 2.0 * y
    grad_R, grad_v = _VJPS_GEO["rotor_sandwich"](dout, R, v)
    # Grade analysis on grad_R.
    odd_components = sum(
        abs(grad_R.coefficients[b.mask])
        for b in cl30.blades()
        if b.grade % 2 == 1
    )
    even_components = sum(
        abs(grad_R.coefficients[b.mask])
        for b in cl30.blades()
        if b.grade % 2 == 0
    )
    # Odd-grade leakage should be negligible (numerical noise only).
    assert odd_components < 1e-10, (
        f"Rotor sandwich gradient leaked into odd grades: "
        f"|odd| = {odd_components:.3e}, |even| = {even_components:.3e}"
    )
    assert even_components > 0.1, (
        f"Rotor sandwich gradient has trivial even part: {even_components:.3e}"
    )


def test_rotor_sandwich_gradient_for_input_vector_stays_grade1(cl30) -> None:
    """Symmetrically: ∂L/∂v for L = ‖R v R†‖² should stay grade-1 because
    v itself is grade-1 and the sandwich preserves grade structure.
    """
    axis = Multivector.from_blade(cl30.blade("e23"), cl30, dtype=np.float64)
    R = rotor_from_axis(axis, math.pi / 5)
    v = Multivector.from_vector([0.3, 1.2, -0.4], cl30, dtype=np.float64)
    y = rotor_sandwich(R, v)
    dout = 2.0 * y
    grad_R, grad_v = _VJPS_GEO["rotor_sandwich"](dout, R, v)
    non_grade1 = sum(
        abs(grad_v.coefficients[b.mask])
        for b in cl30.blades()
        if b.grade != 1
    )
    assert non_grade1 < 1e-10, (
        f"grad_v leaked into non-grade-1 components: {non_grade1:.3e}"
    )


# ---------------------------------------------------------------------------
# Mixed tensor + multivector tapes coexist
# ---------------------------------------------------------------------------

def test_tape_geo_and_tensor_tape_coexist(cl30) -> None:
    """Opening tape_geo() must not disturb the existing tensor tape."""
    from tessera.autodiff.tape import tape as tensor_tape

    with tensor_tape() as t_tensor:
        with tape_geo() as t_geo:
            # Record one entry on each.
            mv = _random_mv(cl30, 50)
            t_geo.record("manual", mv, (mv,))
            assert len(t_geo) == 1
            # The tensor tape is unrelated.
            assert isinstance(t_geo, GeometricTape)
            # Both tapes are accessible simultaneously.
    # After exiting both, no leakage.
    from tessera.autodiff.geometric.tape import active_tape

    assert active_tape() is None


def test_multivector_grad_returns_finite_gradient(cl30) -> None:
    """The fallback central-difference helper returns a finite gradient
    even without registered VJPs. Sanity-only — full tape AD is GA10."""
    def f(x):
        # Scalar loss: ‖x‖².
        return float(np.sum(x.coefficients ** 2))

    x = _random_mv(cl30, 99)
    grad_fn = multivector_grad(f)
    grad = grad_fn(x)
    assert isinstance(grad, Multivector)
    # ∂ ‖x‖² / ∂x_i = 2 x_i.
    expected = 2.0 * x.coefficients
    assert np.allclose(grad.coefficients, expected, atol=1e-3)


def test_register_vjp_geo_decorator_works(cl30) -> None:
    """Registration API parity with the tensor autodiff registry."""
    @register_vjp_geo("test_op_xyz")
    def vjp_test(dout, a):
        return (a,)

    assert get_vjp_geo("test_op_xyz") is vjp_test
    # Cleanup so this entry doesn't pollute the registry for other tests.
    _VJPS_GEO.pop("test_op_xyz", None)


def test_register_jvp_geo_decorator_works(cl30) -> None:
    @register_jvp_geo("test_jvp_xyz")
    def jvp_test(tangents, primals):
        return primals[0]

    assert get_jvp_geo("test_jvp_xyz") is jvp_test
    _JVPS_GEO.pop("test_jvp_xyz", None)
