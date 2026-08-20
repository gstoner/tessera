"""The matrix-function / factorization family (MC1) and the Kronecker cost identity (MC4).

`docs/audit/compiler/MATRIX_CALCULUS_REVIEW.md` MC1: of 379 public `tessera.ops`
attributes, the linear-algebra family was `cholesky`, `cholesky_solve`, `lu`,
`qr`, `svd`, `tri_solve`, `tridiagonal_solve` — and nothing else. `det`,
`logdet`, `inv`, `solve`, `trace`, `eigh`, `kron`, `vec`, `matrix_power` and
`norm` were all absent, which is the third derivative family: neither
elementwise nor multilinear, so `autodiff/linear.py` cannot derive them and the
pointwise-ODE family cannot reach them.

Every rule is checked three ways, because each catches a different error class:

* **order of accuracy** — the JVP against a central difference, checking the
  *rate* (2.0) rather than a tolerance. A rule wrong by a constant factor
  passes a loose tolerance and flattens the slope immediately.
* **the adjoint identity** — `<Jv, u> == <v, J^T u>`, which localizes a
  disagreement between the mode pair with no reference implementation.
* **closed forms from the notes** — `grad det = det(A) A^-T`, `grad logdet =
  A^-T`, `grad ||A||_F = A/||A||_F`, `grad(sum eigenvalues) = I`.

The law harness in `test_autodiff_laws.py` sweeps all ten as well, via the
`LAW_INPUT_SPECS` entries added alongside them.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import debug, ops
from tessera.autodiff import degeneracy as deg
from tessera.autodiff import grad
from tessera.autodiff.jvp import _JVPS
from tessera.autodiff.vjp import _VJPS

FAMILY = ("det", "logdet", "inv", "solve", "trace", "eigh",
          "kron", "vec", "matrix_power", "norm")


def _rng() -> np.random.Generator:
    return np.random.default_rng(20260820)


def _well_conditioned(n: int = 4) -> np.ndarray:
    return _rng().standard_normal((n, n)) + 4.0 * np.eye(n)


def _spd(n: int = 4) -> np.ndarray:
    M = _rng().standard_normal((n, n))
    return M @ M.T + n * np.eye(n)


def _flat(x) -> np.ndarray:
    if isinstance(x, tuple):
        return np.concatenate([np.ravel(np.asarray(t)) for t in x])
    return np.ravel(np.asarray(x))


# ── the family exists, in both registries (Decision #24) ────────────────────
@pytest.mark.parametrize("name", FAMILY)
def test_op_is_public(name) -> None:
    assert hasattr(ops, name), f"tessera.ops.{name} is missing"


@pytest.mark.parametrize("name", FAMILY)
def test_op_has_both_derivative_modes(name) -> None:
    assert name in _VJPS, f"{name} has no VJP"
    assert name in _JVPS, f"{name} has no JVP"


@pytest.mark.parametrize("name", FAMILY)
def test_op_is_in_the_coverage_registry(name) -> None:
    from tessera.compiler.primitive_coverage import all_primitive_coverages

    coverage = all_primitive_coverages()
    assert name in coverage, (
        f"{name} is not in primitive_coverage — Decision #24 requires a new "
        f"primitive to land in the catalog AND the coverage registry"
    )
    entry = coverage[name]
    assert entry.contract_status["vjp"] == "complete"
    assert entry.contract_status["jvp"] == "complete"


# ── the adjoint identity, over the whole family ─────────────────────────────
_ADJOINT_CASES = [
    ("det", lambda: (_well_conditioned(),), {}),
    ("logdet", lambda: (_spd(),), {}),
    ("inv", lambda: (_well_conditioned(),), {}),
    ("trace", lambda: (_well_conditioned(),), {}),
    ("vec", lambda: (_rng().standard_normal((4, 3)),), {}),
    ("kron", lambda: (_rng().standard_normal((2, 3)),
                      _rng().standard_normal((3, 2))), {}),
    ("solve", lambda: (_well_conditioned(), _rng().standard_normal(4)), {}),
    ("matrix_power", lambda: (_well_conditioned(),), {"n": 3}),
    ("norm", lambda: (_rng().standard_normal((4, 3)),), {"ord": "fro"}),
    ("norm", lambda: (_rng().standard_normal((4, 3)),), {"ord": "nuc"}),
    ("eigh", lambda: (np.diag([5.0, 3.0, 1.0])
                      + 0.1 * np.eye(3),), {}),
]


@pytest.mark.parametrize("op,make,kwargs", _ADJOINT_CASES,
                         ids=[f"{c[0]}{c[2] or ''}" for c in _ADJOINT_CASES])
def test_adjoint_identity_holds(op, make, kwargs) -> None:
    """`<J v, u> == <v, J^T u>` — the VJP must be the adjoint of the JVP."""
    rng = np.random.default_rng(11)
    primals = make()
    worst = 0.0
    for _ in range(4):
        tangents = tuple(rng.standard_normal(np.shape(p)) for p in primals)
        if op == "eigh":                       # input manifold is symmetric
            tangents = tuple(0.5 * (t + t.T) for t in tangents)
        y, dy = _JVPS[op](primals, tangents, **kwargs)
        if isinstance(y, tuple):
            u = tuple(rng.standard_normal(np.shape(t)) for t in y)
        else:
            u = rng.standard_normal(np.shape(y))
        lhs = float(_flat(dy) @ _flat(u))
        d_in = _VJPS[op](u, *primals, **kwargs)
        rhs = sum(float(np.sum(np.asarray(g) * t))
                  for g, t in zip(d_in, tangents) if g is not None)
        worst = max(worst, 2 * abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-12))
    assert worst < 1e-9, f"{op} adjoint residual {worst:.3e}"


# ── order of accuracy of the forward rules ──────────────────────────────────
_ORDER_CASES = [
    ("det", lambda X: ops.det(X), _well_conditioned),
    ("logdet", lambda X: ops.logdet(X), _spd),
    ("inv", lambda X: ops.inv(X), _well_conditioned),
    ("trace", lambda X: ops.trace(X), _well_conditioned),
    ("vec", lambda X: ops.vec(X), _well_conditioned),
    ("matrix_power", lambda X: ops.matrix_power(X, n=3), _well_conditioned),
    ("norm_fro", lambda X: ops.norm(X, ord="fro"), _well_conditioned),
    ("norm_nuc", lambda X: ops.norm(X, ord="nuc"), _well_conditioned),
]


@pytest.mark.parametrize("name,fn,make", _ORDER_CASES,
                         ids=[c[0] for c in _ORDER_CASES])
def test_jvp_converges_at_second_order(name, fn, make) -> None:
    from tessera.autodiff import jvp

    A = make()
    result = debug.check_order_of_accuracy(
        fn, [A], lambda X, V: jvp(fn, X, V)[1], scheme="central",
    )
    assert result.passed, result.format()
    # trace/vec are exactly linear, so central differences have no truncation
    # term at all — that is a pass with no slope, not an order-0 failure.
    if name in ("trace", "vec"):
        assert result.exact, "a linear map must be reproduced exactly"
    else:
        assert not result.exact and abs(result.measured_order - 2.0) < 0.25


# ── the closed forms from the notes ─────────────────────────────────────────
def test_det_gradient_is_the_transposed_adjugate() -> None:
    A = _well_conditioned()
    g = grad(lambda X: ops.det(X))(A)
    np.testing.assert_allclose(g, np.linalg.det(A) * np.linalg.inv(A).T, rtol=1e-10)


def test_logdet_gradient_is_the_inverse_transpose() -> None:
    A = _spd()
    g = grad(lambda X: ops.logdet(X))(A)
    np.testing.assert_allclose(g, np.linalg.inv(A).T, rtol=1e-10)


def test_frobenius_norm_gradient() -> None:
    A = _well_conditioned()
    g = grad(lambda X: ops.norm(X, ord="fro"))(A)
    np.testing.assert_allclose(g, A / np.linalg.norm(A), rtol=1e-10)


def test_trace_gradient_is_the_identity() -> None:
    A = _well_conditioned()
    np.testing.assert_allclose(grad(lambda X: ops.trace(X))(A), np.eye(4))


def test_eigenvalue_sum_gradient_is_the_identity() -> None:
    """sum(eigenvalues) == trace, so its gradient must be I — even though the
    per-eigenvalue rule is a completely different code path."""
    S = _spd(3)
    g = grad(lambda X: ops.reduce(ops.eigh(X)[0], op="sum"))(S)
    np.testing.assert_allclose(g, np.eye(3), atol=1e-9)


def test_solve_gradient_matches_the_adjoint_derivation() -> None:
    """The notes' §6.3 result: db = A^-T g, dA = -db x^T. Two solves, total."""
    A, b = _well_conditioned(), _rng().standard_normal(4)
    c = _rng().standard_normal(4)
    dA, db = grad(lambda M, v: ops.reduce(ops.mul(c, ops.solve(M, v)), op="sum"),
                  argnums=(0, 1))(A, b)
    x = np.linalg.solve(A, b)
    expected_db = np.linalg.solve(A.T, c)
    np.testing.assert_allclose(db, expected_db, rtol=1e-10)
    np.testing.assert_allclose(dA, -np.outer(expected_db, x), rtol=1e-10)


def test_inv_derivative_is_the_operator_form() -> None:
    """d(A^-1)[dA] = -A^-1 dA A^-1 — two products, not an n^2 x n^2 Jacobian."""
    from tessera.autodiff import jvp

    A = _well_conditioned()
    dA = _rng().standard_normal((4, 4))
    _, tangent = jvp(lambda X: ops.inv(X), A, dA)
    Ainv = np.linalg.inv(A)
    np.testing.assert_allclose(tangent, -Ainv @ dA @ Ainv, rtol=1e-10)


# ── MC4: the Kronecker identity is a rewrite, not a materialization ─────────
def test_kronecker_vec_identity() -> None:
    """`(B kron C) vec(Y) == vec(C Y B^T)` (notes §3.3.1).

    This is the identity that makes the `Theta(n^4)` form avoidable. It holds
    only for **column-major** vec, which is why `ops.vec` stacks columns.
    """
    rng = np.random.default_rng(5)
    B, C, Y = rng.standard_normal((3, 3)), rng.standard_normal((4, 4)), rng.standard_normal((4, 3))
    lhs = ops.kron(B, C) @ ops.vec(Y)
    rhs = ops.vec(C @ Y @ B.T)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


def test_vec_is_column_major() -> None:
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(ops.vec(A), [1.0, 3.0, 2.0, 4.0])


def test_vec_roundtrips_through_its_own_adjoint() -> None:
    A = _rng().standard_normal((4, 3))
    (recovered,) = _VJPS["vec"](ops.vec(A), A)
    np.testing.assert_allclose(recovered, A)


# ── fail-closed contracts ───────────────────────────────────────────────────
def test_singular_matrix_fails_closed_rather_than_returning_inf() -> None:
    singular = np.array([[1.0, 2.0], [2.0, 4.0]])
    for fn in (lambda: ops.inv(singular),
               lambda: ops.solve(singular, np.ones(2))):
        with pytest.raises(ValueError, match="E_LINALG_CONTRACT"):
            fn()


def test_logdet_refuses_a_negative_determinant() -> None:
    A = np.diag([-1.0, 1.0])
    with pytest.raises(ValueError, match="positive determinant"):
        ops.logdet(A)


def test_non_square_input_fails_closed() -> None:
    with pytest.raises(ValueError, match="square"):
        ops.det(np.ones((2, 3)))


def test_norm_ord_is_a_semantic_key() -> None:
    """`ord` selects which function is differentiated, so it never defaults."""
    A = _well_conditioned()
    with pytest.raises(ValueError, match="semantic key"):
        ops.norm(A, ord="spectral")


def test_frobenius_norm_is_not_differentiable_at_zero() -> None:
    with pytest.raises(ValueError, match="not\\s+differentiable"):
        _VJPS["norm"](np.ones(()), np.zeros((3, 3)), ord="fro")


def test_eigh_degeneracy_uses_the_declared_policy() -> None:
    S = np.diag([2.0, 2.0, 1.0])
    with pytest.raises(deg.TesseraDegeneracyError, match="E_DEGENERATE_FACTORIZATION"):
        _VJPS["eigh"]((np.ones(3), np.ones((3, 3))), S)
