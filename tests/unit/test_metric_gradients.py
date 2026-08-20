"""The gradient is `metric^-1 . differential` (MC3).

`docs/audit/compiler/MATRIX_CALCULUS_REVIEW.md` MC3, from Bright/Edelman/Johnson
§5.1 and §13: the differential `df = f'(x)[dx]` is a linear *form* and is
invariant, while a gradient is whatever you take an inner product with to
recover it. Change the inner product and you change the vector — which is what
preconditioning, natural gradient and Muon all are.

Before this, `manifold` was a declared semantic key with no consumer, and three
places in-tree had independently paid for the same missing abstraction
(`ebm/geo_sampling.py`'s hand-rolled sphere projection, `ga/manifold.py`'s
quadrature-only "manifolds", `optim.muon`'s unremarked spectral geometry). The
consumer is `grad(..., metric=)`.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import ops
from tessera.autodiff import grad
from tessera.metric import (
    Euclidean,
    Metric,
    Orthogonal,
    Sphere,
    Weighted,
    riemannian_gradient,
)

A = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]])


def quadratic(x):
    """f(x) = x^T A x, with x as a column so matmul stays 2-D."""
    return ops.reduce(ops.mul(x, ops.matmul(A, x)), op="sum")


def _unit_column() -> np.ndarray:
    return np.array([[1.0], [0.0], [0.0]])


# ── every instance really is a Metric ───────────────────────────────────────
@pytest.mark.parametrize("metric", [Euclidean(), Weighted([1.0, 2.0, 3.0]),
                                    Sphere(), Orthogonal()])
def test_instances_satisfy_the_protocol(metric) -> None:
    assert isinstance(metric, Metric)


def test_non_metric_is_rejected() -> None:
    with pytest.raises(TypeError, match="E_METRIC_CONTRACT"):
        riemannian_gradient(object(), np.ones(3), np.ones(3))


# ── the Euclidean default is exactly today's behaviour ──────────────────────
def test_euclidean_metric_changes_nothing() -> None:
    x = _unit_column()
    np.testing.assert_allclose(grad(quadratic)(x),
                               grad(quadratic, metric=Euclidean())(x))


def test_euclidean_gradient_is_the_textbook_result() -> None:
    """grad(x^T A x) = (A + A^T) x (notes §2.3)."""
    x = _unit_column()
    np.testing.assert_allclose(grad(quadratic)(x), (A + A.T) @ x)


# ── weighted inner product: grad_W f = W^-1 grad f (§5.1) ───────────────────
def test_weighted_metric_raises_the_index() -> None:
    x = _unit_column()
    w = np.array([1.0, 4.0, 9.0])
    g_e = grad(quadratic)(x)
    g_w = grad(quadratic, metric=Weighted(w))(x)
    np.testing.assert_allclose(g_w, g_e / w.reshape(3, 1))


def test_weighted_matrix_metric_solves_rather_than_inverting() -> None:
    rng = np.random.default_rng(3)
    M = rng.standard_normal((3, 3))
    W = M @ M.T + 3.0 * np.eye(3)
    g = rng.standard_normal(3)
    np.testing.assert_allclose(Weighted(W).sharp(g), np.linalg.solve(W, g))


def test_the_differential_is_invariant_across_metrics() -> None:
    """The whole point: different gradient vectors, one differential.

    `<grad_W f, dx>_W` must equal `<grad_E f, dx>_E` — the number `df` does not
    depend on which inner product you chose to represent it in.
    """
    x = _unit_column()
    w = np.array([1.0, 4.0, 9.0])
    dx = np.array([[0.0], [1.0], [-1.0]])
    g_e = grad(quadratic)(x)
    g_w = grad(quadratic, metric=Weighted(w))(x)
    df_euclid = float(np.sum(g_e * dx))
    df_weighted = Weighted(w).inner(g_w, dx)
    np.testing.assert_allclose(df_weighted, df_euclid, rtol=1e-12)


@pytest.mark.parametrize("bad", [[1.0, -2.0], [0.0, 1.0]])
def test_indefinite_weights_are_rejected(bad) -> None:
    with pytest.raises(ValueError, match="positive"):
        Weighted(bad)


def test_non_symmetric_metric_matrix_is_rejected() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        Weighted(np.array([[1.0, 2.0], [0.0, 1.0]]))


def test_indefinite_metric_matrix_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        Weighted(np.diag([1.0, -1.0]))


# ── the sphere: (I - x x^T) g (§13.1.2) ─────────────────────────────────────
def test_sphere_gradient_is_tangent() -> None:
    x = _unit_column()
    g = grad(quadratic, metric=Sphere())(x)
    assert abs(float(np.sum(x * g))) < 1e-12, "gradient must be perpendicular to x"


def test_sphere_gradient_is_the_projection_of_the_ambient_one() -> None:
    x = _unit_column()
    g_e = grad(quadratic)(x)
    g_s = grad(quadratic, metric=Sphere())(x)
    np.testing.assert_allclose(g_s, g_e - x * float(np.sum(x * g_e)))


def test_sphere_retraction_returns_to_the_manifold() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal(4)
    x = x / np.linalg.norm(x)
    v = Sphere().project_tangent(x, rng.standard_normal(4))
    moved = Sphere().retract(x, 0.3 * v)
    np.testing.assert_allclose(np.linalg.norm(moved), 1.0, rtol=1e-12)


def test_sphere_retraction_at_zero_is_the_identity() -> None:
    x = np.array([0.6, 0.8])
    np.testing.assert_allclose(Sphere().retract(x, np.zeros(2)), x)


# ── the orthogonal group: Q^T dQ is antisymmetric (§13.2) ───────────────────
def _orthogonal(n: int = 3) -> np.ndarray:
    Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((n, n)))
    return Q


def test_orthogonal_gradient_is_antisymmetric_in_the_lie_algebra() -> None:
    Q = _orthogonal()
    g = grad(lambda M: ops.trace(ops.matmul(A, M)), metric=Orthogonal())(Q)
    M = Q.T @ g
    np.testing.assert_allclose(M, -M.T, atol=1e-12)


def test_orthogonal_tangent_space_has_the_right_dimension() -> None:
    """n(n-1)/2 free parameters, not n^2 — the count §13.2 derives."""
    n = 4
    Q = _orthogonal(n)
    rng = np.random.default_rng(1)
    basis = [Orthogonal().project_tangent(Q, rng.standard_normal((n, n)))
             for _ in range(n * n)]
    rank = np.linalg.matrix_rank(np.stack([b.ravel() for b in basis]))
    assert rank == n * (n - 1) // 2, f"tangent dimension {rank}, expected {n*(n-1)//2}"


def test_orthogonal_retraction_returns_to_the_group() -> None:
    Q = _orthogonal()
    V = Orthogonal().project_tangent(Q, np.random.default_rng(2).standard_normal((3, 3)))
    moved = Orthogonal().retract(Q, 0.2 * V)
    np.testing.assert_allclose(moved.T @ moved, np.eye(3), atol=1e-12)


def test_orthogonal_retraction_at_zero_is_the_identity() -> None:
    Q = _orthogonal()
    np.testing.assert_allclose(Orthogonal().retract(Q, np.zeros((3, 3))), Q, atol=1e-12)


# ── the sphere gradient of x^T A x /2, worked in the notes (§13.1.2) ────────
def test_sphere_gradient_of_the_rayleigh_form() -> None:
    """The notes' own worked result: (I - x x^T) A x for symmetric A."""
    S = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    x = np.array([[0.0], [1.0], [0.0]])

    def half_rayleigh(v):
        return ops.mul(ops.reduce(ops.mul(v, ops.matmul(S, v)), op="sum"),
                       scalar=0.5)

    g = grad(half_rayleigh, metric=Sphere())(x)
    expected = (np.eye(3) - x @ x.T) @ (S @ x)
    np.testing.assert_allclose(g, expected, atol=1e-10)
