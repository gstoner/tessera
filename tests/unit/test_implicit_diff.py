"""T3 — implicit differentiation: CG, IHVP, custom_root, adjoint state.

Every piece is checked against a closed-form or finite-difference ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.autodiff.implicit import (
    TesseraImplicitDiffError,
    cg_solve,
    ihvp,
    root_vjp,
    root_jvp,
    custom_root,
    adjoint_state_grad,
)


# ── conjugate gradient ───────────────────────────────────────────────────────

def test_cg_solves_spd_system():
    rng = np.random.default_rng(0)
    n = 8
    M = rng.standard_normal((n, n))
    A = M @ M.T + n * np.eye(n)  # SPD
    b = rng.standard_normal(n)
    x = cg_solve(lambda v: A @ v, b)
    np.testing.assert_allclose(x, np.linalg.solve(A, b), atol=1e-6)


def test_cg_raises_on_nonconvergence():
    # A tiny iteration budget on a poorly-scaled system -> honest failure.
    A = np.diag([1.0, 1e8])
    with pytest.raises(TesseraImplicitDiffError):
        cg_solve(lambda v: A @ v, np.array([1.0, 1.0]), tol=1e-14, maxiter=1)


# ── inverse-Hessian vector product ───────────────────────────────────────────

def test_ihvp_matches_dense_inverse_for_quadratic():
    from tessera import ops
    rng = np.random.default_rng(1)
    n = 5
    d = rng.uniform(1.0, 4.0, size=n)  # positive diagonal → SPD Hessian 2·diag(d)

    def f(x):
        # scalar Σ dᵢ xᵢ² through ops so grad/hvp can tape it. Hessian = 2·diag(d).
        return ops.reduce(ops.mul(ops.mul(x, x), d))

    x = rng.standard_normal(n)
    u = rng.standard_normal(n)
    got = ihvp(f, x, u, eps=1e-3)
    # Hessian = 2·diag(d), so ihvp = u / (2 d).
    np.testing.assert_allclose(got, u / (2.0 * d), atol=1e-3)


# ── custom_root: scalar sqrt via x² - θ = 0 ─────────────────────────────────

def test_root_vjp_scalar_sqrt():
    # x*(θ) = √θ, so ∂x*/∂θ = 1/(2√θ). VJP with u=1 returns exactly that.
    theta = np.array([4.0, 9.0, 16.0])
    xstar = np.sqrt(theta)

    def F(x, th):
        return x * x - th  # zero at x = √θ

    g = root_vjp(F, xstar, (theta,), np.ones_like(xstar))
    np.testing.assert_allclose(g, 1.0 / (2.0 * xstar), atol=1e-5)


def test_root_jvp_matches_vjp_duality():
    theta = np.array([4.0, 9.0])
    xstar = np.sqrt(theta)

    def F(x, th):
        return x * x - th

    v = np.array([1.0, 1.0])
    t = root_jvp(F, xstar, (theta,), (v,))
    # dx* = (∂x*/∂θ) v = v / (2√θ)
    np.testing.assert_allclose(t, v / (2 * xstar), atol=1e-5)


def test_custom_root_decorator_linear_system():
    # x*(θ): A x = θ  ⇒  F(x, θ) = A x - θ,  ∂x*/∂θ = A⁻¹.
    A = np.array([[3.0, 1.0], [0.0, 2.0]])

    @custom_root(lambda x, th: A @ x - th)
    def solve(th):
        return np.linalg.solve(A, th)

    theta = np.array([1.0, 4.0])
    xs = solve(theta)
    # VJP: ∂L/∂θ = (A⁻¹)ᵀ u  (since ∂x*/∂θ = A⁻¹, so -Bᵀr with B=-I gives A⁻ᵀu)
    u = np.array([1.0, 1.0])
    g = solve.vjp(xs, (theta,), u)
    np.testing.assert_allclose(g, np.linalg.solve(A.T, u), atol=1e-6)


def test_root_vjp_reports_singular_jacobian():
    # F(x,θ) = 0·x - θ has singular ∂_xF -> IFT does not apply; must raise.
    def F(x, th):
        return np.zeros_like(x) - th

    with pytest.raises(TesseraImplicitDiffError):
        root_vjp(F, np.array([1.0]), (np.array([0.0]),), np.array([1.0]))


# ── adjoint-state method ─────────────────────────────────────────────────────

def test_adjoint_state_matches_finite_difference():
    # Constraint c(s, w) = s - w²  ⇒  s*(w) = w².  Objective L = sum(s²).
    # Reduced objective h(w) = sum(w⁴), so ∇h = 4 w³.
    w = np.array([1.0, 2.0, 3.0])
    s_star = w * w

    def L(s, ww):
        return float(np.sum(s * s))

    def c(s, ww):
        return s - ww * ww

    g = adjoint_state_grad(L, c, s_star, w)
    np.testing.assert_allclose(g, 4.0 * w**3, atol=1e-4)


def test_adjoint_state_linear_constraint():
    # c(s,w) = M s - w  ⇒  s*(w) = M⁻¹ w. L = 0.5 |s|². h(w)=0.5 wᵀ M⁻ᵀM⁻¹ w.
    M = np.array([[2.0, 0.0], [1.0, 3.0]])
    w = np.array([1.0, -1.0])
    s_star = np.linalg.solve(M, w)

    def L(s, ww):
        return float(0.5 * np.sum(s * s))

    def c(s, ww):
        return M @ s - ww

    g = adjoint_state_grad(L, c, s_star, w)
    # ∇h(w) = M⁻ᵀ M⁻¹ w
    expected = np.linalg.solve(M.T, np.linalg.solve(M, w))
    np.testing.assert_allclose(g, expected, atol=1e-5)
