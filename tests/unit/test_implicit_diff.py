"""T3 — implicit differentiation: CG, IHVP, custom_root, adjoint state.

Every piece is checked against a closed-form or finite-difference ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.autodiff.implicit import (
    TesseraImplicitDiffError,
    ResidualLinearization,
    cg_solve,
    gmres_solve,
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


def test_restarted_gmres_solves_nonsymmetric_system_and_reports_true_work():
    A = np.array(
        [
            [4.0, 3.0, 0.0, 0.0],
            [-1.0, 4.0, 2.0, 0.0],
            [0.0, -2.0, 4.0, 1.0],
            [0.0, 0.0, -1.0, 3.0],
        ]
    )
    b = np.array([1.0, -2.0, 3.0, 0.5])

    x, info = gmres_solve(lambda v: A @ v, b, restart=2, maxiter=40, return_info=True)

    np.testing.assert_allclose(x, np.linalg.solve(A, b), atol=1e-7)
    assert info.converged
    assert info.iterations > 0
    assert info.restarts > 0
    assert info.matvecs >= info.iterations + 1
    assert info.residual_norm == pytest.approx(np.linalg.norm(b - A @ x))


def test_gmres_fails_closed_on_arnoldi_breakdown():
    with pytest.raises(TesseraImplicitDiffError, match="breakdown"):
        gmres_solve(lambda v: np.zeros_like(v), np.ones(3), maxiter=3)


# ── the convergence criterion is relative, so a solve is scale-invariant ─────


@pytest.mark.parametrize("solve", [cg_solve, gmres_solve])
@pytest.mark.parametrize("magnitude", [1e-9, 1e-12])
def test_solvers_do_not_return_zero_for_a_small_right_hand_side(solve, magnitude):
    """A cotangent smaller than `tol` is not a solved system. The absolute
    residual floor `tol * max(1, ||b||)` made x = 0 pass the initial check
    for every ||b|| <= tol and reported it converged — a 100% relative error
    on the routine late-training case (2026-08-29 review, P2)."""
    rng = np.random.default_rng(5)
    M = rng.standard_normal((5, 5))
    A = M @ M.T + 5 * np.eye(5)  # SPD, so both solvers apply
    b = rng.standard_normal(5)
    b = b / np.linalg.norm(b) * magnitude

    x = np.asarray(solve(lambda v: A @ v, b))
    truth = np.linalg.solve(A, b)
    assert np.linalg.norm(x - truth) <= 1e-6 * np.linalg.norm(truth)


@pytest.mark.parametrize("solve", [cg_solve, gmres_solve])
def test_solves_are_scale_invariant_in_the_right_hand_side(solve):
    """A_inv is linear, so scaling b must scale the answer exactly."""
    rng = np.random.default_rng(6)
    M = rng.standard_normal((4, 4))
    A = M @ M.T + 4 * np.eye(4)
    b = rng.standard_normal(4)

    unit = np.asarray(solve(lambda v: A @ v, b))
    tiny = np.asarray(solve(lambda v: A @ v, 1e-9 * b))
    np.testing.assert_allclose(tiny, 1e-9 * unit, rtol=1e-6)


def test_zero_right_hand_side_still_returns_the_zero_solution():
    assert np.all(cg_solve(lambda v: 2.0 * v, np.zeros(3)) == 0.0)
    assert np.all(gmres_solve(lambda v: 2.0 * v, np.zeros(3)) == 0.0)


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


@pytest.mark.parametrize("magnitude", [1.0, 1e3, 1e6])
def test_root_jvp_is_linear_in_an_unnormalized_tangent(magnitude):
    """`_partial_jacobian_matvecs` finite-differences along `eps * v` with a
    fixed step, so its truncation error grows as (eps*||v||)^2 — it stopped
    being a linear map for a large tangent and made GMRES break down on a
    solvable system. Scaling the step by ||v||_inf restores exactness
    (2026-08-29 review, P2)."""
    theta = np.array([4.0, 9.0])
    xstar = np.sqrt(theta)

    def F(x, th):
        return x * x + np.sin(x) - th - np.sin(np.sqrt(th))

    v = np.array([1.0, -0.5]) * magnitude
    t = np.asarray(root_jvp(F, xstar, (theta,), (v,)))
    # dF/dth = -(1 + cos(sqrt(th)) / (2 sqrt(th))), dF/dx = 2x + cos(x).
    expected = v * (1.0 + np.cos(xstar) / (2 * xstar)) / (2 * xstar + np.cos(xstar))
    np.testing.assert_allclose(t, expected, rtol=1e-5)


def test_general_root_products_expose_matrix_free_solver_work() -> None:
    A = np.array([[3.0, 2.0], [-1.0, 4.0]])
    theta = np.array([2.0, -3.0])
    solution = np.linalg.solve(A, theta)

    def residual(x, parameter):
        return A @ x - parameter

    vjp, vjp_info = root_vjp(
        residual,
        solution,
        (theta,),
        np.array([0.5, -1.0]),
        return_solve_info=True,
    )
    jvp, jvp_info = root_jvp(
        residual,
        solution,
        (theta,),
        (np.array([1.0, 2.0]),),
        return_solve_info=True,
    )

    np.testing.assert_allclose(vjp, np.linalg.solve(A.T, [0.5, -1.0]), atol=1e-6)
    np.testing.assert_allclose(jvp, np.linalg.solve(A, [1.0, 2.0]), atol=1e-6)
    for info in (vjp_info, jvp_info):
        assert info.algorithm == "gmres"
        assert info.converged
        assert info.matvecs >= info.iterations + 1
        assert info.residual_norm <= 1e-7


def test_general_root_products_consume_exact_compiler_linearization() -> None:
    A = np.array([[4.0, -1.0], [2.0, 3.0]])
    B = np.array([[1.0, 2.0], [-2.0, 1.0]])
    theta = np.array([0.5, -1.5])
    solution = np.linalg.solve(A, -B @ theta)

    def must_not_run(*_args):
        raise AssertionError("finite-difference residual oracle was re-entered")

    products = ResidualLinearization(
        residual_shape=(2,),
        solution_jvp=lambda direction: A @ direction,
        solution_vjp=lambda cotangent: A.T @ cotangent,
        parameter_jvp=lambda index, direction: B @ direction,
        parameter_vjp=lambda index, cotangent: B.T @ cotangent,
        provenance="paired_graph_ir:sha256:test",
    )
    direction = np.array([1.0, -0.25])
    cotangent = np.array([0.75, 2.0])
    tangent, jvp_info = root_jvp(
        must_not_run, solution, (theta,), (direction,),
        linearization=products, return_solve_info=True,
    )
    gradient, vjp_info = root_vjp(
        must_not_run, solution, (theta,), cotangent,
        linearization=products, return_solve_info=True,
    )

    np.testing.assert_allclose(tangent, -np.linalg.solve(A, B @ direction), atol=1e-8)
    np.testing.assert_allclose(
        gradient, -B.T @ np.linalg.solve(A.T, cotangent), atol=1e-8
    )
    assert jvp_info.product_source == "paired_graph_ir:sha256:test"
    assert vjp_info.product_source == "paired_graph_ir:sha256:test"


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

    # The decorated solver must be a tape operation, not merely expose a
    # manual helper: it composes with the public reverse-mode transform.
    from tessera import ops
    from tessera.autodiff import grad

    transform_grad = grad(lambda th: ops.reduce(solve(th)))(theta)
    np.testing.assert_allclose(transform_grad, np.linalg.solve(A.T, np.ones_like(theta)), atol=1e-6)


def test_custom_root_transform_selects_each_declared_parameter():
    # F(x, a, b) = x - a - 2b: the decorated root has two tape inputs, so its
    # VJP must return None/grad in the same arity and order as the declaration.
    @custom_root(lambda x, a, b: x - a - 2.0 * b, argnums=(0, 1))
    def solve(a, b):
        return a + 2.0 * b

    from tessera import ops
    from tessera.autodiff import grad

    a = np.array([1.0, 3.0])
    b = np.array([2.0, -1.0])
    da, db = grad(lambda aa, bb: ops.reduce(solve(aa, bb)), argnums=(0, 1))(a, b)
    np.testing.assert_allclose(da, np.ones_like(a), atol=1e-6)
    np.testing.assert_allclose(db, 2.0 * np.ones_like(b), atol=1e-6)


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

    g, info = adjoint_state_grad(L, c, s_star, w, return_solve_info=True)
    np.testing.assert_allclose(g, 4.0 * w**3, atol=1e-4)
    assert info.algorithm == "gmres" and info.converged


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


# ── the FD adjoint is built once, not once per solver iteration ─────────────
def _counted_linear_root(n: int):
    """A linear residual plus a counter for the residual evaluations."""
    rng = np.random.default_rng(0)
    A = np.eye(n) * 3.0 + rng.standard_normal((n, n)) * 0.2
    b = np.random.default_rng(1).standard_normal(n)
    calls = [0]

    def F(x, theta):
        calls[0] += 1
        return A @ x - theta * b

    return F, np.linalg.solve(A, 1.7 * b), [np.array(1.7)], calls


def test_fd_adjoint_is_memoized_across_gmres_iterations():
    """Review finding: `rmatvec` rebuilt all n finite-difference Jacobian
    columns on every call, and GMRES applies the adjoint 2*iters + 1 times.
    The 'matrix-free' default therefore cost O(n * iters) residual evaluations
    against O(n) for the dense oracle it ships alongside — strictly dominated
    by the path it was meant to improve on. The columns do not depend on the
    cotangent, so building them once makes the two paths cost the same.
    """
    for n in (12, 40):
        F, xstar, params, calls = _counted_linear_root(n)
        u = np.ones(n)

        calls[0] = 0
        gmres_grad = root_vjp(F, xstar, params, u, linear_solver="gmres")
        gmres_evals = calls[0]

        calls[0] = 0
        dense_grad = root_vjp(F, xstar, params, u, linear_solver="dense")
        dense_evals = calls[0]

        # Same F-eval budget as the dense oracle: one basis sweep for A, one
        # for B. Before the fix this was (2*iters + 1) sweeps for A alone.
        assert gmres_evals == dense_evals, (
            f"n={n}: gmres path used {gmres_evals} residual evaluations vs "
            f"{dense_evals} for dense — the FD columns are being rebuilt")
        assert np.allclose(gmres_grad, dense_grad, rtol=1e-9), (
            gmres_grad, dense_grad)


def test_fd_adjoint_memo_agrees_with_the_streaming_sweep():
    """Memoizing must not change the answer: the cached columns are exactly
    the ones the per-call sweep computed (the FD step for a unit basis vector
    is a constant `eps`, so the columns are a well-defined matrix)."""
    from tessera.autodiff import implicit as impl

    n = 10
    F, xstar, params, _ = _counted_linear_root(n)
    u = np.linspace(-1.0, 1.0, n)

    memoized = root_vjp(F, xstar, params, u, linear_solver="gmres")

    budget = impl._FD_JACOBIAN_ELEMENT_BUDGET
    impl._FD_JACOBIAN_ELEMENT_BUDGET = 0  # force the streaming fallback
    try:
        with pytest.warns(RuntimeWarning, match="not memoized"):
            streamed = root_vjp(F, xstar, params, u, linear_solver="gmres")
    finally:
        impl._FD_JACOBIAN_ELEMENT_BUDGET = budget

    assert np.allclose(memoized, streamed, rtol=1e-10, atol=1e-12), (
        memoized, streamed)
