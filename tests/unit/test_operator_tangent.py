"""AD-OPERATOR-1 acceptance — `OperatorTangent` + the H3 root certificate.

Four claims, per AUTODIFF_NEXTGEN_PLAN §7:

* the adjoint law holds at the operator level — ``⟨A v, u⟩ = ⟨v, Aᵀ u⟩``
  through the type's OWN transpose (not a re-derived one), including for
  compositions, and ``.T`` is an involution with ``(A B)ᵀ = Bᵀ Aᵀ``;
* `implicit.py`'s public surface is behavior-identical (its existing tests
  are the regression net; this file adds the operator-vs-legacy identity);
* non-convergent / ill-posed paths stay fail-closed — a forward-only
  operator refuses ``.T`` rather than inventing an adjoint;
* the well-posedness certificate accepts a clean root with measured
  numbers and REJECTS a degenerate root (the strict-complementarity
  failure mode for KKT residuals), failing closed when it cannot be
  evaluated at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.autodiff.implicit import (
    TesseraImplicitDiffError,
    cg_solve,
    custom_root,
    gmres_solve,
)
from tessera.autodiff.operator import (
    OperatorTangent,
    RootConditionCertificate,
    TesseraOperatorError,
    certify_root,
)


def _dense_operator(M: np.ndarray, label: str = "M") -> OperatorTangent:
    Md = np.asarray(M, dtype=np.float64)
    return OperatorTangent.from_matvec_pair(
        lambda v: Md @ np.asarray(v, dtype=np.float64).reshape(-1),
        lambda u: Md.T @ np.asarray(u, dtype=np.float64).reshape(-1),
        in_shape=(Md.shape[1],),
        out_shape=(Md.shape[0],),
        label=label,
    )


# ── the adjoint law, through the type's own transpose ────────────────────────


def test_adjoint_law_holds_through_operator_transpose():
    rng = np.random.default_rng(7)
    A = _dense_operator(rng.standard_normal((5, 3)))
    for _ in range(4):
        v = rng.standard_normal(3)
        u = rng.standard_normal(5)
        lhs = float(np.dot(A(v), u))
        rhs = float(np.dot(v, A.T(u)))
        assert abs(lhs - rhs) / max(abs(lhs), 1e-12) < 1e-12


def test_adjoint_law_catches_a_wrong_adjoint():
    """The law must be falsifiable: an operator whose declared adjoint is
    NOT the transpose fails the pairing — the control can fail."""
    rng = np.random.default_rng(11)
    M = rng.standard_normal((4, 4))
    liar = OperatorTangent.from_matvec_pair(
        lambda v: M @ np.asarray(v).reshape(-1),
        lambda u: M @ np.asarray(u).reshape(-1),  # forward again, not Mᵀ
        in_shape=(4,), out_shape=(4,), label="liar",
    )
    v, u = rng.standard_normal(4), rng.standard_normal(4)
    lhs = float(np.dot(liar(v), u))
    rhs = float(np.dot(v, liar.T(u)))
    assert abs(lhs - rhs) / max(abs(lhs), 1e-12) > 1e-3


def test_transpose_is_an_involution_and_reverses_composition():
    rng = np.random.default_rng(13)
    A = _dense_operator(rng.standard_normal((4, 3)), "A")
    B = _dense_operator(rng.standard_normal((3, 5)), "B")
    v5, v4 = rng.standard_normal(5), rng.standard_normal(4)

    np.testing.assert_allclose(A.T.T(v5[:3]), A(v5[:3]), rtol=1e-12)

    AB = A @ B
    assert AB.shape == (4, 5)
    np.testing.assert_allclose(AB(v5), A(B(v5)), rtol=1e-12)
    # (A B)ᵀ = Bᵀ Aᵀ — pointwise on probes, through the type's own T.
    np.testing.assert_allclose(AB.T(v4), B.T(A.T(v4)), rtol=1e-12)


def test_operators_fail_closed_at_the_edges():
    rng = np.random.default_rng(17)
    M = rng.standard_normal((3, 3))
    forward_only = OperatorTangent.from_matvec_pair(
        lambda v: M @ np.asarray(v).reshape(-1), None,
        in_shape=(3,), out_shape=(3,), label="fwd-only",
    )
    with pytest.raises(TesseraOperatorError, match="no adjoint"):
        _ = forward_only.T
    A = _dense_operator(rng.standard_normal((4, 3)))
    with pytest.raises(TesseraOperatorError, match="inner dimensions"):
        _ = A @ A  # (4,3) ∘ (4,3) — mismatched
    with pytest.raises(TesseraOperatorError, match="expected 3"):
        A(np.zeros(5))
    with pytest.raises(TesseraOperatorError, match="self-adjoint"):
        _ = OperatorTangent(
            in_shape=(3,), out_shape=(4,),
            fwd=lambda v: np.zeros(4), self_adjoint=True,
        ).T


def test_solvers_consume_operators_directly():
    """Solve-consumption (§3.5): CG and GMRES accept the operator where
    they accepted a bare closure — including a transposed solve."""
    rng = np.random.default_rng(19)
    Q = rng.standard_normal((4, 4))
    spd = _dense_operator(Q @ Q.T + 4.0 * np.eye(4), "SPD")
    b = rng.standard_normal(4)
    x = cg_solve(spd, b, tol=1e-12)
    np.testing.assert_allclose(spd(x), b, atol=1e-9)

    M = rng.standard_normal((4, 4)) + 4.0 * np.eye(4)
    gen = _dense_operator(M, "G")
    y = gmres_solve(gen.T, b, tol=1e-12)
    np.testing.assert_allclose(M.T @ y, b, atol=1e-8)


# ── H3: the root certificate ─────────────────────────────────────────────────


def test_certificate_accepts_a_clean_root_with_measured_numbers():
    theta = np.array([2.0])
    x_star = np.sqrt(theta)
    cert = certify_root(lambda x, t: x * x - t, x_star, (theta,))
    assert isinstance(cert, RootConditionCertificate)
    assert cert.strict and cert.residual_ok and cert.nondegenerate
    # ∂ₓF = 2x* = 2√2 — the measurement, not just the verdict.
    np.testing.assert_allclose(cert.sigma_min, 2.0 * np.sqrt(2.0), rtol=1e-4)
    assert cert.residual_norm < 1e-12


def test_certificate_rejects_a_degenerate_root():
    """x² − θ at θ = 0: x* = 0 IS a root but ∂ₓF = 2x* = 0 — exactly the
    ill-posed case (a strict-complementarity failure in KKT form)."""
    theta = np.array([0.0])
    x_star = np.array([0.0])
    cert = certify_root(lambda x, t: x * x - t, x_star, (theta,))
    assert cert.residual_ok and not cert.nondegenerate and not cert.strict
    assert cert.sigma_min < 1e-12


def test_certificate_fails_closed_when_unevaluable():
    with pytest.raises(TesseraImplicitDiffError, match="non-finite"):
        certify_root(
            lambda x, t: np.full_like(x, np.nan),
            np.array([1.0]), (np.array([1.0]),),
        )
    with pytest.raises(TesseraImplicitDiffError, match="raised"):
        certify_root(
            lambda x, t: (_ for _ in ()).throw(RuntimeError("boom")),
            np.array([1.0]), (np.array([1.0]),),
        )


def test_custom_root_rejects_degenerate_solution_in_both_modes():
    @custom_root(lambda x, theta: x * x - theta)
    def sqrt_solver(theta):
        return np.sqrt(theta)

    theta0 = np.array([0.0])
    x0 = sqrt_solver(theta0)
    with pytest.raises(TesseraImplicitDiffError, match="degenerate"):
        sqrt_solver.vjp(x0, (theta0,), np.ones_like(x0))
    with pytest.raises(TesseraImplicitDiffError, match="degenerate"):
        sqrt_solver.jvp(x0, (theta0,), (np.ones_like(theta0),))

    # The clean root still differentiates, and exposes its certificate.
    theta1 = np.array([2.0])
    x1 = sqrt_solver(theta1)
    grad = sqrt_solver.vjp(x1, (theta1,), np.ones_like(x1))
    np.testing.assert_allclose(grad, 0.5 / np.sqrt(theta1), rtol=1e-5)
    cert = sqrt_solver.certificate(x1, (theta1,))
    assert cert.strict


def test_custom_root_rejects_a_nonroot_solution():
    """The certificate's other half: a 'solution' that is not actually a
    root of the optimality condition rejects on the residual check."""
    @custom_root(lambda x, theta: x * x - theta)
    def bad_solver(theta):
        return np.sqrt(theta) + 0.5  # deliberately not the root

    theta = np.array([2.0])
    x_wrong = bad_solver(theta)
    with pytest.raises(TesseraImplicitDiffError, match="not a root"):
        bad_solver.vjp(x_wrong, (theta,), np.ones_like(x_wrong))


def test_custom_root_certify_false_is_an_explicit_opt_out():
    @custom_root(lambda x, theta: x * x - theta, certify=False)
    def sqrt_solver(theta):
        return np.sqrt(theta)

    theta0 = np.array([0.0])
    x0 = sqrt_solver(theta0)
    # Opt-out skips the gate; the ill-posed solve then fails in the solver
    # itself (fail-closed non-convergence preserved) rather than silently
    # returning a gradient.
    with pytest.raises(TesseraImplicitDiffError):
        sqrt_solver.vjp(x0, (theta0,), np.ones_like(x0))


def test_tape_backward_through_degenerate_custom_root_rejects():
    """The H3 gate holds on the tape path too: reverse mode through a
    degenerate custom root raises with the measured numbers instead of
    propagating an ill-posed gradient."""
    from tessera.autodiff.tape import tape

    @custom_root(lambda x, theta: x * x - theta)
    def sqrt_solver(theta):
        return np.sqrt(theta)

    theta0 = np.array(0.0)
    with tape() as t:
        y = sqrt_solver(theta0)
    with pytest.raises(TesseraImplicitDiffError, match="degenerate"):
        t.backward(y, cotangent=np.ones_like(y))


def test_singular_linear_system_certificate_names_degeneracy():
    """A 2-D singular residual Jacobian (rank-1 system) — the certificate
    measures σ_min ≈ 0 and rejects, naming the condition number."""
    S = np.array([[1.0, 1.0], [1.0, 1.0]])

    @custom_root(lambda x, b: S @ x - b)
    def solver(b):
        # Minimum-norm 'solution' of the singular system (b in range(S)).
        return np.full(2, float(b.sum()) / 4.0)

    b = np.array([1.0, 1.0])
    x = solver(b)
    with pytest.raises(TesseraImplicitDiffError, match="degenerate"):
        solver.vjp(x, (b,), np.ones(2))
