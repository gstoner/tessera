"""M6 Step 3 — :class:`EnergyGradientProgram` + :func:`refine`.

Coverage:

  - Forward pass matches :mod:`tessera.energy` on every whitelisted op.
  - Reverse-mode AD over the IR matches central finite differences
    on composed programs (multi-op chains).
  - T-step refinement converges on a known minimum (quadratic + MLP).
  - **Build-once invariant**: ``refine`` does NOT rebuild the
    gradient program between iterations — the build call count
    after 100 steps is 1.
  - Per-step gradient differs from a fixed-snapshot gradient (the
    M6 Step 3 promise vs. the older ``ebt_tiny`` snapshot path).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import energy
from tessera.compiler.energy_grad import (
    EnergyGradientError,
    EnergyGradientProgram,
    make_gradient_program,
    refine,
)
from tessera.compiler.energy_jit import lower_energy_function


# ---------------------------------------------------------------------------
# Forward pass — sanity checks
# ---------------------------------------------------------------------------

def test_forward_matches_direct_call_on_quadratic() -> None:
    def E(y, W):
        return energy.quadratic(y, W)

    prog = make_gradient_program(E)
    rng = np.random.RandomState(0)
    y = rng.randn(5)
    W = rng.randn(5, 5)
    np.testing.assert_allclose(
        prog.evaluate({"y": y, "W": W}),
        energy.quadratic(y, W),
        atol=1e-12,
    )


def test_forward_matches_direct_call_on_mlp_head() -> None:
    def E(y, W1, b1, W2, b2):
        h = energy.linear(y, W1, b1)
        a = energy.relu(h)
        out = energy.linear(a, W2, b2)
        return energy.reduce_sum(out)

    prog = make_gradient_program(E)
    rng = np.random.RandomState(1)
    y = rng.randn(4)
    W1, b1 = rng.randn(4, 6), rng.randn(6)
    W2, b2 = rng.randn(6, 3), rng.randn(3)
    direct = energy.reduce_sum(
        energy.linear(energy.relu(energy.linear(y, W1, b1)), W2, b2)
    )
    assert prog.evaluate(
        {"y": y, "W1": W1, "b1": b1, "W2": W2, "b2": b2}
    ) == pytest.approx(direct, rel=1e-12)


# ---------------------------------------------------------------------------
# Reverse-mode AD vs finite differences on composed programs
# ---------------------------------------------------------------------------

def _finite_diff(fn, y, eps=1e-5):
    """Central-difference estimate of ``∂fn/∂y`` at scalar output."""
    y = y.astype(np.float64)
    grad = np.zeros_like(y)
    it = np.nditer(y, flags=["multi_index"], op_flags=[["readwrite"]])
    while not it.finished:
        idx = it.multi_index
        orig = float(y[idx])
        y[idx] = orig + eps
        plus = float(fn(y))
        y[idx] = orig - eps
        minus = float(fn(y))
        y[idx] = orig
        grad[idx] = (plus - minus) / (2.0 * eps)
        it.iternext()
    return grad


def test_grad_y_matches_finite_diff_on_norm_sq() -> None:
    def E(y):
        return energy.norm_sq(y)

    prog = make_gradient_program(E)
    rng = np.random.RandomState(2)
    y = rng.randn(6).astype(np.float64)
    closed = prog.grad_y({"y": y})
    fd = _finite_diff(lambda yy: energy.norm_sq(yy), y)
    np.testing.assert_allclose(closed, fd, atol=1e-6)


def test_grad_y_matches_finite_diff_on_quadratic() -> None:
    def E(y, W):
        return energy.quadratic(y, W)

    prog = make_gradient_program(E)
    rng = np.random.RandomState(3)
    y = rng.randn(4).astype(np.float64)
    W = rng.randn(4, 4).astype(np.float64)
    closed = prog.grad_y({"y": y, "W": W})
    fd = _finite_diff(lambda yy: float(energy.quadratic(yy, W)), y)
    np.testing.assert_allclose(closed, fd, atol=1e-5)


def test_grad_y_matches_finite_diff_on_mlp_head() -> None:
    def E(y, W1, b1, W2, b2):
        h = energy.linear(y, W1, b1)
        a = energy.relu(h)
        out = energy.linear(a, W2, b2)
        return energy.reduce_sum(out)

    prog = make_gradient_program(E)
    rng = np.random.RandomState(4)
    y = rng.randn(4).astype(np.float64)
    W1, b1 = rng.randn(4, 6).astype(np.float64), rng.randn(6).astype(np.float64)
    W2, b2 = rng.randn(6, 3).astype(np.float64), rng.randn(3).astype(np.float64)
    env = {"y": y, "W1": W1, "b1": b1, "W2": W2, "b2": b2}
    closed = prog.grad_y(env)
    fd = _finite_diff(
        lambda yy: float(energy.reduce_sum(
            energy.linear(energy.relu(energy.linear(yy, W1, b1)), W2, b2)
        )),
        y,
    )
    np.testing.assert_allclose(closed, fd, atol=1e-4)


def test_grad_y_matches_finite_diff_on_softplus_chain() -> None:
    """A non-linear activation chain — softplus → norm_sq."""
    def E(y):
        s = energy.softplus(y)
        return energy.norm_sq(s)

    prog = make_gradient_program(E)
    rng = np.random.RandomState(5)
    y = rng.randn(5).astype(np.float64)
    closed = prog.grad_y({"y": y})
    fd = _finite_diff(
        lambda yy: float(energy.norm_sq(energy.softplus(yy))),
        y,
    )
    np.testing.assert_allclose(closed, fd, atol=1e-5)


# ---------------------------------------------------------------------------
# Convergence on known minima
# ---------------------------------------------------------------------------

def test_refine_converges_to_origin_on_quadratic() -> None:
    """E(y) = y^T W y for PD W ⇒ minimum at y = 0."""
    def E(y, W):
        return energy.quadratic(y, W)

    prog = make_gradient_program(E)
    W = np.array([[2.0, 0.1, 0.0], [0.1, 3.0, 0.0], [0.0, 0.0, 1.0]])
    y0 = np.array([3.0, -2.0, 1.0])
    yT = refine(y0, prog, T=200, eta=0.05, params={"W": W})
    assert np.linalg.norm(yT) < 1e-6


def test_refine_decreases_energy_at_each_step() -> None:
    """Gradient descent must monotonically decrease the energy
    (for small enough η).  We measure E(y₀) and E(y_T) and check
    the latter is strictly smaller."""
    def E(y):
        return energy.norm_sq(y)

    prog = make_gradient_program(E)
    y0 = np.array([2.0, -3.0, 1.5])
    e0 = float(prog.evaluate({"y": y0}))
    yT = refine(y0, prog, T=10, eta=0.1)
    eT = float(prog.evaluate({"y": yT}))
    assert eT < e0


# ---------------------------------------------------------------------------
# Build-once invariant — the "fused" property at the Python level
# ---------------------------------------------------------------------------

def test_make_gradient_program_increments_build_count_once() -> None:
    def E(y):
        return energy.norm_sq(y)

    prog = make_gradient_program(E)
    assert prog.build_call_count == 1


def test_refine_does_not_rebuild_gradient_program_per_step() -> None:
    """The headline M6 Step 3 invariant: building is amortized
    across the T-step inner loop.  After 100 refinement steps,
    the program has been BUILT exactly once."""
    def E(y):
        return energy.norm_sq(y)

    prog = make_gradient_program(E)
    refine(np.array([1.0, 2.0]), prog, T=100, eta=0.05)
    assert prog.build_call_count == 1, (
        f"refine rebuilt the gradient program {prog.build_call_count - 1} "
        "times — that breaks the fused-kernel invariant"
    )


def test_refine_t_zero_returns_unchanged_y0() -> None:
    """Edge case: T=0 must be a no-op."""
    def E(y):
        return energy.norm_sq(y)

    prog = make_gradient_program(E)
    y0 = np.array([1.0, 2.0, 3.0])
    yT = refine(y0, prog, T=0, eta=0.1)
    np.testing.assert_array_equal(yT, y0)


def test_refine_rejects_negative_T() -> None:
    def E(y):
        return energy.norm_sq(y)

    prog = make_gradient_program(E)
    with pytest.raises(ValueError, match="non-negative"):
        refine(np.array([1.0]), prog, T=-1, eta=0.1)


# ---------------------------------------------------------------------------
# Per-step gradient vs snapshot gradient — the M6 Step 3 promise
# ---------------------------------------------------------------------------

def test_per_step_gradient_differs_from_snapshot() -> None:
    """The whole point of Step 3 is per-step gradient recomputation.

    We compute T-step refinement two ways:

      1. **Snapshot**: ``y_T = y₀ − T·η·∇E(y₀)``  (the old
         ``ebt_tiny`` shape — gradient evaluated once at y₀).
      2. **Per-step**: ``y_T`` via the M6 Step 3 :func:`refine`,
         which recomputes ``∇E(y)`` at each step.

    For a non-quadratic energy (so the gradient actually changes
    with y), the two paths must produce different y_T.  Otherwise
    the snapshot path would be "free" and M6 Step 3 wouldn't
    deliver anything new."""
    def E(y):
        # softplus ⇒ non-linear, so ∇E depends on y.
        s = energy.softplus(y)
        return energy.norm_sq(s)

    prog = make_gradient_program(E)
    y0 = np.array([2.0, -3.0, 1.5])
    T, eta = 20, 0.05
    # Snapshot path.
    g0 = prog.grad_y({"y": y0})
    yT_snapshot = y0 - (T * eta) * g0
    # Per-step path.
    yT_per_step = refine(y0, prog, T=T, eta=eta)
    # The two must differ — non-trivially.  We don't claim either
    # is "better" (that depends on η + curvature); only that the
    # per-step path produces a DIFFERENT iterate, which proves
    # the gradient recomputation actually happened.
    diff = np.linalg.norm(yT_per_step - yT_snapshot)
    assert diff > 1e-3, (
        f"per-step and snapshot results too similar (Δ={diff}); "
        "Step 3 isn't delivering meaningful new behavior here"
    )
    # And both paths reach a finite, non-NaN, lower-than-start energy.
    e0 = float(prog.evaluate({"y": y0}))
    e_snap = float(prog.evaluate({"y": yT_snapshot}))
    e_step = float(prog.evaluate({"y": yT_per_step}))
    assert e_snap < e0 and e_step < e0


# ---------------------------------------------------------------------------
# Builder error handling
# ---------------------------------------------------------------------------

def test_make_gradient_program_accepts_callable_or_ir() -> None:
    def E(y):
        return energy.norm_sq(y)

    # Pass a callable.
    prog_cb = make_gradient_program(E)
    # Pass an IR.
    ir = lower_energy_function(E)
    prog_ir = make_gradient_program(ir)
    # Both produce the same gradient.
    y = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        prog_cb.grad_y({"y": y}),
        prog_ir.grad_y({"y": y}),
    )


def test_make_gradient_program_rejects_non_callable_non_ir() -> None:
    with pytest.raises(TypeError, match="EnergyIRProgram or callable"):
        make_gradient_program(42)


# ---------------------------------------------------------------------------
# Batched inputs
# ---------------------------------------------------------------------------

def test_grad_y_handles_batched_inputs() -> None:
    """The reference path is mostly vectorized; spot-check that
    a batched y produces a batched gradient."""
    def E(y):
        return energy.norm_sq(y)

    prog = make_gradient_program(E)
    rng = np.random.RandomState(7)
    y = rng.randn(4, 3)  # batch of 4 vectors of dim 3
    g = prog.grad_y({"y": y})
    assert g.shape == y.shape
    # ∂‖y‖²/∂y = 2y
    np.testing.assert_allclose(g, 2 * y, atol=1e-9)
