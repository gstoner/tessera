"""T2 — Fenchel-Young loss template.

Checks the two headline properties: (1) the softmax FY loss equals the existing
cross-entropy for one-hot targets, and the sparsemax loss is its l2 sibling; and
(2) the FY gradient is *exactly* ŷ(θ) - y, verified against finite differences.
Plus the Bregman-divergence property (≥ 0, zero at a perfect prediction).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.losses import (
    fenchel_young_loss,
    fy_loss_and_grad,
    sparsemax_loss,
    softmax_fy_loss,
    cross_entropy_loss,
)
from tessera.relaxation import _sparsemax_forward


def _softmax(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    e = np.exp(z - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def _onehot(idx, n):
    e = np.zeros(n)
    e[idx] = 1.0
    return e


# ── equivalence with existing losses ────────────────────────────────────────

def test_softmax_fy_equals_cross_entropy_for_onehot():
    rng = np.random.default_rng(0)
    theta = rng.standard_normal((4, 5))
    y = np.stack([_onehot(i, 5) for i in rng.integers(0, 5, size=4)])
    fy = softmax_fy_loss(theta, y, reduction="mean")
    ce = cross_entropy_loss(theta, y, reduction="mean")  # probability targets
    assert fy == pytest.approx(float(ce), abs=1e-9)


def test_sparsemax_loss_is_l2_fy():
    theta = np.array([2.0, 1.0, 0.0, -1.0])
    y = _onehot(0, 4)
    a = sparsemax_loss(theta, y, reduction="none")
    b = fenchel_young_loss(theta, y, omega="l2", reduction="none")
    np.testing.assert_allclose(np.asarray(a), np.asarray(b))


# ── exact gradient ŷ - y ─────────────────────────────────────────────────────

@pytest.mark.parametrize("omega,pred", [("entropy", _softmax), ("l2", _sparsemax_forward)])
def test_fy_gradient_is_yhat_minus_y(omega, pred):
    rng = np.random.default_rng(1)
    theta = rng.standard_normal(6)
    y = _onehot(2, 6)
    _, grad = fy_loss_and_grad(theta, y, omega=omega, reduction="none")
    np.testing.assert_allclose(grad, pred(theta) - y, atol=1e-9)


@pytest.mark.parametrize("omega", ["entropy", "l2"])
def test_fy_gradient_matches_finite_difference(omega):
    rng = np.random.default_rng(2)
    theta = rng.standard_normal(6)
    y = _onehot(3, 6)
    _, grad = fy_loss_and_grad(theta, y, omega=omega, reduction="none")
    eps = 1e-6
    fd = np.zeros_like(theta)
    for i in range(theta.size):
        hi = theta.copy(); hi[i] += eps
        lo = theta.copy(); lo[i] -= eps
        fd[i] = (
            fenchel_young_loss(hi, y, omega=omega, reduction="none")
            - fenchel_young_loss(lo, y, omega=omega, reduction="none")
        ) / (2 * eps)
    # entropy is smooth everywhere; l2/sparsemax is smooth away from support
    # changes — this theta is generic, so no kink is straddled.
    np.testing.assert_allclose(grad, fd, atol=1e-5)


# ── Bregman-divergence properties ───────────────────────────────────────────

@pytest.mark.parametrize("omega", ["entropy", "l2"])
def test_fy_loss_nonnegative(omega):
    rng = np.random.default_rng(3)
    for _ in range(20):
        theta = rng.standard_normal(5) * 3
        y = _onehot(int(rng.integers(0, 5)), 5)
        assert fenchel_young_loss(theta, y, omega=omega, reduction="none") >= -1e-9


def test_fy_loss_zero_at_perfect_prediction_l2():
    # If θ already induces ŷ = y exactly (sparsemax picks a vertex), loss = 0.
    theta = np.array([10.0, 0.0, 0.0])  # sparsemax → [1, 0, 0]
    y = _onehot(0, 3)
    assert _sparsemax_forward(theta)[0] == pytest.approx(1.0)
    assert sparsemax_loss(theta, y, reduction="none") == pytest.approx(0.0, abs=1e-9)
