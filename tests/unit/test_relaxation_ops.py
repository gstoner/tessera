"""T1 — smoothing / relaxation operator family.

Forwards are checked against defining properties (simplex membership, sparsity,
limits) and every VJP against central finite differences.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera.relaxation as relax
from tessera.relaxation import (
    _sparsemax_forward,
    _sparsemax_vjp,
    _entmax15_forward,
    _entmax15_vjp,
    _soft_top_k_forward,
    _soft_top_k_vjp,
    _gumbel_softmax_forward,
    _gumbel_softmax_vjp,
    _perturbed_argmax_forward,
    _perturbed_argmax_vjp,
)
from tessera import rng as trng
from tessera import ops


def _fd_vjp(forward, z, dout, *, eps=1e-6, **kw):
    """Central-difference VJP: (Jᵀ dout)_i = Σ_j dout_j ∂f_j/∂z_i."""
    z = np.asarray(z, dtype=np.float64)
    dout = np.asarray(dout, dtype=np.float64)
    g = np.zeros_like(z)
    it = np.nditer(z, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        hi = z.copy(); hi[i] += eps
        lo = z.copy(); lo[i] -= eps
        dfi = (forward(hi, **kw) - forward(lo, **kw)) / (2 * eps)
        g[i] = np.sum(dout * dfi)
        it.iternext()
    return g


# ── sparsemax ────────────────────────────────────────────────────────────────

def test_sparsemax_is_on_simplex_and_sparse():
    z = np.array([3.0, 0.1, 0.2, -1.0, 2.5])
    p = _sparsemax_forward(z)
    assert p.min() >= 0.0
    assert p.sum() == pytest.approx(1.0)
    assert np.any(p == 0.0)  # sparsemax produces exact zeros


def test_sparsemax_vjp_matches_fd():
    # A well-conditioned point with a stable multi-element support (values close
    # enough that ≥2 survive, none exactly at the threshold), so FD is valid.
    # (At a degenerate vertex the true VJP is 0 while FD straddles a kink.)
    z = np.array([1.0, 1.2, 0.9, 0.7, 1.1])
    p = _sparsemax_forward(z)
    assert int((p > 0).sum()) >= 3  # multi-element support at this point
    dout = np.array([0.2, -0.5, 1.0, 0.3, -0.1])
    (g,) = _sparsemax_vjp(dout, z)
    np.testing.assert_allclose(g, _fd_vjp(_sparsemax_forward, z, dout), atol=1e-5)


def test_sparsemax_vjp_zero_at_vertex():
    # At a one-hot vertex sparsemax is locally constant → exact zero subgradient.
    z = np.array([1.0, 5.0, 0.5])
    (g,) = _sparsemax_vjp(np.array([0.2, -0.5, 1.0]), z)
    np.testing.assert_array_equal(g, np.zeros_like(z))


def test_sparsemax_batched_axis():
    z = np.random.default_rng(0).standard_normal((4, 6))
    p = _sparsemax_forward(z, axis=-1)
    np.testing.assert_allclose(p.sum(axis=-1), np.ones(4), atol=1e-9)


# ── entmax15 ─────────────────────────────────────────────────────────────────

def test_entmax15_is_on_simplex():
    z = np.array([2.0, 1.0, 0.0, -1.0])
    p = _entmax15_forward(z)
    assert p.min() >= 0.0
    assert p.sum() == pytest.approx(1.0, abs=1e-6)


def test_entmax15_vjp_matches_fd():
    z = np.array([1.2, 0.4, -0.6, 0.9, 0.1])
    dout = np.array([0.3, -0.2, 0.5, -0.4, 0.1])
    (g,) = _entmax15_vjp(dout, z)
    np.testing.assert_allclose(g, _fd_vjp(_entmax15_forward, z, dout), atol=1e-4)


# ── soft top-k ───────────────────────────────────────────────────────────────

def test_soft_top_k_gate_selects_k_at_low_tau():
    z = np.array([5.0, 4.0, 1.0, 0.0, -2.0])
    gate = _soft_top_k_forward(z, k=2, tau=0.05)
    # The two largest gates approach one, all others approach zero, and the
    # continuous relaxation retains exactly k units of selection mass.
    assert gate[0] > 0.9 and gate[1] > 0.9
    assert gate[3] < 0.1 and gate[4] < 0.1
    np.testing.assert_allclose(gate.sum(), 2.0, atol=1e-8)


def test_soft_top_k_vjp_matches_fd():
    # The threshold is implicitly determined by the exact-k mass constraint, so
    # finite differences differentiate both the scores and the moving threshold.
    z = np.array([1.0, 2.0, 0.5, 3.0, -0.5])
    dout = np.array([0.1, 0.2, -0.3, 0.4, 0.05])
    tau = 0.7
    (g,) = _soft_top_k_vjp(dout, z, k=2, tau=tau)
    np.testing.assert_allclose(
        g, _fd_vjp(_soft_top_k_forward, z, dout, k=2, tau=tau), atol=1e-4
    )


def test_entmax15_vjp_honors_forward_iteration_count():
    z = np.array([1.2, 0.4, -0.6, 0.9, 0.1])
    dout = np.array([0.3, -0.2, 0.5, -0.4, 0.1])
    (g,) = _entmax15_vjp(dout, z, n_iter=2)
    p = _entmax15_forward(z, n_iter=2)
    s = np.sqrt(p)
    expected = s * (dout - np.sum(s * dout) / np.sum(s))
    np.testing.assert_allclose(
        g, expected, atol=1e-12
    )


def test_soft_top_k_rejects_nonpositive_temperature():
    with pytest.raises(ValueError, match="tau > 0"):
        _soft_top_k_forward(np.array([1.0, 0.0]), k=1, tau=0.0)


# ── gumbel_softmax ───────────────────────────────────────────────────────────

def test_gumbel_softmax_noise_free_is_tempered_softmax():
    z = np.array([1.0, 2.0, 3.0])
    p = _gumbel_softmax_forward(z, tau=0.5)
    y = z / 0.5
    ref = np.exp(y - y.max()) / np.sum(np.exp(y - y.max()))
    np.testing.assert_allclose(p, ref, atol=1e-9)


def test_gumbel_softmax_with_noise_is_on_simplex():
    key = trng.RNGKey.from_seed(3)
    z = np.array([0.5, -1.0, 2.0, 0.3])
    g = trng.gumbel(key, z.shape, dtype="fp64")
    p = _gumbel_softmax_forward(z, tau=1.0, noise=g)
    assert p.sum() == pytest.approx(1.0)
    assert p.min() >= 0.0


def test_gumbel_softmax_vjp_matches_fd():
    z = np.array([0.7, -0.2, 1.1, 0.3])
    dout = np.array([0.2, 0.1, -0.4, 0.3])
    (g,) = _gumbel_softmax_vjp(dout, z, tau=0.8)
    np.testing.assert_allclose(
        g, _fd_vjp(lambda a: _gumbel_softmax_forward(a, tau=0.8), z, dout), atol=1e-5
    )


# ── perturbed argmax ─────────────────────────────────────────────────────────

def test_perturbed_argmax_is_on_simplex_and_approaches_argmax():
    z = np.array([3.0, 0.0, -1.0])
    p = _perturbed_argmax_forward(z, sigma=0.3, n_samples=2000, seed=1)
    assert p.sum() == pytest.approx(1.0)
    assert np.argmax(p) == 0  # mass concentrates on the true argmax


def test_perturbed_argmax_vjp_matches_fd():
    # Monte-Carlo VJP vs a matched-seed finite difference. Both use the SAME
    # seed so the perturbation sequence is common-random-number consistent.
    z = np.array([1.0, 0.4, -0.2, 0.7])
    dout = np.array([0.3, -0.1, 0.2, -0.05])
    kw = dict(sigma=0.5, n_samples=4000, seed=7)
    (g,) = _perturbed_argmax_vjp(dout, z, **kw)
    fd = _fd_vjp(lambda a, **k: _perturbed_argmax_forward(a, **k), z, dout, eps=1e-2, **kw)
    # stochastic estimator → loose tolerance, but same ballpark and sign.
    np.testing.assert_allclose(g, fd, atol=0.15)


# ── registration surface ─────────────────────────────────────────────────────

def test_relaxation_ops_registered_with_vjp():
    from tessera.autodiff.vjp import get_vjp
    from tessera import ops
    for name in ("sparsemax", "entmax15", "soft_top_k", "gumbel_softmax", "perturbed_argmax"):
        assert get_vjp(name) is not None, f"{name} has no VJP"
        assert hasattr(ops, name), f"ops.{name} missing"


def test_public_relaxation_forward_surface_matches_defining_implementations():
    """Exercise the public primitives, not only their private VJP oracles."""
    z = np.array([1.0, 0.4, -0.2, 0.7], dtype=np.float64)
    np.testing.assert_allclose(ops.sparsemax(z), _sparsemax_forward(z))
    np.testing.assert_allclose(ops.entmax15(z), _entmax15_forward(z))
    np.testing.assert_allclose(
        ops.soft_top_k(z, k=2, tau=0.7),
        _soft_top_k_forward(z, k=2, tau=0.7),
    )
    np.testing.assert_allclose(
        ops.gumbel_softmax(z, tau=0.8),
        _gumbel_softmax_forward(z, tau=0.8),
    )
    np.testing.assert_allclose(
        ops.perturbed_argmax(z, sigma=0.5, n_samples=128, seed=11),
        _perturbed_argmax_forward(z, sigma=0.5, n_samples=128, seed=11),
    )
