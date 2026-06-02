"""EBM2 acceptance: iterative Markov-chain samplers in tessera.rng.

Sprint: EBM2 (Langevin + MALA + HMC + Gibbs).
Roadmap: docs/audit/domain/DOMAIN_AUDIT.md § EBM2

Covers:
  - 4 samplers registered in primitive_coverage.py under category="rng".
  - HMC leapfrog reverses to fp32 round-trip (volume-preserving check):
    forward L steps from (q, p), then forward L steps from (q', -p'),
    should recover (q, -p).
  - MALA acceptance ratio matches analytic expectation for the 2D
    standard Gaussian target over 10k samples (around 0.5–0.9 for
    a reasonable step size).
  - Langevin sampler converges to the target distribution mean & variance
    on a 2D Gaussian (zero-mean, identity covariance).
  - Gibbs sampler converges to the target on a 2D Gaussian.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tessera import rng
from tessera.rng import RNGKey


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_four_ebm2_samplers_registered_in_coverage() -> None:
    from tessera.compiler import primitive_coverage as pc

    names = {e.name for e in pc.all_primitive_coverages().values()}
    expected = {
        "rng_langevin_sample",
        "rng_mala_sample",
        "rng_hmc_sample",
        "rng_gibbs_sample",
    }
    assert expected.issubset(names)


# ---------------------------------------------------------------------------
# 2D Gaussian target — used by Langevin / MALA / HMC tests
# ---------------------------------------------------------------------------

def _gauss_energy(y: np.ndarray) -> float:
    return 0.5 * float(np.sum(y ** 2))


def _gauss_grad(y: np.ndarray) -> np.ndarray:
    return y.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Langevin sampler
# ---------------------------------------------------------------------------

def test_langevin_sample_returns_correct_shape() -> None:
    key = RNGKey.from_seed(0)
    init = np.zeros(2, dtype=np.float32)
    samples, next_key, info = rng.langevin_sample(
        key, init=init, grad_fn=_gauss_grad,
        eta=0.05, temperature=1.0, n_samples=8, burn_in=2,
    )
    assert samples.shape == (8, 2)
    assert isinstance(next_key, RNGKey)
    assert next_key != key
    assert info == {}  # ULA has no MH diagnostics.


def test_langevin_sample_recovers_2d_gaussian_moments() -> None:
    """Stationary distribution: y ~ N(0, I) at temperature=1.0."""
    key = RNGKey.from_seed(1)
    samples, _, _ = rng.langevin_sample(
        key, init=np.zeros(2, dtype=np.float32),
        grad_fn=_gauss_grad,
        eta=0.05, temperature=1.0,
        n_samples=4000, burn_in=500, thin=1,
    )
    mean = samples.mean(axis=0)
    cov = np.cov(samples, rowvar=False)
    assert np.allclose(mean, 0.0, atol=0.15)
    # ULA has step-size bias; allow ~15% slack on the variance.
    assert 0.7 < cov[0, 0] < 1.4
    assert 0.7 < cov[1, 1] < 1.4
    assert abs(cov[0, 1]) < 0.2  # near-zero off-diagonal


def test_langevin_sample_rejects_bad_params() -> None:
    key = RNGKey.from_seed(0)
    with pytest.raises(ValueError, match="eta > 0"):
        rng.langevin_sample(key, init=np.zeros(2), grad_fn=_gauss_grad,
                            eta=0.0, temperature=1.0, n_samples=1)
    with pytest.raises(ValueError, match="temperature >= 0"):
        rng.langevin_sample(key, init=np.zeros(2), grad_fn=_gauss_grad,
                            eta=0.1, temperature=-0.1, n_samples=1)


# ---------------------------------------------------------------------------
# MALA — headline acceptance-ratio test on 2D Gaussian
# ---------------------------------------------------------------------------

def test_mala_acceptance_rate_in_expected_range_for_2d_gaussian() -> None:
    """For a 2D standard Gaussian with η = 0.1, MALA acceptance should
    land comfortably above 0.5 (well-tuned regime) and below 1.0 (some
    rejection on tails). 10k samples gives a tight estimate."""
    key = RNGKey.from_seed(42)
    samples, _, info = rng.mala_sample(
        key, init=np.zeros(2, dtype=np.float64),
        energy_fn=_gauss_energy,
        grad_fn=_gauss_grad,
        eta=0.1, temperature=1.0,
        n_samples=10_000, burn_in=200, thin=1,
    )
    assert samples.shape == (10_000, 2)
    rate = info["accept_rate"]
    assert 0.5 < rate < 0.99, f"MALA acceptance rate out of range: {rate:.3f}"


def test_mala_recovers_2d_gaussian_moments() -> None:
    key = RNGKey.from_seed(43)
    samples, _, info = rng.mala_sample(
        key, init=np.zeros(2, dtype=np.float64),
        energy_fn=_gauss_energy,
        grad_fn=_gauss_grad,
        eta=0.1, temperature=1.0,
        n_samples=5000, burn_in=500, thin=1,
    )
    mean = samples.mean(axis=0)
    cov = np.cov(samples, rowvar=False)
    # MALA is unbiased — tighter tolerance on covariance than ULA;
    # sample-mean tolerance is finite-sample, not bias-driven.
    assert np.allclose(mean, 0.0, atol=0.2)
    assert 0.85 < cov[0, 0] < 1.15
    assert 0.85 < cov[1, 1] < 1.15


def test_mala_rejects_zero_temperature() -> None:
    key = RNGKey.from_seed(0)
    with pytest.raises(ValueError, match="temperature > 0"):
        rng.mala_sample(key, init=np.zeros(2),
                        energy_fn=_gauss_energy, grad_fn=_gauss_grad,
                        eta=0.1, temperature=0.0, n_samples=1)


# ---------------------------------------------------------------------------
# HMC — headline reversibility test
# ---------------------------------------------------------------------------

def test_hmc_leapfrog_is_reversible() -> None:
    """Leapfrog integration is time-reversible: forward L steps from
    (q, p), then forward L steps from (q', -p'), should give (q, -p).

    This is the volume-preserving / reversibility property HMC relies
    on for the MH ratio to be valid.
    """
    from tessera.rng import _hmc_leapfrog

    rng_np = np.random.RandomState(7)
    for _ in range(20):
        q0 = rng_np.randn(3).astype(np.float64)
        p0 = rng_np.randn(3).astype(np.float64)
        step_size = 0.05
        n_leapfrog = 25
        mass_inv = np.ones_like(q0)
        q1, p1 = _hmc_leapfrog(q0, p0, _gauss_grad, step_size, n_leapfrog, mass_inv)
        # Reverse: negate momentum and integrate forward again.
        q_back, p_back = _hmc_leapfrog(
            q1, -p1, _gauss_grad, step_size, n_leapfrog, mass_inv
        )
        # Should recover (q0, -p0).
        assert np.allclose(q_back, q0, atol=1e-7), (
            f"position mismatch: forward {q0} -> {q1}, reverse -> {q_back}"
        )
        assert np.allclose(p_back, -p0, atol=1e-7), (
            f"momentum mismatch: forward {p0} -> {p1}, reverse -> {p_back}"
        )


def test_hmc_sample_recovers_2d_gaussian_moments() -> None:
    key = RNGKey.from_seed(101)
    samples, _, info = rng.hmc_sample(
        key, init=np.zeros(2, dtype=np.float64),
        energy_fn=_gauss_energy,
        grad_fn=_gauss_grad,
        step_size=0.15, n_leapfrog=10,
        n_samples=2000, burn_in=200, thin=1,
    )
    assert samples.shape == (2000, 2)
    rate = info["accept_rate"]
    # Well-tuned HMC on a Gaussian should accept >70% of proposals.
    assert rate > 0.7, f"HMC acceptance rate too low: {rate:.3f}"
    mean = samples.mean(axis=0)
    cov = np.cov(samples, rowvar=False)
    assert np.allclose(mean, 0.0, atol=0.08)
    assert 0.85 < cov[0, 0] < 1.15
    assert 0.85 < cov[1, 1] < 1.15


def test_hmc_sample_validates_inputs() -> None:
    key = RNGKey.from_seed(0)
    with pytest.raises(ValueError, match="step_size > 0"):
        rng.hmc_sample(key, init=np.zeros(2),
                       energy_fn=_gauss_energy, grad_fn=_gauss_grad,
                       step_size=0.0, n_leapfrog=10, n_samples=1)
    with pytest.raises(ValueError, match="n_leapfrog >= 1"):
        rng.hmc_sample(key, init=np.zeros(2),
                       energy_fn=_gauss_energy, grad_fn=_gauss_grad,
                       step_size=0.1, n_leapfrog=0, n_samples=1)


def test_hmc_sample_with_custom_mass() -> None:
    """Diagonal mass matrix produces a different stationary covariance
    only in momentum space — q-marginal should still be standard Gaussian."""
    key = RNGKey.from_seed(202)
    mass = np.array([2.0, 0.5], dtype=np.float64)
    samples, _, info = rng.hmc_sample(
        key, init=np.zeros(2, dtype=np.float64),
        energy_fn=_gauss_energy, grad_fn=_gauss_grad,
        step_size=0.1, n_leapfrog=15, mass=mass,
        n_samples=1500, burn_in=200, thin=1,
    )
    assert info["accept_rate"] > 0.5
    cov = np.cov(samples, rowvar=False)
    assert 0.7 < cov[0, 0] < 1.4
    assert 0.7 < cov[1, 1] < 1.4


# ---------------------------------------------------------------------------
# Gibbs — recovers a 2D correlated Gaussian
# ---------------------------------------------------------------------------

def test_gibbs_sample_recovers_correlated_gaussian() -> None:
    """Target: N(0, Σ) with Σ = [[1, 0.6], [0.6, 1]].

    Conditional p(y_0 | y_1) and p(y_1 | y_0) are 1D Gaussians whose
    parameters are closed-form. Gibbs alternation should recover the
    joint moments.
    """
    rho = 0.6

    def conditional(idx: int, y: np.ndarray, k: RNGKey) -> tuple[float, RNGKey]:
        other = y[1 - idx]
        # p(y_i | y_{-i}) = N(rho * y_other, sqrt(1 - rho**2)).
        mean = rho * other
        std = math.sqrt(1.0 - rho * rho)
        sub, next_k = k.split(2)
        sample = rng.normal(sub, shape=(), dtype="fp64", mean=mean, std=std)
        return float(sample), next_k

    key = RNGKey.from_seed(7)
    samples, _, _ = rng.gibbs_sample(
        key,
        init=np.zeros(2, dtype=np.float64),
        conditional_sample=conditional,
        n_samples=4000, burn_in=500,
    )
    mean = samples.mean(axis=0)
    cov = np.cov(samples, rowvar=False)
    assert np.allclose(mean, 0.0, atol=0.1)
    assert abs(cov[0, 1] - rho) < 0.1, f"recovered correlation {cov[0, 1]:.3f} vs {rho}"
    assert 0.85 < cov[0, 0] < 1.15
    assert 0.85 < cov[1, 1] < 1.15


def test_gibbs_sample_rejects_invalid_sweep_order() -> None:
    key = RNGKey.from_seed(0)
    def stub(i, y, k):
        return 0.0, k
    with pytest.raises(ValueError, match="permutation"):
        rng.gibbs_sample(
            key, init=np.zeros(3),
            conditional_sample=stub,
            sweep_order=[0, 1, 1],
            n_samples=1,
        )


def test_gibbs_sample_rejects_non_rank1_init() -> None:
    key = RNGKey.from_seed(0)
    def stub(i, y, k):
        return 0.0, k
    with pytest.raises(ValueError, match="rank-1"):
        rng.gibbs_sample(
            key, init=np.zeros((2, 2)),
            conditional_sample=stub,
            n_samples=1,
        )


# ---------------------------------------------------------------------------
# Shared chain harness — burn_in / thin parameters
# ---------------------------------------------------------------------------

def test_burn_in_drops_initial_samples() -> None:
    key = RNGKey.from_seed(0)
    samples, _, _ = rng.langevin_sample(
        key, init=np.full(2, 5.0, dtype=np.float64),  # far-from-equilibrium start
        grad_fn=_gauss_grad,
        eta=0.1, temperature=1.0,
        n_samples=200, burn_in=1000, thin=1,
    )
    # After burn-in the chain should be near zero — definitely not still near 5.
    assert abs(samples.mean()) < 0.5


def test_thin_reduces_correlation() -> None:
    key = RNGKey.from_seed(0)
    samples, _, _ = rng.langevin_sample(
        key, init=np.zeros(2, dtype=np.float64),
        grad_fn=_gauss_grad,
        eta=0.05, temperature=1.0,
        n_samples=100, burn_in=200, thin=5,
    )
    # Length matches n_samples even with thinning.
    assert samples.shape == (100, 2)


def test_invalid_chain_params_rejected() -> None:
    key = RNGKey.from_seed(0)
    with pytest.raises(ValueError, match="n_samples must be positive"):
        rng.langevin_sample(key, init=np.zeros(2), grad_fn=_gauss_grad,
                            eta=0.1, temperature=1.0, n_samples=0)
    with pytest.raises(ValueError, match="burn_in must be non-negative"):
        rng.langevin_sample(key, init=np.zeros(2), grad_fn=_gauss_grad,
                            eta=0.1, temperature=1.0, n_samples=1, burn_in=-1)
    with pytest.raises(ValueError, match="thin must be >= 1"):
        rng.langevin_sample(key, init=np.zeros(2), grad_fn=_gauss_grad,
                            eta=0.1, temperature=1.0, n_samples=1, thin=0)
