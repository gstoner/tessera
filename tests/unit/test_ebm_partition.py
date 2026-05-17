"""EBM3 acceptance: partition function estimators.

Sprint: EBM3.
Roadmap: docs/audit/ga_ebm_roadmap.md § EBM3

Covers the EBM3 acceptance criteria:
  - 3 partition-function primitives registered in primitive_coverage.
  - Exact partition on a 4-visible RBM matches the brute-force sum
    (verified against an independent enumeration).
  - AIS estimate on a 2D Gaussian matches the analytic Z = (2π)^(d/2)·σ²
    to ≤ 5% with 1000 chains.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from tessera import ebm
from tessera.ebm.partition import (
    partition_function,
    partition_function_ais,
    partition_function_exact,
    partition_function_monte_carlo,
)
from tessera.rng import RNGKey, normal


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_three_ebm3_primitives_registered_in_coverage() -> None:
    from tessera.compiler import primitive_coverage as pc

    names = {e.name for e in pc.all_primitive_coverages().values()}
    expected = {"ebm_partition_exact", "ebm_partition_monte_carlo", "ebm_partition_ais"}
    assert expected.issubset(names)


# ---------------------------------------------------------------------------
# Exact — 4-visible RBM headline test
# ---------------------------------------------------------------------------

def _rbm_energy(W: np.ndarray, b_v: np.ndarray, b_h: np.ndarray):
    """Free energy of a binary RBM over visible units only, with hidden
    units summed analytically.

    For a Bernoulli-Bernoulli RBM with weights W (n_v × n_h) and biases
    b_v, b_h, the free energy is

        F(v) = -b_v · v - Σ_j log(1 + exp(b_h_j + Σ_i W_{ij} v_i))

    The marginal partition function is Z = Σ_v exp(-F(v)).
    """
    def energy_fn(v):
        v = np.asarray(v, dtype=np.float64)
        bv_term = float(np.dot(b_v, v))
        h_pre = b_h + W.T @ v
        log_one_plus_exp = np.log1p(np.exp(h_pre))
        return -bv_term - float(log_one_plus_exp.sum())

    return energy_fn


def test_exact_partition_on_4_visible_rbm_matches_brute_force() -> None:
    """4-visible × 3-hidden RBM. 16 visible states; tractable."""
    rng = np.random.RandomState(0)
    n_v, n_h = 4, 3
    W = rng.randn(n_v, n_h) * 0.5
    b_v = rng.randn(n_v) * 0.3
    b_h = rng.randn(n_h) * 0.3

    energy_fn = _rbm_energy(W, b_v, b_h)
    all_states = list(itertools.product([0.0, 1.0], repeat=n_v))

    Z_tessera = partition_function_exact(energy_fn, all_states)

    # Independent brute-force reference: Z = Σ_v Σ_h exp(-E(v, h)).
    Z_brute = 0.0
    for v in itertools.product([0.0, 1.0], repeat=n_v):
        v_arr = np.array(v, dtype=np.float64)
        for h in itertools.product([0.0, 1.0], repeat=n_h):
            h_arr = np.array(h, dtype=np.float64)
            energy_joint = (
                -float(np.dot(b_v, v_arr))
                - float(np.dot(b_h, h_arr))
                - float(v_arr @ W @ h_arr)
            )
            Z_brute += math.exp(-energy_joint)

    assert Z_tessera == pytest.approx(Z_brute, rel=1e-10)


def test_exact_partition_uses_logsumexp_for_numerical_stability() -> None:
    """Two states with large negative energies — Z should not overflow."""
    def energy_fn(s):
        return -500.0 if s == "a" else -502.0

    Z = partition_function_exact(energy_fn, ["a", "b"])
    expected = math.exp(500.0) + math.exp(502.0)
    assert Z == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def test_monte_carlo_partition_matches_analytic_for_2d_gaussian() -> None:
    """Target: p(y) ∝ exp(-½ ‖y‖²). Z_analytic = 2π.
    Use a wider Gaussian proposal q = N(0, σ_q² I) with σ_q = 1.5."""
    sigma_q = 1.5
    log_norm_q = -math.log(2.0 * math.pi * sigma_q * sigma_q)  # log(1/(2π σ²)) in 2D

    def energy_fn(y):
        return 0.5 * float(y @ y)

    def proposal_sampler(key):
        sub, next_key = key.split(2)
        y = normal(sub, shape=(2,), dtype="fp64", std=sigma_q).astype(np.float64, copy=False)
        return y, next_key

    def proposal_log_density(y):
        return float(-0.5 * (y @ y) / (sigma_q * sigma_q) + log_norm_q)

    key = RNGKey.from_seed(0)
    Z_est, info = partition_function_monte_carlo(
        energy_fn,
        key=key,
        proposal_sampler=proposal_sampler,
        proposal_log_density=proposal_log_density,
        n_samples=20_000,
    )
    Z_analytic = 2.0 * math.pi
    rel_err = abs(Z_est - Z_analytic) / Z_analytic
    assert rel_err < 0.05, (
        f"MC Z estimate {Z_est:.4f} vs analytic {Z_analytic:.4f} "
        f"(rel err {rel_err:.4f}); ESS={info['ess']:.0f}"
    )
    assert info["ess"] > 1000


# ---------------------------------------------------------------------------
# AIS — headline acceptance test
# ---------------------------------------------------------------------------

def test_ais_partition_for_2d_gaussian_matches_analytic_within_5pct() -> None:
    """Target: p(y) ∝ exp(-½ ‖y/σ‖²) with σ = 2. Reference: standard N(0, I).

    Z_target = (2π σ²) ^ (d/2), Z_ref = (2π)^(d/2). Ratio = σ^d = 16 for d=2.
    """
    d = 2
    sigma = 2.0

    def energy_fn(y):
        return 0.5 * float(y @ y) / (sigma * sigma)

    def grad_fn(y):
        return y / (sigma * sigma)

    def ref_sampler(key):
        sub, next_key = key.split(2)
        y = normal(sub, shape=(d,), dtype="fp64").astype(np.float64, copy=False)
        return y, next_key

    def ref_log_density(y):
        return float(-0.5 * (y @ y) - 0.5 * d * math.log(2.0 * math.pi))

    def ref_grad_fn(y):
        return y  # gradient of -log p_ref = 0.5 ||y||² is y

    # ref_log_density is normalized (returns log p_ref(y) with the
    # normalizing constant baked in), so the bridge distribution at β=0
    # already has partition function 1 — Z_ref = 1 here. The AIS
    # estimator returns Z_target / Z_ref = Z_target directly.
    key = RNGKey.from_seed(7)
    Z_target, info = partition_function_ais(
        energy_fn,
        key=key,
        ref_sampler=ref_sampler,
        ref_log_density=ref_log_density,
        grad_fn=grad_fn,
        ref_grad_fn=ref_grad_fn,
        Z_ref=1.0,
        n_chains=1000,
        n_steps=32,
        schedule="linear",
        mcmc_step_size=0.2,
        mcmc_n_leapfrog=8,
    )
    Z_analytic = (2.0 * math.pi * sigma * sigma) ** (d / 2)
    rel_err = abs(Z_target - Z_analytic) / Z_analytic
    assert rel_err < 0.05, (
        f"AIS Z estimate {Z_target:.4f} vs analytic {Z_analytic:.4f} "
        f"(rel err {rel_err:.4f}); ESS={info['ess']:.0f}"
    )


def test_ais_without_grad_fn_runs_without_mcmc_moves() -> None:
    """Without grad_fn / ref_grad_fn, AIS skips HMC transitions but still
    runs the importance-weighted ratio (with higher variance)."""
    d = 2

    def energy_fn(y):
        return 0.5 * float(y @ y)

    def ref_sampler(key):
        sub, next_key = key.split(2)
        y = normal(sub, shape=(d,), dtype="fp64").astype(np.float64, copy=False)
        return y, next_key

    def ref_log_density(y):
        return float(-0.5 * (y @ y) - 0.5 * d * math.log(2.0 * math.pi))

    key = RNGKey.from_seed(0)
    Z_target, info = partition_function_ais(
        energy_fn,
        key=key,
        ref_sampler=ref_sampler,
        ref_log_density=ref_log_density,
        Z_ref=1.0,  # normalized reference
        n_chains=200,
        n_steps=16,
    )
    # No MCMC means the chain is a sequence of unrefreshed samples — still
    # a valid estimator but high-variance. Just check it returns a positive
    # finite estimate.
    assert math.isfinite(Z_target) and Z_target > 0


def test_ais_rejects_invalid_params() -> None:
    def energy(y): return 0.0
    def sampler(key):
        sub, k = key.split(2)
        return normal(sub, shape=(2,), dtype="fp64"), k
    def density(y): return 0.0
    key = RNGKey.from_seed(0)
    with pytest.raises(ValueError, match="n_chains must be positive"):
        partition_function_ais(energy, key=key, ref_sampler=sampler,
                                ref_log_density=density, n_chains=0)
    with pytest.raises(ValueError, match="n_steps must be >= 2"):
        partition_function_ais(energy, key=key, ref_sampler=sampler,
                                ref_log_density=density, n_steps=1)


# ---------------------------------------------------------------------------
# Dispatch wrapper
# ---------------------------------------------------------------------------

def test_partition_function_dispatch_exact() -> None:
    def energy_fn(s): return -1.0 if s == "a" else 0.0
    Z = partition_function(energy_fn, method="exact", states=["a", "b"])
    assert Z == pytest.approx(math.e + 1.0)


def test_partition_function_dispatch_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        partition_function(lambda s: 0.0, method="bogus")


def test_partition_function_exact_method_requires_states_kwarg() -> None:
    with pytest.raises(TypeError, match="requires `states`"):
        partition_function(lambda s: 0.0, method="exact")


def test_partition_function_exposed_through_ebm_namespace() -> None:
    """All four partition functions are accessible as tessera.ebm.*."""
    assert ebm.partition_function is partition_function
    assert ebm.partition_function_exact is partition_function_exact
    assert ebm.partition_function_monte_carlo is partition_function_monte_carlo
    assert ebm.partition_function_ais is partition_function_ais
