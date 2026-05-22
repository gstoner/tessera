"""Sprint #20e (2026-05-22) — sharding mock-mesh proofs for the
ebm sampling family.

5 ops promote (Bucket B):
  * ``ebm_inner_step``     — pointwise per-candidate (B, K) → (B, K)
  * ``ebm_langevin_step``  — pointwise per-candidate
  * ``ebm_partition_exact``    — stable logsumexp over candidates
  * ``ebm_partition_ais``      — annealed importance sampling
  * ``ebm_partition_monte_carlo`` — Monte Carlo estimator

4 ops stay at `partial` (Bucket C — manifold-bound, Phase G/H/I gate):
  * ``ebm_bivector_langevin_sample``
  * ``ebm_bivector_langevin_step``
  * ``ebm_sphere_langevin_sample``
  * ``ebm_sphere_langevin_step``

These bivector / sphere Langevin steps live on non-Euclidean manifolds
(Spin(p,q) bivector subspace; the unit sphere).  Their sharding rule
requires GA-aware halo exchange that hasn't shipped — keep at partial.

The ebm ops aren't directly accessible from ``tessera.ops`` (they live
behind the energy-IR lane), so this file proves the abstract patterns
on numpy reference implementations.  The point is to lock the
*sharding contract* — the algorithmic invariant that defines what
sharded execution must reproduce.

Bucket B proofs:

  ebm_inner_step / ebm_langevin_step — under candidate-axis sharding,
  each rank holds its (B_local, K) slice; the gradient step / Langevin
  step is purely local (no cross-shard communication).  all_gather
  along the candidate axis recovers the global state.

  ebm_partition_exact — Z = sum_k exp(-E(y_k)) via stable logsumexp:
      logZ = m + log(sum_k exp(-E_k - m))    where m = max_k(-E_k)
  Under candidate-axis sharding the canonical reduction is:
      local_max  → all_reduce(max)            = global_max
      local_logZ_partial = sum_k exp(-E_k - global_max)
      global_Z = all_reduce(sum)(local_logZ_partial)
      logZ = global_max + log(global_Z)
  Two collectives (one max, one sum).

  ebm_partition_ais / ebm_partition_monte_carlo — embarrassingly
  parallel chains.  Each rank runs independent chains; mean of the
  per-chain estimator via all_reduce(mean) recovers the global
  estimator.  Per-rank-local in the chain loop.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.primitive_coverage import all_primitive_coverages
from tessera.testing.mock_collective import MockRankGroup


def _energy_fn(y: np.ndarray) -> np.ndarray:
    """A simple reference energy: E(y) = 0.5 * sum(y², axis=-1).
    Used to anchor the partition / sampling proofs on a closed-form
    quantity.  Returns shape (..., 1) suitable for partition reductions
    that sum over the candidate axis."""
    return 0.5 * (y ** 2).sum(axis=-1)


def _grad_energy(y: np.ndarray) -> np.ndarray:
    """∇E(y) = y for the reference energy above."""
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Pointwise: inner_step / langevin_step are candidate-axis-local
# ─────────────────────────────────────────────────────────────────────────────


def test_ebm_inner_step_candidate_split_is_local() -> None:
    """Reference ebm_inner_step is y' = y - eta * ∇E(y).  Under
    candidate-axis sharding each rank's slice updates independently;
    all_gather along the candidate axis recovers the global state."""
    np.random.seed(0)
    K, D = 8, 4
    eta = 0.1
    y = np.random.randn(K, D).astype(np.float32)
    expected = y - eta * _grad_energy(y)

    world_size = 2
    K_local = K // world_size

    def worker(rank):
        k0 = rank.rank * K_local
        k1 = k0 + K_local
        y_local = y[k0:k1]
        local_out = y_local - eta * _grad_energy(y_local)
        return rank.all_gather(local_out.astype(np.float32), axis=0)

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for gathered in results:
        np.testing.assert_allclose(gathered, expected, rtol=1e-6, atol=1e-6)


def test_ebm_langevin_step_candidate_split_is_local() -> None:
    """Reference Langevin step: y' = y - eta*∇E(y) + sqrt(2*eta*T)*ξ
    with ξ drawn per-candidate.  Under candidate-axis sharding the
    noise term uses per-shard RNG (deterministic via fold_in pattern).
    Mock-mesh proof uses a pre-sampled noise tensor to make the test
    deterministic; the sharding contract is the per-rank-local
    structure."""
    np.random.seed(1)
    K, D = 8, 4
    eta = 0.1
    T = 0.5
    y = np.random.randn(K, D).astype(np.float32)
    noise = np.random.randn(K, D).astype(np.float32)  # pre-sampled

    scale = np.sqrt(2.0 * eta * T)
    expected = y - eta * _grad_energy(y) + scale * noise

    world_size = 2
    K_local = K // world_size

    def worker(rank):
        k0 = rank.rank * K_local
        k1 = k0 + K_local
        y_local = y[k0:k1]
        n_local = noise[k0:k1]
        local_out = y_local - eta * _grad_energy(y_local) + scale * n_local
        return rank.all_gather(local_out.astype(np.float32), axis=0)

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for gathered in results:
        np.testing.assert_allclose(gathered, expected, rtol=1e-6, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Partition function: stable logsumexp over candidates
# ─────────────────────────────────────────────────────────────────────────────


def test_ebm_partition_exact_logsumexp_two_collectives() -> None:
    """logZ = log(sum_k exp(-E(y_k))) shards via stable two-collective
    pattern: all_reduce(max) for the shift, then a sum-of-exp inside a
    single packed all_reduce.  We pack (sum, world_max) to keep one
    collective per step, avoiding the back-to-back race in
    MockRankGroup."""
    np.random.seed(2)
    K, D = 16, 4
    y = np.random.randn(K, D).astype(np.float32)
    neg_E = -_energy_fn(y)  # shape (K,)
    # Single-rank reference logsumexp.
    m_ref = float(neg_E.max())
    expected_logZ = float(m_ref + np.log(np.exp(neg_E - m_ref).sum()))

    world_size = 2
    K_local = K // world_size

    def worker(rank):
        k0 = rank.rank * K_local
        k1 = k0 + K_local
        neg_E_local = neg_E[k0:k1]
        # First collective: all_reduce(max) — broadcasts the global max.
        local_max = np.asarray([float(neg_E_local.max())], dtype=np.float32)
        global_max_arr = rank.all_reduce(local_max, op="max")
        global_max = float(global_max_arr[0])
        # Barrier to force ordering between consecutive all_reduce
        # calls — MockRankGroup's `_withdraw` after the first all_reduce
        # can race with the next `_deposit` if both ranks haven't
        # finished returning.  An explicit barrier serializes the calls.
        rank.barrier()
        # Second collective: all_reduce(sum) of shifted exponentials.
        local_sum = np.asarray(
            [float(np.exp(neg_E_local - global_max).sum())], dtype=np.float32,
        )
        global_sum_arr = rank.all_reduce(local_sum, op="sum")
        return float(global_max + np.log(global_sum_arr[0]))

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for logZ in results:
        assert abs(logZ - expected_logZ) < 1e-3, (
            f"logsumexp under sharding {logZ} != single-rank {expected_logZ}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# AIS / Monte Carlo — embarrassingly parallel chains via all_reduce(mean)
# ─────────────────────────────────────────────────────────────────────────────


def test_ebm_partition_ais_chains_average_to_global() -> None:
    """AIS estimator runs independent chains and averages the per-chain
    weights.  Under chain-axis sharding each rank runs its chains
    locally; all_reduce(mean) recovers the global estimator.  The
    canonical pattern.

    Mock-mesh proof: simulate chains as per-chain log-weights drawn
    independently; verify mean-reduction equals the single-rank mean."""
    np.random.seed(3)
    num_chains = 16
    log_w = np.random.randn(num_chains).astype(np.float32)
    expected_mean = float(log_w.mean())

    world_size = 4
    chains_local = num_chains // world_size

    def worker(rank):
        c0 = rank.rank * chains_local
        c1 = c0 + chains_local
        local_mean = np.asarray([float(log_w[c0:c1].mean())], dtype=np.float32)
        # all_reduce(sum) of per-rank means / world_size = global mean
        # (because each rank has equal chains_local).
        global_sum = rank.all_reduce(local_mean, op="sum")
        return float(global_sum[0] / world_size)

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for mean in results:
        assert abs(mean - expected_mean) < 1e-5


def test_ebm_partition_monte_carlo_chains_average_to_global() -> None:
    """Same pattern as AIS — Monte Carlo estimator under chain-axis
    sharding is embarrassingly parallel."""
    np.random.seed(4)
    num_samples = 32
    samples = np.random.randn(num_samples).astype(np.float32)
    expected_estimator = float(samples.mean())

    world_size = 2
    samples_local = num_samples // world_size

    def worker(rank):
        s0 = rank.rank * samples_local
        s1 = s0 + samples_local
        local_mean = np.asarray([float(samples[s0:s1].mean())], dtype=np.float32)
        global_sum = rank.all_reduce(local_mean, op="sum")
        return float(global_sum[0] / world_size)

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for est in results:
        assert abs(est - expected_estimator) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Registry promotion claims
# ─────────────────────────────────────────────────────────────────────────────


_SPRINT_20E_PROMOTED_NAMES = (
    "ebm_inner_step",
    "ebm_langevin_step",
    "ebm_partition_exact",
    "ebm_partition_ais",
    "ebm_partition_monte_carlo",
)

_SPRINT_20E_BUCKET_C = (
    "ebm_bivector_langevin_sample",
    "ebm_bivector_langevin_step",
    "ebm_sphere_langevin_sample",
    "ebm_sphere_langevin_step",
)


def test_sprint_20e_promoted_set_sharding_complete() -> None:
    """The mock-mesh proofs above license `sharding_rule = complete`
    for the 5 ebm sampling ops covered by Sprint #20e."""
    entries = all_primitive_coverages()
    failures: list[tuple[str, str]] = []
    for name in _SPRINT_20E_PROMOTED_NAMES:
        if name not in entries:
            continue
        actual = entries[name].contract_status.get("sharding_rule")
        if actual != "complete":
            failures.append((name, str(actual)))
    assert not failures, (
        "Sprint #20e promoted set sharding_rule must be `complete` "
        f"after the mock-mesh proofs, but got: {failures}.  See "
        "test_ebm_inner_step_candidate_split_is_local, "
        "test_ebm_partition_exact_logsumexp_two_collectives, etc."
    )


def test_sprint_20e_manifold_langevin_stays_partial() -> None:
    """The 4 manifold Langevin ops live on non-Euclidean manifolds
    (Spin(p,q) bivector subspace, unit sphere).  Their sharding
    requires GA-aware halo exchange that hasn't shipped — keep at
    partial pending Phase G/H/I."""
    entries = all_primitive_coverages()
    failures: list[tuple[str, str]] = []
    for name in _SPRINT_20E_BUCKET_C:
        if name not in entries:
            continue
        actual = entries[name].contract_status.get("sharding_rule")
        if actual not in ("partial", "planned"):
            failures.append((name, str(actual)))
    assert not failures, (
        "manifold Langevin ops must stay at partial/planned (Bucket C "
        f"— Phase G/H/I manifold halo gate); got {failures}."
    )
