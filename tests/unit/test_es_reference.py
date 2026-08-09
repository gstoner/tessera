"""Oracles for the Evolution-Strategies reference tier (``tessera.stdlib.es``).

W1 of the EGGROLL plan (``docs/audit/compiler/EGGROLL_SUPPORT_PLAN.md``): the
host-free reference the Graph-IR op ``es_low_rank_correction`` must reproduce.
Two families — A exact (any r), B statistical (expectation) — mirroring the
``verify_synthesized_*`` split so asserting the ES limit as equality (the "G1"
error) never happens.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera import rng
from tessera.stdlib import es


def _cos(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


# ============================================================ FAMILY A — EXACT
@pytest.mark.parametrize("r", [1, 2, 4])
def test_A1_forward_identity(r):
    key = rng.RNGKey.from_seed(12345)
    x = np.random.default_rng(0).standard_normal((6, 16))
    M = np.random.default_rng(1).standard_normal((24, 16))
    members = list(range(6))
    got = es.population_forward(x, M, key, members, sigma=0.03, rank=r)
    dense = np.stack([
        x[p] @ (M + es.low_rank_perturbation(key, mem, 24, 16, r, sigma=0.03,
                                             dtype="fp64")).T
        for p, mem in enumerate(members)])
    assert np.allclose(got, dense, atol=1e-10)


@pytest.mark.parametrize("r", [1, 3])
def test_A2_update_matches_naive_sum(r):
    key = rng.RNGKey.from_seed(999)
    members = list(range(32))
    f = np.random.default_rng(2).standard_normal(32)
    got = es.es_update(key, members, f, 20, 12, sigma=0.02, rank=r)
    naive = np.zeros((20, 12))
    for p, mem in enumerate(members):
        A, B, s = es.reconstruct_factors(key, mem, 20, 12, r)
        naive += (f[p] * s / np.sqrt(r) / len(members)) * (
            A.astype(np.float64) @ B.astype(np.float64).T)   # match es_update's fp64 accum
    assert np.allclose(got, naive, atol=1e-9)


@pytest.mark.parametrize("P,r,anti,exp", [(2, 1, True, 1), (5, 2, True, 6),
                                          (40, 1, True, 12), (8, 1, False, 8)])
def test_A3_aggregate_rank_antithetic_aware(P, r, anti, exp):
    # antithetic halves distinct directions: rank = min(#pairs·r, out, in)  (Thm 3, refined)
    key = rng.RNGKey.from_seed(7)
    members = list(range(P))
    f = np.random.default_rng(3).standard_normal(P) + 0.5
    upd = es.es_update(key, members, f, 20, 12, sigma=0.02, rank=r, antithetic=anti)
    assert np.linalg.matrix_rank(upd, tol=1e-9) == exp


def test_A4_rng_member_keyed_not_rank_keyed():           # G2
    key = rng.RNGKey.from_seed(1)
    a0, b0, _ = es.reconstruct_factors(key, 5, 24, 16, 1)
    a1, b1, _ = es.reconstruct_factors(key, 5, 24, 16, 1)     # repeat = identical
    assert np.array_equal(a0, a1) and np.array_equal(b0, b1)
    # a worker on another rank folds NO rank in -> same member, same factors
    A0, B0, s0 = es.reconstruct_factors(key, 0, 24, 16, 1)
    A1, B1, s1 = es.reconstruct_factors(key, 1, 24, 16, 1)
    E0 = (s0 * A0 @ B0.T).ravel(); E1 = (s1 * A1 @ B1.T).ravel()
    assert np.corrcoef(E0, E1)[0, 1] < -0.999                # antithetic pair anti-correlated
    reps = [(lambda A, B: (A @ B.T).ravel())(*es.reconstruct_factors(key, 2 * k, 24, 16, 1)[:2])
            for k in range(48)]
    off = np.abs(np.corrcoef(np.array(reps)) - np.eye(48))
    assert off.max() < 0.4                                   # distinct directions ~uncorrelated


def test_A5_accum_fp32_contract():                           # I6 / Decision #32
    key = rng.RNGKey.from_seed(5)
    members = list(range(4000))
    f = np.random.default_rng(4).standard_normal(4000)
    ref = es.es_update(key, members, f, 8, 8, sigma=0.02, rank=1, accum="fp32")

    def bf16(x):
        u = np.asarray(x, np.float32).view(np.uint32)
        return ((u + 0x8000) & 0xFFFF0000).view(np.float32).astype(np.float64)
    acc = np.zeros((8, 8))
    for p, mem in enumerate(members):
        A, B, s = es.reconstruct_factors(key, mem, 8, 8, 1)
        acc = bf16(acc + bf16((f[p] * s / len(members)) * (A @ B.T)))
    assert np.linalg.norm(acc - ref) / np.linalg.norm(ref) > 1e-3

    with pytest.raises(ValueError):
        es.es_update(key, members, f, 8, 8, sigma=0.02, rank=1, accum="bf16")
    # s32 (integer lane) is reserved, not silently aliased to fp32 (review P2)
    with pytest.raises(NotImplementedError):
        es.es_update(key, members, f, 8, 8, sigma=0.02, rank=1, accum="s32")


def test_A_storage_dtype_is_honored_not_widened():           # review P2 (fp16)
    key = rng.RNGKey.from_seed(0)
    import ml_dtypes
    assert es.low_rank_perturbation(key, 0, 8, 8, 1, sigma=0.1, dtype="fp16").dtype == np.float16
    assert es.low_rank_perturbation(key, 0, 8, 8, 1, sigma=0.1, dtype="f16").dtype == np.float16
    assert es.low_rank_perturbation(key, 0, 8, 8, 1, sigma=0.1,
                                    dtype="bf16").dtype == ml_dtypes.bfloat16
    with pytest.raises((ValueError, Exception)):               # unsupported spelling rejected
        es.low_rank_perturbation(key, 0, 8, 8, 1, sigma=0.1, dtype="fp8_e4m3")


def test_A_sigma_is_required_semantic():                     # G4
    key = rng.RNGKey.from_seed(0)
    with pytest.raises(ValueError):
        es.low_rank_perturbation(key, 0, 8, 8, 1, sigma=None)
    with pytest.raises(ValueError):
        es.es_update(key, [0, 1], [1.0, -1.0], 8, 8, sigma=None, rank=1)


# ================================================== FAMILY B — STATISTICAL
def test_B1_moment_match_at_rank1():
    g = np.random.default_rng(1)
    A = g.standard_normal((200000, 16)); B = g.standard_normal((200000, 10))
    E = (A[:, :, None] * B[:, None, :]).reshape(200000, -1)   # a_i b_j, r=1
    assert abs(E.var(0).mean() - 1) < 0.02
    C = np.cov(E[:, [0, 1, 16, 17]].T)
    assert np.abs(C - np.diag(np.diag(C))).max() < 0.03


def test_B2_unbiased_on_linear_f():
    out, inn = 12, 10
    G = np.random.default_rng(5).standard_normal((out, inn))
    M = np.zeros((out, inn))
    acc = np.zeros((out, inn)); REP, N, sigma = 20, 1500, 0.05
    for k in range(REP):
        key = rng.RNGKey.from_seed(1000 + k)
        members = list(range(N))
        fit = np.array([np.sum(G * (M + es.low_rank_perturbation(key, m, out, inn, 1,
                                                                 sigma=sigma, dtype="fp64")))
                        for m in members])
        acc += es.es_update(key, members, fit, out, inn, sigma=sigma, rank=1)
    assert _cos(acc / REP, G) > 0.98


def test_B3_bias_requires_third_derivative():
    # quadratic f (∇³f=0) => EGGROLL unbiased at r=1  (Thm 5 mechanism)
    out, inn = 12, 10
    M = 0.3 * np.random.default_rng(6).standard_normal((out, inn))
    Q = np.random.default_rng(7).standard_normal((out, inn))
    acc = np.zeros((out, inn)); REP, N, sigma = 20, 1500, 0.05
    for k in range(REP):
        key = rng.RNGKey.from_seed(2000 + k)
        members = list(range(N))
        fit = np.array([np.sum(0.5 * Q * (M + es.low_rank_perturbation(
            key, m, out, inn, 1, sigma=sigma, dtype="fp64")) ** 2) for m in members])
        acc += es.es_update(key, members, fit, out, inn, sigma=sigma, rank=1)
    assert _cos(acc / REP, Q * M) > 0.98


def test_B_fitness_shaping_shares_grpo_normalizer():
    # O6: zscore delegates to rl.normalize_group_advantages (ES ≡ GRPO scoring)
    from tessera.rl import normalize_group_advantages
    r = np.random.default_rng(9).standard_normal((4, 8))
    assert np.allclose(es.fitness_shaping(r, method="zscore", group_axis=1),
                       normalize_group_advantages(r, group_axis=1))
    cr = es.fitness_shaping(np.array([3.0, 1.0, 2.0, 0.0]), method="centered_rank")
    assert abs(cr.mean()) < 1e-12 and cr.argmax() == 0 and cr.argmin() == 3
    assert np.array_equal(es.antithetic_sign([1.0, 0.0, 2.0], [0.0, 1.0, 2.0]),
                          [1.0, -1.0, 0.0])
