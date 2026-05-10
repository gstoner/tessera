"""Tests for `tessera.rng` (S-series sprint S4).

S4 acceptance criteria (from execution_roadmap.md):
  - Typed RNG keys with split / fold_in / clone, named streams, and
    deterministic replay metadata.
  - Samplers: uniform, normal, truncated_normal, bernoulli, categorical,
    multinomial, randint, permutation, gamma, beta, dirichlet, poisson.
  - Determinism across single-device, sharded, checkpointed, resumed runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.rng import (
    RNGKey,
    bernoulli,
    beta,
    categorical,
    dirichlet,
    gamma,
    multinomial,
    normal,
    permutation,
    poisson,
    randint,
    truncated_normal,
    uniform,
)


# ── RNGKey: construction, naming, equality ─────────────────────────────────


def test_rng_key_from_seed_is_deterministic():
    a = RNGKey.from_seed(42)
    b = RNGKey.from_seed(42)
    assert a.seed_high == b.seed_high
    assert a.seed_low == b.seed_low


def test_rng_key_different_seeds_yield_different_keys():
    a = RNGKey.from_seed(0)
    b = RNGKey.from_seed(1)
    assert a.seed_high != b.seed_high


def test_rng_key_supports_optional_name_for_debugging():
    k = RNGKey.from_seed(7, name="dropout")
    assert k.name == "dropout"
    # Name must NOT influence the sample stream.
    k_unnamed = RNGKey.from_seed(7)
    np.testing.assert_array_equal(
        uniform(k, (4,), dtype="fp32"),
        uniform(k_unnamed, (4,), dtype="fp32"),
    )


# ── split: independence + reproducibility ──────────────────────────────────


def test_split_is_deterministic():
    parent = RNGKey.from_seed(123)
    a1, a2, a3 = parent.split(3)
    b1, b2, b3 = parent.split(3)
    assert (a1.seed_high, a1.seed_low) == (b1.seed_high, b1.seed_low)
    assert (a2.seed_high, a2.seed_low) == (b2.seed_high, b2.seed_low)
    assert (a3.seed_high, a3.seed_low) == (b3.seed_high, b3.seed_low)


def test_split_children_are_distinct():
    parent = RNGKey.from_seed(0)
    a, b, c = parent.split(3)
    s_a = uniform(a, (1024,))
    s_b = uniform(b, (1024,))
    s_c = uniform(c, (1024,))
    # No two children should produce the same draws.
    assert not np.array_equal(s_a, s_b)
    assert not np.array_equal(s_a, s_c)
    assert not np.array_equal(s_b, s_c)


def test_split_requires_positive_count():
    with pytest.raises(ValueError, match="num > 0"):
        RNGKey.from_seed(0).split(0)


# ── fold_in: deterministic per-shard / per-step keys ───────────────────────


def test_fold_in_is_deterministic_and_distinct():
    base = RNGKey.from_seed(99)
    epoch_0 = base.fold_in(0)
    epoch_1 = base.fold_in(1)
    epoch_0_again = base.fold_in(0)
    assert epoch_0.seed_low == epoch_0_again.seed_low
    assert epoch_0.seed_low != epoch_1.seed_low
    # Streams must differ.
    assert not np.array_equal(uniform(epoch_0, (32,)), uniform(epoch_1, (32,)))


def test_fold_in_accepts_strings_and_ints():
    base = RNGKey.from_seed(7)
    layer3 = base.fold_in("layer_3_dropout")
    rank1 = base.fold_in(1)
    assert layer3.seed_low != rank1.seed_low


# ── clone: equal state, no advancement ─────────────────────────────────────


def test_clone_yields_equivalent_stream():
    a = RNGKey.from_seed(42)
    b = a.clone()
    np.testing.assert_array_equal(uniform(a, (16,)), uniform(b, (16,)))


# ── samplers: shape + dtype + determinism ──────────────────────────────────


def test_uniform_shape_and_dtype():
    k = RNGKey.from_seed(0)
    out = uniform(k, (2, 3), low=-1.0, high=2.0, dtype="fp32")
    assert out.shape == (2, 3)
    assert out.dtype == np.float32
    assert (out >= -1.0).all() and (out < 2.0).all()


def test_uniform_rejects_high_le_low():
    k = RNGKey.from_seed(0)
    with pytest.raises(ValueError, match="high > low"):
        uniform(k, (4,), low=1.0, high=1.0)


def test_normal_mean_and_std_approximately_match():
    k = RNGKey.from_seed(0)
    samples = normal(k, (50_000,), mean=3.0, std=2.0, dtype="fp64")
    assert abs(samples.mean() - 3.0) < 0.05
    assert abs(samples.std() - 2.0) < 0.05


def test_truncated_normal_stays_in_bounds():
    k = RNGKey.from_seed(0)
    samples = truncated_normal(k, (5000,), lower=-1.0, upper=1.0)
    assert (samples >= -1.0).all()
    assert (samples <= 1.0).all()


def test_bernoulli_returns_bool_dtype_by_default():
    k = RNGKey.from_seed(0)
    samples = bernoulli(k, (1024,), p=0.7)
    assert samples.dtype == np.bool_
    # Sample mean should be near 0.7.
    assert abs(samples.astype(np.float64).mean() - 0.7) < 0.05


def test_bernoulli_rejects_invalid_p():
    k = RNGKey.from_seed(0)
    with pytest.raises(ValueError):
        bernoulli(k, (4,), p=1.5)


def test_categorical_returns_indices_in_range():
    k = RNGKey.from_seed(0)
    # Sharply peaked on idx 1 so the test isn't flaky.
    # softmax([0,5,0,0]) puts ~99% mass on class 1.
    logits = np.tile(np.array([0.0, 5.0, 0.0, 0.0]), (256, 1))
    out = categorical(k, logits, axis=-1)
    assert out.shape == (256,)
    assert (out >= 0).all() and (out < 4).all()
    assert (out == 1).mean() > 0.9


def test_multinomial_sums_to_n():
    k = RNGKey.from_seed(0)
    out = multinomial(k, n=100, p=np.array([0.2, 0.3, 0.5]))
    assert out.sum() == 100


def test_randint_range_and_dtype():
    k = RNGKey.from_seed(0)
    out = randint(k, (256,), low=5, high=10, dtype="i64")
    assert out.dtype == np.int64
    assert (out >= 5).all() and (out < 10).all()


def test_permutation_of_int_is_a_valid_perm():
    k = RNGKey.from_seed(0)
    out = permutation(k, 100)
    assert sorted(out.tolist()) == list(range(100))


def test_permutation_of_array_preserves_elements():
    k = RNGKey.from_seed(0)
    a = np.arange(20)
    out = permutation(k, a)
    assert sorted(out.tolist()) == sorted(a.tolist())


def test_gamma_beta_dirichlet_poisson_smoke():
    k = RNGKey.from_seed(0)
    g = gamma(k, (8,), concentration=2.0, rate=1.0)
    assert g.shape == (8,) and (g > 0).all()

    b = beta(k, (8,), alpha=2.0, beta_param=5.0)
    assert b.shape == (8,) and ((b > 0) & (b < 1)).all()

    d = dirichlet(k, np.array([1.0, 2.0, 3.0]), shape=(4,))
    assert d.shape == (4, 3)
    np.testing.assert_allclose(d.sum(axis=-1), 1.0, atol=1e-5)

    p = poisson(k, (32,), rate=4.0, dtype="i32")
    assert p.dtype == np.int32 and (p >= 0).all()


# ── End-to-end determinism scenarios ───────────────────────────────────────


def test_determinism_under_split_then_sample():
    # Two independent runs must produce identical samples if seeded identically.
    def run(seed):
        root = RNGKey.from_seed(seed)
        ka, kb = root.split(2)
        return uniform(ka, (256,)), normal(kb, (256,))

    a1, b1 = run(2026)
    a2, b2 = run(2026)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(b1, b2)


def test_determinism_under_fold_in_resume():
    # Simulates resuming a run at step 100: same key, same fold_in -> same draw.
    base = RNGKey.from_seed(42)

    def step(step_id):
        return uniform(base.fold_in(step_id), (8,))

    first_pass = [step(i) for i in range(100, 105)]
    second_pass = [step(i) for i in range(100, 105)]
    for a, b in zip(first_pass, second_pass):
        np.testing.assert_array_equal(a, b)


def test_per_shard_determinism_via_fold_in():
    # Simulates 4 ranks each folding in their rank index — every rank has a
    # different stream, but each rank is reproducible.
    base = RNGKey.from_seed(7)
    streams = {rank: uniform(base.fold_in(rank), (64,)) for rank in range(4)}
    # All distinct.
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(streams[i], streams[j])
    # Reproducible.
    again = uniform(base.fold_in(2), (64,))
    np.testing.assert_array_equal(streams[2], again)
