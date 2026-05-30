"""Apple GPU Gumbel-max inference sampler (2026-05-30).

`tessera_apple_gpu_gumbel_argmax_f32` draws one token per row via
``argmax(logits/T + g)`` with Gumbel noise ``g = -log(-log(u))`` supplied from
the canonical Philox stream — an exact draw from ``softmax(logits/T)`` with the
per-row vocab argmax on-GPU. Reproducible (same key ⇒ same tokens), #18-safe
(no on-GPU RNG), and a throughput win for batched sampling of concurrent
sequences. Validated for distribution correctness, greedy/top-k/top-p, and
determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as R
from tessera import rng as TR


def _softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def test_gumbel_symbol_exported():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_gumbel_argmax_f32")
    assert R._apple_gpu_gumbel_argmax_f32() is not None


def test_greedy_equals_argmax():
    rng = np.random.RandomState(0)
    logits = rng.randn(4, 50).astype(np.float32)
    ids = R._apple_gpu_gumbel_sample(logits, np, greedy=True)
    assert ids.shape == (4,)
    np.testing.assert_array_equal(ids, np.argmax(logits, axis=-1))


def test_temperature_zero_is_greedy():
    rng = np.random.RandomState(1)
    logits = rng.randn(3, 40).astype(np.float32)
    ids = R._apple_gpu_gumbel_sample(logits, np, temperature=0.0)
    np.testing.assert_array_equal(ids, np.argmax(logits, axis=-1))


def test_distribution_matches_softmax():
    """Empirical sample frequencies converge to softmax(logits/T). One batched
    call: the same logits tiled to [N, V] with N independent Gumbel draws."""
    rng = np.random.RandomState(2)
    V = 6
    logits = (rng.randn(V) * 1.2).astype(np.float32)
    T = 1.0
    probs = _softmax(logits / T)
    N = 20000
    tiled = np.broadcast_to(logits, (N, V)).astype(np.float32)
    ids = R._apple_gpu_gumbel_sample(tiled, np, key=TR.RNGKey.from_seed(123),
                                     temperature=T)
    freq = np.bincount(ids, minlength=V) / N
    # ~0.7% std at N=20k; allow generous slack
    np.testing.assert_allclose(freq, probs, atol=0.02)


def test_reproducible_same_key():
    rng = np.random.RandomState(3)
    logits = rng.randn(8, 100).astype(np.float32)
    key = TR.RNGKey.from_seed(7)
    a = R._apple_gpu_gumbel_sample(logits, np, key=key, temperature=0.8)
    b = R._apple_gpu_gumbel_sample(logits, np, key=key, temperature=0.8)
    np.testing.assert_array_equal(a, b)


def test_different_keys_differ():
    rng = np.random.RandomState(4)
    logits = (rng.randn(16, 200) * 0.5).astype(np.float32)  # low-confidence
    a = R._apple_gpu_gumbel_sample(logits, np, key=TR.RNGKey.from_seed(1))
    b = R._apple_gpu_gumbel_sample(logits, np, key=TR.RNGKey.from_seed(2))
    assert not np.array_equal(a, b)


def test_top_k_restricts_candidates():
    rng = np.random.RandomState(5)
    V, k = 100, 5
    logits = (rng.randn(32, V) * 2.0).astype(np.float32)
    topk_sets = [set(np.argsort(-row)[:k].tolist()) for row in logits]
    key = TR.RNGKey.from_seed(9)
    for trial in range(20):
        ids = R._apple_gpu_gumbel_sample(logits, np, key=key.fold_in(trial),
                                         temperature=1.0, top_k=k)
        for r, tok in enumerate(ids):
            assert int(tok) in topk_sets[r]


def test_top_p_restricts_candidates():
    rng = np.random.RandomState(6)
    V = 50
    logits = (rng.randn(8, V) * 2.0).astype(np.float32)
    p = 0.9
    # reference nucleus sets
    nucleus = []
    for row in logits:
        order = np.argsort(-row)
        probs = _softmax(row[order])
        cum = np.cumsum(probs)
        keep_n = int(np.searchsorted(cum, p) + 1)
        nucleus.append(set(order[:keep_n].tolist()))
    key = TR.RNGKey.from_seed(11)
    for trial in range(20):
        ids = R._apple_gpu_gumbel_sample(logits, np, key=key.fold_in(trial),
                                         temperature=1.0, top_p=p)
        for r, tok in enumerate(ids):
            assert int(tok) in nucleus[r]


def test_batched_sampling_shapes():
    """A [B, V] batch (e.g. concurrent decode) samples B tokens in one call."""
    rng = np.random.RandomState(7)
    B, V = 64, 4096
    logits = rng.randn(B, V).astype(np.float32)
    ids = R._apple_gpu_gumbel_sample(logits, np, key=TR.RNGKey.from_seed(0),
                                     temperature=1.0)
    assert ids.shape == (B,)
    assert ids.dtype == np.int64
    assert ids.min() >= 0 and ids.max() < V


def test_single_row_scalar_out():
    rng = np.random.RandomState(8)
    logits = rng.randn(32).astype(np.float32)  # [V] -> scalar id
    idx = R._apple_gpu_gumbel_sample(logits, np, greedy=True)
    assert idx.shape == ()
    assert int(idx) == int(np.argmax(logits))
