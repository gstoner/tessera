"""Apple GPU R3 — resident decode loop (logits + activations resident across
steps; only the token id reads back).

`tessera.cache.ResidentMLADecoder` runs each per-token decode step entirely
on-device in one command buffer, keeps the model weights resident across steps
(uploaded once), and reads back only the sampled token id. These tests validate
a multi-step loop against a numpy reference and check the residency invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as R
from tessera import rng as TR
from tessera.cache import ResidentMLADecoder


def _softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def _rmsnorm(x, gamma, eps=1e-6):
    d = x.astype(np.float64)
    return d / np.sqrt((d * d).mean(-1, keepdims=True) + eps) * gamma.astype(np.float64)


def _ref_step(x, k_t, v, gamma, w_logit, noise, inv_temp):
    H, D = x.shape
    qn = _rmsnorm(x, gamma)
    scores = np.einsum("hd,hdk->hk", qn, k_t.astype(np.float64))
    attn = _softmax(scores)
    ctx = np.einsum("hk,hkd->hd", attn, v.astype(np.float64))
    logits = ctx.reshape(1, H * D) @ w_logit.astype(np.float64)
    return int(np.argmax(logits * inv_temp + noise.astype(np.float64), axis=-1)[0])


def _make(H=2, D=8, V=40, seed=0):
    rng = np.random.RandomState(seed)
    gamma = rng.randn(D).astype(np.float32)
    w_logit = (rng.randn(H * D, V) * 0.2).astype(np.float32)
    return ResidentMLADecoder(num_heads=H, head_dim=D, vocab=V,
                              rmsnorm_gamma=gamma, w_logit=w_logit), gamma, w_logit, rng


def test_single_step_greedy_matches_numpy():
    dec, gamma, w_logit, rng = _make(seed=1)
    H, D, V = dec.H, dec.D, dec.V
    Skv = 5
    x = rng.randn(H, D).astype(np.float32)
    k_t = rng.randn(H, D, Skv).astype(np.float32)
    v = rng.randn(H, Skv, D).astype(np.float32)
    tok = dec.step(x, k_t, v, greedy=True)
    ref = _ref_step(x, k_t, v, gamma, w_logit, np.zeros((1, V), np.float32), 1.0)
    assert tok == ref
    dec.free()


def test_multistep_growing_cache_matches_numpy():
    """A decode loop where the KV window grows each step; every sampled token
    matches the numpy reference, and weights stay uploaded once."""
    dec, gamma, w_logit, rng = _make(seed=2)
    H, D, V = dec.H, dec.D, dec.V
    for step in range(8):
        Skv = 2 + step
        x = rng.randn(H, D).astype(np.float32)
        k_t = rng.randn(H, D, Skv).astype(np.float32)
        v = rng.randn(H, Skv, D).astype(np.float32)
        key = TR.RNGKey.from_seed(100 + step)
        noise = R._gumbel_noise_from_key((1, V), key, np)
        tok = dec.step(x, k_t, v, key=key, temperature=0.8)
        ref = _ref_step(x, k_t, v, gamma, w_logit, noise, 1.0 / 0.8)
        assert tok == ref, f"step {step}: {tok} != {ref}"
    assert dec.weight_uploads == 1  # uploaded once across the whole loop
    dec.free()


def test_greedy_temperature_zero():
    dec, gamma, w_logit, rng = _make(seed=3)
    H, D, V = dec.H, dec.D, dec.V
    x = rng.randn(H, D).astype(np.float32)
    k_t = rng.randn(H, D, 4).astype(np.float32)
    v = rng.randn(H, 4, D).astype(np.float32)
    t0 = dec.step(x, k_t, v, temperature=0.0)
    tg = dec.step(x, k_t, v, greedy=True)
    ref = _ref_step(x, k_t, v, gamma, w_logit, np.zeros((1, V), np.float32), 1.0)
    assert t0 == ref and tg == ref
    dec.free()


def test_reproducible_same_key():
    dec, *_ = _make(seed=4)
    H, D = dec.H, dec.D
    rng = np.random.RandomState(9)
    x = rng.randn(H, D).astype(np.float32)
    k_t = rng.randn(H, D, 6).astype(np.float32)
    v = rng.randn(H, 6, D).astype(np.float32)
    key = TR.RNGKey.from_seed(5)
    a = dec.step(x, k_t, v, key=key)
    b = dec.step(x, k_t, v, key=key)
    assert a == b
    dec.free()


def test_weight_validation():
    rng = np.random.RandomState(0)
    with pytest.raises(ValueError):
        ResidentMLADecoder(num_heads=2, head_dim=8, vocab=40,
                           rmsnorm_gamma=rng.randn(7).astype(np.float32),  # wrong
                           w_logit=rng.randn(16, 40).astype(np.float32))
    with pytest.raises(ValueError):
        ResidentMLADecoder(num_heads=2, head_dim=8, vocab=40,
                           rmsnorm_gamma=rng.randn(8).astype(np.float32),
                           w_logit=rng.randn(10, 40).astype(np.float32))  # wrong


def test_token_ids_in_range():
    dec, *_ = _make(H=4, D=16, V=128, seed=6)
    H, D, V = dec.H, dec.D, dec.V
    rng = np.random.RandomState(11)
    for step in range(5):
        Skv = 3 + step
        tok = dec.step(rng.randn(H, D).astype(np.float32),
                       rng.randn(H, D, Skv).astype(np.float32),
                       rng.randn(H, Skv, D).astype(np.float32),
                       key=TR.RNGKey.from_seed(step))
        assert 0 <= tok < V
    dec.free()
