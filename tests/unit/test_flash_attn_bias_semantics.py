"""Regression tests for attn_bias semantics (PR #67 Codex review fixes).

Locks three behaviors the review flagged:
  1. causal + attn_bias applies BOTH masks (not bias-only).
  2. flash_attn VJP arity: kwarg bias -> 3 cotangents (tape records only Q/K/V);
     positional bias -> 4 cotangents (bias is a recorded input).
  3. broadcast (1, Sq, Sk) bias is handled (numpy-broadcast on the reference
     path; the Apple lowering rejects it so it never hits the (B,Sq,Sk) kernel).
"""
import sys

import numpy as np
import pytest

from tessera import ops
from tessera.autodiff.vjp import _VJPS

try:
    from tessera._apple_gpu_dispatch import apple_gpu_available
except Exception:  # pragma: no cover
    apple_gpu_available = lambda: False  # noqa: E731

DARWIN = sys.platform == "darwin"


def _ref(Q, K, V, bias=None, causal=False):
    B, Sq, D = Q.shape
    Sk = K.shape[1]
    s = np.einsum("bqd,bkd->bqk", Q, K) * (D ** -0.5)
    if bias is not None:
        s = s + np.asarray(bias)            # numpy broadcasts (1,Sq,Sk)
    if causal:
        off = max(Sk - Sq, 0)
        m = np.triu(np.ones((Sq, Sk), bool), k=1 + off)
        s = np.where(m, -np.inf, s)
    s = s - s.max(-1, keepdims=True)
    a = np.exp(s); a /= a.sum(-1, keepdims=True)
    return np.einsum("bqk,bkd->bqd", a, V)


# ── 1. causal + bias (eager / CPU reference path; always runs) ──────────────

def test_eager_causal_plus_bias_applies_both():
    rng = np.random.default_rng(1)
    B, Sq, Sk, D = 2, 6, 6, 16
    Q, K, V = (rng.standard_normal((B, Sq, D)).astype(np.float32) for _ in range(3))
    bias = rng.standard_normal((B, Sq, Sk)).astype(np.float32)
    got = ops.flash_attn(Q, K, V, attn_bias=bias, causal=True)
    assert np.allclose(np.asarray(got), _ref(Q, K, V, bias, causal=True), rtol=1e-5, atol=1e-5)
    # and bias-only differs from bias+causal (proves causal isn't dropped)
    assert not np.allclose(np.asarray(got), _ref(Q, K, V, bias, causal=False))


# ── 2. VJP arity contract ───────────────────────────────────────────────────

def test_vjp_arity_kwarg_bias_returns_three():
    rng = np.random.default_rng(2)
    Q, K, V = (rng.standard_normal((2, 4, 8)) for _ in range(3))
    bias = rng.standard_normal((2, 4, 4))
    dout = rng.standard_normal((2, 4, 8))
    g_kw = _VJPS["flash_attn"](dout, Q, K, V, attn_bias=bias)
    g_none = _VJPS["flash_attn"](dout, Q, K, V)
    g_pos = _VJPS["flash_attn"](dout, Q, K, V, bias)
    assert len(g_kw) == 3 and len(g_none) == 3      # tape records only Q/K/V
    assert len(g_pos) == 4                          # positional bias -> dbias
    # kwarg bias still affects the recompute (dQ depends on the bias-shifted softmax)
    assert not np.allclose(g_kw[0], g_none[0])


def test_vjp_positional_bias_gradient_matches_finite_diff():
    rng = np.random.default_rng(3)
    Q, K, V = (rng.standard_normal((1, 4, 6)).astype(np.float64) for _ in range(3))
    bias = rng.standard_normal((1, 4, 4)).astype(np.float64)
    dout = rng.standard_normal((1, 4, 6)).astype(np.float64)
    dbias = _VJPS["flash_attn"](dout, Q, K, V, bias, scale=6 ** -0.5)[3]
    eps = 1e-6
    fd = np.zeros_like(bias)
    for idx in np.ndindex(bias.shape):
        bp = bias.copy(); bp[idx] += eps
        bm = bias.copy(); bm[idx] -= eps
        fd[idx] = ((_ref(Q, K, V, bp) - _ref(Q, K, V, bm)) * dout).sum() / (2 * eps)
    assert np.abs(np.asarray(dbias) - fd).max() < 1e-7


# ── 3. broadcast (1, Sq, Sk) bias (eager numpy-broadcast; always runs) ───────

def test_eager_broadcast_bias():
    rng = np.random.default_rng(4)
    B, Sq, Sk, D = 3, 5, 5, 16
    Q, K, V = (rng.standard_normal((B, Sq, D)).astype(np.float32) for _ in range(3))
    bias1 = rng.standard_normal((1, Sq, Sk)).astype(np.float32)
    got = ops.flash_attn(Q, K, V, attn_bias=bias1)
    assert np.allclose(np.asarray(got), _ref(Q, K, V, bias1), rtol=1e-5, atol=1e-5)


# ── Apple GPU lane (Darwin-gated): causal+bias and broadcast bias ───────────

@pytest.mark.skipif(not (DARWIN and apple_gpu_available()), reason="Metal device required")
def test_gpu_dispatch_causal_plus_bias_and_broadcast():
    from tessera.runtime import _apple_gpu_dispatch_flash_attn

    rng = np.random.default_rng(5)
    B, Sq, Sk, D = 2, 8, 8, 16
    Q, K, V = (np.ascontiguousarray(rng.standard_normal((B, Sq, D)).astype(np.float32)) for _ in range(3))
    bias = rng.standard_normal((B, Sq, Sk)).astype(np.float32)
    # causal + bias routes to the reference inside the bias symbol; must match numpy.
    got = _apple_gpu_dispatch_flash_attn(
        "tessera.flash_attn", [Q, K, V, bias], {"causal": True}, np)
    assert np.allclose(np.asarray(got), _ref(Q, K, V, bias, causal=True), rtol=1e-3, atol=1e-3)
    # broadcast (1,Sq,Sk) bias: out of the GPU envelope -> reference fallback, still correct.
    bias1 = rng.standard_normal((1, Sq, Sk)).astype(np.float32)
    got_b = _apple_gpu_dispatch_flash_attn(
        "tessera.flash_attn", [Q, K, V, bias1], {}, np)
    assert np.allclose(np.asarray(got_b), _ref(Q, K, V, bias1), rtol=1e-3, atol=1e-3)
