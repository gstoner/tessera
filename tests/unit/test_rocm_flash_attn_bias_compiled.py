"""ROCm (gfx1151) additive-attention-bias flash lane.

The compiler-generated WMMA FA-2 forward kernel (`generate-wmma-flash-attn-kernel`)
gains an optional additive `attn_bias`: ``O = softmax(scale·Q@K^T + attn_bias)·V``.
The bias is a trailing f32 `[bh,Sq,Sk]` memref runtime arg, host-broadcast from the
caller's `(B,Sq,Sk)` / `(1,Sq,Sk)` / `(Sq,Sk)` bias, added to the scaled score after
soft-cap and before masking. This is the DFlash substrate. Validated on real
gfx1151 vs a numpy reference (f16 storage → ~1e-2 tolerance).
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest


def _rocm_or_skip():
    from tessera import runtime as rt
    if not (shutil.which("hipcc") or os.path.exists("/opt/rocm/bin/hipcc")):
        pytest.skip("no hipcc")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no live gfx1151")
    return rt


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def _ref(q, k, v, scale, bias, causal):
    qf, kf, vf = (a.astype(np.float32) for a in (q, k, v))
    s = scale * (qf @ kf.transpose(0, 2, 1))
    if bias is not None:
        s = s + bias
    if causal:
        n = s.shape[-1]
        s = np.where(np.triu(np.ones((n, n), bool), 1), -1e30, s)
    return _softmax(s) @ vf


def test_rocm_flash_attn_bias_matches_numpy():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(0)
    f16 = np.float16

    def mk(bh, s, d):
        return rng.standard_normal((bh, s, d)).astype(f16)

    # (bh, S, D, causal, bias-shape) — bias trailing dims are (S, S).
    cases = [
        (2, 16, 16, False, (2, 16, 16)),   # per-row bias
        (2, 16, 16, True, (1, 16, 16)),    # broadcast over batch, causal
        (4, 32, 16, False, (32, 32)),      # rank-2 (Sq,Sk) broadcast
        (2, 48, 32, True, (2, 48, 48)),    # causal + per-row, D=32
        (3, 64, 16, False, (1, 64, 64)),
    ]
    for bh, s, d, causal, bshape in cases:
        q, k, v = mk(bh, s, d), mk(bh, s, d), mk(bh, s, d)
        bias = rng.standard_normal(bshape).astype(np.float32)
        scale = 1.0 / np.sqrt(d)
        out = rt._rocm_flash_attn(q, k, v, scale=scale, causal=causal,
                                  attn_bias=bias)
        ref = _ref(q, k, v, scale, np.broadcast_to(bias, (bh, s, s)), causal)
        np.testing.assert_allclose(out, ref, rtol=0, atol=2e-2)


def test_rocm_flash_attn_bias_gqa():
    # GQA (H query heads, G<H kv heads) + additive bias: the bias arg index must
    # account for the two trailing gqa runtime args (heads, kv_ratio).
    rt = _rocm_or_skip()
    rng = np.random.default_rng(1)
    f16 = np.float16
    b, hq, hkv, s, d = 1, 4, 2, 32, 16
    q = rng.standard_normal((b, hq, s, d)).astype(f16)
    k = rng.standard_normal((b, hkv, s, d)).astype(f16)
    v = rng.standard_normal((b, hkv, s, d)).astype(f16)
    bias = rng.standard_normal((b * hq, s, s)).astype(np.float32)
    scale = 1.0 / np.sqrt(d)
    out = rt._rocm_flash_attn(q, k, v, scale=scale, causal=True, attn_bias=bias)
    # numpy reference: expand kv heads to query heads, fold to [bh,S,D].
    rep = hq // hkv
    kf = np.repeat(k, rep, axis=1).reshape(b * hq, s, d)
    vf = np.repeat(v, rep, axis=1).reshape(b * hq, s, d)
    qf = q.reshape(b * hq, s, d)
    ref = _ref(qf, kf, vf, scale, bias, True).reshape(b, hq, s, d)
    np.testing.assert_allclose(out, ref, rtol=0, atol=2e-2)


def test_rocm_flash_attn_no_bias_unchanged():
    # The no-bias path (3 operands) must still run the plain kernel.
    rt = _rocm_or_skip()
    rng = np.random.default_rng(2)
    f16 = np.float16
    q, k, v = (rng.standard_normal((2, 16, 16)).astype(f16) for _ in range(3))
    scale = 1.0 / np.sqrt(16)
    out = rt._rocm_flash_attn(q, k, v, scale=scale, causal=False)
    ref = _ref(q, k, v, scale, None, False)
    np.testing.assert_allclose(out, ref, rtol=0, atol=2e-2)
