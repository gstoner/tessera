"""Apple GPU MLA weight-absorption decode — native f16 / bf16 (2026-05-30).

`tessera_apple_gpu_mla_absorb_decode_{f16,bf16}` extend the absorbed decode to
half precision: f16 carries f16 I/O on-GPU (half the cache-read bandwidth) with
fp32 accumulation; bf16 runs via a host fp32 round-trip. The Python dispatcher
routes by the dtype of ``q_nope`` and the six tensor inputs share that dtype;
cos/sin tables stay f32. Validated against the f32 path / numpy reference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tessera import runtime as R


def _rope(x, cos, sin, style):
    dr = x.shape[-1]
    half = dr // 2
    out = np.empty_like(x)
    if style == "interleaved":
        a, b = x[..., 0::2], x[..., 1::2]
        out[..., 0::2] = a * cos - b * sin
        out[..., 1::2] = a * sin + b * cos
    else:
        a, b = x[..., :half], x[..., half:]
        out[..., :half] = a * cos - b * sin
        out[..., half:] = b * cos + a * sin
    return out


def _ref(q_nope, q_rope, c_kv, k_rope, Wuk_t, Wuv, cosQ, sinQ, cosK, sinK, style):
    q_nope, q_rope, c_kv, k_rope = (a.astype(np.float64)
                                    for a in (q_nope, q_rope, c_kv, k_rope))
    Wuk_t, Wuv = Wuk_t.astype(np.float64), Wuv.astype(np.float64)
    B, H, Sq, dn = q_nope.shape
    dr = q_rope.shape[-1]
    Skv, Dl = c_kv.shape[-2], c_kv.shape[-1]
    dv = Wuv.shape[-1]
    scale = 1.0 / math.sqrt(dn + dr)
    qrR = _rope(q_rope, cosQ[None, None].astype(np.float64),
                sinQ[None, None].astype(np.float64), style)
    krR = _rope(k_rope, cosK[None].astype(np.float64),
                sinK[None].astype(np.float64), style)
    O = np.empty((B, H, Sq, dv))
    for b in range(B):
        for h in range(H):
            qabs = q_nope[b, h] @ Wuk_t[h]
            s = qabs @ c_kv[b].T + qrR[b, h] @ krR[b].T
            s = s * scale
            s = s - s.max(-1, keepdims=True)
            e = np.exp(s)
            O[b, h] = (e / e.sum(-1, keepdims=True)) @ (c_kv[b] @ Wuv[h])
    return O


def _make(B=2, H=4, Sq=2, Skv=8, dn=16, dr=8, dv=16, Dl=32, seed=0):
    rng = np.random.RandomState(seed)
    f = lambda *s: (rng.randn(*s) * 0.3).astype(np.float32)
    q_nope, q_rope = f(B, H, Sq, dn), f(B, H, Sq, dr)
    c_kv, k_rope = f(B, Skv, Dl), f(B, Skv, dr)
    Wuk_t, Wuv = f(H, dn, Dl), f(H, Dl, dv)
    half = dr // 2
    inv = 10000.0 ** (-(np.arange(half) * 2.0 / dr))
    cosQ = np.cos(np.arange(Sq)[:, None] * inv[None, :]).astype(np.float32)
    sinQ = np.sin(np.arange(Sq)[:, None] * inv[None, :]).astype(np.float32)
    cosK = np.cos(np.arange(Skv)[:, None] * inv[None, :]).astype(np.float32)
    sinK = np.sin(np.arange(Skv)[:, None] * inv[None, :]).astype(np.float32)
    return q_nope, q_rope, c_kv, k_rope, Wuk_t, Wuv, cosQ, sinQ, cosK, sinK


@pytest.mark.parametrize("style", ["interleaved", "half"])
def test_absorb_f16_native(style):
    d = _make(seed=1)
    ref = _ref(*d, style)
    d16 = [d[i].astype(np.float16) for i in range(6)] + list(d[6:])
    out = R._apple_gpu_mla_absorb_decode(*d16, np, rotation_style=style)
    assert out is not None and out.dtype == np.float16
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("style", ["interleaved", "half"])
def test_absorb_bf16(style):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    bf16 = ml_dtypes.bfloat16
    d = _make(seed=2)
    ref = _ref(*d, style)
    dbf = [d[i].astype(bf16) for i in range(6)] + list(d[6:])
    out = R._apple_gpu_mla_absorb_decode(*dbf, np, rotation_style=style)
    assert out is not None and out.dtype == bf16
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=6e-2, atol=6e-2)


def test_absorb_f16_close_to_f32_path():
    """f16 result tracks the f32 GPU result (both fp32-accumulated)."""
    d = _make(seed=3)
    f32 = R._apple_gpu_mla_absorb_decode(*d, np)
    d16 = [d[i].astype(np.float16) for i in range(6)] + list(d[6:])
    f16 = R._apple_gpu_mla_absorb_decode(*d16, np)
    assert f32 is not None and f16 is not None
    np.testing.assert_allclose(f16.astype(np.float32), f32, rtol=3e-2, atol=3e-2)


def test_absorb_dtype_symbols_exported():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_mla_absorb_decode_f16")
    assert hasattr(rt, "tessera_apple_gpu_mla_absorb_decode_bf16")
    assert R._apple_gpu_mla_absorb_decode_half("f16") is not None
    assert R._apple_gpu_mla_absorb_decode_half("bf16") is not None
