"""Apple GPU — native f16/bf16 for the two secondary MLA decode kernels
(2026-05-30).

Extends the dtype matrix to the compressed-KV `mla_decode` and the explicit-K
decoupled-RoPE `mla_decode_rope` kernels (the absorbed kernel already had it).
f16 runs native f16 I/O on-GPU (fp32 accumulation); bf16 runs via a host fp32
round-trip. Validated against numpy / the f32 path.
"""

from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

from tessera import runtime as R


# --------------------------------------------------------------------------
# compressed-KV mla_decode
# --------------------------------------------------------------------------
def _ref_mla_decode(X, Wdkv, Wuk, Wuv, Q):
    X = X.astype(np.float64)
    c = X @ Wdkv.astype(np.float64)
    K = c @ Wuk.astype(np.float64)
    V = c @ Wuv.astype(np.float64)
    Q = Q.astype(np.float64)
    B, S_q, D_h = Q.shape
    scale = 1.0 / math.sqrt(D_h)
    O = np.empty_like(Q)
    for b in range(B):
        s = (Q[b] @ K.T) * scale
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s)
        O[b] = (e / e.sum(-1, keepdims=True)) @ V
    return O


def _make_decode(B=2, S_kv=6, D_x=8, D_lat=16, S_q=3, D_h=8, seed=0):
    rng = np.random.RandomState(seed)
    f = lambda *s: (rng.randn(*s) * 0.3).astype(np.float32)
    return f(S_kv, D_x), f(D_x, D_lat), f(D_lat, D_h), f(D_lat, D_h), f(B, S_q, D_h)


def _call_decode(suffix, view_dtype, X, Wdkv, Wuk, Wuv, Q):
    sym = R._apple_gpu_mla_decode_sym(suffix)
    assert sym is not None
    B, S_q, D_h = Q.shape
    S_kv, D_x = X.shape
    D_lat = Wdkv.shape[1]
    if suffix == "f32":
        p = lambda a: np.ascontiguousarray(a, np.float32).ctypes.data_as(
            ctypes.POINTER(ctypes.c_float))
        O = np.zeros((B, S_q, D_h), np.float32)
        op = O.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    else:
        p = lambda a: np.ascontiguousarray(a).view(np.uint16).ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint16))
        O = np.zeros((B, S_q, D_h), np.uint16)
        op = O.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    sym(p(X), p(Wdkv), p(Wuk), p(Wuv), p(Q), op, B, S_kv, D_x, D_lat, S_q, D_h)
    return O if suffix == "f32" else O.view(view_dtype)


def test_mla_decode_f16():
    d = _make_decode(seed=1)
    ref = _ref_mla_decode(*d)
    d16 = [a.astype(np.float16) for a in d]
    out = _call_decode("f16", np.float16, *d16)
    assert out.dtype == np.float16
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=3e-2, atol=3e-2)


def test_mla_decode_bf16():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    bf16 = ml_dtypes.bfloat16
    d = _make_decode(seed=2)
    ref = _ref_mla_decode(*d)
    dbf = [a.astype(bf16) for a in d]
    out = _call_decode("bf16", bf16, *dbf)
    assert out.dtype == bf16
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=6e-2, atol=6e-2)


def test_mla_decode_dtype_symbols_exported():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_mla_decode_f16")
    assert hasattr(rt, "tessera_apple_gpu_mla_decode_bf16")


# --------------------------------------------------------------------------
# decoupled-RoPE mla_decode_rope
# --------------------------------------------------------------------------
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


def _ref_rope(Qn, Qr, Kn, Kr, V, cosQ, sinQ, cosK, sinK, style):
    Qn, Qr, Kn, Kr, V = (a.astype(np.float64) for a in (Qn, Qr, Kn, Kr, V))
    B, H, Sq, dn = Qn.shape
    dr = Qr.shape[-1]
    Skv, dv = Kn.shape[-2], V.shape[-1]
    scale = 1.0 / math.sqrt(dn + dr)
    QrR = _rope(Qr, cosQ[None, None].astype(np.float64),
                sinQ[None, None].astype(np.float64), style)
    KrR = _rope(Kr, cosK[None].astype(np.float64),
                sinK[None].astype(np.float64), style)
    Qf = np.concatenate([Qn, QrR], -1)
    Kf = np.concatenate([Kn, np.broadcast_to(KrR[:, None], (B, H, Skv, dr))], -1)
    O = np.empty((B, H, Sq, dv))
    for b in range(B):
        for h in range(H):
            s = (Qf[b, h] @ Kf[b, h].T) * scale
            s = s - s.max(-1, keepdims=True)
            e = np.exp(s)
            O[b, h] = (e / e.sum(-1, keepdims=True)) @ V[b, h]
    return O


def _make_rope(B=2, H=4, Sq=2, Skv=6, dn=16, dr=8, dv=16, seed=0):
    rng = np.random.RandomState(seed)
    f = lambda *s: (rng.randn(*s) * 0.3).astype(np.float32)
    Qn, Qr, Kn, Kr, V = (f(B, H, Sq, dn), f(B, H, Sq, dr), f(B, H, Skv, dn),
                         f(B, Skv, dr), f(B, H, Skv, dv))
    half = dr // 2
    inv = 10000.0 ** (-(np.arange(half) * 2.0 / dr))
    cosQ = np.cos(np.arange(Sq)[:, None] * inv[None, :]).astype(np.float32)
    sinQ = np.sin(np.arange(Sq)[:, None] * inv[None, :]).astype(np.float32)
    cosK = np.cos(np.arange(Skv)[:, None] * inv[None, :]).astype(np.float32)
    sinK = np.sin(np.arange(Skv)[:, None] * inv[None, :]).astype(np.float32)
    return Qn, Qr, Kn, Kr, V, cosQ, sinQ, cosK, sinK


@pytest.mark.parametrize("style", ["interleaved", "half"])
def test_mla_rope_f16(style):
    d = _make_rope(seed=1)
    ref = _ref_rope(*d, style)
    d16 = [d[i].astype(np.float16) for i in range(5)] + list(d[5:])
    out = R._apple_gpu_mla_decode_rope(*d16, np, rotation_style=style)
    assert out is not None and out.dtype == np.float16
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("style", ["interleaved", "half"])
def test_mla_rope_bf16(style):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    bf16 = ml_dtypes.bfloat16
    d = _make_rope(seed=2)
    ref = _ref_rope(*d, style)
    dbf = [d[i].astype(bf16) for i in range(5)] + list(d[5:])
    out = R._apple_gpu_mla_decode_rope(*dbf, np, rotation_style=style)
    assert out is not None and out.dtype == bf16
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=6e-2, atol=6e-2)


def test_mla_rope_dtype_symbols_exported():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_mla_decode_rope_f16")
    assert hasattr(rt, "tessera_apple_gpu_mla_decode_rope_bf16")
    assert R._apple_gpu_mla_decode_rope_half("f16") is not None
    assert R._apple_gpu_mla_decode_rope_half("bf16") is not None
