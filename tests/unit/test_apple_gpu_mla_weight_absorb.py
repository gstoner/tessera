"""Apple GPU MLA decode with weight absorption — the real bandwidth win
(2026-05-30).

DeepSeek MLA's up-projection weights absorb into the query / output so attention
runs directly against the cached compressed latent ``c_kv`` (shared across all
heads) — per-head K/V are never materialized, and the KV cache stores only
``c_kv [Skv, Dl]`` + the shared ``k_rope [Skv, dr]``.

`tessera_apple_gpu_mla_absorb_decode_f32`:
  q_abs  = q_nope @ Wukᵀ
  s_nope = q_abs @ c_kvᵀ ;  s_rope = rope(q_rope) @ rope(k_rope)ᵀ
  attn   = softmax((s_nope + s_rope)·scale)
  ctx    = attn @ c_kv ;  O = ctx @ Wuv

This is mathematically identical to the explicit-K decoupled-RoPE kernel
(`_apple_gpu_mla_decode_rope`), which is the primary cross-check here. The
incremental-decode test demonstrates KV-cache integration: the cache is just a
growing ``c_kv`` + ``k_rope``. See docs/audit/backend/apple/archive/apple_gpu_tier2_tier3_plan.md.
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


def _ref_absorb(q_nope, q_rope, c_kv, k_rope, Wuk, Wuv, cosQ, sinQ, cosK, sinK,
                style):
    """Direct numpy of the absorbed math. Wuk is [H,Dl,dn], Wuv [H,Dl,dv]."""
    q_nope, q_rope, c_kv, k_rope = (a.astype(np.float64)
                                    for a in (q_nope, q_rope, c_kv, k_rope))
    Wuk, Wuv = Wuk.astype(np.float64), Wuv.astype(np.float64)
    cosQ, sinQ, cosK, sinK = (a.astype(np.float64)
                              for a in (cosQ, sinQ, cosK, sinK))
    B, H, Sq, dn = q_nope.shape
    dr = q_rope.shape[-1]
    Skv, Dl = c_kv.shape[-2], c_kv.shape[-1]
    dv = Wuv.shape[-1]
    scale = 1.0 / math.sqrt(dn + dr)
    qrR = _rope(q_rope, cosQ[None, None], sinQ[None, None], style)
    krR = _rope(k_rope, cosK[None], sinK[None], style)            # [B,Skv,dr]
    O = np.empty((B, H, Sq, dv))
    for b in range(B):
        for h in range(H):
            # Wuk[h] is [Dl,dn]; absorb Wukᵀ into the query: q_nope @ Wuk[h]ᵀ.
            qabs = q_nope[b, h] @ Wuk[h].T                          # [Sq,Dl]
            s = qabs @ c_kv[b].T + qrR[b, h] @ krR[b].T            # [Sq,Skv]
            s = s * scale
            s = s - s.max(-1, keepdims=True)
            e = np.exp(s)
            attn = e / e.sum(-1, keepdims=True)
            ctx = attn @ c_kv[b]                                   # [Sq,Dl]
            O[b, h] = ctx @ Wuv[h]                                 # [Sq,dv]
    return O


def _make(B, H, Sq, Skv, dn, dr, dv, Dl, seed=0, base=10000.0):
    rng = np.random.RandomState(seed)
    f = lambda *s: (rng.randn(*s) * 0.3).astype(np.float32)
    q_nope = f(B, H, Sq, dn)
    q_rope = f(B, H, Sq, dr)
    c_kv = f(B, Skv, Dl)
    k_rope = f(B, Skv, dr)
    Wuk = f(H, Dl, dn)        # K up-proj: K_nope[h] = c_kv @ Wuk[h]
    Wuv = f(H, Dl, dv)        # V up-proj: V[h]      = c_kv @ Wuv[h]
    half = dr // 2
    inv = base ** (-(np.arange(half, dtype=np.float64) * 2.0 / dr))
    cosQ = np.cos(np.arange(Sq)[:, None] * inv[None, :]).astype(np.float32)
    sinQ = np.sin(np.arange(Sq)[:, None] * inv[None, :]).astype(np.float32)
    cosK = np.cos(np.arange(Skv)[:, None] * inv[None, :]).astype(np.float32)
    sinK = np.sin(np.arange(Skv)[:, None] * inv[None, :]).astype(np.float32)
    return q_nope, q_rope, c_kv, k_rope, Wuk, Wuv, cosQ, sinQ, cosK, sinK


_CASES = [
    pytest.param(2, 4, 3, 6, 16, 8, 16, 32, id="base"),
    pytest.param(1, 8, 1, 64, 16, 8, 16, 48, id="decode_step_long_ctx"),
    pytest.param(2, 8, 5, 10, 32, 16, 24, 64, id="deepseek_shaped"),
    pytest.param(1, 1, 4, 4, 8, 4, 8, 16, id="single_head"),
]


@pytest.mark.parametrize("style", ["interleaved", "half"])
@pytest.mark.parametrize("B,H,Sq,Skv,dn,dr,dv,Dl", _CASES)
def test_absorb_matches_numpy(B, H, Sq, Skv, dn, dr, dv, Dl, style):
    qn, qr, ckv, kr, Wuk, Wuv, cQ, sQ, cK, sK = _make(B, H, Sq, Skv, dn, dr, dv, Dl)
    Wuk_t = np.ascontiguousarray(np.swapaxes(Wuk, 1, 2))   # [H,dn,Dl]
    out = R._apple_gpu_mla_absorb_decode(qn, qr, ckv, kr, Wuk_t, Wuv, cQ, sQ,
                                         cK, sK, np, rotation_style=style)
    assert out is not None and out.shape == (B, H, Sq, dv)
    ref = _ref_absorb(qn, qr, ckv, kr, Wuk, Wuv, cQ, sQ, cK, sK, style)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("style", ["interleaved", "half"])
def test_absorb_equals_explicit_kernel(style):
    """Absorption is exact: derive explicit per-head K_nope = c_kv @ Wuk and
    V = c_kv @ Wuv, run the explicit decoupled-RoPE kernel, and require the
    absorbed kernel to match it bit-for-bit (up to fp tolerance)."""
    B, H, Sq, Skv, dn, dr, dv, Dl = 2, 4, 3, 8, 16, 8, 16, 24
    qn, qr, ckv, kr, Wuk, Wuv, cQ, sQ, cK, sK = _make(B, H, Sq, Skv, dn, dr, dv,
                                                      Dl, seed=2)
    # explicit per-head K_nope / V from the latent
    Kn = np.einsum("bsl,hld->bhsd", ckv, Wuk).astype(np.float32)   # [B,H,Skv,dn]
    V = np.einsum("bsl,hld->bhsd", ckv, Wuv).astype(np.float32)    # [B,H,Skv,dv]
    explicit = R._apple_gpu_mla_decode_rope(qn, qr, Kn, kr, V, cQ, sQ, cK, sK,
                                            np, rotation_style=style)
    Wuk_t = np.ascontiguousarray(np.swapaxes(Wuk, 1, 2))
    absorbed = R._apple_gpu_mla_absorb_decode(qn, qr, ckv, kr, Wuk_t, Wuv, cQ,
                                              sQ, cK, sK, np,
                                              rotation_style=style)
    assert explicit is not None and absorbed is not None
    np.testing.assert_allclose(absorbed, explicit, rtol=2e-4, atol=2e-4)


def test_absorb_styles_differ():
    args = _make(1, 2, 4, 6, 8, 4, 8, 16, seed=3)
    Wuk_t = np.ascontiguousarray(np.swapaxes(args[4], 1, 2))
    a = R._apple_gpu_mla_absorb_decode(args[0], args[1], args[2], args[3],
                                       Wuk_t, args[5], *args[6:], np,
                                       rotation_style="interleaved")
    b = R._apple_gpu_mla_absorb_decode(args[0], args[1], args[2], args[3],
                                       Wuk_t, args[5], *args[6:], np,
                                       rotation_style="half")
    assert a is not None and b is not None
    assert not np.allclose(a, b, rtol=1e-3, atol=1e-3)


def test_absorb_incremental_kv_cache():
    """KV-cache integration: the cache is just a growing c_kv + k_rope. Decoding
    a query against the first ``t`` cached tokens must equal a full decode over
    those ``t`` tokens — proving the latent cache is all you need to store."""
    B, H, dn, dr, dv, Dl = 1, 4, 16, 8, 16, 24
    T = 8
    rng = np.random.RandomState(7)
    f = lambda *s: (rng.randn(*s) * 0.3).astype(np.float32)
    q_nope = f(B, H, 1, dn)        # single decode-step query
    q_rope = f(B, H, 1, dr)
    Wuk = f(H, Dl, dn)
    Wuv = f(H, Dl, dv)
    Wuk_t = np.ascontiguousarray(np.swapaxes(Wuk, 1, 2))
    full_ckv = f(B, T, Dl)         # T tokens' worth of cached latent
    full_kr = f(B, T, dr)
    half = dr // 2
    inv = 10000.0 ** (-(np.arange(half, dtype=np.float64) * 2.0 / dr))
    cosQ = np.cos(np.zeros((1, 1)) * inv[None, :]).astype(np.float32)  # pos 0 query
    sinQ = np.sin(np.zeros((1, 1)) * inv[None, :]).astype(np.float32)

    for t in range(1, T + 1):
        ckv = np.ascontiguousarray(full_ckv[:, :t])     # grow the cache
        kr = np.ascontiguousarray(full_kr[:, :t])
        cosK = np.cos(np.arange(t)[:, None] * inv[None, :]).astype(np.float32)
        sinK = np.sin(np.arange(t)[:, None] * inv[None, :]).astype(np.float32)
        out = R._apple_gpu_mla_absorb_decode(q_nope, q_rope, ckv, kr, Wuk_t,
                                             Wuv, cosQ, sinQ, cosK, sinK, np)
        ref = _ref_absorb(q_nope, q_rope, ckv, kr, Wuk, Wuv, cosQ, sinQ, cosK,
                          sinK, "interleaved")
        assert out is not None and out.shape == (B, H, 1, dv)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_absorb_cache_is_smaller_than_explicit_kv():
    """The headline win: the latent cache (c_kv + k_rope, shared across heads)
    is far smaller than explicit per-head K/V for realistic DeepSeek dims."""
    H, dn, dr, dv, Dl = 128, 128, 64, 128, 512
    latent_per_tok = Dl + dr                       # shared across all heads
    explicit_per_tok = H * (dn + dr + dv)          # per-head K_nope + k_rope + V
    assert latent_per_tok < explicit_per_tok
    assert explicit_per_tok / latent_per_tok > 8.0


def test_absorb_symbol_exported():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_mla_absorb_decode_f32")
    assert R._apple_gpu_mla_absorb_decode_f32() is not None
