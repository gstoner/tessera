"""Apple GPU MLA decode with decoupled RoPE — explicit per-head K (2026-05-30).

DeepSeek-style MLA splits each head's query/key into a no-position-encoding part
(dim ``dn``) and a RoPE-carrying part (dim ``dr``); the key RoPE part is shared
across heads. `tessera_apple_gpu_mla_decode_rope_f32` applies RoPE to the rope
parts (switchable interleaved/half convention), concatenates ``[nope ; rope]``
per head, and runs the resulting standard MHA on-GPU via the fused ``bsmm``
kernel. Validated against a numpy reference. See docs/apple_gpu_tier2_tier3_plan.md.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tessera import runtime as R


def _rope(x, cos, sin, style):
    """x [..., dr]; cos/sin broadcast to [..., dr/2]."""
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


def _ref_mla_rope(Qn, Qr, Kn, Kr, V, cosQ, sinQ, cosK, sinK, style):
    Qn, Qr, Kn, Kr, V = (a.astype(np.float64) for a in (Qn, Qr, Kn, Kr, V))
    cosQ, sinQ, cosK, sinK = (a.astype(np.float64)
                              for a in (cosQ, sinQ, cosK, sinK))
    B, H, Sq, dn = Qn.shape
    dr = Qr.shape[-1]
    Skv, dv = Kn.shape[-2], V.shape[-1]
    dh = dn + dr
    scale = 1.0 / math.sqrt(dh)
    QrR = _rope(Qr, cosQ[None, None], sinQ[None, None], style)      # [B,H,Sq,dr]
    KrR = _rope(Kr, cosK[None], sinK[None], style)                  # [B,Skv,dr]
    Qfull = np.concatenate([Qn, QrR], axis=-1)                      # [B,H,Sq,dh]
    KrR_b = np.broadcast_to(KrR[:, None], (B, H, Skv, dr))
    Kfull = np.concatenate([Kn, KrR_b], axis=-1)                    # [B,H,Skv,dh]
    O = np.empty((B, H, Sq, dv))
    for b in range(B):
        for h in range(H):
            s = (Qfull[b, h] @ Kfull[b, h].T) * scale
            s = s - s.max(-1, keepdims=True)
            e = np.exp(s)
            O[b, h] = (e / e.sum(-1, keepdims=True)) @ V[b, h]
    return O


def _make(B, H, Sq, Skv, dn, dr, dv, seed=0, base=10000.0):
    rng = np.random.RandomState(seed)
    f = lambda *s: (rng.randn(*s) * 0.3).astype(np.float32)
    Qn = f(B, H, Sq, dn)
    Qr = f(B, H, Sq, dr)
    Kn = f(B, H, Skv, dn)
    Kr = f(B, Skv, dr)            # shared across heads
    V = f(B, H, Skv, dv)
    half = dr // 2
    inv = base ** (-(np.arange(half, dtype=np.float64) * 2.0 / dr))
    posQ = np.arange(Sq)[:, None] * inv[None, :]
    posK = np.arange(Skv)[:, None] * inv[None, :]
    cosQ = np.cos(posQ).astype(np.float32)
    sinQ = np.sin(posQ).astype(np.float32)
    cosK = np.cos(posK).astype(np.float32)
    sinK = np.sin(posK).astype(np.float32)
    return Qn, Qr, Kn, Kr, V, cosQ, sinQ, cosK, sinK


_CASES = [
    pytest.param(2, 4, 3, 6, 16, 8, 16, id="base_H4"),
    pytest.param(1, 2, 1, 8, 12, 4, 12, id="decode_step_Sq1"),
    pytest.param(2, 8, 5, 10, 32, 16, 24, id="deepseek_shaped"),
    pytest.param(1, 1, 4, 4, 8, 2, 8, id="single_head"),
]


@pytest.mark.parametrize("style", ["interleaved", "half"])
@pytest.mark.parametrize("B,H,Sq,Skv,dn,dr,dv", _CASES)
def test_mla_rope_matches_numpy(B, H, Sq, Skv, dn, dr, dv, style):
    args = _make(B, H, Sq, Skv, dn, dr, dv)
    out = R._apple_gpu_mla_decode_rope(*args, np, rotation_style=style)
    assert out is not None and out.shape == (B, H, Sq, dv)
    ref = _ref_mla_rope(*args, style)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_mla_rope_styles_differ():
    """Interleaved vs half are genuinely different rotations (so the switch
    actually does something) — same inputs, different outputs."""
    args = _make(1, 2, 4, 5, 8, 4, 8, seed=3)
    a = R._apple_gpu_mla_decode_rope(*args, np, rotation_style="interleaved")
    b = R._apple_gpu_mla_decode_rope(*args, np, rotation_style="half")
    assert a is not None and b is not None
    assert not np.allclose(a, b, rtol=1e-3, atol=1e-3)


def test_mla_rope_key_shared_across_heads():
    """Kr carries no head axis ([B,Skv,dr]); the kernel broadcasts it across
    heads. A reference that broadcasts the same way must match."""
    args = _make(2, 4, 3, 6, 16, 8, 16, seed=5)
    out = R._apple_gpu_mla_decode_rope(*args, np, rotation_style="interleaved")
    ref = _ref_mla_rope(*args, "interleaved")
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_mla_rope_no_rope_dims_is_plain_mha():
    """dr=0 (no rope dims) degenerates to standard multi-head attention."""
    B, H, Sq, Skv, dn, dv = 2, 3, 4, 5, 16, 16
    rng = np.random.RandomState(9)
    f = lambda *s: (rng.randn(*s) * 0.3).astype(np.float32)
    Qn, Qr = f(B, H, Sq, dn), np.zeros((B, H, Sq, 0), np.float32)
    Kn, Kr = f(B, H, Skv, dn), np.zeros((B, Skv, 0), np.float32)
    V = f(B, H, Skv, dv)
    empty = np.zeros((Sq, 0), np.float32)
    emptyK = np.zeros((Skv, 0), np.float32)
    out = R._apple_gpu_mla_decode_rope(Qn, Qr, Kn, Kr, V, empty, empty,
                                       emptyK, emptyK, np)
    assert out is not None
    # plain MHA reference
    scale = 1.0 / math.sqrt(dn)
    ref = np.empty((B, H, Sq, dv))
    for b in range(B):
        for h in range(H):
            s = (Qn[b, h].astype(np.float64) @ Kn[b, h].T.astype(np.float64)) * scale
            s = s - s.max(-1, keepdims=True)
            e = np.exp(s)
            ref[b, h] = (e / e.sum(-1, keepdims=True)) @ V[b, h]
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_mla_rope_symbol_exported():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_mla_decode_rope_f32")
    assert R._apple_gpu_mla_decode_rope_f32() is not None
