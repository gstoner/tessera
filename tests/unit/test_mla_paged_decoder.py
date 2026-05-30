"""MLA paged-cache decoder — production-serving wiring (2026-05-30).

`tessera.cache.MLAPagedDecoder` bundles two `LatentKVCacheHandle`s (compressed
latent + shared decoupled-RoPE key slice) and drives the weight-absorbed MLA
decode kernel. These tests run a real serving loop (prefill → step-by-step
decode) and validate each step against a from-scratch full-window reference,
plus eviction / sliding-window RoPE-position correctness. The decoder runs on
the Apple GPU kernel when available and a numpy reference otherwise, so the
suite is portable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tessera.cache import MLAPagedDecoder, LatentKVCacheHandle


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


def _full_window_ref(q_nope, q_rope, c_kv, k_rope, Wuk, Wuv, key_pos, q_pos,
                     rope_base, style):
    """From-scratch reference: rebuild explicit per-head K/V from the latent and
    run plain decoupled-RoPE attention. Independent of the decoder internals."""
    H, dn = q_nope.shape
    dr = q_rope.shape[-1]
    S, Dl = c_kv.shape
    dv = Wuv.shape[-1]
    half = dr // 2
    inv = rope_base ** (-(np.arange(half) * 2.0 / dr))
    cosK = np.cos(key_pos[:, None] * inv[None, :])
    sinK = np.sin(key_pos[:, None] * inv[None, :])
    cosQ = np.cos(np.asarray([q_pos])[:, None] * inv[None, :])
    sinQ = np.sin(np.asarray([q_pos])[:, None] * inv[None, :])
    scale = 1.0 / math.sqrt(dn + dr)
    krR = _rope(k_rope.astype(np.float64), cosK, sinK, style)
    O = np.empty((H, dv))
    for h in range(H):
        Kn = c_kv.astype(np.float64) @ Wuk[h].astype(np.float64)     # [S,dn]
        V = c_kv.astype(np.float64) @ Wuv[h].astype(np.float64)      # [S,dv]
        qrR = _rope(q_rope[h].astype(np.float64), cosQ[0], sinQ[0], style)
        s = (q_nope[h].astype(np.float64) @ Kn.T) + (qrR @ krR.T)
        s = s * scale
        s = s - s.max()
        e = np.exp(s)
        O[h] = (e / e.sum()) @ V
    return O


def _setup(H=4, dn=16, dr=8, dv=16, Dl=32, seed=0, **kw):
    rng = np.random.RandomState(seed)
    f = lambda *s: (rng.randn(*s) * 0.3).astype(np.float32)
    Wuk = f(H, Dl, dn)                         # K up-proj
    Wuv = f(H, Dl, dv)                         # V up-proj
    Wuk_t = np.ascontiguousarray(np.swapaxes(Wuk, 1, 2))
    dec = MLAPagedDecoder(num_heads=H, nope_dim=dn, rope_dim=dr, v_dim=dv,
                          latent_dim=Dl, Wuk_t=Wuk_t, Wuv=Wuv, **kw)
    return dec, Wuk, Wuv, rng


def test_paged_decode_matches_full_window_each_step():
    dec, Wuk, Wuv, rng = _setup(max_seq=64, rope_base=10000.0)
    f = lambda *s: (rng.randn(*s) * 0.3).astype(np.float32)
    H, dn, dr, dv, Dl = dec.num_heads, dec.nope_dim, dec.rope_dim, dec.v_dim, dec.latent_dim

    all_ckv, all_kr = [], []
    for step in range(6):
        c_t = f(1, Dl)
        r_t = f(1, dr)
        dec.append(c_t, r_t)
        all_ckv.append(c_t)
        all_kr.append(r_t)

        q_nope = f(H, dn)
        q_rope = f(H, dr)
        out = dec.decode(q_nope, q_rope)
        assert out.shape == (H, dv)

        c_kv = np.concatenate(all_ckv, 0)
        k_rope = np.concatenate(all_kr, 0)
        S = c_kv.shape[0]
        ref = _full_window_ref(q_nope, q_rope, c_kv, k_rope, Wuk, Wuv,
                               np.arange(S), S - 1, 10000.0, "interleaved")
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_paged_prefill_then_decode():
    dec, Wuk, Wuv, rng = _setup(max_seq=64, seed=1)
    f = lambda *s: (rng.randn(*s) * 0.3).astype(np.float32)
    H, dn, dr, dv, Dl = dec.num_heads, dec.nope_dim, dec.rope_dim, dec.v_dim, dec.latent_dim
    # prefill a 5-token prompt in one append
    c_pre, r_pre = f(5, Dl), f(5, dr)
    dec.append(c_pre, r_pre)
    assert dec.current_seq == 5
    q_nope, q_rope = f(H, dn), f(H, dr)
    out = dec.decode(q_nope, q_rope)
    ref = _full_window_ref(q_nope, q_rope, c_pre, r_pre, Wuk, Wuv,
                           np.arange(5), 4, 10000.0, "interleaved")
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("style", ["interleaved", "half"])
def test_paged_rotation_style(style):
    dec, Wuk, Wuv, rng = _setup(max_seq=32, seed=2, rotation_style=style)
    f = lambda *s: (rng.randn(*s) * 0.3).astype(np.float32)
    H, dn, dr, dv, Dl = dec.num_heads, dec.nope_dim, dec.rope_dim, dec.v_dim, dec.latent_dim
    dec.append(f(4, Dl), f(4, dr))
    q_nope, q_rope = f(H, dn), f(H, dr)
    out = dec.decode(q_nope, q_rope)
    c_kv, k_rope = dec.latent_cache.read(0, 4), dec.rope_cache.read(0, 4)
    ref = _full_window_ref(q_nope, q_rope, c_kv, k_rope, Wuk, Wuv,
                           np.arange(4), 3, 10000.0, style)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_paged_sliding_window_positions():
    """After eviction the remaining keys keep their ABSOLUTE RoPE positions, so
    the decode must match a reference using those absolute positions."""
    dec, Wuk, Wuv, rng = _setup(max_seq=4, seed=3, auto_evict=True)
    f = lambda *s: (rng.randn(*s) * 0.3).astype(np.float32)
    H, dn, dr, dv, Dl = dec.num_heads, dec.nope_dim, dec.rope_dim, dec.v_dim, dec.latent_dim

    tokens_c, tokens_r = [], []
    for _ in range(7):                  # 7 tokens into a 4-slot window
        c_t, r_t = f(1, Dl), f(1, dr)
        dec.append(c_t, r_t)
        tokens_c.append(c_t)
        tokens_r.append(r_t)
    # window holds the last 4 tokens; their absolute positions are 3..6
    assert dec.current_seq == 4
    assert dec._abs_base == 3
    q_nope, q_rope = f(H, dn), f(H, dr)
    out = dec.decode(q_nope, q_rope)

    c_kv = np.concatenate(tokens_c[-4:], 0)
    k_rope = np.concatenate(tokens_r[-4:], 0)
    ref = _full_window_ref(q_nope, q_rope, c_kv, k_rope, Wuk, Wuv,
                           np.arange(3, 7), 6, 10000.0, "interleaved")
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_paged_explicit_evict():
    dec, Wuk, Wuv, rng = _setup(max_seq=16, seed=4)
    f = lambda *s: (rng.randn(*s) * 0.3).astype(np.float32)
    H, dn, dr, dv, Dl = dec.num_heads, dec.nope_dim, dec.rope_dim, dec.v_dim, dec.latent_dim
    dec.append(f(6, Dl), f(6, dr))
    dec.evict_oldest(2)
    assert dec.current_seq == 4 and dec._abs_base == 2
    q_nope, q_rope = f(H, dn), f(H, dr)
    out = dec.decode(q_nope, q_rope)
    c_kv, k_rope = dec.latent_cache.read(0, 4), dec.rope_cache.read(0, 4)
    ref = _full_window_ref(q_nope, q_rope, c_kv, k_rope, Wuk, Wuv,
                           np.arange(2, 6), 5, 10000.0, "interleaved")
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_paged_cache_footprint():
    dec, *_ = _setup(max_seq=32, dn=128, dr=64, dv=128, Dl=512, H=128)
    # latent + rope (shared across heads) vs explicit per-head K/V
    latent = dec.cache_bytes_per_token()
    explicit = dec.num_heads * (dec.nope_dim + dec.rope_dim + dec.v_dim) * 4
    assert latent == (512 + 64) * 4
    assert explicit / latent > 8.0


def test_paged_uses_two_latent_handles():
    dec, *_ = _setup(max_seq=16)
    assert isinstance(dec.latent_cache, LatentKVCacheHandle)
    assert isinstance(dec.rope_cache, LatentKVCacheHandle)
    assert dec.latent_cache.latent_dim == dec.latent_dim
    assert dec.rope_cache.latent_dim == dec.rope_dim


def test_paged_validates_shapes():
    rng = np.random.RandomState(0)
    Wuk_t = rng.randn(4, 16, 32).astype(np.float32)
    Wuv = rng.randn(4, 32, 16).astype(np.float32)
    with pytest.raises(ValueError):
        MLAPagedDecoder(num_heads=4, nope_dim=16, rope_dim=7,  # odd rope_dim
                        v_dim=16, latent_dim=32, Wuk_t=Wuk_t, Wuv=Wuv, max_seq=8)
    with pytest.raises(ValueError):
        MLAPagedDecoder(num_heads=4, nope_dim=16, rope_dim=8, v_dim=16,
                        latent_dim=32, Wuk_t=Wuk_t[:, :, :8],  # wrong latent dim
                        Wuv=Wuv, max_seq=8)
    dec = MLAPagedDecoder(num_heads=4, nope_dim=16, rope_dim=8, v_dim=16,
                          latent_dim=32, Wuk_t=Wuk_t, Wuv=Wuv, max_seq=8)
    with pytest.raises(ValueError):
        dec.decode(np.zeros((4, 16), np.float32), np.zeros((4, 8), np.float32))
