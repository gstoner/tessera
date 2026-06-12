"""P1 — DFlash block-diffusion attention keystone.

Validates ``tessera.nn.functional.block_diffusion_attention`` against an
independent numpy transcription of DFlash ``DFlashAttention.__call__``
(z-lab/dflash ``dflash/model_mlx.py`` lines 82-116): QK-norm, KV injection
(``concat([context_KV, proposal_KV])``), GQA, rope offsets, the sliding-window
mask routed through the ``attn_bias`` substrate, and accumulated context cache.
"""
import numpy as np
import pytest

from tessera.nn import functional as F


def _rope(t, offset):
    # t: (B, H, T, Dh) — rotate even/odd pairs by (offset+pos)*freq.
    Dh = t.shape[-1]
    half = Dh // 2
    pos = (offset + np.arange(t.shape[2]))[None, None, :, None]
    freq = 1.0 / (10000.0 ** (np.arange(half) / half))
    ang = pos * freq
    e, o = t[..., 0::2], t[..., 1::2]
    out = np.empty_like(t)
    out[..., 0::2] = e * np.cos(ang) - o * np.sin(ang)
    out[..., 1::2] = e * np.sin(ang) + o * np.cos(ang)
    return out


def _ref(x, x_ctx, qw, kw, vw, ow, Hq, Hkv, Dh, qn, kn, *,
         sliding=None, cache_k=None, cache_v=None, rope=None, cache_offset=0):
    """Independent numpy port of DFlashAttention.__call__."""
    B, L, _ = x.shape
    S = x_ctx.shape[1]

    def heads(t, W, h):
        y = t @ W
        T = y.shape[1]
        return y.reshape(B, T, h, Dh).transpose(0, 2, 1, 3)

    def rms(t, w):
        y = t / np.sqrt((t * t).mean(-1, keepdims=True) + 1e-6)
        return y * w if w is not None else y

    q = rms(heads(x, qw, Hq), qn)
    ck = rms(heads(x_ctx, kw, Hkv), kn)
    cv = heads(x_ctx, vw, Hkv)
    pk = rms(heads(x, kw, Hkv), kn)
    pv = heads(x, vw, Hkv)
    if rope is not None:
        q = rope(q, cache_offset + S)
        ck = rope(ck, cache_offset)
        pk = rope(pk, cache_offset + S)
    if cache_k is not None:
        ck = np.concatenate([cache_k.transpose(0, 2, 1, 3), ck], axis=2)
        cv = np.concatenate([cache_v.transpose(0, 2, 1, 3), cv], axis=2)
    ctx_len = ck.shape[2]
    K = np.concatenate([ck, pk], axis=2)
    V = np.concatenate([cv, pv], axis=2)
    Sk = K.shape[2]
    if Hkv != Hq:
        K = np.repeat(K, Hq // Hkv, axis=1)
        V = np.repeat(V, Hq // Hkv, axis=1)
    scale = Dh ** -0.5
    s = np.einsum("bhqd,bhkd->bhqk", q, K) * scale
    if sliding is not None:
        qpos = ctx_len + np.arange(L)[:, None]
        kpos = np.arange(Sk)[None, :]
        allow = (kpos <= qpos) & (kpos > qpos - sliding)
        s = np.where(allow, s, -1e30)
    s = s - s.max(-1, keepdims=True)
    a = np.exp(s); a /= a.sum(-1, keepdims=True)
    o = np.einsum("bhqk,bhkd->bhqd", a, V)
    o = o.transpose(0, 2, 1, 3).reshape(B, L, Hq * Dh)
    return o @ ow


def _weights(rng, D, Hq, Hkv, Dh):
    return dict(
        q_proj=rng.standard_normal((D, Hq * Dh)).astype(np.float32) * 0.1,
        k_proj=rng.standard_normal((D, Hkv * Dh)).astype(np.float32) * 0.1,
        v_proj=rng.standard_normal((D, Hkv * Dh)).astype(np.float32) * 0.1,
        o_proj=rng.standard_normal((Hq * Dh, D)).astype(np.float32) * 0.1,
        q_norm=(rng.standard_normal(Dh).astype(np.float32) * 0.1 + 1.0),
        k_norm=(rng.standard_normal(Dh).astype(np.float32) * 0.1 + 1.0),
    )


@pytest.mark.parametrize("Hq,Hkv", [(4, 4), (4, 2), (4, 1)])
def test_full_attention_mha_and_gqa(Hq, Hkv):
    rng = np.random.default_rng(Hq * 10 + Hkv)
    B, L, S, Dh, D = 2, 4, 6, 8, 16
    x = rng.standard_normal((B, L, D)).astype(np.float32)
    x_ctx = rng.standard_normal((B, S, D)).astype(np.float32)
    w = _weights(rng, D, Hq, Hkv, Dh)
    got = F.block_diffusion_attention(
        x, x_ctx, num_heads=Hq, num_kv_heads=Hkv, head_dim=Dh, **w)
    ref = _ref(x, x_ctx, w["q_proj"], w["k_proj"], w["v_proj"], w["o_proj"],
               Hq, Hkv, Dh, w["q_norm"], w["k_norm"])
    assert got.shape == (B, L, D)
    assert np.abs(got - ref).max() < 1e-4


def test_sliding_window_via_attn_bias():
    rng = np.random.default_rng(7)
    B, L, S, Dh, D, Hq, Hkv = 2, 5, 7, 8, 16, 4, 2
    x = rng.standard_normal((B, L, D)).astype(np.float32)
    x_ctx = rng.standard_normal((B, S, D)).astype(np.float32)
    w = _weights(rng, D, Hq, Hkv, Dh)
    got = F.block_diffusion_attention(
        x, x_ctx, num_heads=Hq, num_kv_heads=Hkv, head_dim=Dh,
        sliding_window=3, **w)
    ref = _ref(x, x_ctx, w["q_proj"], w["k_proj"], w["v_proj"], w["o_proj"],
               Hq, Hkv, Dh, w["q_norm"], w["k_norm"], sliding=3)
    assert np.abs(got - ref).max() < 1e-4


def test_with_rope_offsets_and_cache():
    rng = np.random.default_rng(11)
    B, L, S, Dh, D, Hq, Hkv = 2, 4, 5, 8, 16, 4, 2
    Sc = 6  # prior accumulated context length
    x = rng.standard_normal((B, L, D)).astype(np.float32)
    x_ctx = rng.standard_normal((B, S, D)).astype(np.float32)
    cache_k = rng.standard_normal((B, Sc, Hkv, Dh)).astype(np.float32)
    cache_v = rng.standard_normal((B, Sc, Hkv, Dh)).astype(np.float32)
    w = _weights(rng, D, Hq, Hkv, Dh)
    got = F.block_diffusion_attention(
        x, x_ctx, num_heads=Hq, num_kv_heads=Hkv, head_dim=Dh,
        rope_fn=_rope, cache_offset=Sc, cache_keys=cache_k, cache_values=cache_v, **w)
    ref = _ref(x, x_ctx, w["q_proj"], w["k_proj"], w["v_proj"], w["o_proj"],
               Hq, Hkv, Dh, w["q_norm"], w["k_norm"],
               rope=_rope, cache_offset=Sc, cache_k=cache_k, cache_v=cache_v)
    assert np.abs(got - ref).max() < 1e-4


def test_mask_token_block():
    blk = F.mask_token_block(np.array([[101]]), block_size=4, mask_token_id=0)
    assert blk.shape == (1, 1, 4)
    assert blk[0, 0, 0] == 101 and (blk[0, 0, 1:] == 0).all()
    # scalar prev token
    blk2 = F.mask_token_block(5, block_size=3, mask_token_id=9)
    assert blk2.tolist() == [5, 9, 9]
