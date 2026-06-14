"""M3 (MLA) + M4 (DSA) attention pillars — oracle-gated.

MLA oracles: absorb ≡ no-absorb (the production latent-only path must equal the
explicit-K/V reference); prefill-then-decode ≡ full-prefill (autoregressive
consistency over the paged latent cache).
DSA oracle: select-all ≡ dense causal (DESIL cross-path); an explicit top-1
hand check; selection determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.cache.latent import LatentKVCacheHandle
from tessera.stdlib import attention as attn


def _mla_weights(rng, H=32, Hh=4, d_c=16, d_nope=8, d_rope=8, d_v=8):
    s = 1.0 / np.sqrt(H)
    return attn.MLAWeights(
        w_dkv=(rng.standard_normal((H, d_c)) * s),
        w_uk=(rng.standard_normal((d_c, Hh * d_nope)) / np.sqrt(d_c)),
        w_uv=(rng.standard_normal((d_c, Hh * d_v)) / np.sqrt(d_c)),
        w_q=(rng.standard_normal((H, Hh * (d_nope + d_rope))) * s),
        w_kr=(rng.standard_normal((H, d_rope)) * s),
        num_heads=Hh, d_nope=d_nope, d_rope=d_rope, d_v=d_v)


# ── shared RoPE ───────────────────────────────────────────────────────────────
def test_rope_position_zero_is_identity():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 8))
    np.testing.assert_allclose(attn.apply_rope(x, np.zeros(5)), x, atol=1e-12)


# ── M3 MLA ────────────────────────────────────────────────────────────────────
def test_mla_absorb_equals_no_absorb():
    """Headline MLA oracle: the absorbed (latent-only) path is numerically
    identical to the explicit nope-K / V reference."""
    rng = np.random.default_rng(1)
    w = _mla_weights(rng)
    x = rng.standard_normal((7, 32))
    c, kr = attn.compress_latent(x, w)
    o_absorb = attn.mla_attention(x, c, kr, w, q_positions=np.arange(7), absorb=True)
    o_plain = attn.mla_attention(x, c, kr, w, q_positions=np.arange(7), absorb=False)
    np.testing.assert_allclose(o_absorb, o_plain, rtol=1e-9, atol=1e-9)
    assert o_absorb.shape == (7, w.num_heads * w.d_v)


def test_mla_prefill_then_decode_matches_full_prefill():
    """Autoregressive consistency over the paged latent cache: prefilling p
    tokens then decoding the rest one-by-one reproduces a full prefill."""
    rng = np.random.default_rng(2)
    w = _mla_weights(rng)
    S, p = 9, 4
    x = rng.standard_normal((S, 32))
    o_full, _, _ = attn.mla_prefill(x, w)

    o_pre, c0, kr0 = attn.mla_prefill(x[:p], w)
    lat = LatentKVCacheHandle(latent_dim=w.d_c, max_seq=S, dtype="fp64")
    rope = LatentKVCacheHandle(latent_dim=w.d_rope, max_seq=S, dtype="fp64")
    lat.append(c0); rope.append(kr0)
    outs = [o_pre]
    for t in range(p, S):
        outs.append(attn.mla_decode_step(x[t:t + 1], lat, rope, w))
    o_decoded = np.concatenate(outs, axis=0)
    np.testing.assert_allclose(o_decoded, o_full, rtol=1e-8, atol=1e-8)
    # the latent cache holds only d_c + d_rope per token (the MLA memory win)
    assert lat.latent_dim + rope.latent_dim < w.num_heads * w.d_nope


def test_mla_decode_chunk_equals_token_by_token():
    rng = np.random.default_rng(3)
    w = _mla_weights(rng)
    x = rng.standard_normal((6, 32))
    o_full, _, _ = attn.mla_prefill(x, w)
    # decode the whole thing as one chunk from an empty cache
    lat = LatentKVCacheHandle(latent_dim=w.d_c, max_seq=6, dtype="fp64")
    rope = LatentKVCacheHandle(latent_dim=w.d_rope, max_seq=6, dtype="fp64")
    o_chunk = attn.mla_decode_step(x, lat, rope, w)
    np.testing.assert_allclose(o_chunk, o_full, rtol=1e-8, atol=1e-8)


# ── M4 DSA ────────────────────────────────────────────────────────────────────
def _qkv(rng, B=2, Hq=4, Hkv=2, S=16, D=8):
    return (rng.standard_normal((B, Hq, S, D)),
            rng.standard_normal((B, Hkv, S, D)),
            rng.standard_normal((B, Hkv, S, D)))


def test_dsa_index_shape():
    rng = np.random.default_rng(4)
    Q, K, _ = _qkv(rng)
    idx = attn.dsa_block_index(Q, K, block_size=4)
    assert idx.shape == (2, 2, 16, 4)


def test_dsa_select_all_equals_dense():
    """DESIL cross-path: with top-k = num_blocks the sparse path is dense."""
    rng = np.random.default_rng(5)
    Q, K, V = _qkv(rng, S=16, D=8)
    num_blocks = 16 // 4
    sparse = attn.dsa_block_sparse_attention(
        Q, K, V, top_k_blocks=num_blocks, block_size=4, causal=True)
    dense = attn.dense_causal_attention(Q, K, V)
    np.testing.assert_allclose(sparse, dense, rtol=1e-9, atol=1e-9)


def test_dsa_top1_matches_manual():
    """Explicit small check: top-1 (+ forced local) selection then exact
    attention over those blocks equals an independent recomputation."""
    rng = np.random.default_rng(6)
    Q, K, V = _qkv(rng, B=1, Hq=2, Hkv=1, S=8, D=4)
    out = attn.dsa_block_sparse_attention(Q, K, V, top_k_blocks=1, block_size=4, causal=True)
    # independent: recompute via the public index/select then masked softmax
    idx = attn.dsa_block_index(Q, K, block_size=4)
    keep = attn.dsa_select_blocks(idx, top_k=1, block_size=4, causal=True)
    tok_keep = np.repeat(keep, 4, axis=-1)
    kpos = np.arange(8)[None, None, None, :]
    qpos = np.arange(8)[None, None, :, None]
    tok_keep = tok_keep & (kpos <= qpos)
    ref = np.zeros_like(out)
    for h in range(2):
        s = (Q[:, h] @ np.swapaxes(K[:, 0], -1, -2)) / np.sqrt(4)
        s = np.where(tok_keep[:, 0], s, -np.inf)
        e = np.exp(s - s.max(-1, keepdims=True)); w = e / e.sum(-1, keepdims=True)
        ref[:, h] = w @ V[:, 0]
    np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-9)


def test_dsa_selection_is_deterministic():
    rng = np.random.default_rng(7)
    Q, K, _ = _qkv(rng)
    idx = attn.dsa_block_index(Q, K, block_size=4)
    a = attn.dsa_select_blocks(idx, top_k=2, block_size=4)
    b = attn.dsa_select_blocks(idx, top_k=2, block_size=4)
    assert np.array_equal(a, b)


def test_dsa_sparse_differs_from_dense_when_restricted():
    rng = np.random.default_rng(8)
    Q, K, V = _qkv(rng, S=16, D=8)
    sparse = attn.dsa_block_sparse_attention(Q, K, V, top_k_blocks=1, block_size=4, causal=True)
    dense = attn.dense_causal_attention(Q, K, V)
    assert not np.allclose(sparse, dense)


# ── M3.1 / M4.1 — Apple GPU composed execution lanes ─────────────────────────
def _apple_gpu_available() -> bool:
    try:
        from tessera import _apple_gpu_backend as agb
        agb.gpu_matmul(np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32))
        return True
    except Exception:                                  # noqa: BLE001
        return False


@pytest.mark.skipif(not _apple_gpu_available(), reason="Apple GPU / Metal not available")
def test_mla_apple_gpu_matches_reference():
    rng = np.random.default_rng(10)
    w = _mla_weights(rng)
    x = rng.standard_normal((7, 32))
    c, kr = attn.compress_latent(x, w)
    ref = attn.mla_attention(x, c, kr, w, q_positions=np.arange(7), backend="reference")
    gpu = attn.mla_attention(x, c, kr, w, q_positions=np.arange(7), backend="apple_gpu")
    np.testing.assert_allclose(gpu, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _apple_gpu_available(), reason="Apple GPU / Metal not available")
def test_dsa_apple_gpu_matches_reference():
    rng = np.random.default_rng(11)
    Q, K, V = _qkv(rng, S=16, D=8)
    ref = attn.dsa_block_sparse_attention(Q, K, V, top_k_blocks=2, block_size=4, backend="reference")
    gpu = attn.dsa_block_sparse_attention(Q, K, V, top_k_blocks=2, block_size=4, backend="apple_gpu")
    np.testing.assert_allclose(gpu, ref, rtol=1e-3, atol=1e-3)
