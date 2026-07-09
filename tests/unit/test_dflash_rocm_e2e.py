"""DFlash attention core on the ROCm gfx1151 WMMA flash lane (Phase 2 seam).

`tessera.dflash.rocm_attention_fn` routes block_diffusion_attention's rank-3
`flash_attn(+attn_bias)` core onto the gfx1151 WMMA kernel (the #328 additive-bias
variant), casting f32→f16 for the lane. This proves the DFlash draft attention —
GQA-folded, concatenated context+proposal KV, optional sliding-window bias —
executes on the AMD GPU and matches the numpy reference to f16 precision, and that
the whole draft forward preserves the greedy tokens (the DFlash invariant). Device
tests skip without a live gfx1151; the head-dim fallback runs everywhere.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from tessera import dflash as Df
from tessera.nn import functional as F


def _rocm_or_skip():
    from tessera import runtime as rt
    if not (shutil.which("hipcc") or os.path.exists("/opt/rocm/bin/hipcc")):
        pytest.skip("no hipcc")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no live gfx1151")


def _weights(rng, d, hq, hkv, dh):
    return dict(
        q_proj=rng.standard_normal((d, hq * dh)).astype(np.float32) * 0.1,
        k_proj=rng.standard_normal((d, hkv * dh)).astype(np.float32) * 0.1,
        v_proj=rng.standard_normal((d, hkv * dh)).astype(np.float32) * 0.1,
        o_proj=rng.standard_normal((hq * dh, d)).astype(np.float32) * 0.1,
        q_norm=rng.standard_normal(dh).astype(np.float32) * 0.1 + 1.0,
        k_norm=rng.standard_normal(dh).astype(np.float32) * 0.1 + 1.0,
    )


def test_rocm_attention_fn_head_dim_fallback():
    # head_dim not a multiple of 16 → the WMMA lane can't run it, so the seam
    # must fall back to the numpy reference EXACTLY (runs on any box).
    rng = np.random.default_rng(1)
    bh, ql, kl, dh = 2, 6, 5, 8   # dh=8 is not a multiple of 16
    q = rng.standard_normal((bh, ql, dh)).astype(np.float32)
    k = rng.standard_normal((bh, kl, dh)).astype(np.float32)
    v = rng.standard_normal((bh, kl, dh)).astype(np.float32)
    from tessera import ops
    out = Df.rocm_attention_fn(q, k, v, scale=1.0 / np.sqrt(dh), causal=False)
    ref = ops.flash_attn(q, k, v, scale=1.0 / np.sqrt(dh), causal=False)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


def test_public_rocm_attention_fn_seam():
    # block_diffusion_attention driven onto gfx1151 matches numpy to f16.
    _rocm_or_skip()
    rng = np.random.default_rng(99)
    b, hq, hkv, dh, d = 1, 4, 2, 16, 32
    ln, s = 12, 6
    x = rng.standard_normal((b, ln, d)).astype(np.float32)
    x_ctx = rng.standard_normal((b, s, d)).astype(np.float32)
    common = dict(num_heads=hq, num_kv_heads=hkv, head_dim=dh,
                  sliding_window=3, **_weights(rng, d, hq, hkv, dh))
    ref = F.block_diffusion_attention(x, x_ctx, **common)
    gpu = F.block_diffusion_attention(
        x, x_ctx, attention_fn=Df.rocm_attention_fn, **common)
    np.testing.assert_allclose(np.asarray(gpu), np.asarray(ref),
                               rtol=0, atol=2e-2)


def test_whole_draft_attention_on_rocm():
    # The whole draft forward runs its attention on gfx1151 via the seam threaded
    # through every decoder layer — logits close to numpy AND the greedy tokens
    # identical (the DFlash greedy-invariant survives the f16 draft precision).
    _rocm_or_skip()
    rng = np.random.default_rng(13)
    cfg = Df.DFlashConfig(hidden_size=32, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2,
                          head_dim=16, intermediate_size=64, vocab_size=41,
                          block_size=16, target_layer_ids=(0, 1, 2))
    dm, hq, hkv, dh, inter, vocab, nl = 32, 4, 2, 16, 64, 41, 3
    s = lambda *sh: rng.standard_normal(sh).astype(np.float32) * 0.1
    layers = [Df.DFlashLayerWeights(
        q_proj=s(dm, hq * dh), k_proj=s(dm, hkv * dh), v_proj=s(dm, hkv * dh),
        o_proj=s(hq * dh, dm), q_norm=s(dh) + 1.0, k_norm=s(dh) + 1.0,
        input_layernorm=s(dm) + 1.0, post_attention_layernorm=s(dm) + 1.0,
        mlp_gate=s(dm, inter), mlp_up=s(dm, inter),
        mlp_down=s(inter, dm)) for _ in range(2)]
    w = Df.DFlashWeights(embed_tokens=s(vocab, dm), fc=s(nl * dm, dm),
                         hidden_norm=s(dm) + 1.0, layers=layers,
                         final_norm=s(dm) + 1.0, lm_head=s(dm, vocab))
    block = rng.integers(0, vocab, (1, cfg.block_size))
    th = rng.standard_normal((1, 8, nl * dm)).astype(np.float32)
    rope = Df.make_rope(cfg.head_dim)
    cpu = np.asarray(Df.dflash_draft_forward(
        block, th, w, cfg, logits_start=1, rope_fn=rope))
    gpu = np.asarray(Df.dflash_draft_forward(
        block, th, w, cfg, logits_start=1, rope_fn=rope,
        attention_fn=Df.rocm_attention_fn))
    assert gpu.shape == cpu.shape
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=5e-3)
    assert (cpu.argmax(-1) == gpu.argmax(-1)).all(), "greedy tokens diverged"
