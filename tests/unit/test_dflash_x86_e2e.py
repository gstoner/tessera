"""DFlash attention core on the native x86 AVX-512 flash lane (Phase 2 seam).

`tessera.dflash.x86_attention_fn` routes block_diffusion_attention's rank-3
`flash_attn(+attn_bias)` core onto the AVX-512 online-softmax kernel
(`tessera_x86_flash_attn_ext_f32`). Unlike the ROCm/Apple GPU seams this is
**f32-native with no head-dim constraint**, so it matches the numpy reference to
f32 epsilon (not an f16 tolerance) and runs any head_dim. Skipped when the x86
elementwise lib isn't built; the fallback path runs everywhere.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from tessera import dflash as Df
from tessera.nn import functional as F


def _x86_or_skip():
    from tessera import runtime as rt
    if not (shutil.which("clang") or shutil.which("cc")
            or os.path.exists("/usr/bin/cc")):
        pytest.skip("no C toolchain")
    if rt._load_x86_elementwise() is None:
        pytest.skip("libtessera_x86_elementwise.so not built")


def _weights(rng, d, hq, hkv, dh):
    return dict(
        q_proj=rng.standard_normal((d, hq * dh)).astype(np.float32) * 0.1,
        k_proj=rng.standard_normal((d, hkv * dh)).astype(np.float32) * 0.1,
        v_proj=rng.standard_normal((d, hkv * dh)).astype(np.float32) * 0.1,
        o_proj=rng.standard_normal((hq * dh, d)).astype(np.float32) * 0.1,
        q_norm=rng.standard_normal(dh).astype(np.float32) * 0.1 + 1.0,
        k_norm=rng.standard_normal(dh).astype(np.float32) * 0.1 + 1.0,
    )


@pytest.mark.parametrize("dh", [16, 8])  # x86 has no WMMA multiple-of-16 gate
def test_public_x86_attention_fn_seam(dh):
    _x86_or_skip()
    rng = np.random.default_rng(99)
    b, hq, hkv, d = 1, 4, 2, 32
    ln, s = 12, 6
    x = rng.standard_normal((b, ln, d)).astype(np.float32)
    x_ctx = rng.standard_normal((b, s, d)).astype(np.float32)
    common = dict(num_heads=hq, num_kv_heads=hkv, head_dim=dh,
                  sliding_window=3, **_weights(rng, d, hq, hkv, dh))
    ref = F.block_diffusion_attention(x, x_ctx, **common)
    cpu = F.block_diffusion_attention(
        x, x_ctx, attention_fn=Df.x86_attention_fn, **common)
    np.testing.assert_allclose(np.asarray(cpu), np.asarray(ref),
                               rtol=0, atol=1e-5)


def test_whole_draft_attention_on_x86():
    # The whole draft forward runs its attention on the AVX-512 lane via the seam
    # threaded through every decoder layer — logits match numpy to f32 epsilon AND
    # the greedy tokens are identical (the DFlash greedy-invariant holds).
    _x86_or_skip()
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
    ref = np.asarray(Df.dflash_draft_forward(
        block, th, w, cfg, logits_start=1, rope_fn=rope))
    cpu = np.asarray(Df.dflash_draft_forward(
        block, th, w, cfg, logits_start=1, rope_fn=rope,
        attention_fn=Df.x86_attention_fn))
    assert cpu.shape == ref.shape
    np.testing.assert_allclose(cpu, ref, rtol=0, atol=1e-5)
    assert (ref.argmax(-1) == cpu.argmax(-1)).all(), "greedy tokens diverged"
