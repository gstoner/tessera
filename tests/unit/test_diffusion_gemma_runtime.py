"""DiffusionGemma block-diffusion region — native execution path.

The denoiser layer composes through ops.flash_attn (+ moe_swiglu_block), so its
attention runs on the Apple GPU metal_runtime lane via the attn_bias substrate.
Validates that the GPU attention backend matches the numpy reference and that
the layer is shape-correct + faithful to the GQA / KV-injection contract.
"""
import sys

import numpy as np
import pytest

from tessera.models import DiffusionGemmaConfig
from tessera.models import block_diffusion_runtime as BR

try:
    from tessera._apple_gpu_dispatch import apple_gpu_available
except Exception:  # pragma: no cover
    apple_gpu_available = lambda: False  # noqa: E731
DARWIN = sys.platform == "darwin"


def _small_cfg():
    # Small-but-faithful: GQA 4:2 query:kv heads, head_dim 16 (<=256 kernel cap),
    # 4 experts / 2 active, tiny MoE FFN. Same structure as the production card.
    return DiffusionGemmaConfig(
        hidden_size=64, num_attention_heads=4, num_kv_heads=2, head_dim=16,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=48, num_layers=2, canvas_size=12,
        vocab_size=50)


def test_denoise_layer_shapes_and_residual():
    cfg = _small_cfg()
    rng = np.random.default_rng(0)
    T, S = cfg.canvas_size, 8
    w = BR.synthetic_layer_weights(cfg, seed=1)
    ck, cv = BR.synthetic_encoder_kv(cfg, context_len=S, seed=2)
    h = rng.standard_normal((T, cfg.hidden_size)).astype(np.float32)
    out = BR.denoise_layer(h, ck, cv, w, cfg, top_k=cfg.num_experts_per_tok)
    assert out.shape == (T, cfg.hidden_size)
    # residual structure: a zero-weight layer would return h unchanged; here it
    # must differ (real projections) but stay finite.
    assert np.isfinite(out).all() and not np.allclose(out, h)


def test_gpu_attention_backend_matches_numpy_reference():
    """ops.flash_attn (numpy) vs the Apple GPU dispatcher must agree — off-Darwin
    the dispatcher falls back to numpy, so this always holds; on Darwin it proves
    the canvas-denoiser attention runs correctly on metal_runtime."""
    from tessera import dflash as D
    from tessera import runtime as R

    def gpu_attn(q, k, v, *, scale, causal, attn_bias=None):
        operands = [q, k, v] if attn_bias is None else [q, k, v, attn_bias]
        return R._apple_gpu_dispatch_flash_attn(
            "tessera.flash_attn", operands, {"scale": scale, "causal": causal}, np)

    cfg = _small_cfg()
    rng = np.random.default_rng(3)
    T, S = cfg.canvas_size, 10
    w = BR.synthetic_layer_weights(cfg, seed=4)
    ck, cv = BR.synthetic_encoder_kv(cfg, context_len=S, seed=5)
    h = rng.standard_normal((T, cfg.hidden_size)).astype(np.float32)

    ref = BR.denoise_layer(h, ck, cv, w, cfg, top_k=cfg.num_experts_per_tok)
    gpu = BR.denoise_layer(h, ck, cv, w, cfg, top_k=cfg.num_experts_per_tok,
                           attention_fn=gpu_attn)
    # public seam too
    gpu2 = BR.denoise_layer(h, ck, cv, w, cfg, top_k=cfg.num_experts_per_tok,
                            attention_fn=D.apple_gpu_attention_fn)
    assert np.allclose(ref, gpu, rtol=1e-3, atol=1e-3)
    assert np.allclose(ref, gpu2, rtol=1e-3, atol=1e-3)


def test_run_denoise_stacks_layers_deterministically():
    cfg = _small_cfg()
    rng = np.random.default_rng(6)
    T, S = cfg.canvas_size, 6
    ws = [BR.synthetic_layer_weights(cfg, seed=10 + i) for i in range(cfg.num_layers)]
    enc = BR.synthetic_encoder_kv(cfg, context_len=S, seed=7)
    canvas = rng.standard_normal((T, cfg.hidden_size)).astype(np.float32)
    a = BR.run_denoise(canvas, enc, ws, cfg, num_layers=cfg.num_layers,
                       top_k=cfg.num_experts_per_tok)
    b = BR.run_denoise(canvas, enc, ws, cfg, num_layers=cfg.num_layers,
                       top_k=cfg.num_experts_per_tok)
    assert a.shape == (T, cfg.hidden_size)
    assert np.array_equal(a, b)


def test_bidirectional_canvas_sees_all_positions():
    """Bidirectional denoising: perturbing a later canvas position changes the
    output at an earlier position (would not happen under causal attention)."""
    cfg = _small_cfg()
    rng = np.random.default_rng(8)
    T, S = cfg.canvas_size, 4
    w = BR.synthetic_layer_weights(cfg, seed=9)
    enc = BR.synthetic_encoder_kv(cfg, context_len=S, seed=11)
    h = rng.standard_normal((T, cfg.hidden_size)).astype(np.float32)
    out_a = BR.denoise_layer(h, *enc, w, cfg, top_k=cfg.num_experts_per_tok)
    h2 = h.copy(); h2[-1] += 5.0
    out_b = BR.denoise_layer(h2, *enc, w, cfg, top_k=cfg.num_experts_per_tok)
    assert not np.allclose(out_a[0], out_b[0])


@pytest.mark.skipif(not (DARWIN and apple_gpu_available()), reason="Metal device required")
def test_denoiser_attention_envelope_on_metal():
    """At production head_dim=256 the per-head attention is in the GPU kernel
    envelope; confirm the metal path runs and matches numpy."""
    from tessera import dflash as D
    cfg = DiffusionGemmaConfig()  # production: head_dim=256, 10:2 heads
    rng = np.random.default_rng(12)
    T, S = 16, 8
    w = BR.synthetic_layer_weights(cfg, seed=13)
    ck, cv = BR.synthetic_encoder_kv(cfg, context_len=S, seed=14)
    h = (rng.standard_normal((T, cfg.hidden_size)) * 0.1).astype(np.float32)
    ref = BR.denoise_layer(h, ck, cv, w, cfg, top_k=cfg.num_experts_per_tok)
    gpu = BR.denoise_layer(h, ck, cv, w, cfg, top_k=cfg.num_experts_per_tok,
                           attention_fn=D.apple_gpu_attention_fn)
    assert gpu.shape == (T, cfg.hidden_size)
    assert np.allclose(ref, gpu, rtol=1e-2, atol=1e-2)
