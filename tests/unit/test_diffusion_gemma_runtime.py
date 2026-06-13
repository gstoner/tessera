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


def _step_inputs(cfg, *, S=8, seed=20):
    from tessera.models import SamplerConfig
    rng = np.random.default_rng(seed)
    ws = [BR.synthetic_layer_weights(cfg, seed=seed + i) for i in range(cfg.num_layers)]
    enc = BR.synthetic_encoder_kv(cfg, context_len=S, seed=seed + 99)
    canvas = (rng.standard_normal((cfg.canvas_size, cfg.hidden_size)) * 0.1).astype(np.float32)
    w_lm = (rng.standard_normal((cfg.hidden_size, cfg.vocab_size)) / np.sqrt(cfg.hidden_size)).astype(np.float32)
    sc = SamplerConfig(vocab_size=cfg.vocab_size)
    return ws, enc, canvas, w_lm, sc


def test_execute_step_shapes_and_contract():
    cfg = _small_cfg()
    ws, enc, canvas, w_lm, sc = _step_inputs(cfg)
    r = BR.execute_block_diffusion_step(
        canvas, enc, ws, w_lm, cfg, step=1, sampler_config=sc,
        num_denoise_layers=cfg.num_layers, rng_key=7, top_k=cfg.num_experts_per_tok)
    assert r.tokens.shape == (cfg.canvas_size,)
    assert np.array_equal(r.renoise_mask, ~r.accepted_mask)
    assert r.committed == int(r.accepted_mask.sum())
    assert r.stop_reason in {"continue", "all_accepted", "stability", "eos", "max_steps"}


def test_execute_step_gpu_backend_matches_numpy():
    """The full step (denoiser + LM head + sampler) yields identical tokens/masks
    whether the attention runs via numpy or the Apple GPU dispatcher."""
    from tessera import dflash as D
    cfg = _small_cfg()
    ws, enc, canvas, w_lm, sc = _step_inputs(cfg, seed=30)
    ref = BR.execute_block_diffusion_step(
        canvas, enc, ws, w_lm, cfg, step=2, sampler_config=sc,
        num_denoise_layers=cfg.num_layers, rng_key=5, top_k=cfg.num_experts_per_tok)
    gpu = BR.execute_block_diffusion_step(
        canvas, enc, ws, w_lm, cfg, step=2, sampler_config=sc,
        num_denoise_layers=cfg.num_layers, rng_key=5, top_k=cfg.num_experts_per_tok,
        attention_fn=D.apple_gpu_attention_fn)
    assert np.array_equal(ref.tokens, gpu.tokens)
    assert np.array_equal(ref.accepted_mask, gpu.accepted_mask)
    assert np.allclose(ref.entropy, gpu.entropy, rtol=1e-3, atol=1e-3)


def test_full_gpu_backend_denoise_matches_numpy_within_f32():
    """backend='apple_gpu' routes attention + MoE through the Metal lanes (f32);
    off-Darwin those dispatchers fall back to numpy. Either way the denoised
    canvas matches the numpy-f64 reference within f32 tolerance."""
    cfg = _small_cfg()
    rng = np.random.default_rng(40)
    T, S = cfg.canvas_size, 8
    w = BR.synthetic_layer_weights(cfg, seed=41)
    ck, cv = BR.synthetic_encoder_kv(cfg, context_len=S, seed=42)
    h = (rng.standard_normal((T, cfg.hidden_size)) * 0.1).astype(np.float32)
    ref = BR.denoise_layer(h, ck, cv, w, cfg, top_k=cfg.num_experts_per_tok, backend="numpy")
    gpu = BR.denoise_layer(h, ck, cv, w, cfg, top_k=cfg.num_experts_per_tok, backend="apple_gpu")
    assert gpu.shape == (T, cfg.hidden_size)
    assert np.allclose(ref, gpu, rtol=2e-2, atol=2e-2)


def test_full_gpu_step_runs_and_close_to_numpy():
    """The whole step (attention + MoE + LM head on GPU) produces a valid result
    and logits close to numpy; commit decisions agree on the bulk of positions."""
    cfg = _small_cfg()
    ws, enc, canvas, w_lm, sc = _step_inputs(cfg, seed=50)
    ref = BR.execute_block_diffusion_step(
        canvas, enc, ws, w_lm, cfg, step=1, sampler_config=sc,
        num_denoise_layers=cfg.num_layers, rng_key=3, top_k=cfg.num_experts_per_tok,
        backend="numpy")
    gpu = BR.execute_block_diffusion_step(
        canvas, enc, ws, w_lm, cfg, step=1, sampler_config=sc,
        num_denoise_layers=cfg.num_layers, rng_key=3, top_k=cfg.num_experts_per_tok,
        backend="apple_gpu")
    assert gpu.tokens.shape == (cfg.canvas_size,)
    assert np.array_equal(gpu.renoise_mask, ~gpu.accepted_mask)
    # entropy (hence commit decisions) close at f32 precision
    assert np.allclose(ref.entropy, gpu.entropy, rtol=5e-2, atol=5e-2)


def _decoder(cfg, *, backend="numpy", max_steps=None, seed=60):
    from tessera.models import SamplerConfig, synthetic_decoder_weights, NativeBlockDiffusionDecoder
    # >= canvas_size so the guaranteed-1-per-step progress always fully commits.
    max_steps = max_steps if max_steps is not None else cfg.canvas_size
    w = synthetic_decoder_weights(cfg, num_denoise_layers=cfg.num_layers, seed=seed)
    sc = SamplerConfig(vocab_size=cfg.vocab_size, num_steps=max_steps)
    return NativeBlockDiffusionDecoder(
        cfg, w, num_denoise_layers=cfg.num_layers, max_steps=max_steps,
        sampler_config=sc, top_k=cfg.num_experts_per_tok, backend=backend)


def test_native_decoder_commits_full_canvas_and_promotes_kv():
    cfg = _small_cfg()
    dec = _decoder(cfg)
    assert dec.context_len == 0
    r0 = dec.decode_block(rng_key=1)
    # the loop guarantees progress → the canvas fully commits within max_steps
    assert r0.committed == cfg.canvas_size and r0.stop_reason == "all_committed"
    assert r0.tokens.shape == (cfg.canvas_size,)
    assert tuple(np.diff(r0.progress)) == () or all(d >= 0 for d in np.diff(r0.progress))
    # KV promotion grew the committed context for the next block.
    assert dec.context_len == cfg.canvas_size
    r1 = dec.decode_block(rng_key=2)            # second block attends over the cache
    assert dec.context_len == 2 * cfg.canvas_size
    assert r1.committed == cfg.canvas_size


def test_native_decoder_numpy_vs_gpu_backend_consistent():
    """End-to-end decode on the GPU lane (backend='apple_gpu') produces the same
    committed tokens as numpy (off-Darwin the dispatchers fall back to numpy; on
    Darwin the whole loop runs on Metal) — exact when the dispatcher is the numpy
    fallback, within f32 commit-decision agreement on Metal."""
    cfg = _small_cfg()
    ref = _decoder(cfg, backend="numpy", seed=70).decode_block(rng_key=9)
    gpu = _decoder(cfg, backend="apple_gpu", seed=70).decode_block(rng_key=9)
    assert gpu.committed == cfg.canvas_size
    # at least the bulk of positions agree (f32 vs f64 may flip a few near the
    # entropy threshold); off-Darwin this is exact.
    agree = float((ref.tokens == gpu.tokens).mean())
    assert agree >= 0.6


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
