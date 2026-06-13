"""DiffusionGemma Phase D — block-diffusion step region + reference runner.

The step is a compiler-visible region (causal encoder KV read → bidirectional
canvas denoise → lm_head → entropy sample → token update → re-noise → stop). The
NumPy runner is reference-only.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tessera.models.diffusion_gemma import DiffusionGemmaConfig, DiffusionGemmaDimError
from tessera.models import block_diffusion as bd
from tessera.models import moe_routing as mr
from tessera.models.sampler import SamplerConfig


def _tiny_cfg():
    return dataclasses.replace(
        DiffusionGemmaConfig(), hidden_size=64, num_attention_heads=4,
        num_kv_heads=2, head_dim=16, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=32, shared_expert_intermediate_size=64,
        vocab_size=128, canvas_size=16)


def _runner_inputs(cfg, seed=0):
    rng = np.random.default_rng(seed)
    w = {"moe": mr.synthetic_moe_weights(cfg, seed=seed + 1),
         "w_lm": (rng.standard_normal((cfg.hidden_size, cfg.vocab_size)) / 8).astype(np.float32)}
    canvas = (rng.standard_normal((cfg.canvas_size, cfg.hidden_size)) / 8).astype(np.float32)
    enc = ((rng.standard_normal((10, cfg.hidden_size)) / 8),
           (rng.standard_normal((10, cfg.hidden_size)) / 8))
    return canvas, enc, w


# ── Compiler-visible step region ─────────────────────────────────────────────

def test_step_region_contract():
    g = bd.build_block_diffusion_step(DiffusionGemmaConfig(), num_denoise_layers=4)
    bd.verify_block_diffusion_step(g)
    assert g.region_names() == (
        "encoder_kv_read", "canvas_denoise", "lm_head", "entropy_stats",
        "entropy_sample", "token_update", "renoise_mask", "stop_flags")
    # encoder read is causal; every canvas-denoiser layer is bidirectional.
    assert g.encoder_read.attrs["causal"] is True
    assert len(g.denoiser) == 4
    assert all(d.causal is False for d in g.denoiser)
    assert all(d.find("attention").attrs["causal"] is False for d in g.denoiser)
    assert g.lm_head.output[-1] == DiffusionGemmaConfig().vocab_size


def test_verify_rejects_causal_canvas():
    cfg = DiffusionGemmaConfig()
    g = bd.build_block_diffusion_step(cfg, num_denoise_layers=2)
    # Tamper one denoiser layer to be causal → contract must reject.
    bad_layer = dataclasses.replace(g.denoiser[0], causal=True)
    bad = dataclasses.replace(g, denoiser=(bad_layer,) + g.denoiser[1:])
    with pytest.raises(DiffusionGemmaDimError, match="bidirectional"):
        bd.verify_block_diffusion_step(bad)


def test_step_distinguishes_causal_prefill_from_bidirectional_canvas():
    # The encoder read is causal; the canvas denoiser is bidirectional — the
    # block-diffusion distinction, surfaced in one compiler-visible region.
    g = bd.build_block_diffusion_step(DiffusionGemmaConfig(), num_denoise_layers=2)
    assert g.encoder_read.attrs["causal"] is True
    assert g.denoiser[0].find("attention").attrs["causal"] is False


# ── Bidirectional attention mechanism ────────────────────────────────────────

def test_bidirectional_attention_sees_future_positions():
    rng = np.random.default_rng(1)
    q = rng.standard_normal((4, 8))
    k = rng.standard_normal((4, 8))
    v = rng.standard_normal((4, 8))
    # Perturb the LAST key/value (a "future" position).
    k2, v2 = k.copy(), v.copy()
    k2[3] += 1.0
    v2[3] += 1.0
    bi_a = bd._attention(q, k, v, causal=False)
    bi_b = bd._attention(q, k2, v2, causal=False)
    ca_a = bd._attention(q, k, v, causal=True)
    ca_b = bd._attention(q, k2, v2, causal=True)
    # Bidirectional: position 0's output changes when a later position changes.
    assert not np.allclose(bi_a[0], bi_b[0])
    # Causal: position 0 cannot see position 3, so its output is unchanged.
    np.testing.assert_allclose(ca_a[0], ca_b[0], atol=1e-12)


# ── Reference runner ─────────────────────────────────────────────────────────

def test_runner_executes_one_step():
    cfg = _tiny_cfg()
    canvas, enc, w = _runner_inputs(cfg, seed=3)
    sc = SamplerConfig(vocab_size=cfg.vocab_size, num_steps=48)
    r = bd.run_block_diffusion_step(
        canvas, enc, w, step=3, sampler_config=sc, num_denoise_layers=2,
        rng_key=7, top_k=cfg.num_experts_per_tok)
    assert r.tokens.shape == (cfg.canvas_size,)
    assert np.array_equal(r.renoise_mask, ~r.accepted_mask)
    assert r.committed == int(r.accepted_mask.sum())
    assert r.stop_reason in {"continue", "all_accepted", "stability", "eos", "max_steps"}


def test_runner_deterministic_by_key():
    cfg = _tiny_cfg()
    canvas, enc, w = _runner_inputs(cfg, seed=4)
    sc = SamplerConfig(vocab_size=cfg.vocab_size, num_steps=48, entropy_threshold=100.0)
    a = bd.run_block_diffusion_step(canvas, enc, w, step=1, sampler_config=sc,
                                    num_denoise_layers=2, rng_key=11, top_k=2)
    b = bd.run_block_diffusion_step(canvas, enc, w, step=1, sampler_config=sc,
                                    num_denoise_layers=2, rng_key=11, top_k=2)
    np.testing.assert_array_equal(a.tokens, b.tokens)


def test_runner_last_step_reports_max_steps():
    cfg = _tiny_cfg()
    canvas, enc, w = _runner_inputs(cfg, seed=5)
    # tight thresholds + high-entropy logits → nothing accepted, last step.
    sc = SamplerConfig(vocab_size=cfg.vocab_size, num_steps=6,
                       entropy_threshold=1e-6, stability_entropy=1e-9)
    r = bd.run_block_diffusion_step(canvas, enc, w, step=5, sampler_config=sc,
                                    num_denoise_layers=2, rng_key=0, top_k=2)
    assert r.stop_reason == "max_steps"
