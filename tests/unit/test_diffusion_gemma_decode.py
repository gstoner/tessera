"""DiffusionGemma Phase E — KV-cache promotion + decode loop.

single block → multi-step (≤48) → multi-block, over a real KVCacheHandle. This
is reference orchestration; the "no host-only round-trips in the native path"
property is NOT claimed here (it needs native kernels — Phase G).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from tessera.models.diffusion_gemma import DiffusionGemmaConfig
from tessera.models import moe_routing as mr
from tessera.models.sampler import SamplerConfig
from tessera.models.decode import BlockDiffusionDecoder


def _cfg(**over):
    base = dict(hidden_size=64, num_attention_heads=4, num_kv_heads=2, head_dim=16,
                num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
                shared_expert_intermediate_size=64, vocab_size=128, canvas_size=16)
    base.update(over)
    return dataclasses.replace(DiffusionGemmaConfig(), **base)


def _decoder(cfg, *, max_steps=48, entropy_threshold=0.5, seed=0):
    rng = np.random.default_rng(seed)
    w = {"moe": mr.synthetic_moe_weights(cfg, seed=seed + 1),
         "w_lm": (rng.standard_normal((cfg.hidden_size, cfg.vocab_size)) / 8).astype(np.float32)}
    embed = rng.standard_normal((cfg.vocab_size, cfg.hidden_size)) / 8
    sc = SamplerConfig(vocab_size=cfg.vocab_size, num_steps=max_steps,
                       entropy_threshold=entropy_threshold)
    return BlockDiffusionDecoder(cfg, w, embed, num_denoise_layers=2,
                                 max_steps=max_steps, sampler_config=sc, top_k=2)


def test_single_block_prefill_and_step():
    cfg = _cfg()
    dec = _decoder(cfg, seed=1)
    assert dec.context_len == 0
    r = dec.decode_block(rng_key=10)
    assert r.tokens.shape == (cfg.canvas_size,)
    assert dec.context_len == cfg.canvas_size       # KV promotion appended the block
    assert r.committed >= 1


def test_multi_step_converges_within_budget():
    cfg = _cfg()
    dec = _decoder(cfg, max_steps=48, seed=2)
    r = dec.decode_block(rng_key=5)
    assert r.steps <= 48
    assert r.stop_reason == "all_committed"
    assert r.committed == cfg.canvas_size
    # progress is monotone non-decreasing and ends at the full canvas.
    assert all(b >= a for a, b in zip(r.progress, r.progress[1:]))
    assert r.progress[-1] == cfg.canvas_size


def test_multi_block_kv_promotion_grows_context():
    cfg = _cfg()
    dec = _decoder(cfg, seed=3)
    dec.decode_block(rng_key=1)
    assert dec.context_len == cfg.canvas_size
    dec.decode_block(rng_key=2)
    assert dec.context_len == 2 * cfg.canvas_size   # second block read a non-empty cache
    assert len(dec.blocks) == 2


def test_max_steps_caps_decoding():
    cfg = _cfg()
    # Only 8 steps for a 16-position canvas → forced progress commits ≤8 → capped.
    dec = _decoder(cfg, max_steps=8, entropy_threshold=1e-9, seed=4)
    r = dec.decode_block(rng_key=0)
    assert r.steps == 8
    assert r.stop_reason == "max_steps"
    assert r.committed <= 8


def test_max_steps_budget_is_enforced():
    cfg = _cfg()
    with pytest.raises(ValueError, match="<= 48"):
        _decoder(cfg, max_steps=64)


def test_sliding_window_context_read_is_bounded():
    cfg = _cfg(canvas_size=8, sliding_window=10)
    dec = _decoder(cfg, seed=6)
    for key in range(4):                            # 4 blocks × 8 = 32 context tokens
        dec.decode_block(rng_key=key)
    assert dec.context_len == 32
    k_full, _ = dec.context_kv(sliding=False)
    k_slide, _ = dec.context_kv(sliding=True)
    assert k_full.shape[0] == 32
    assert k_slide.shape[0] == cfg.sliding_window   # windowed read is bounded


def test_native_no_round_trip_claim_is_not_made():
    # Honesty guard: this phase is reference orchestration, not a native-runtime
    # proof. The module must say so (mirrors the audit "real proof only" rule).
    src = (Path(__file__).resolve().parents[2]
           / "python" / "tessera" / "models" / "decode.py").read_text()
    assert "reference orchestration" in src
    assert "does **not** prove" in src and "no host-only round-trips" in src
