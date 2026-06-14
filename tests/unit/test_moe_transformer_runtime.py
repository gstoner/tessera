"""M5 — full MoE-transformer stack + autoregressive decode loop.

Headline oracle: KV-cached greedy decode ≡ full recompute (a non-circular,
whole-model cache-consistency proof), run for all three scaled frontier models
(DeepSeek-V3.2 MLA+MoE, GLM-5 GQA+MoE, Kimi-K2 MLA+MoE).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.models import deepseek_v32, glm5, kimi_k2
from tessera.models import moe_transformer_runtime as rt

SCALED = [deepseek_v32.scaled_config, glm5.scaled_config, kimi_k2.scaled_config]
IDS = ["deepseek_v32", "glm5", "kimi_k2"]


def _recompute_generate(cfg, weights, prompt, n):
    """Reference: re-run the full forward on the growing prefix each step."""
    seq = list(prompt)
    out = []
    for _ in range(n):
        logits = rt.forward(cfg, weights, seq)
        tok = int(np.argmax(logits[-1]))
        out.append(tok)
        seq.append(tok)
    return out


@pytest.mark.parametrize("make_cfg", SCALED, ids=IDS)
def test_forward_shape_and_finite(make_cfg):
    cfg = make_cfg()
    w = rt.synthetic_weights(cfg, seed=0)
    logits = rt.forward(cfg, w, [1, 5, 9, 2, 7])
    assert logits.shape == (5, cfg.vocab_size)
    assert np.isfinite(logits).all()


@pytest.mark.parametrize("make_cfg", SCALED, ids=IDS)
def test_kv_cached_decode_equals_recompute(make_cfg):
    """The M5 capstone oracle."""
    cfg = make_cfg()
    w = rt.synthetic_weights(cfg, seed=1)
    prompt = [3, 1, 4, 1, 5, 9]
    n = 6
    cached = rt.greedy_generate(cfg, w, prompt, n)
    recompute = _recompute_generate(cfg, w, prompt, n)
    assert cached == recompute
    assert len(cached) == n


@pytest.mark.parametrize("make_cfg", SCALED, ids=IDS)
def test_decode_step_logits_match_recompute(make_cfg):
    """Per-step logits (not just argmax) from the cache match the recompute, so
    the equality isn't an argmax-tie coincidence."""
    cfg = make_cfg()
    w = rt.synthetic_weights(cfg, seed=2)
    prompt = [2, 4, 6, 8]
    logits, state = rt.prefill(cfg, w, prompt, max_seq=len(prompt) + 4)
    np.testing.assert_allclose(logits, rt.forward(cfg, w, prompt)[-1],
                               rtol=1e-9, atol=1e-9)
    seq = list(prompt)
    for _ in range(4):
        tok = int(np.argmax(logits))
        seq.append(tok)
        logits, state = rt.decode_step(cfg, w, state, tok)
        np.testing.assert_allclose(logits, rt.forward(cfg, w, seq)[-1],
                                   rtol=1e-8, atol=1e-8)


def test_dsa_is_genuinely_engaged_not_dense():
    """With a long-enough sequence (multiple blocks), the DSA layers restrict
    attention — the DSA forward must differ from the same model run dense,
    proving the block-sparsity is actually wired in (not degenerating to dense).
    """
    import dataclasses
    cfg = deepseek_v32.scaled_config()           # MLA + DSA, block_size=4, top_k=2
    w = rt.synthetic_weights(cfg, seed=5)
    seq = list(range(20))                         # 20 tokens → 5 blocks > top_k+local
    dsa_logits = rt.forward(cfg, w, seq)
    dense_cfg = dataclasses.replace(cfg, sparse=None)
    dense_logits = rt.forward(dense_cfg, w, seq)
    assert not np.allclose(dsa_logits, dense_logits), "DSA collapsed to dense"


def test_dsa_decode_equals_recompute_long():
    """Decode ≡ recompute with real multi-block sparsity (offset-aware indexer)."""
    cfg = deepseek_v32.scaled_config()
    w = rt.synthetic_weights(cfg, seed=6)
    prompt = list(range(11))                      # spans multiple DSA blocks
    cached = rt.greedy_generate(cfg, w, prompt, 6)
    recompute = _recompute_generate(cfg, w, prompt, 6)
    assert cached == recompute


def test_first_layer_dense_rest_moe():
    """The scaled configs keep a leading dense layer then MoE (DeepSeek convention)."""
    cfg = deepseek_v32.scaled_config()
    w = rt.synthetic_weights(cfg)
    assert w.layers[0].is_moe is False
    assert all(lw.is_moe for lw in w.layers[cfg.first_k_dense:])
