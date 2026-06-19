"""M5 — full MoE-transformer stack + autoregressive decode loop.

Headline oracle: KV-cached greedy decode ≡ full recompute (a non-circular,
whole-model cache-consistency proof), run for the scaled frontier models
(DeepSeek-V3.2 MLA+MoE, GLM-5 GQA+MoE, Kimi-K2 MLA+MoE, MiniMax-M3 GQA+MSA+MoE).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tessera.models import deepseek_v32, glm5, kimi_k2, minimax_m3
from tessera.models import moe_transformer as mt
from tessera.models import moe_transformer_runtime as rt

SCALED = [deepseek_v32.scaled_config, glm5.scaled_config, kimi_k2.scaled_config, minimax_m3.scaled_config]
IDS = ["deepseek_v32", "glm5", "kimi_k2", "minimax_m3"]


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
def test_forward_embeds_matches_token_forward(make_cfg):
    cfg = make_cfg()
    w = rt.synthetic_weights(cfg, seed=20)
    ids = [1, 5, 9, 2, 7]
    embeds = rt.embed_tokens(w, ids)

    np.testing.assert_allclose(
        rt.forward_embeds(cfg, w, embeds),
        rt.forward(cfg, w, ids),
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.parametrize("make_cfg", SCALED, ids=IDS)
def test_prefill_embeds_matches_token_prefill(make_cfg):
    cfg = make_cfg()
    w = rt.synthetic_weights(cfg, seed=21)
    ids = [2, 4, 6, 8]
    embeds = rt.embed_tokens(w, ids)

    logits_from_embeds, state_from_embeds = rt.prefill_embeds(cfg, w, embeds, max_seq=10)
    logits_from_ids, state_from_ids = rt.prefill(cfg, w, ids, max_seq=10)

    np.testing.assert_allclose(logits_from_embeds, logits_from_ids, rtol=1e-9, atol=1e-9)
    assert state_from_embeds.position == state_from_ids.position == len(ids)
    assert [cache[0] for cache in state_from_embeds.caches] == [cache[0] for cache in state_from_ids.caches]


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


def _lsa_config():
    """A scaled MoE-transformer that runs with Lookahead Sparse Attention."""
    return mt.MoETransformerConfig(
        name="lsa_scaled", hidden_size=256, num_layers=4, vocab_size=1024,
        attn_kind="gqa", num_attention_heads=8, num_kv_heads=2, head_dim=32,
        rope_head_dim=32, sparse="lsa", lsa_window_size=4, dsa_block_size=4,
        lsa_threshold=0.5, num_experts=8, num_experts_per_tok=2,
        num_shared_experts=1, moe_intermediate_size=256,
        shared_expert_intermediate_size=256, first_k_dense=1,
        dense_intermediate_size=512)


def test_lsa_decode_equals_recompute():
    """A model running with LSA attention: KV-cached greedy decode ≡ recompute
    (offset-aware lookahead selection is decode-loop-consistent)."""
    cfg = _lsa_config()
    w = rt.synthetic_weights(cfg, seed=7)
    prompt = list(range(11))
    cached = rt.greedy_generate(cfg, w, prompt, 6)
    recompute = _recompute_generate(cfg, w, prompt, 6)
    assert cached == recompute


def test_lsa_is_genuinely_engaged_not_dense():
    """Over a long-enough sequence the LSA layers prune history — the LSA forward
    must differ from a dense run (threshold below 1 would otherwise select all)."""
    cfg = dataclasses.replace(_lsa_config(), lsa_threshold=0.9, lsa_window_size=2)
    w = rt.synthetic_weights(cfg, seed=8)
    seq = list(range(20))
    lsa_logits = rt.forward(cfg, w, seq)
    dense_logits = rt.forward(dataclasses.replace(cfg, sparse=None), w, seq)
    assert not np.allclose(lsa_logits, dense_logits), "LSA collapsed to dense"


def test_lsa_graph_builds_and_verifies():
    cfg = _lsa_config()
    g = mt.build_block(cfg, layer_index=cfg.num_layers - 1)
    assert "lookahead_sparse_attention" in g.op_sequence()
    mt.verify_block(g, cfg)


def test_msa_decode_equals_recompute_long():
    """MiniMax-style MSA decode uses offset-aware block selection, so cached
    greedy decode matches full recompute across several selected-block changes."""
    cfg = minimax_m3.scaled_config()
    w = rt.synthetic_weights(cfg, seed=9)
    prompt = list(range(11))                      # spans multiple MSA blocks
    cached = rt.greedy_generate(cfg, w, prompt, 6)
    recompute = _recompute_generate(cfg, w, prompt, 6)
    assert cached == recompute


def test_msa_is_genuinely_engaged_not_dense():
    """Over multiple blocks, MSA layers must prune history rather than falling
    through to dense GQA attention."""
    cfg = minimax_m3.scaled_config()
    w = rt.synthetic_weights(cfg, seed=10)
    seq = list(range(20))
    msa_logits = rt.forward(cfg, w, seq)
    dense_logits = rt.forward(dataclasses.replace(cfg, sparse=None), w, seq)
    assert not np.allclose(msa_logits, dense_logits), "MSA collapsed to dense"


def test_minimax_m3_dense_warmup_runtime_cache_is_dense_then_msa():
    """The runtime honors MiniMax's dense warmup layers before sparse MSA."""
    cfg = minimax_m3.scaled_config()
    w = rt.synthetic_weights(cfg, seed=11)
    _, state = rt.prefill(cfg, w, list(range(9)), max_seq=16)
    assert state.caches[0][0] == "gqa"
    assert all(cache[0] == "msa" for cache in state.caches[1:])


def test_first_layer_dense_rest_moe():
    """The scaled configs keep a leading dense layer then MoE (DeepSeek convention)."""
    cfg = deepseek_v32.scaled_config()
    w = rt.synthetic_weights(cfg)
    assert w.layers[0].is_moe is False
    assert all(lw.is_moe for lw in w.layers[cfg.first_k_dense:])
