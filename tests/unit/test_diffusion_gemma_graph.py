"""DiffusionGemma Phase A — production text block graph + config-aware verifier.

Covers the work plan's leading "Graph/spec tests" group:
  * shape-only production graph for one sliding and one full-attention layer;
  * verifier rejection for Q/K/V head mismatch, expert-count mismatch, wrong
    moe_intermediate_size, and wrong vocab size;
  * golden test proving causal prefill and bidirectional canvas are represented
    as different attention modes.
"""

from __future__ import annotations

import dataclasses

import pytest

from tessera.models import diffusion_gemma as dg
from tessera.models.diffusion_gemma import (
    DiffusionGemmaConfig,
    DiffusionGemmaDimError,
    GraphNode,
    TextBlockGraph,
    build_lm_head,
    build_text_block,
    estimated_param_counts,
    verify_config,
    verify_lm_head,
    verify_param_budget,
    verify_text_block,
)


_EXPECTED_OPS = (
    "rmsnorm", "q_proj", "k_proj", "v_proj", "rope", "rope", "attention",
    "o_proj", "residual_add", "rmsnorm", "router", "moe_swiglu_block",
    "shared_expert", "moe_combine", "residual_add",
)


# ── Shape-only production graph ──────────────────────────────────────────────

def test_card_facts():
    cfg = DiffusionGemmaConfig()
    # Gemma 4 26B A4B model-card facts.
    assert cfg.num_layers == 30
    assert cfg.num_experts == 128
    assert cfg.num_experts_per_tok == 8
    assert cfg.num_shared_experts == 1
    assert cfg.vocab_size == 262144
    assert cfg.context_length == 262144
    assert cfg.sliding_window == 1024
    assert cfg.final_layer_global is True
    assert cfg.global_unified_kv is True
    assert cfg.rope_variant == "p_rope"
    assert cfg.modalities == ("text", "image")
    # recommended sampling defaults
    assert (cfg.sample_temperature, cfg.sample_top_p, cfg.sample_top_k) == (1.0, 0.95, 64)
    assert cfg.canvas_size == 256  # block-diffusion design param (not a Gemma spec)


def test_full_attention_layer_shape_only():
    cfg = DiffusionGemmaConfig()
    g = build_text_block(cfg, layer_index=5, causal=False)  # layer 5 → "full" (global)
    assert g.attention_mode == "full"
    assert g.op_sequence() == _EXPECTED_OPS
    attn = g.find("attention")
    assert attn.attrs["mode"] == "full"
    assert attn.attrs["sliding_window"] is None
    # global layer: unified KV (single head) + p-RoPE
    assert attn.attrs["unified_kv"] is True
    assert attn.attrs["num_kv_heads"] == 1
    assert attn.attrs["rope_variant"] == "p_rope"
    # residual-shaped: starts and ends at (T, hidden)
    assert g.nodes[0].inputs[0][-1] == cfg.hidden_size
    assert g.nodes[-1].output[-1] == cfg.hidden_size
    # Q is full width; K/V unified to one head on a global layer
    assert g.find("q_proj").output[-1] == cfg.attn_dim
    assert g.find("k_proj").output[-1] == cfg.head_dim
    # router → num_experts; expert path carries the FFN width
    assert g.find("router").output[-1] == cfg.num_experts
    assert g.find("moe_swiglu_block").inputs[1] == (cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size)


def test_sliding_attention_layer_shape_only():
    cfg = DiffusionGemmaConfig()
    g = build_text_block(cfg, layer_index=0, causal=True)   # layer 0 → "sliding"
    assert g.attention_mode == "sliding"
    attn = g.find("attention")
    assert attn.attrs["mode"] == "sliding"
    assert attn.attrs["sliding_window"] == cfg.sliding_window
    # sliding layer keeps GQA KV heads + standard RoPE
    assert attn.attrs["num_kv_heads"] == cfg.num_kv_heads
    assert attn.attrs["unified_kv"] is False
    assert attn.attrs["rope_variant"] == "rope"
    assert g.find("k_proj").output[-1] == cfg.kv_dim


def test_final_layer_is_always_global():
    cfg = DiffusionGemmaConfig()
    last = build_text_block(cfg, layer_index=cfg.num_layers - 1)
    assert last.attention_mode == "full"


def test_lm_head_vocab_projection():
    cfg = DiffusionGemmaConfig()
    head = build_lm_head(cfg)
    assert head.output[-1] == cfg.vocab_size
    assert head.attrs["logit_softcap"] == cfg.final_logit_softcap
    verify_lm_head(head, cfg)  # no raise


# ── Verifier rejection tests (the four named contracts) ──────────────────────

def test_reject_qkv_head_mismatch():
    # 15 query heads is not a multiple of 8 KV heads → GQA invalid.
    bad = dataclasses.replace(DiffusionGemmaConfig(), num_attention_heads=15, num_kv_heads=8)
    with pytest.raises(DiffusionGemmaDimError, match="GQA head mismatch"):
        verify_config(bad)
    with pytest.raises(DiffusionGemmaDimError):
        build_text_block(bad)


def test_reject_expert_count_mismatch():
    bad = dataclasses.replace(DiffusionGemmaConfig(), num_experts=128, num_experts_per_tok=200)
    with pytest.raises(DiffusionGemmaDimError, match="num_experts_per_tok"):
        verify_config(bad)


def test_reject_wrong_moe_intermediate_size():
    # Build a valid graph, then corrupt the expert-weight FFN width — the
    # graph-level verifier must reject it.
    cfg = DiffusionGemmaConfig()
    g = build_text_block(cfg)
    moe = g.find("moe_swiglu_block")
    bad_F = cfg.moe_intermediate_size + 1
    bad_moe = dataclasses.replace(
        moe,
        inputs=((("T"), cfg.hidden_size),
                (cfg.num_experts, cfg.hidden_size, bad_F),
                (cfg.num_experts, cfg.hidden_size, bad_F),
                (cfg.num_experts, bad_F, cfg.hidden_size),
                (cfg.num_experts,)),
    )
    bad_nodes = tuple(bad_moe if n.op == "moe_swiglu_block" else n for n in g.nodes)
    bad_graph = dataclasses.replace(g, nodes=bad_nodes)
    with pytest.raises(DiffusionGemmaDimError, match="moe_intermediate_size"):
        verify_text_block(bad_graph, cfg)


def test_reject_wrong_vocab_size():
    cfg = DiffusionGemmaConfig()
    head = build_lm_head(cfg)
    bad_head = dataclasses.replace(head, output=((("T"), cfg.vocab_size - 1)))
    with pytest.raises(DiffusionGemmaDimError, match="vocab_size"):
        verify_lm_head(bad_head, cfg)


def test_reject_router_width_mismatch():
    # A graph-level guard distinct from the config check: router out ≠ num_experts.
    cfg = DiffusionGemmaConfig()
    g = build_text_block(cfg)
    router = g.find("router")
    bad_router = dataclasses.replace(router, output=((("T"), cfg.num_experts + 1)))
    bad_graph = dataclasses.replace(
        g, nodes=tuple(bad_router if n.op == "router" else n for n in g.nodes))
    with pytest.raises(DiffusionGemmaDimError, match="router out width"):
        verify_text_block(bad_graph, cfg)


# ── Golden: causal prefill vs bidirectional canvas are different modes ────────

def test_causal_prefill_vs_bidirectional_canvas_differ():
    cfg = DiffusionGemmaConfig()
    prefill = build_text_block(cfg, layer_index=5, causal=True)   # encoder prefill
    canvas = build_text_block(cfg, layer_index=5, causal=False)   # 256-token canvas
    # Same op sequence and window policy ...
    assert prefill.op_sequence() == canvas.op_sequence()
    assert prefill.attention_mode == canvas.attention_mode == "full"
    # ... but the attention role differs: causal prefill vs bidirectional canvas.
    assert prefill.find("attention").attrs["causal"] is True
    assert canvas.find("attention").attrs["causal"] is False
    assert prefill.causal is True and canvas.causal is False


# ── Parameter-budget estimator (self-check vs published model size) ──────────

def test_param_estimate_internally_consistent():
    est = estimated_param_counts(DiffusionGemmaConfig())
    assert est["active"] <= est["total"]
    assert est["total_b"] == round(est["total"] / 1e9, 2)
    assert est["bf16_gb"] == round(est["total"] * 2 / 1e9, 1)
    # active uses top-k experts, total uses all — so total expert mass dominates.
    assert est["experts_all_per_layer"] > 0


def test_default_config_matches_26b_a4b_budget():
    # The calibrated defaults land within 10% of the card's 25.2B/3.8B budget.
    cfg = DiffusionGemmaConfig()
    verify_param_budget(cfg, total_b=cfg.total_params_b, active_b=cfg.active_params_b)
    est = estimated_param_counts(cfg)
    assert abs(est["total_b"] - 25.2) / 25.2 < 0.05
    assert abs(est["active_b"] - 3.8) / 3.8 < 0.05


def test_budget_check_rejects_miscalibrated_config():
    import dataclasses
    # Doubling the expert FFN blows the total far past budget → rejected.
    bad = dataclasses.replace(DiffusionGemmaConfig(), moe_intermediate_size=4096)
    with pytest.raises(DiffusionGemmaDimError, match="param budget miss"):
        verify_param_budget(bad, total_b=25.2, active_b=3.8)


def test_param_budget_accepts_matching_target():
    # Accept path: passing the estimator's own numbers as the target validates.
    cfg = DiffusionGemmaConfig()
    est = estimated_param_counts(cfg)
    verify_param_budget(cfg, total_b=est["total_b"], active_b=est["active_b"], rel_tol=0.01)


def test_sliding_and_full_layers_differ_by_policy():
    cfg = DiffusionGemmaConfig()
    sliding = build_text_block(cfg, layer_index=0)
    full = build_text_block(cfg, layer_index=5)
    assert sliding.find("attention").attrs["mode"] == "sliding"
    assert full.find("attention").attrs["mode"] == "full"
    assert sliding.find("attention").attrs["sliding_window"] == cfg.sliding_window
    assert full.find("attention").attrs["sliding_window"] is None
