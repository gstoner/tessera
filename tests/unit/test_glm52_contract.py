"""GLM-5.2 contract tests: config truth + DSA IndexShare semantics."""

from __future__ import annotations

import numpy as np

from tessera.models import glm5
from tessera.models import moe_transformer as mt


def test_glm52_full_config_matches_released_shape():
    cfg = glm5.glm52_config()
    assert cfg.name == "glm5_2"
    assert cfg.num_layers == 78
    assert cfg.hidden_size == 6144
    assert cfg.context_length == 1_048_576
    assert cfg.attn_kind == "mla"
    assert cfg.num_attention_heads == 64
    assert cfg.num_kv_heads == 64
    assert cfg.head_dim == 192
    assert cfg.qk_head_dim == 256
    assert cfg.qk_nope_head_dim == 192
    assert cfg.qk_rope_head_dim == 64
    assert cfg.v_head_dim == 256
    assert cfg.q_lora_rank == 2048
    assert cfg.kv_lora_rank == 512
    assert cfg.num_experts == 256
    assert cfg.num_experts_per_tok == 8
    assert cfg.first_k_dense == 3
    assert cfg.index_n_heads == 32
    assert cfg.index_head_dim == 128
    assert cfg.index_topk == 2048
    assert cfg.index_topk_freq == 4
    assert cfg.mtp_num_layers == 1
    assert cfg.mtp_num_steps == 4
    assert cfg.mtp_index_share is True
    assert cfg.mtp_kv_share is True
    assert cfg.dtype == "bf16"
    assert cfg.rollout_kv_dtype == "fp8"
    mt.verify_config(cfg)
    mt.verify_param_budget(cfg)


def test_glm52_index_share_groups_are_four_layer_after_warmup():
    cfg = glm5.glm52_config()
    groups = mt.shared_topk_index_groups(cfg)
    assert groups[0].producer_layer == 0
    assert groups[0].consumer_layers == ()
    assert groups[2].producer_layer == 2
    assert groups[2].consumer_layers == (3, 4, 5)
    assert groups[3].producer_layer == 6
    assert groups[3].consumer_layers == (7, 8, 9)
    assert all(g.top_k == 2048 for g in groups)
    assert all(g.storage_policy == "current_query_only" for g in groups)


def test_glm52_graph_distinguishes_index_producer_and_consumer():
    cfg = glm5.scaled_config()
    producer = mt.build_block(cfg, layer_index=6)
    consumer = mt.build_block(cfg, layer_index=7)
    assert "dsa_topk_indexer" in producer.op_sequence()
    assert "shared_topk_index" not in producer.op_sequence()
    assert "shared_topk_index" in consumer.op_sequence()
    assert "dsa_topk_indexer" not in consumer.op_sequence()
    prod_attn = producer.find("deepseek_sparse_attention")
    cons_attn = consumer.find("deepseek_sparse_attention")
    assert prod_attn.attrs["indexer_mode"] == "full"
    assert cons_attn.attrs["indexer_mode"] == "shared"
    assert cons_attn.attrs["index_storage_policy"] == "current_query_only"


def test_glm52_graph_uses_qk_and_value_dimensions():
    cfg = glm5.glm52_config()
    graph = mt.build_block(cfg, layer_index=6)
    assert graph.find("q_proj").output == ("T", cfg.num_attention_heads * cfg.qk_head_dim)
    assert graph.find("latent_kv_expand_k").output == (
        "T", cfg.num_attention_heads * cfg.qk_head_dim)
    assert graph.find("latent_kv_expand_v").output == (
        "T", cfg.num_attention_heads * cfg.v_head_dim)
    assert graph.find("deepseek_sparse_attention").output == (
        "T", cfg.num_attention_heads * cfg.v_head_dim)
    assert graph.find("o_proj").inputs[0] == ("T", cfg.num_attention_heads * cfg.v_head_dim)


def test_deterministic_topk_tie_breaks_lowest_index():
    scores = np.array([[0.5, 1.0, 1.0, 0.7], [2.0, 2.0, 1.0, 2.0]])
    idx = mt.deterministic_topk_indices(scores, 2)
    np.testing.assert_array_equal(idx, np.array([[1, 2], [0, 1]]))


def test_index_share_memory_guard_is_current_query_only():
    cfg = glm5.glm52_config()
    groups = mt.shared_topk_index_groups(cfg)
    stored_indices_per_query = sum(g.top_k for g in groups if g.consumer_layers)
    replay_indices_per_token = cfg.num_layers * cfg.index_topk
    assert stored_indices_per_query < replay_indices_per_token
    assert all(g.storage_policy == "current_query_only" for g in groups)
