"""MiniMax-M3 contract tests: GQA + MSA + MoE text tower, staged multimodal metadata."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tessera.models import minimax_m3
from tessera.models import moe_transformer as mt


def test_minimax_m3_full_config_matches_hf_shape():
    cfg = minimax_m3.config()
    assert cfg.name == "minimax_m3"
    assert cfg.hidden_size == 6144
    assert cfg.num_layers == 60
    assert cfg.vocab_size == 200064
    assert cfg.context_length == 1_048_576
    assert cfg.attn_kind == "gqa"
    assert cfg.num_attention_heads == 64
    assert cfg.num_kv_heads == 4
    assert cfg.head_dim == 128
    assert cfg.rope_head_dim == 64
    assert cfg.rope_theta == 5_000_000.0
    assert cfg.sparse == "msa"
    assert cfg.msa_top_k_blocks == 16
    assert cfg.msa_block_size == 128
    assert cfg.msa_index_dim == 128
    assert cfg.msa_num_index_heads == 4
    assert cfg.msa_score_type == "max"
    assert cfg.msa_sparse_layer_freq == (0, 0, 0) + (1,) * 57
    assert cfg.num_experts == 128
    assert cfg.num_experts_per_tok == 4
    assert cfg.num_shared_experts == 1
    assert cfg.first_k_dense == 3
    assert cfg.dtype == "bf16"
    mt.verify_config(cfg)


def test_minimax_m3_dense_warmup_and_sparse_moe_layer_shapes():
    cfg = minimax_m3.config()
    dense = mt.build_block(cfg, layer_index=0)
    sparse = mt.build_block(cfg, layer_index=cfg.num_layers - 1)

    assert dense.is_moe is False
    assert "attention" in dense.op_sequence()
    assert "msa_sparse_attention" not in dense.op_sequence()
    assert "dense_ffn" in dense.op_sequence()

    assert sparse.is_moe is True
    assert "msa_sparse_attention" in sparse.op_sequence()
    assert "router" in sparse.op_sequence()
    assert "moe_swiglu_block" in sparse.op_sequence()
    assert "shared_expert" in sparse.op_sequence()
    assert sparse.find("rope").attrs["rope_head_dim"] == 64


def test_minimax_m3_all_full_layers_build_and_verify():
    cfg = minimax_m3.config()
    for layer in range(cfg.num_layers):
        graph = mt.build_block(cfg, layer_index=layer)
        mt.verify_block(graph, cfg)


def test_minimax_m3_sparse_frequency_matches_every_full_layer():
    cfg = minimax_m3.config()
    dense_layers = []
    sparse_layers = []
    for layer in range(cfg.num_layers):
        graph = mt.build_block(cfg, layer_index=layer)
        ops = graph.op_sequence()
        if layer < cfg.first_k_dense:
            dense_layers.append(layer)
            assert cfg.uses_msa_layer(layer) is False
            assert graph.is_moe is False
            assert "attention" in ops
            assert "dense_ffn" in ops
            assert "msa_sparse_attention" not in ops
            assert "router" not in ops
        else:
            sparse_layers.append(layer)
            assert cfg.uses_msa_layer(layer) is True
            assert graph.is_moe is True
            assert "msa_sparse_attention" in ops
            assert "attention" not in ops
            assert "router" in ops
            assert "moe_swiglu_block" in ops
            assert "shared_expert" in ops

    assert tuple(dense_layers) == (0, 1, 2)
    assert tuple(sparse_layers) == tuple(range(3, 60))


def test_minimax_m3_param_estimate_matches_released_scale():
    cfg = minimax_m3.config()
    est = mt.estimated_param_counts(cfg)
    assert abs(est["total_b"] - 427.04) <= 0.05 * 427.04
    assert abs(est["active_b"] - 23.0) <= 0.15 * 23.0
    mt.verify_param_budget(cfg)


def test_minimax_m3_scaled_config_builds_every_layer():
    cfg = minimax_m3.scaled_config()
    mt.verify_config(cfg)
    for layer in range(cfg.num_layers):
        mt.build_block(cfg, layer_index=layer)


def test_minimax_m3_multimodal_metadata_is_importer_only():
    meta = minimax_m3.VISION_METADATA
    assert meta.image_token_index == 200025
    assert meta.video_token_index == 200026
    assert meta.image_seq_length == 576
    assert meta.patch_size == 14
    assert meta.image_size == 2016
    assert meta.projector_hidden_size == 6144
    assert meta.spatial_merge_size == 2
    assert meta.temporal_patch_size == 2
    assert meta.vision_segment_max_frames == 4
    assert meta.vision_execution_supported is False


def test_minimax_m3_full_multimodal_graph_builds_for_image_and_video():
    cfg = minimax_m3.config()
    graph = minimax_m3.build_multimodal_graph(cfg, frames=4)

    minimax_m3.verify_multimodal_graph(graph)
    ops = graph.op_sequence()
    assert ops.count("image_preprocess") == 1
    assert ops.count("video_frame_sample") == 1
    assert ops.count("patch_embed") == 2
    assert ops.count("patch_merge") == 2
    assert ops.count("media_project") == 2
    assert ops[-1] == "splice_embeddings"

    image_project, video_project = graph.find_all("media_project")
    assert image_project.output == (576, 6144)
    assert video_project.output == (4 * 576, 6144)
    assert graph.find("image_preprocess").attrs["image_size"] == 2016
    assert graph.find("image_preprocess").attrs["patch_size"] == 14
    assert graph.find("splice_embeddings").attrs["vision_execution_supported"] is False


def test_minimax_m3_scaled_multimodal_graph_preserves_executable_shape():
    cfg = minimax_m3.scaled_config()
    vision = minimax_m3.scaled_vision_metadata()
    graph = minimax_m3.build_multimodal_graph(cfg, vision=vision, frames=2)

    minimax_m3.verify_multimodal_graph(graph)
    image_project, video_project = graph.find_all("media_project")
    assert image_project.output == (vision.image_seq_length, cfg.hidden_size)
    assert video_project.output == (2 * vision.image_seq_length, cfg.hidden_size)
    assert graph.find("splice_embeddings").output == ("T+media", cfg.hidden_size)


def test_msa_config_rejects_missing_or_bad_shape_contracts():
    cfg = minimax_m3.scaled_config()
    with pytest.raises(mt.MoETransformerDimError, match="msa_top_k_blocks"):
        mt.verify_config(replace(cfg, msa_top_k_blocks=0))
    with pytest.raises(mt.MoETransformerDimError, match="msa_block_size"):
        mt.verify_config(replace(cfg, msa_block_size=0))
    with pytest.raises(mt.MoETransformerDimError, match="msa_sparse_layer_freq"):
        mt.verify_config(replace(cfg, msa_sparse_layer_freq=(0, 1)))
    with pytest.raises(mt.MoETransformerDimError, match="msa_score_type"):
        mt.verify_config(replace(cfg, msa_score_type="softmax"))


def test_msa_attrs_on_emitted_graph_match_config():
    cfg = minimax_m3.scaled_config()
    dense = mt.build_block(cfg, layer_index=0)
    sparse = mt.build_block(cfg, layer_index=1)

    assert dense.find("attention").attrs["attn_kind"] == "gqa"
    attn = sparse.find("msa_sparse_attention")
    assert attn.attrs["top_k_blocks"] == cfg.msa_top_k_blocks
    assert attn.attrs["block_size"] == cfg.msa_block_size
    assert attn.attrs["index_dim"] == cfg.msa_index_dim
    assert attn.attrs["num_index_heads"] == cfg.msa_num_index_heads
    assert attn.attrs["score_type"] == cfg.msa_score_type
    assert attn.attrs["force_local_block"] is True
    assert attn.attrs["causal"] is True
