from __future__ import annotations

import json

import numpy as np
import pytest

from tessera import dflash_io as safeio
from tessera.data import VocabTokenizer
from tessera.models import minimax_m3
from tessera.models import minimax_m3_importer as imp
from tessera.models import moe_transformer_runtime as rt


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _hf_config(cfg):
    return {
        "model_type": "minimax_m3_vl",
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_layers,
        "vocab_size": cfg.vocab_size,
        "max_position_embeddings": cfg.context_length,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_kv_heads,
        "head_dim": cfg.head_dim,
        "rope_head_dim": cfg.rope_head_dim,
        "rope_theta": cfg.rope_theta,
        "n_routed_experts": cfg.num_experts,
        "num_experts_per_tok": cfg.num_experts_per_tok,
        "num_shared_experts": cfg.num_shared_experts,
        "first_k_dense": cfg.first_k_dense,
        "msa_top_k_blocks": cfg.msa_top_k_blocks,
        "msa_block_size": cfg.msa_block_size,
        "msa_index_dim": cfg.msa_index_dim,
        "msa_num_index_heads": cfg.msa_num_index_heads,
        "msa_score_type": cfg.msa_score_type,
        "sparse_attention_freq": list(cfg.msa_sparse_layer_freq),
    }


def _write_tokenizer(root):
    _write_json(root / "tokenizer_config.json", {
        "unk_token": "<unk>",
        "unk_token_id": 0,
        "chat_template": "{{ messages }}",
        "added_tokens_decoder": {
            "101": {"content": "<image>"},
            "102": {"content": "<video>"},
        },
    })
    _write_json(root / "tokenizer.json", {
        "model": {
            "type": "BPE",
            "vocab": {"<unk>": 0, "hello": 1, "world": 2, "<image>": 101, "<video>": 102},
        }
    })


def _write_safetensors(root, cfg):
    rng = np.random.default_rng(0)
    shapes = imp.expected_hf_text_tensor_shapes(cfg, layers=(0, 1))
    tensors = {
        name: (rng.standard_normal(shape).astype(np.float16) if shape else np.asarray(0, dtype=np.float16))
        for name, shape in shapes.items()
    }
    safeio.save_safetensors(root / "model-00001-of-00001.safetensors", tensors)
    _write_json(root / "model.safetensors.index.json", {
        "metadata": {"total_size": int(sum(arr.nbytes for arr in tensors.values()))},
        "weight_map": {name: "model-00001-of-00001.safetensors" for name in tensors},
    })
    return shapes


def _hf_fixture(tmp_path):
    cfg = minimax_m3.scaled_config()
    _write_json(tmp_path / "config.json", _hf_config(cfg))
    _write_tokenizer(tmp_path)
    shapes = _write_safetensors(tmp_path, cfg)
    return cfg, shapes


def test_import_minimax_m3_hf_manifest_reads_config_tokenizer_and_safetensors(tmp_path):
    cfg, shapes = _hf_fixture(tmp_path)

    manifest = imp.import_minimax_m3_hf(tmp_path, expected_config=cfg)

    assert manifest.config == cfg
    assert manifest.tokenizer is not None
    assert manifest.tokenizer.spec.kind == "bpe"
    assert manifest.tokenizer.spec.chat_template_present is True
    assert manifest.tokenizer.tokenizer.encode("hello world") == [1, 2]
    assert manifest.safetensors is not None
    assert "model.embed_tokens.weight" in manifest.safetensors.tensors
    imp.validate_safetensors_shapes(manifest.safetensors, shapes)
    assert manifest.vision_execution_supported is False


def test_minimax_m3_importer_rejects_bad_hf_config(tmp_path):
    cfg = minimax_m3.scaled_config()
    bad = _hf_config(cfg)
    bad["hidden_size"] = cfg.hidden_size + 1
    _write_json(tmp_path / "config.json", bad)
    _write_tokenizer(tmp_path)

    with pytest.raises(imp.MiniMaxM3ImportError, match="hidden_size"):
        imp.import_minimax_m3_hf(tmp_path, expected_config=cfg, require_weights=False)


def test_minimax_m3_importer_requires_tokenizer_files_when_requested(tmp_path):
    cfg = minimax_m3.scaled_config()
    _write_json(tmp_path / "config.json", _hf_config(cfg))

    with pytest.raises(imp.MiniMaxM3ImportError, match="missing tokenizer"):
        imp.import_minimax_m3_hf(tmp_path, expected_config=cfg)

    manifest = imp.import_minimax_m3_hf(tmp_path, expected_config=cfg, require_tokenizer=False)
    assert manifest.tokenizer is None


def test_safetensors_shape_validation_rejects_bad_text_weight(tmp_path):
    cfg, shapes = _hf_fixture(tmp_path)
    manifest = imp.read_safetensors_manifest(tmp_path)
    bad = dict(shapes)
    bad["model.embed_tokens.weight"] = (cfg.vocab_size + 1, cfg.hidden_size)

    with pytest.raises(imp.MiniMaxM3ImportError, match="shape mismatches"):
        imp.validate_safetensors_shapes(manifest, bad)


def test_selected_safetensors_tensor_loading(tmp_path):
    cfg, _ = _hf_fixture(tmp_path)

    tensors = imp.load_safetensors_tensors(tmp_path, names=["model.norm.weight"])

    assert set(tensors) == {"model.norm.weight"}
    assert tensors["model.norm.weight"].shape == (cfg.hidden_size,)
    assert tensors["model.norm.weight"].dtype == np.float16

    with pytest.raises(imp.MiniMaxM3ImportError, match="requested tensors"):
        imp.load_safetensors_tensors(tmp_path, names=["missing.weight"])


def test_prepare_multimodal_prompt_tracks_image_and_video_spans():
    tok = VocabTokenizer({"<unk>": 0, "hello": 1, "world": 2})
    vision = minimax_m3.MiniMaxM3VisionMetadata(
        image_token_index=101,
        video_token_index=102,
        image_seq_length=3,
        vision_segment_max_frames=2,
    )

    prepared = imp.prepare_multimodal_prompt(
        ["hello", imp.PromptSegment("image"), {"kind": "video", "frames": 2}, "world"],
        tok,
        vision=vision,
    )

    assert prepared.token_ids == (1, 101, 101, 101, 102, 102, 102, 102, 102, 102, 2)
    assert prepared.spans[0].kind == "image"
    assert (prepared.spans[0].start, prepared.spans[0].end) == (1, 4)
    assert prepared.spans[1].kind == "video"
    assert (prepared.spans[1].start, prepared.spans[1].end, prepared.spans[1].frames) == (4, 10, 2)
    assert prepared.vision_execution_supported is False

    with pytest.raises(imp.MiniMaxM3ImportError, match="video frames"):
        imp.prepare_multimodal_prompt([{"kind": "video", "frames": 3}], tok, vision=vision)


def test_text_only_multimodal_prompt_executes_through_text_tower():
    cfg = minimax_m3.scaled_config()
    weights = rt.synthetic_weights(cfg, seed=12)
    tok = VocabTokenizer({"<unk>": 0, "hello": 1, "world": 2})
    prepared = imp.prepare_multimodal_prompt(["hello world"], tok)

    logits = imp.execute_multimodal_prompt(cfg, weights, prepared)

    assert logits.shape == (2, cfg.vocab_size)
    assert np.isfinite(logits).all()


def test_image_prompt_requires_real_vision_execution():
    cfg = minimax_m3.scaled_config()
    weights = rt.synthetic_weights(cfg, seed=13)
    tok = VocabTokenizer({"<unk>": 0, "hello": 1})
    vision = minimax_m3.MiniMaxM3VisionMetadata(image_token_index=101, image_seq_length=2)
    prepared = imp.prepare_multimodal_prompt(["hello", imp.PromptSegment("image")], tok, vision=vision)

    with pytest.raises(imp.MiniMaxM3VisionExecutionError, match="vision/video execution is not implemented"):
        imp.execute_multimodal_prompt(cfg, weights, prepared)
