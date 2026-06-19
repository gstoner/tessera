from __future__ import annotations

import json

import numpy as np
import pytest

from tessera import dflash_io as safeio
from tessera.data import VocabTokenizer
from tessera.models import minimax_m3
from tessera.models import minimax_m3_importer as imp
from tessera.models import moe_transformer_runtime as rt
from tessera.models import vision_transformer as vt


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
    assert manifest.processor is not None
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


def test_image_prompt_requires_projected_media_embeddings():
    cfg = minimax_m3.scaled_config()
    weights = rt.synthetic_weights(cfg, seed=13)
    tok = VocabTokenizer({"<unk>": 0, "hello": 1})
    vision = minimax_m3.MiniMaxM3VisionMetadata(image_token_index=101, image_seq_length=2)
    prepared = imp.prepare_multimodal_prompt(["hello", imp.PromptSegment("image")], tok, vision=vision)

    with pytest.raises(imp.MiniMaxM3VisionExecutionError, match="requires projected media embeddings"):
        imp.execute_multimodal_prompt(cfg, weights, prepared)


def test_projected_media_embeddings_splice_and_execute():
    cfg = minimax_m3.scaled_config()
    weights = rt.synthetic_weights(cfg, seed=14)
    tok = VocabTokenizer({"<unk>": 0, "hello": 1, "world": 2})
    vision = minimax_m3.MiniMaxM3VisionMetadata(image_token_index=200061, image_seq_length=2)
    prepared = imp.prepare_multimodal_prompt(
        ["hello", imp.PromptSegment("image"), "world"], tok, vision=vision)
    media = [np.full((2, cfg.hidden_size), 0.125, dtype=np.float32)]

    spliced = imp.splice_media_embeddings(cfg, weights, prepared, media_embeddings=media)
    logits = imp.execute_multimodal_prompt(cfg, weights, prepared, media_embeddings=media)

    assert spliced.shape == (4, cfg.hidden_size)
    np.testing.assert_allclose(spliced[1:3], media[0])
    np.testing.assert_allclose(logits, rt.forward_embeds(cfg, weights, spliced))


def test_projected_media_prefill_can_continue_decode():
    cfg = minimax_m3.scaled_config()
    weights = rt.synthetic_weights(cfg, seed=15)
    tok = VocabTokenizer({"<unk>": 0, "hello": 1})
    vision = minimax_m3.MiniMaxM3VisionMetadata(image_token_index=200061, image_seq_length=2)
    prepared = imp.prepare_multimodal_prompt(["hello", imp.PromptSegment("image")], tok, vision=vision)
    media = {"image": np.full((2, cfg.hidden_size), -0.25, dtype=np.float32)}
    spliced = imp.splice_media_embeddings(cfg, weights, prepared, media_embeddings=media)

    logits, state = imp.prefill_multimodal_prompt(
        cfg, weights, prepared, media_embeddings=media, max_seq=spliced.shape[0] + 1)
    next_token = int(np.argmax(logits))
    decoded_logits, _ = rt.decode_step(cfg, weights, state, next_token)
    recompute_embeds = np.concatenate([spliced, rt.embed_tokens(weights, [next_token])], axis=0)

    np.testing.assert_allclose(
        decoded_logits,
        rt.forward_embeds(cfg, weights, recompute_embeds)[-1],
        rtol=1e-8,
        atol=1e-8,
    )


def test_projected_media_embedding_shape_is_validated():
    cfg = minimax_m3.scaled_config()
    weights = rt.synthetic_weights(cfg, seed=16)
    tok = VocabTokenizer({"<unk>": 0, "hello": 1})
    vision = minimax_m3.MiniMaxM3VisionMetadata(image_token_index=101, image_seq_length=2)
    prepared = imp.prepare_multimodal_prompt(["hello", imp.PromptSegment("image")], tok, vision=vision)

    with pytest.raises(imp.MiniMaxM3ImportError, match="media embedding"):
        imp.splice_media_embeddings(
            cfg, weights, prepared,
            media_embeddings=[np.zeros((1, cfg.hidden_size), dtype=np.float32)],
        )


def test_processor_config_imports_hf_metadata(tmp_path):
    vision = minimax_m3.scaled_vision_metadata()
    _write_json(tmp_path / "processor_config.json", {
        "size": {"height": 8, "width": 8},
        "patch_size": 2,
        "image_seq_length": 4,
        "spatial_merge_size": 2,
        "temporal_patch_size": 1,
        "max_frames": 2,
        "num_channels": 3,
        "image_mean": [0.1, 0.2, 0.3],
        "image_std": [1.0, 1.0, 1.0],
    })

    processor = imp.import_processor_config(tmp_path, vision=vision)

    assert processor.image_size == 8
    assert processor.patch_size == 2
    assert processor.image_seq_length == 4
    assert processor.max_frames == 2
    assert processor.image_mean == (0.1, 0.2, 0.3)


def test_reference_vision_runtime_executes_image_and_video():
    cfg = minimax_m3.scaled_config()
    vision = minimax_m3.scaled_vision_metadata()
    runtime = imp.synthetic_vision_runtime(cfg, vision=vision, seed=17)
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    video = np.stack([image, image[::-1]], axis=0)

    image_emb = imp.encode_media_inputs(
        cfg,
        imp.prepare_multimodal_prompt([imp.PromptSegment("image")], VocabTokenizer({"<unk>": 0}), vision=vision),
        media_inputs=[image],
        vision_runtime=runtime,
    )
    video_emb = imp.encode_media_inputs(
        cfg,
        imp.prepare_multimodal_prompt([imp.PromptSegment("video", frames=2)], VocabTokenizer({"<unk>": 0}), vision=vision),
        media_inputs=[video],
        vision_runtime=runtime,
    )

    assert image_emb.embeddings[0].shape == (vision.image_seq_length, cfg.hidden_size)
    assert video_emb.embeddings[0].shape == (2 * vision.image_seq_length, cfg.hidden_size)
    assert np.isfinite(image_emb.embeddings[0]).all()
    assert np.isfinite(video_emb.embeddings[0]).all()


def test_raw_media_inputs_execute_through_reference_vision_tower():
    cfg = minimax_m3.scaled_config()
    weights = rt.synthetic_weights(cfg, seed=18)
    vision = minimax_m3.scaled_vision_metadata()
    runtime = imp.synthetic_vision_runtime(cfg, vision=vision, seed=19)
    tok = VocabTokenizer({"<unk>": 0, "hello": 1, "world": 2})
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    prepared = imp.prepare_multimodal_prompt(["hello", imp.PromptSegment("image"), "world"], tok, vision=vision)

    projected = imp.encode_media_inputs(cfg, prepared, media_inputs={"image": image}, vision_runtime=runtime)
    logits = imp.execute_multimodal_prompt(
        cfg, weights, prepared, media_inputs={"image": image}, vision_runtime=runtime)
    spliced = imp.splice_media_embeddings(cfg, weights, prepared, media_embeddings=projected.embeddings)

    assert logits.shape == (len(prepared.token_ids), cfg.vocab_size)
    np.testing.assert_allclose(logits, rt.forward_embeds(cfg, weights, spliced), rtol=1e-9, atol=1e-9)


def test_vision_safetensors_roundtrip_into_typed_runtime_weights(tmp_path):
    cfg = minimax_m3.scaled_config()
    vision = minimax_m3.scaled_vision_metadata()
    processor = minimax_m3.processor_config(vision)
    vision_cfg = minimax_m3.vision_transformer_config(
        text_hidden_size=cfg.hidden_size,
        vision=vision,
        num_layers=1,
        num_heads=4,
    )
    shapes = imp.expected_hf_vision_tensor_shapes(cfg, vision_cfg, layers=(0,))
    rng = np.random.default_rng(20)
    tensors = {name: rng.standard_normal(shape).astype(np.float16) for name, shape in shapes.items()}
    safeio.save_safetensors(tmp_path / "vision.safetensors", tensors)

    direct = imp.load_vision_runtime_weights(tensors, vision_cfg)
    loaded_tensors = imp.load_safetensors_tensors(tmp_path / "vision.safetensors", names=sorted(tensors))
    loaded = imp.load_vision_runtime_weights(loaded_tensors, vision_cfg)
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)

    direct_out = vt.encode_image(image, vision_cfg, direct, processor=processor)
    loaded_out = vt.encode_image(image, vision_cfg, loaded, processor=processor)

    np.testing.assert_allclose(loaded_out, direct_out, rtol=0, atol=0)
