"""Validate captured model inputs before any checkpoint download or GPU work."""

from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from benchmarks.rocm import benchmark_rocm_nvfp4_checkpoint as checkpoint


def load_capture(path, requested_rows):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if (
        metadata.get("schema") != "tessera.checkpoint.model_activations.v1"
        or metadata.get("capture_kind") != "pinned_bf16_model_layer_input"
        or metadata.get("repo") != checkpoint.BF16_REPO
        or metadata.get("revision") != checkpoint.BF16_REVISION
        or metadata.get("tensor") != "model.layers.0.mlp.input"
        or metadata.get("capture_boundary") != "upstream_decoder_mlp_forward_pre_hook"
        or metadata.get("reference_execution") != "torch_cpu_bfloat16_eager"
        or metadata.get("repeat_bitwise_equal") is not True
        or metadata.get("model_quality_acceptance") is not False
    ):
        raise ValueError("captured activation source contract differs")
    source = metadata.get("weight_source", {})
    tensors = source.get("tensors", {})
    required = {
        "model.layers.0." + name
        for name in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        )
    }

    def sha(value):
        return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None

    if (
        source.get("repo") != checkpoint.BF16_REPO
        or source.get("revision") != checkpoint.BF16_REVISION
        or not sha(source.get("index_sha256"))
        or set(tensors) != required | {"model.embed_tokens.weight"}
        or any(
            not sha(tensors[name].get("sha256")) or tensors[name].get("storage", {}).get("dtype") != "BF16"
            for name in required
        )
        or not sha(tensors["model.embed_tokens.weight"].get("selected_row_sha256"))
        or set(metadata.get("unmaterialized_unexecuted_parameters", []))
        != {"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}
        or any(
            not sha(metadata.get("sources_sha256", {}).get(key))
            for key in ("capture", "checkpoint_loader", "upstream_qwen3", "upstream_masking")
        )
    ):
        raise ValueError("captured activation weight/reference provenance differs")
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != metadata.get("npz_sha256"):
        raise ValueError("captured activation archive hash differs")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"activations", "token_ids"}:
            raise ValueError("captured activation fields differ")
        activations = archive["activations"]
        token_ids = archive["token_ids"]
    if (
        activations.dtype != np.float32
        or activations.ndim != 2
        or not all(activations.shape)
        or not np.isfinite(activations).all()
        or list(activations.shape) != metadata.get("shape")
        or token_ids.dtype != np.int64
        or token_ids.ndim != 2
        or not all(token_ids.shape)
        or np.any(token_ids < 0)
        or token_ids.size != activations.shape[0]
        or list(token_ids.shape) != metadata.get("batch_sequence_shape")
        or hashlib.sha256(activations.tobytes()).hexdigest() != metadata.get("activation_sha256")
        or hashlib.sha256(token_ids.tobytes()).hexdigest() != metadata.get("token_ids_sha256")
    ):
        raise ValueError("captured activation tensor identity differs")
    if not requested_rows or any(type(m) is not int or not 0 < m <= len(activations) for m in requested_rows):
        raise ValueError("requested rows exceed the captured model input envelope")
    return np.ascontiguousarray(activations), metadata
