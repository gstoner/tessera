"""Captured quality inputs are checked independently of Torch, network or GPU."""

import hashlib
import json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import ml_dtypes
import pytest
from benchmarks.rocm import checkpoint_activations as source
from benchmarks.rocm import capture_qwen3_layer0_activations as capture
from benchmarks.rocm import benchmark_gfx1201_checkpoint_formats as recorder


def fake_capture(tmp_path):
    path = tmp_path / "capture.npz"
    a = np.arange(8, dtype=np.float32).reshape(2, 4)
    ids = np.asarray([[1, 2]], np.int64)
    np.savez_compressed(path, activations=a, token_ids=ids)

    def sha(x):
        return hashlib.sha256(x).hexdigest()

    parameters = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    )
    tensors = {"model.layers.0." + name: {"sha256": "0" * 64, "storage": {"dtype": "BF16"}} for name in parameters}
    tensors["model.embed_tokens.weight"] = {"selected_row_sha256": "0" * 64}
    metadata = {
        "schema": "tessera.checkpoint.model_activations.v1",
        "capture_kind": "pinned_bf16_model_layer_input",
        "repo": source.checkpoint.BF16_REPO,
        "revision": source.checkpoint.BF16_REVISION,
        "tensor": "model.layers.0.mlp.input",
        "capture_boundary": "upstream_decoder_mlp_forward_pre_hook",
        "reference_execution": "torch_cpu_bfloat16_eager",
        "repeat_bitwise_equal": True,
        "model_quality_acceptance": False,
        "npz_sha256": sha(path.read_bytes()),
        "activation_sha256": sha(a.tobytes()),
        "token_ids_sha256": sha(ids.tobytes()),
        "shape": [2, 4],
        "batch_sequence_shape": [1, 2],
        "weight_source": {
            "repo": source.checkpoint.BF16_REPO,
            "revision": source.checkpoint.BF16_REVISION,
            "index_sha256": "0" * 64,
            "tensors": tensors,
        },
        "sources_sha256": dict.fromkeys(
            ("capture", "checkpoint_loader", "upstream_qwen3", "upstream_masking"), "0" * 64
        ),
        "unmaterialized_unexecuted_parameters": ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"],
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return path, a, ids, metadata


def test_captured_arrays_keep_their_exact_identity(tmp_path):
    path, a, _, metadata = fake_capture(tmp_path)
    actual, info = source.load_capture(path, [1, 2])
    np.testing.assert_array_equal(actual, a)
    assert info == metadata


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "wrong"),
        ("capture_kind", "synthetic"),
        ("model_quality_acceptance", True),
        ("npz_sha256", "f" * 64),
        ("activation_sha256", "f" * 64),
        ("token_ids_sha256", "f" * 64),
    ],
)
def test_capture_rejects_mismatched_source_or_bytes(tmp_path, field, value):
    path, _, _, metadata = fake_capture(tmp_path)
    metadata[field] = value
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        source.load_capture(path, [2])


def test_capture_requires_executed_parameter_provenance(tmp_path):
    path, _, _, metadata = fake_capture(tmp_path)
    del metadata["weight_source"]["tensors"]["model.layers.0.self_attn.q_norm.weight"]
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="provenance"):
        source.load_capture(path, [2])


def test_capture_rejects_nonfinite_values_even_with_matching_hashes(tmp_path):
    path, a, ids, metadata = fake_capture(tmp_path)
    a[0, 0] = np.nan
    np.savez_compressed(path, activations=a, token_ids=ids)
    metadata["npz_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata["activation_sha256"] = hashlib.sha256(a.tobytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="tensor identity"):
        source.load_capture(path, [2])


def test_requested_capture_extent_is_checked_before_gpu_or_download(tmp_path, monkeypatch):
    path, *_ = fake_capture(tmp_path)
    monkeypatch.setattr(recorder.rt, "_rocm_live_arch", lambda: pytest.fail("invalid capture reached GPU"))
    monkeypatch.setattr(
        recorder.checkpoint, "_load_projection", lambda *a: pytest.fail("invalid capture reached network")
    )
    with pytest.raises(ValueError, match="envelope"):
        recorder.measure(SimpleNamespace(windows=3, m=[128], activations=path))


def test_embedding_windows_preserve_duplicates_and_token_order(monkeypatch):
    reader = object.__new__(capture.PinnedBF16Reader)
    name = "model.embed_tokens.weight"
    values = np.arange(21, dtype=np.float32).astype(ml_dtypes.bfloat16).reshape(7, 3)
    raw = values.tobytes()
    spec = {"dtype": "BF16", "shape": [7, 3], "data_offsets": [0, len(raw)]}
    reader.weight_map = {name: "weights"}
    reader.headers = {"weights": ("url", {name: spec}, 32)}
    reader.provenance = {"tensors": {}}
    requests = []

    def fetch(url, first, last):
        requests.append((first, last))
        return raw[first - 32 : last - 32 + 1]

    monkeypatch.setattr(capture.checkpoint, "_fetch_range", fetch)
    ids = np.asarray([[6, 1, 6], [3, 0, 2]], np.int64)
    actual = reader.embedding_rows(ids, block_rows=2)
    np.testing.assert_array_equal(actual, values[ids])
    assert requests == [(32, 43), (44, 55), (68, 73)]
    with pytest.raises(ValueError, match="vocabulary"):
        reader.embedding_rows(np.asarray([7]))


@pytest.mark.parametrize("path,length", [("wrong.json", 128), ("valid.npz", 0), ("valid.npz", 4097)])
def test_invalid_capture_arguments_precede_torch_import_and_network(path, length):
    with pytest.raises(ValueError):
        capture.capture(Path(path), length)
