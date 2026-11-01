"""Host-free guards for the Qwen3 all-layer MLP quality harness."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from benchmarks.rocm import capture_qwen3_all_layer_mlp_inputs as capture
from benchmarks.rocm import qwen3_local_checkpoint as local
from benchmarks.rocm.record_qwen3_mlp_layer_quality import _Accumulator


def _safetensors(tensors: dict[str, np.ndarray], dtypes: dict[str, str]) -> bytes:
    header, payload, offset = {}, b"", 0
    for name, value in tensors.items():
        raw = np.ascontiguousarray(value).tobytes()
        header[name] = {"dtype": dtypes[name], "shape": list(value.shape), "data_offsets": [offset, offset + len(raw)]}
        payload += raw
        offset += len(raw)
    encoded = json.dumps(header).encode()
    return len(encoded).to_bytes(8, "little") + encoded + payload


def _snapshot(tmp_path: Path, repo: str, shards: dict[str, bytes], weight_map: dict[str, str]) -> Path:
    base = tmp_path / ("models--" + repo.replace("/", "--"))
    snap = base / "snapshots" / local.PINS[repo]
    snap.mkdir(parents=True)
    (base / "blobs").mkdir()
    for shard, data in shards.items():
        blob = base / "blobs" / hashlib.sha256(data).hexdigest()
        blob.write_bytes(data)
        (snap / shard).symlink_to(blob)
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return snap


def test_local_snapshot_reads_verified_tensor(tmp_path: Path) -> None:
    value = np.arange(12, dtype=np.float32).reshape(3, 4).astype(ml_dtypes.bfloat16)
    data = _safetensors({"w": value}, {"w": "BF16"})
    _snapshot(tmp_path, local.BF16_REPO, {"a.safetensors": data}, {"w": "a.safetensors"})
    snapshot = local.LocalSnapshot.open(tmp_path, local.BF16_REPO)
    read, record = snapshot.tensor("w")
    np.testing.assert_array_equal(read, value)
    assert record["shard_sha256"] == hashlib.sha256(data).hexdigest()
    assert record["sha256"] == hashlib.sha256(value.tobytes()).hexdigest()


def test_local_snapshot_refuses_tampered_blob(tmp_path: Path) -> None:
    value = np.ones((2, 2), np.float32)
    snap = _snapshot(tmp_path, local.BF16_REPO, {"a.safetensors": _safetensors({"w": value}, {"w": "F32"})},
                     {"w": "a.safetensors"})
    blob = (snap / "a.safetensors").resolve()
    blob.write_bytes(blob.read_bytes()[:-4] + b"\0\0\0\0")
    with pytest.raises(RuntimeError, match="differs from its blob name"):
        local.LocalSnapshot.open(tmp_path, local.BF16_REPO).tensor("w")


def test_unpinned_revision_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a pinned"):
        local.snapshot_dir(tmp_path, local.BF16_REPO, "0" * 40)


def test_load_layer_mlp_checks_nvfp4_contract(tmp_path: Path) -> None:
    n, k = 4, 32
    nv_tensors, nv_types, bf_tensors, bf_types = {}, {}, {}, {}
    for role in local.MLP_ROLES:
        prefix = f"model.layers.0.mlp.{role}"
        nv_tensors[prefix + ".weight"] = np.zeros((n, k // 2), np.uint8)
        nv_types[prefix + ".weight"] = "U8"
        nv_tensors[prefix + ".weight_scale"] = np.ones((n, k // 16), ml_dtypes.float8_e4m3fn)
        nv_types[prefix + ".weight_scale"] = "F8_E4M3"
        nv_tensors[prefix + ".weight_scale_2"] = np.array([0.5], np.float32)
        nv_types[prefix + ".weight_scale_2"] = "F32"
        bf_tensors[prefix + ".weight"] = np.zeros((n, k), ml_dtypes.bfloat16)
        bf_types[prefix + ".weight"] = "BF16"
    _snapshot(tmp_path, local.NVFP4_REPO, {"q.safetensors": _safetensors(nv_tensors, nv_types)},
              {name: "q.safetensors" for name in nv_tensors})
    _snapshot(tmp_path, local.BF16_REPO, {"b.safetensors": _safetensors(bf_tensors, bf_types)},
              {name: "b.safetensors" for name in bf_tensors})
    loaded = local.load_layer_mlp(tmp_path, 0)
    assert [p.name for p in loaded["projections"]] == list(local.MLP_ROLES)
    assert all(p.global_scale == 0.5 for p in loaded["projections"])
    with pytest.raises(KeyError, match="no tensor"):
        local.load_layer_mlp(tmp_path, 1)


def _capture_dir(tmp_path: Path, bits: np.ndarray) -> Path:
    np.save(tmp_path / "layer_00.npy", bits)
    metadata = {"schema": capture.SCHEMA, "repeat_bitwise_equal": True, "revision": local.BF16_REVISION,
                "model_quality_acceptance": False,
                "layers": [{"layer": 0, "file": "layer_00.npy", "shape": list(bits.shape),
                            "bf16_bits_sha256": hashlib.sha256(bits.tobytes()).hexdigest()}]}
    (tmp_path / "capture.json").write_text(json.dumps(metadata))
    return tmp_path


def test_capture_layer_round_trips_bf16_bits(tmp_path: Path) -> None:
    values = np.linspace(-3, 3, 32, dtype=np.float32).reshape(4, 8).astype(ml_dtypes.bfloat16)
    loaded = capture.load_layer(_capture_dir(tmp_path, values.view(np.uint16)), 0)
    assert loaded.dtype == ml_dtypes.bfloat16
    np.testing.assert_array_equal(loaded.view(np.uint16), values.view(np.uint16))


def test_capture_layer_refuses_changed_bits(tmp_path: Path) -> None:
    bits = np.arange(32, dtype=np.uint16).reshape(4, 8)
    directory = _capture_dir(tmp_path, bits)
    np.save(directory / "layer_00.npy", bits + 1)
    with pytest.raises(ValueError, match="identity differs"):
        capture.load_layer(directory, 0)


def test_accumulator_matches_whole_array_relative_rms() -> None:
    rng = np.random.default_rng(3)
    reference = rng.normal(size=(6, 5))
    candidate = reference + rng.normal(scale=0.1, size=reference.shape)
    acc = _Accumulator()
    acc.add(reference[:3], candidate[:3])
    acc.add(reference[3:], candidate[3:])
    expected = np.sqrt(np.sum((reference - candidate) ** 2) / np.sum(reference ** 2))
    assert acc.result()["relative_rms_error"] == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError, match="non-finite"):
        acc.add(np.array([np.nan]), np.array([0.0]))


def test_worker_frames_round_trip_and_refuse_truncation() -> None:
    import io

    from benchmarks.rocm import qwen3_native_mlp_worker as worker

    stream = io.BytesIO()
    worker.write_frame(stream, {"op": "mlp", "rows": 2}, b"\x01\x02\x03")
    stream.seek(0)
    header, payload = worker.read_frame(stream)
    assert header == {"op": "mlp", "rows": 2, "nbytes": 3} and payload == b"\x01\x02\x03"
    with pytest.raises(EOFError, match="pipe closed"):
        worker.read_frame(io.BytesIO(stream.getvalue()[:-1]))
    with pytest.raises(ValueError, match="multiple of"):
        worker._rows(b"\0" * (worker.HIDDEN * 2), 1)


def test_perplexity_verdict_applies_the_fixed_ratio() -> None:
    from benchmarks.rocm.record_qwen3_native_mlp_perplexity import THRESHOLD_RATIO, verdict

    assert THRESHOLD_RATIO == 1.5
    passed = verdict({"bf16": 10.0, "shipped": 10.4, "native": 10.6})
    assert passed["passed"] and passed["ratio"] == pytest.approx(1.5)
    assert not verdict({"bf16": 10.0, "shipped": 10.4, "native": 10.61})["passed"]
    undefined = verdict({"bf16": 10.0, "shipped": 9.9, "native": 10.0})
    assert not undefined["passed"] and undefined["ratio"] is None
    with pytest.raises(ValueError, match="finite"):
        verdict({"bf16": float("nan"), "shipped": 1.0, "native": 1.0})


def test_resume_refuses_changed_measurement_settings() -> None:
    from benchmarks.rocm.record_qwen3_mlp_layer_quality import resume_mismatches

    identity = {"compiler_sha256": "a", "shape_mhkn": [128, 8, 4, 4], "rows_per_layer": 2048}
    existing = {**identity, "layers": [{"layer": 0, "rows": 2048, "chunk_rows": 128}]}
    assert resume_mismatches(existing, identity) == []
    smoke = {**existing, "rows_per_layer": 128, "layers": [{"layer": 0, "rows": 128, "chunk_rows": 128}]}
    assert resume_mismatches(smoke, identity) == ["rows_per_layer", "layer 0 rows/chunk"]
    assert resume_mismatches({**existing, "compiler_sha256": "b"}, identity) == ["compiler_sha256"]
