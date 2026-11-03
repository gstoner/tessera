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


def test_perplexity_arms_and_candidate_verdicts() -> None:
    from benchmarks.rocm.record_qwen3_native_mlp_perplexity import parse_arms, verdict

    assert parse_arms("native") == ("bf16", "shipped", "native")
    assert parse_arms("mxfp4_direct,mxfp4_ingested") == ("bf16", "shipped", "mxfp4_direct", "mxfp4_ingested")
    for bad in ("", "shipped", "native,native", "mxfp8"):
        with pytest.raises(ValueError, match="candidate arms"):
            parse_arms(bad)
    ppl = {"bf16": 10.0, "shipped": 10.4, "mxfp4_direct": 10.6, "mxfp4_ingested": 11.0}
    assert verdict(ppl, arm="mxfp4_direct")["passed"]
    failed = verdict(ppl, arm="mxfp4_ingested")
    assert not failed["passed"] and failed["ratio"] == pytest.approx(2.5) and failed["arm"] == "mxfp4_ingested"


def _nvfp4_projection(rng: np.random.Generator, n: int, k: int):
    from tessera.compiler import rocm_nvfp4_ingest as ingest

    codes = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
    scales = rng.uniform(0.25, 4.0, (n, k // 16)).astype(ml_dtypes.float8_e4m3fn)
    return ingest.NVFP4Projection("p", codes, scales, 0.01)


def test_weight_only_arms_decode_their_own_formats() -> None:
    from benchmarks.rocm import qwen3_native_mlp_worker as worker
    from benchmarks.rocm.benchmark_checkpoint_quantized_mlp import decode_projection
    from tessera.compiler import rocm_mxfp4 as mx

    rng = np.random.default_rng(0)
    projections = [_nvfp4_projection(rng, 8, 64) for _ in range(3)]
    originals = [decode_projection(p).astype(ml_dtypes.bfloat16) for p in projections]
    loaded = {"projections": projections, "bf16": originals}
    shipped = worker.decode_arm("shipped", loaded)
    assert all(np.array_equal(a, decode_projection(p)) for a, p in zip(shipped, projections))

    def rel(a, b):
        b = np.asarray(b, np.float64)
        return float(np.sqrt(((np.asarray(a, np.float64) - b) ** 2).sum() / (b ** 2).sum()))

    magnitudes = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    for arm, reference in (("mxfp4_direct", originals), ("mxfp4_ingested", shipped)):
        values = worker.decode_arm(arm, loaded)
        assert [v.shape for v in values] == [(8, 64)] * 3
        assert all(0 < rel(v, r) < 0.25 for v, r in zip(values, reference))
        # Every value is an E2M1 magnitude times one power of two per 32-wide block.
        blocks = values[0].reshape(8, 2, 32)
        peak = np.abs(blocks).max(axis=-1, keepdims=True)
        scale = np.exp2(np.ceil(np.log2(peak / 6)))
        assert set(np.unique(np.abs(blocks / scale)).tolist()) <= magnitudes | {m / 2 for m in magnitudes}
    assert mx.MXFP4_GROUP_K == 32
    with pytest.raises(ValueError, match="weight-only arm"):
        worker.decode_arm("native", loaded)


def test_native_mxfp4_is_gated_against_its_own_weights() -> None:
    from benchmarks.rocm.record_qwen3_native_mlp_perplexity import parse_arms, verdict

    assert parse_arms("native_mxfp4,mxfp4_direct") == ("bf16", "shipped", "native_mxfp4", "mxfp4_direct")
    with pytest.raises(ValueError, match="mxfp4_direct, which must run too"):
        parse_arms("native_mxfp4")
    ppl = {"bf16": 10.0, "shipped": 10.2, "mxfp4_direct": 10.4, "native_mxfp4": 10.6}
    passed = verdict(ppl, arm="native_mxfp4")
    assert passed["reference"] == "mxfp4_direct" and passed["ratio"] == pytest.approx(1.5) and passed["passed"]
    # Against shipped the same arm would fail: the reference is what the gate measures.
    assert not verdict(ppl, arm="native_mxfp4", reference="shipped")["passed"]
    assert not verdict({**ppl, "native_mxfp4": 10.61}, arm="native_mxfp4")["passed"]


def test_native_mxfp4_roots_carry_exactly_the_direct_weights() -> None:
    from benchmarks.rocm import qwen3_native_mlp_worker as worker
    from tessera.compiler.nvfp4_expansion import mxfp4_expansion_policy, reference_nvfp4_expand

    rng = np.random.default_rng(4)
    bf16 = [rng.standard_normal(shape).astype(ml_dtypes.bfloat16) for shape in ((64, 96), (64, 96), (96, 64))]
    loaded = {"bf16": bf16}
    direct = worker.decode_arm("mxfp4_direct", loaded)
    roots = worker.native_mxfp4_roots(loaded)
    assert loaded["mxfp4_direct"] is worker.direct_mxfp4(loaded)  # one shared quantization
    for (packed, scales, globals_), offsets, expected in (
            (roots[:3], [0, 64, 128], np.concatenate(direct[:2])), (roots[3:], [0, 96], direct[2])):
        assert scales.dtype == np.uint8 and np.array_equal(globals_, np.ones(len(offsets) - 1))
        values, plane = reference_nvfp4_expand(packed, scales, globals_, row_offsets=offsets,
                                               numeric_policy=mxfp4_expansion_policy())
        decoded = values.astype(np.float64) * np.repeat(plane.T.astype(np.float64), 16, axis=1)
        np.testing.assert_array_equal(decoded, expected)
