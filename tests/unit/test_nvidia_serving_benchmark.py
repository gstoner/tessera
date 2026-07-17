"""Schema and analytical checks for the NVIDIA serving benchmark."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[2] / "benchmarks/nvidia/benchmark_serving.py"
_SPEC = importlib.util.spec_from_file_location("nvidia_serving_benchmark", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def test_replay_traffic_reduction_is_explicit():
    summary = bench.summary_state_bytes_per_token(256, 128)
    replay = bench.replay_state_bytes_per_token(256, 128, 64)
    assert summary == 262144
    assert replay == 4608
    assert summary / replay == pytest.approx(56.8888889)


def test_shape_validation():
    assert bench.parse_shape("2x256x128") == (2, 256, 128)
    with pytest.raises(ValueError, match="BxDxN"):
        bench.parse_shape("256x128")


def test_replay_chunks_are_bounded_into_reusable_slot_waves():
    assert bench.replay_wave_offsets(64, 4, 4) == (
        (0, 4, 8, 12), (16, 20, 24, 28),
        (32, 36, 40, 44), (48, 52, 56, 60))
    with pytest.raises(ValueError, match="positive divisible"):
        bench.replay_wave_offsets(63, 4, 4)


def test_serving_d2_group_selects_device_winner(tmp_path, monkeypatch):
    from tessera.compiler.emit import autotune as at
    path = tmp_path / "corpus.json"
    monkeypatch.setenv("TESSERA_AUTOTUNE_CORPUS", str(path))
    rows = [
        {"op": "paged_kv_decode", "shape": "1x8x128x64", "dtype": "f32",
         "mode": "fused_paged_attention", "device_latency_ms": .1,
         "latency_ms": .4},
        {"op": "paged_kv_decode", "shape": "1x8x128x64", "dtype": "f32",
         "mode": "staged_paged_attention", "device_latency_ms": .3,
         "latency_ms": .2},
    ]
    assert bench.update_d2_corpus(rows) == path
    payload = __import__("json").loads(path.read_text())
    records = {record["timing"]: record for record in payload["records"]}
    assert records[at.TIMING_DEVICE]["winner"] == "fused_paged_attention"
    assert records[at.TIMING_END_TO_END]["winner"] == "staged_paged_attention"

    from tessera.compiler.emit import nvidia_cuda
    assert nvidia_cuda._paged_attention_corpus_winner(1, 8, 128, 64) == "fused"
