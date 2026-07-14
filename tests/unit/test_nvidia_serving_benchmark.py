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


def test_serving_d2_group_selects_device_winner(tmp_path, monkeypatch):
    from tessera.compiler.emit import autotune as at
    path = tmp_path / "corpus.json"
    monkeypatch.setenv("TESSERA_AUTOTUNE_CORPUS", str(path))
    rows = [
        {"op": "paged_kv_decode", "shape": "1x8x128x64", "dtype": "f32",
         "mode": "fused_paged_attention", "device_latency_ms": .1},
        {"op": "paged_kv_decode", "shape": "1x8x128x64", "dtype": "f32",
         "mode": "staged_paged_attention", "device_latency_ms": .3},
    ]
    assert bench.update_d2_corpus(rows) == path
    payload = __import__("json").loads(path.read_text())
    record = payload["records"][0]
    assert record["winner"] == "fused_paged_attention"
    assert record["timing"] == at.TIMING_DEVICE

    from tessera.compiler.emit import nvidia_cuda
    assert nvidia_cuda._paged_attention_corpus_winner(1, 8, 128, 64) == "fused"
