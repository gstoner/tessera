"""Schema and analytical checks for the NVIDIA serving benchmark.

Host-free: this module's name reads like a device lane, and it is not one.
Nothing here loads a module onto a GPU or launches a kernel, so it runs and
means the same on every fleet host. Exact-device proof for this area lives in
the gated lanes that skip when the hardware is absent; if you add a device
call here, move it there instead of deleting this line.
"""
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
    # The serving recorder stores one latency per mode, so its two-candidate
    # rows carry no separation verdict: an unproven ranking, which the lookup
    # now refuses exactly as the ROCm twin and `corpus_winner` do.
    assert nvidia_cuda._paged_attention_corpus_winner(1, 8, 128, 64) is None


@pytest.mark.parametrize("staged_samples,expected", [
    ([.30, .31, .29, .30], "fused"),     # clear margin over tight noise: served
    ([.05, .60, .12, .40], None),        # ranking inside the noise: refused
])
def test_serving_rows_with_samples_earn_a_separation_verdict(
        tmp_path, monkeypatch, staged_samples, expected):
    """The recorder keeps every interleaved rep, so a two-mode row carries a
    noise floor and a verdict; a separated verdict is served again, an
    unseparated one is still refused (AUTOTUNE-TOOLCHAIN-KEY-2026-09-26)."""
    import json
    import statistics
    from tessera.compiler.emit import autotune as at
    from tessera.compiler.emit import nvidia_cuda
    path = tmp_path / "corpus.json"
    monkeypatch.setenv("TESSERA_AUTOTUNE_CORPUS", str(path))
    fused = [.10, .101, .099, .10]
    rows = [{"op": "paged_kv_decode", "shape": "1x8x128x64", "dtype": "f32",
             "mode": f"{mode}_paged_attention",
             "device_latency_ms": statistics.median(samples),
             "device_samples_ms": samples}
            for mode, samples in (("fused", fused), ("staged", staged_samples))]
    bench.update_d2_corpus(rows)
    record = json.loads(path.read_text())["records"][0]
    assert record["separation"] is not None
    assert record["separation"]["separated"] is (expected is not None)
    assert nvidia_cuda._paged_attention_corpus_winner(1, 8, 128, 64) == expected


@pytest.mark.parametrize("separated,expected", [(True, "fused"), (False, None)])
def test_nvidia_paged_attention_lookup_applies_admission(
        tmp_path, monkeypatch, separated, expected):
    from tessera.compiler.emit import autotune as at
    from tessera.compiler.emit import nvidia_cuda
    from tessera.compiler.emit.kernel_emitter import SpecPolicy, bucket_key

    cache = at.MeasureCache()
    cache.put(("nvidia:sm_120", "nvidia", "paged_kv_decode",
               bucket_key((1, 8, 128, 64), SpecPolicy.BUCKET), "f32",
               at.TIMING_DEVICE),
              at.MeasureRecord(
                  winner="fused_paged_attention", latency_ms=0.1,
                  candidates={"fused_paged_attention": 0.1,
                              "staged_paged_attention": 0.3},
                  unmeasured={},
                  separation={"separated": separated, "margin": 0.6,
                              "noise": 0.01, "factor": 2.0,
                              "runner_up": "staged_paged_attention"}),
              fresh=True)
    path = tmp_path / "corpus.json"
    at.save_corpus(path, cache=cache)
    monkeypatch.setenv("TESSERA_AUTOTUNE_CORPUS", str(path))
    assert nvidia_cuda._paged_attention_corpus_winner(1, 8, 128, 64) == expected
