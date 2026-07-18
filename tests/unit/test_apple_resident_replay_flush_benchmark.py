from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "apple_gpu" / "benchmark_resident_replay_flush.py"
    spec = importlib.util.spec_from_file_location("benchmark_resident_replay_flush", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flush_benchmark_keeps_timing_domains_and_unavailable_native_explicit(
    monkeypatch,
):
    module = _module()

    class _Unavailable:
        resident_inputs = False

        def close(self):
            return None

    monkeypatch.setattr(module.rt, "apple_gpu_resident_ssm_replay_state_handle", lambda *args, **kwargs: _Unavailable())
    report = module.run_benchmark(["1x4x3"], tokens=2, warmup=0, reps=1, runs=1)
    assert report["schema"] == "tessera.apple.resident_replay_flush.v1"
    assert report["timing_domains"] == ["end_to_end", "device"]
    reference, native = report["rows"]
    assert reference["route"] == "reference_fold"
    assert reference["correctness"] is True
    assert reference["native_proof"] is False
    assert native["route"] == "resident_native"
    assert native["available"] is False
    assert native["native_proof"] is False
    assert native["correctness"] is None


def test_committed_flush_corpus_is_two_run_native_and_device_timed():
    path = (
        Path(__file__).resolve().parents[2] / "benchmarks" / "baselines" / "apple7_resident_replay_flush_two_run.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["schema"] == "tessera.apple.resident_replay_flush.v1"
    assert report["device"] == "apple7"
    assert report["runs"] == 2
    native = [row for row in report["rows"] if row["route"] == "resident_native"]
    assert {(row["run"], row["shape"]) for row in native} == {
        (1, "1x128x64"),
        (1, "1x256x128"),
        (2, "1x128x64"),
        (2, "1x256x128"),
    }
    assert all(row["available"] and row["native_proof"] and row["correctness"] for row in native)
    assert all(row["device_time_coverage"] == 1.0 for row in native)
    assert all(isinstance(row["timing_domain_device_ns"], int) for row in native)
    assert all(isinstance(row["timing_domain_end_to_end_ns"], int) for row in native)
