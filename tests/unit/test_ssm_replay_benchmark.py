"""Track-R (ReplaySSM) Phase 5-bench — decode benchmark guard.

Validates the benchmark emits the stable JSON schema and that the analytical
state-traffic model reflects the ReplaySSM reduction (replay routes move far
fewer state bytes per token than the summary baseline).
"""

from __future__ import annotations

import json

import pytest

import importlib.util
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "apple_gpu" / "benchmark_ssm_replay.py"
_SUMMARY = (Path(__file__).resolve().parents[2] / "benchmarks" / "apple_gpu"
            / "summarize_replay_stability.py")


def _load():
    spec = importlib.util.spec_from_file_location("benchmark_ssm_replay", _BENCH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_summary():
    spec = importlib.util.spec_from_file_location("summarize_replay_stability", _SUMMARY)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REQUIRED_KEYS = {
    "backend", "op", "shape", "dtype", "latency_ms", "tflops",
    "memory_bw_gb_s", "device", "tessera_version",
}


def test_benchmark_emits_stable_schema():
    bench = _load()
    rows = bench.run_benchmark(["1x32x16"], tokens=8, capacity=4, reps=2)
    assert rows
    for r in rows:
        assert REQUIRED_KEYS.issubset(r), f"missing keys: {REQUIRED_KEYS - set(r)}"
        assert r["backend"] == "apple_gpu"
        assert r["op"] == "ssm_replay_decode"
    modes = {r["mode"] for r in rows}
    assert modes == {"summary", "replay_reference", "replay_fused", "replay_block"}


def test_replay_reduces_state_traffic():
    bench = _load()
    rows = bench.run_benchmark(["1x64x64"], tokens=16, capacity=8, reps=2)
    by_mode = {r["mode"]: r for r in rows}
    summ = by_mode["summary"]["state_bytes_per_token"]
    ref = by_mode["replay_reference"]["state_bytes_per_token"]
    fused = by_mode["replay_fused"]["state_bytes_per_token"]
    assert ref < summ and fused < summ                    # replay moves less state
    assert ref == fused                                   # same analytical model
    assert by_mode["summary"]["state_traffic_ratio"] == 1.0
    assert by_mode["replay_reference"]["state_traffic_ratio"] > 1.0


def test_analytical_traffic_model():
    bench = _load()
    D, N, L = 128, 128, 16
    summ = bench._summary_state_bytes_per_token(D, N)
    repl = bench._replay_state_bytes_per_token(D, N, L)
    assert summ == 2 * 4 * D * N                           # load + store full state
    # flush amortized (D*N/L) + small append (2D+N), all f32.
    assert repl == pytest.approx(4 * D * N / L + 4 * (2 * D + N))
    assert summ / repl > 5.0                               # large reduction at L=16


def test_benchmark_main_writes_json(tmp_path):
    bench = _load()
    out = tmp_path / "ssm_replay.json"
    rc = bench.main(["--shapes", "1x32x16", "--tokens", "8", "--capacity", "4",
                     "--reps", "2", "--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert "runs" in payload and payload["runs"]


def test_replay_stability_requires_native_correct_resource_complete_rows():
    summary = _load_summary()
    row = {
        "shape": "1x4x3", "mode": "replay_block", "latency_ms": 0.1,
        "native_dispatched": True, "numerically_validated": True,
        "device_time_median_ns": 1000, "device_time_coverage": 1.0,
        "resources": {"threadgroup": [4, 1, 1]},
    }
    ledger = summary.summarize([{"runs": [row]}, {"runs": [{**row,
        "latency_ms": 0.11, "device_time_median_ns": 1050}]}])
    assert ledger["selector_status"] == "characterized_not_selector_eligible"
    assert ledger["rows"][0]["evidence_complete"]
    assert ledger["rows"][0]["end_to_end_cross_run_drift_fraction"] == pytest.approx(0.1)
