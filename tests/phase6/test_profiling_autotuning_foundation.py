from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

import tessera as ts
from tessera import profiler
from tessera import autotune


ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_profiler_session_records_metrics_and_report():
    with profiler.session() as p:
        event = p.record(
            "matmul",
            latency_ms=2.0,
            flops=4e12,
            bytes_moved=2e9,
            peak_tflops=4.0,
            counters={"sm_occupancy_pct": 90.0},
        )

    assert event.flops_g == pytest.approx(4000.0)
    assert event.bandwidth_gbps == pytest.approx(1000.0)
    assert event.efficiency_pct == pytest.approx(100.0)
    assert "matmul" in p.report()
    assert "Latency(ms)" in p.report()


def test_profiler_measure_and_timeline_export(tmp_path):
    with profiler.session() as p:
        value = p.measure("sum", lambda: np.arange(4).sum(), flops=4.0)

    assert value == 6
    out = p.timeline(tmp_path / "trace.json")
    payload = json.loads(out.read_text())
    assert payload["traceEvents"][0]["name"] == "sum"


def test_profiler_module_record_requires_active_session():
    with pytest.raises(RuntimeError, match="active profiler.session"):
        profiler.record("orphan", latency_ms=1.0)


def test_roofline_cost_model_identifies_memory_bound_case():
    model = autotune.RooflineCostModel(peak_tflops=100.0, bandwidth_gbps=100.0)

    estimate = model.estimate(flops=1e9, bytes_moved=10e9)

    assert estimate.bound == "memory"
    assert estimate.memory_ms > estimate.compute_ms


def test_public_autotune_callable_persists_and_loads_cache(tmp_path):
    cache = tmp_path / "tuning.db"

    result = autotune(ts.ops.matmul, shapes=(256, 256, 256), max_trials=3, cache_path=cache)
    loaded = autotune.load(ts.ops.matmul, (256, 256, 256), cache_path=cache)

    assert result.latency_ms > 0
    assert loaded is not None
    assert loaded.config.tile_m > 0


def test_autotune_cache_key_and_schedule_artifact(tmp_path):
    cache = tmp_path / "tuning.db"
    result = autotune("matmul", shapes=(128, 128, 128), max_trials=2, cache_path=cache, arch="sm90")

    key = autotune.cache_key("matmul", (128, 128, 128), dtype="bf16", arch="sm90")
    artifact = autotune.schedule_artifact(result, op="matmul", shapes=(128, 128, 128), arch="sm90")

    assert key == ("matmul", (128, 128, 128), "bf16", "sm90")
    assert artifact["arch"] == "sm90"
    assert artifact["hash"]


def test_autotune_rejects_non_gemm_ops():
    with pytest.raises(ValueError, match="GEMM"):
        autotune("softmax", shapes=(1024,), max_trials=1)


def test_profiling_autotuning_guide_is_registered():
    guide = (ROOT / "docs/guides/Tessera_Profiling_And_Autotuning_Guide.md").read_text()
    readme = (ROOT / "docs/README.md").read_text()

    for needle in [
        "Runtime Profiler",
        "Cost Models",
        "Autotuning Workflow",
        "Persistent Caches",
        "On-Device Measurements",
        "Advanced Profiling",
    ]:
        assert needle in guide
    assert "Tessera_Profiling_And_Autotuning_Guide.md" in readme
