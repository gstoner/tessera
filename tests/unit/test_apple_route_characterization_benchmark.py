"""Portable contract checks for the Apple route-characterization producer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "apple_gpu" / "benchmark_route_characterization.py"
SELECTOR = ROOT / "benchmarks" / "apple_gpu" / "select_stable_gemm_routes.py"


def _benchmark_module():
    spec = importlib.util.spec_from_file_location("apple_route_characterization", BENCHMARK)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selector_module():
    spec = importlib.util.spec_from_file_location("apple_stable_route_selector", SELECTOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shape_parser_rejects_ambiguous_or_non_positive_shapes():
    benchmark = _benchmark_module()
    assert benchmark._shape("4x6x5") == (4, 6, 5)
    for invalid in ("", "4x0", "4xnope"):
        with pytest.raises(ValueError):
            benchmark._shape(invalid)


def test_selector_incumbent_override_is_explicit_op_route_pair():
    selector = _selector_module()
    assert selector._incumbent("matmul_relu=mps_mpsgraph_unfused") == (
        "matmul_relu", "mps_mpsgraph_unfused")
    for invalid in ("matmul_relu", "=route", "op="):
        with pytest.raises(selector.argparse.ArgumentTypeError):
            selector._incumbent(invalid)


def test_report_row_carries_selector_proof_fields():
    benchmark = _benchmark_module()
    row = benchmark._row(
        op="softmax", shape="4x8", dtype="f32", route="mpsgraph",
        output=benchmark.np.ones((4, 8), dtype=benchmark.np.float32),
        reference=benchmark.np.ones((4, 8), dtype=benchmark.np.float32),
        latency_ms=0.1, stdev_ms=0.01,
        telemetry={
            "device_time_median_ns": None,
            "device_time_samples": 0,
            "timing_source": None,
            "counter_sampling_supported": None,
            "counter_timestamp_delta_median": None,
        },
    )
    assert row["native_dispatched"]
    assert row["numerically_validated"]
    assert row["route"] == "mpsgraph"
    assert row["telemetry"]["device_time_median_ns"] is None
    assert row["telemetry"]["resources"]["occupancy"] is None
    assert row["telemetry"]["resources"]["spill_evidence"] is None
