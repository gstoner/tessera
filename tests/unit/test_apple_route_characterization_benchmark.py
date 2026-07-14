"""Portable contract checks for the Apple route-characterization producer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "apple_gpu" / "benchmark_route_characterization.py"


def _benchmark_module():
    spec = importlib.util.spec_from_file_location("apple_route_characterization", BENCHMARK)
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


def test_report_row_carries_selector_proof_fields():
    benchmark = _benchmark_module()
    row = benchmark._row(
        op="softmax", shape="4x8", dtype="f32", route="mpsgraph",
        output=benchmark.np.ones((4, 8), dtype=benchmark.np.float32),
        reference=benchmark.np.ones((4, 8), dtype=benchmark.np.float32),
        latency_ms=0.1, stdev_ms=0.01,
    )
    assert row["native_dispatched"]
    assert row["numerically_validated"]
    assert row["route"] == "mpsgraph"
