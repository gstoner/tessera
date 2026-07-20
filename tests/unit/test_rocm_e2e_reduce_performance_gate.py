"""Host-free contract tests for the ROCM-E2E-2 reduction recorder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/rocm/benchmark_rocm_e2e_reduce.py"
BASELINE = ROOT / "benchmarks/baselines/rocm_gfx1151_e2e_reduce_comparison.json"
SPEC = importlib.util.spec_from_file_location("rocm_e2e_reduce_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_benchmark_covers_three_dtypes_kinds_and_axis_layouts() -> None:
    assert [item[0] for item in benchmark.DTYPES] == ["fp32", "fp16", "bf16"]
    assert {item[2] for item in benchmark.CASES} == {"sum", "mean", "max"}
    assert {item[1] for item in benchmark.CASES} == {0, 1}


def test_summary_retains_pairs_and_applies_ten_percent_gate() -> None:
    summary = benchmark._summary([1.0, 1.1, 0.9], [1.05, 1.0, 1.0])
    assert summary["retained_samples_ms"] == [1.0, 1.1, 0.9]
    assert summary["non_regression_10pct"] is True
    assert benchmark._summary([1.0] * 3, [1.2] * 3)["non_regression_10pct"] is False


@pytest.mark.parametrize("old,new", [([], []), ([1.0], []), ([1.0], [1.0, 2.0])])
def test_summary_rejects_unpaired_or_empty_samples(old, new) -> None:
    with pytest.raises(ValueError, match="paired timing samples"):
        benchmark._summary(old, new)


def test_recorder_is_serial_and_keeps_timing_domains_separate() -> None:
    source = SCRIPT.read_text()
    assert "hipEventElapsedTime" in source
    assert "rt.launch" in source
    assert "host_overhead" in source
    assert "device_comparable" in source
    assert "selector_changed" in source
    assert "ThreadPool" not in source and "multiprocessing" not in source


def test_recorded_gfx1151_evidence_closes_comparable_device_and_e2e_gates() -> None:
    data = json.loads(BASELINE.read_text())
    assert data["schema"] == benchmark.SCHEMA
    assert data["evidence_arch"] == "gfx1151"
    assert data["all_correct"] is True
    assert data["selector_changed"] is False
    assert len(data["rows"]) == len(benchmark.CASES) * len(benchmark.DTYPES)
    assert all(row["end_to_end"]["non_regression_10pct"] for row in data["rows"])
    assert all(
        row["device_comparable"] == (row["axis"] == len(row["shape"]) - 1)
        for row in data["rows"]
    )
    assert data["all_non_regression"] is True
    assert all(
        row["device"]["non_regression_10pct"]
        for row in data["rows"]
        if row["device_comparable"]
    )
    assert max(abs(row["host_overhead"]["compiler_minus_retained_ms"]) for row in data["rows"]) < 0.2
