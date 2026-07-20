from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/x86/benchmark_x86_e2e_dtype_matmul.py"
BASELINE = ROOT / "benchmarks/baselines/x86_avx512_e2e_dtype_matmul_comparison.json"
SPEC = importlib.util.spec_from_file_location("x86_e2e_dtype", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_dtype_vertical_slice_has_exact_host_correctness_and_timing() -> None:
    data = json.loads(BASELINE.read_text())
    assert data["schema"] == benchmark.SCHEMA
    assert data["all_correct"] is True
    assert data["selector_changed"] is False
    assert len(data["rows"]) == len(benchmark.DTYPES) * len(benchmark.SHAPES)
    assert {row["dtype"] for row in data["rows"]} == set(benchmark.DTYPES)
    assert all(row["kernel"]["native_median_ms"] > 0 for row in data["rows"])
    assert all(row["end_to_end"]["descriptor_median_ms"] > 0 for row in data["rows"])
