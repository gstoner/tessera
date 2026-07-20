"""Host-free contract for the X86-E2E-1 retained-route comparison."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/x86/benchmark_x86_e2e.py"
BASELINE = ROOT / "benchmarks/baselines/x86_avx512_e2e_comparison.json"
SPEC = importlib.util.spec_from_file_location("x86_e2e_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_recorder_covers_softmax_and_three_reductions() -> None:
    assert len(benchmark.SOFTMAX_SHAPES) == 3
    assert {item[0] for item in benchmark.REDUCTIONS} == {"sum", "mean", "max"}


def test_summary_requires_pairs_and_applies_ten_percent_gate() -> None:
    assert benchmark._summary([1.0, 1.1], [1.02, 1.0])["non_regression_10pct"] is True


def test_recorded_avx512_evidence_is_correct_and_nonregressing() -> None:
    data = json.loads(BASELINE.read_text())
    assert data["schema"] == benchmark.SCHEMA
    assert data["evidence_arch"] == "x86_64_avx512"
    assert data["all_correct"] is True
    assert data["all_non_regression"] is True
    assert data["selector_changed"] is False
    assert len(data["rows"]) == len(benchmark.SOFTMAX_SHAPES) + len(benchmark.REDUCTIONS)
    assert all(row["end_to_end"]["non_regression_10pct"] for row in data["rows"])
