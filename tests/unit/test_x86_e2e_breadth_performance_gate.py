"""Contract gate for X86-E2E-1 matmul and attention retained comparisons."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/x86/benchmark_x86_e2e_breadth.py"
BASELINE = ROOT / "benchmarks/baselines/x86_avx512_e2e_breadth_comparison.json"
SPEC = importlib.util.spec_from_file_location("x86_e2e_breadth", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_breadth_recorder_covers_three_slices() -> None:
    assert len(benchmark.MATMUL_SHAPES) == 2
    assert {extended for extended, _ in benchmark.ATTENTION_CASES} == {False, True}


def test_recorded_breadth_evidence_is_correct_and_nonregressing() -> None:
    data = json.loads(BASELINE.read_text())
    assert data["schema"] == benchmark.SCHEMA
    assert data["evidence_arch"] == "x86_64_avx512"
    assert data["all_correct"] is True
    assert data["all_non_regression"] is True
    assert data["selector_changed"] is False
    assert {row["operation"] for row in data["rows"]} == {"matmul", "attention", "attention_ext"}
    assert all(row["end_to_end"]["non_regression_10pct"] for row in data["rows"])
