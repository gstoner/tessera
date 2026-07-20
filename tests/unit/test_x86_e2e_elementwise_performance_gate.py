"""Recorded performance contract for the X86-E2E-2 elementwise cohort."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/x86/benchmark_x86_e2e_elementwise.py"
BASELINE = ROOT / "benchmarks/baselines/x86_avx512_e2e_elementwise_comparison.json"
SPEC = importlib.util.spec_from_file_location("x86_e2e_elementwise", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_recorder_covers_three_stable_abi_families_and_shape_classes() -> None:
    assert {family for family, _, _ in benchmark.CASES} == {"unary", "binary", "predicate"}
    assert len(benchmark.SHAPES) == 3


def test_recorded_elementwise_evidence_controls_selector_promotion() -> None:
    data = json.loads(BASELINE.read_text())
    assert data["schema"] == benchmark.SCHEMA
    assert data["work_item"] == "X86-E2E-2"
    assert data["evidence_arch"] == "x86_64_avx512"
    assert data["all_correct"] is True
    assert len(data["rows"]) == 9
    assert {row["family"] for row in data["rows"]} == {"unary", "binary", "predicate"}
    assert data["selector_changed"] is True
    assert data["selector_policy_pass"] is True
    assert all(
        row["end_to_end"]["non_regression_10pct"]
        for row in data["rows"] if row["selector_eligible"]
    )
    small_binary = [
        row for row in data["rows"]
        if row["family"] == "binary" and row["elements"] < benchmark.BINARY_PROMOTION_MIN_ELEMENTS
    ]
    assert small_binary and all(not row["selector_eligible"] for row in small_binary)
