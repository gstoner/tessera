"""Recorded selector contract for three X86-E2E-2 typed logic slices."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/x86/benchmark_x86_e2e_typed_logic.py"
BASELINE = ROOT / "benchmarks/baselines/x86_avx512_e2e_typed_logic_comparison.json"
SPEC = importlib.util.spec_from_file_location("x86_e2e_typed_logic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_recorder_covers_three_abi_families_and_shape_classes() -> None:
    assert {case[0] for case in benchmark.CASES} == {"compare", "logical", "bitwise"}
    assert len(benchmark.SHAPES) == 4


def test_recorded_evidence_controls_family_specific_promotion() -> None:
    data = json.loads(BASELINE.read_text())
    assert data["schema"] == benchmark.SCHEMA
    assert data["work_item"] == "X86-E2E-2"
    assert data["evidence_arch"] == "x86_64_avx512"
    assert data["all_correct"] is True
    assert data["selector_policy_pass"] is True
    assert data["selector_changed"] is True
    assert len(data["rows"]) == 12
    assert all(
        row["end_to_end"]["non_regression_10pct"]
        for row in data["rows"] if row["selector_eligible"]
    )
    assert all(
        row["selector_eligible"]
        == (row["elements"] >= benchmark.PROMOTION_MIN_ELEMENTS[row["family"]])
        for row in data["rows"]
    )
