from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/x86/benchmark_x86_e2e_flat_followon.py"
BASELINE = ROOT / "benchmarks/baselines/x86_avx512_e2e_flat_followon_comparison.json"
SPEC = importlib.util.spec_from_file_location("x86_e2e_flat_followon", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_recorded_flat_followon_controls_selector_policy() -> None:
    data = json.loads(BASELINE.read_text())
    assert data["schema"] == benchmark.SCHEMA
    assert data["all_correct"] is True
    assert data["selector_policy_pass"] is True
    assert data["selector_changed"] is True
    assert {row["family"] for row in data["rows"]} == {
        "where", "transcendental", "pow", "silu_mul",
    }
    assert all(
        row["end_to_end"]["non_regression_10pct"]
        for row in data["rows"] if row["selector_eligible"]
    )
    excluded = [row for row in data["rows"] if not row["selector_eligible"]]
    assert {row["family"] for row in excluded} == {"where", "pow", "silu_mul"}
