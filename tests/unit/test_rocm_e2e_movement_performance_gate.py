"""Host-free contract for the ROCM-E2E-2 movement comparison record."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/rocm/benchmark_rocm_e2e_movement.py"
BASELINE = ROOT / "benchmarks/baselines/rocm_gfx1151_e2e_movement_comparison.json"
SPEC = importlib.util.spec_from_file_location("rocm_e2e_movement_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_summary_requires_paired_samples_and_applies_ten_percent_gate() -> None:
    assert benchmark._summary([1.0, 1.1], [1.05, 1.0])["non_regression_10pct"] is True
    assert benchmark._summary([1.0, 1.0], [1.2, 1.2])["non_regression_10pct"] is False


def test_recorded_gfx1151_movement_evidence_retains_nonwinning_route() -> None:
    data = json.loads(BASELINE.read_text())
    assert data["schema"] == benchmark.SCHEMA
    assert data["evidence_arch"] == "gfx1151"
    assert data["all_correct"] is True
    assert data["all_non_regression"] is False
    assert data["selector_changed"] is False
    rows = {row["operation"]: row for row in data["rows"]}
    assert set(rows) == {"paged_kv_read", "moe_dispatch"}
    assert rows["paged_kv_read"]["device"]["non_regression_10pct"] is True
    assert rows["paged_kv_read"]["end_to_end"]["non_regression_10pct"] is False
    assert rows["moe_dispatch"]["end_to_end"]["non_regression_10pct"] is False
    assert "retain production routes" in data["closure_disposition"]


def test_recorder_keeps_device_and_end_to_end_domains_separate() -> None:
    source = SCRIPT.read_text()
    assert "hipEventElapsedTime" in source
    assert "rt.launch" in source
    assert "selector_changed" in source
