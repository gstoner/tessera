"""Committed performance and selector gate for X86-E2E-2 cohorts 3/4."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from tessera.compiler.x86_breadth import GRAPH_PROMOTION_THRESHOLDS, X86_BREADTH_ABIS


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/x86/benchmark_x86_e2e_cohort34.py"
BASELINE = ROOT / "benchmarks/baselines/x86_avx512_e2e_cohort34_comparison.json"
SPEC = importlib.util.spec_from_file_location("x86_e2e_cohort34", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_cohort34_benchmark_is_operation_total() -> None:
    data = json.loads(BASELINE.read_text())
    assert data["schema"] == benchmark.SCHEMA
    assert data["evidence_arch"] == "x86_64_avx512"
    assert data["trials"] == 21
    assert data["all_correct"] is True
    assert data["operation_total"] is True
    dispositions = data["dispositions"]
    assert {row["abi_key"] for row in dispositions} == set(X86_BREADTH_ABIS)
    assert len(dispositions) == 33
    assert all(row["decision"].startswith(("promote_", "retain_")) for row in dispositions)


def test_selector_thresholds_are_the_first_measured_passing_domains() -> None:
    data = json.loads(BASELINE.read_text())
    rows = data["rows"]
    assert {row["family"] for row in rows} == set(GRAPH_PROMOTION_THRESHOLDS)
    for family, threshold in GRAPH_PROMOTION_THRESHOLDS.items():
        passing = [
            row["output_elements"] for row in rows
            if row["family"] == family and row["end_to_end"]["non_regression_10pct"]
        ]
        assert passing
        assert threshold == min(passing)
    promoted = {
        row["abi_key"]: row["threshold"]
        for row in data["dispositions"] if row["decision"] == "promote_measured"
    }
    assert promoted == {
        "gather_f32": 1_048_576,
        "pointwise_loss_f32": 16_384,
        "cholesky_f32": 2_048,
        "tri_solve_f32": 512,
    }


def test_composite_and_specialized_entries_remain_unpromoted() -> None:
    data = json.loads(BASELINE.read_text())
    retained = [row for row in data["dispositions"] if row["decision"].startswith("retain_")]
    assert len(retained) == 29
    assert all(row["threshold"] is None for row in retained)
    by_key = {row["abi_key"]: row["decision"] for row in retained}
    for key in (
        "sddmm_f32", "bitonic_sort_kv_f32", "fft_c2c_f32",
        "clifford_bilinear_f32", "policy_loss_f32", "philox_uniform_f32",
        "optimizer_f32", "kv_cache_append_f32",
    ):
        assert by_key[key] == "retain_composite"
