"""Ledger guard for the APPLE-RASTER-1 exact-device decision."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "benchmarks/baselines/apple7_raster_2026_07_31.json"
BENCHMARK = ROOT / "benchmarks/apple_gpu/benchmark_raster_order.py"


def test_apple7_raster_ledger_retains_row_major_after_matched_ab() -> None:
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert payload["target"] == "Apple M1 Max / apple7"
    assert payload["repetitions_per_round"] == 15
    assert payload["independent_warm_runs"] == 2
    assert payload["selection"]["selected_default"] == "row_major"
    assert len(payload["rows"]) == 4
    for row in payload["rows"]:
        assert row["native_gpu"] and row["numerically_validated"]
        assert len(row["baseline_row_major_ns"]) == 2
        assert len(row["candidate_grouped_m_2_ns"]) == 2
        assert all(value > 0 for value in row["baseline_row_major_ns"])
        assert all(value > 0 for value in row["candidate_grouped_m_2_ns"])


def test_raster_benchmark_records_two_warm_pairs_and_the_shared_axis() -> None:
    source = BENCHMARK.read_text(encoding="utf-8")
    assert "--rounds" in source
    assert 'order="grouped_m"' in source
    assert 'order="row_major"' in source
    assert "raster_order/raster_group only" in source
