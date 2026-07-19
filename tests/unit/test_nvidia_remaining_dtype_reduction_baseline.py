from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "benchmarks/baselines/nvidia_sm120_remaining_dtype_reduction.json"


def test_remaining_dtype_reduction_corpus_is_two_domain_and_resource_linked() -> None:
    report = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert report["schema"] == "tessera.nvidia.remaining-dtype-reduction.v1"
    assert report["stability_policy"]["relative_fraction"] == 0.04
    assert report["stability_policy"]["wsl_acceptance_margin_fraction"] == 0.0015
    assert report["stability_policy"]["margin_rows_selector_eligible"] is False
    assert report["method"]["sampling"] == "two_disjoint_time_interleaved_cohorts"
    assert report["method"]["discarded_launches_per_candidate_sample"] == 1
    assert report["method"]["device_repetitions_per_sample"] == 10
    assert report["method"]["end_to_end_repetitions_per_sample"] == 10
    assert report["selector_promotions"] == []
    rows = report["rows"]
    assert len(rows) == 30
    assert all(row["stable"] for row in rows)
    assert {row["storage"] for row in rows if row["op"] == "fused_epilogue"} == {
        "tf32", "fp8_e4m3", "fp8_e5m2",
    }
    assert {row["candidate"] for row in rows if row["op"] == "reduction"} == {
        "canonical_serial", "canonical_cooperative_128", "production_cuda_composed",
    }
    for row in rows:
        assert len(row["runs"]) == 2
        assert len(row["clock_conditioning_ms"]) == report["method"]["samples_per_run"]
        assert row["resource_fingerprint"]
        assert row["resources"] is not None
        assert row["strict_stable"] == all(
            delta <= report["stability_policy"]["relative_fraction"]
            for delta in row["stability_fraction"].values()
        )
        assert row["stable"] == all(
            delta <= report["stability_policy"]["relative_fraction"]
            + report["stability_policy"]["wsl_acceptance_margin_fraction"]
            for delta in row["stability_fraction"].values()
        )
        assert row["selector_changed"] is False


def test_selector_eligibility_requires_stability_and_cross_domain_consensus() -> None:
    rows = json.loads(BASELINE.read_text(encoding="utf-8"))["rows"]
    for row in rows:
        assert row["selector_eligible"] == (row["strict_stable"] and row["winner_consensus"])
