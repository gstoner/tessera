from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "benchmarks/baselines/nvidia_sm120_bf16_reduction_breadth.json"


def test_bf16_reduction_packet_is_typed_two_domain_and_resource_linked() -> None:
    report = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert report["schema"] == "tessera.nvidia.bf16-reduction-breadth.v1"
    assert report["work_item"] == "NVIDIA-BF16-CANONICAL-BREADTH"
    assert report["stability_policy"]["relative_fraction"] == 0.04
    assert report["selector_promotions"] == []
    assert report["contract"] == {
        "storage": "bf16",
        "accumulation": "fp32",
        "output": "fp32",
        "schedules": ["serial", "cooperative_128"],
        "kinds": ["sum", "mean", "max", "min", "amax", "amin"],
        "arbitrary_axes": True,
        "keepdims": [False, True],
        "ragged_boundaries": True,
        "nonfinite_proof": "tests/device/nvidia/test_e2e_spine_native.py",
    }
    assert report["method"]["discarded_launches_per_candidate_sample"] == 1
    assert report["method"]["end_to_end_repetitions_per_sample"] == 10
    rows = report["rows"]
    assert len(rows) == 12
    assert {row["kind"] for row in rows} == {
        "sum", "mean", "max", "min", "amax", "amin",
    }
    assert {row["candidate"] for row in rows} == {
        "canonical_serial", "canonical_cooperative_128",
    }
    assert {row["storage"] for row in rows} == {"bf16"}
    assert all(row["accumulation"] == "fp32" for row in rows)
    assert all(row["output"] == "fp32" for row in rows)
    assert all(row["numerical"]["passed"] for row in rows)
    assert all(row["resources"] is not None for row in rows)
    assert all(row["resource_fingerprint"] for row in rows)
    assert all(row["compile"]["image_digest_reproducible"] for row in rows)
    assert all(row["compile"]["entry_symbol_reproducible"] for row in rows)
    assert all(row["selector_changed"] is False for row in rows)
    for row in rows:
        assert len(row["runs"]) == 2
        assert all(
            len(run["device_samples_ms"]) == report["method"]["samples_per_run"]
            for run in row["runs"]
        )
        assert all(
            len(run["end_to_end_samples_ms"]) == report["method"]["samples_per_run"]
            for run in row["runs"]
        )


def test_bf16_reduction_selector_eligibility_requires_both_gates() -> None:
    report = json.loads(BASELINE.read_text(encoding="utf-8"))
    for row in report["rows"]:
        expected = row["stable"] and row["winner_consensus"]
        assert row["selector_eligible"] == expected
