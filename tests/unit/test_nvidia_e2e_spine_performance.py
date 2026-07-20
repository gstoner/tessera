from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_comparative_corpus_is_stable_resource_linked_and_non_promoting() -> None:
    report = json.loads((
        ROOT / "benchmarks/baselines/nvidia_sm120_e2e_spine_comparative.json"
    ).read_text(encoding="utf-8"))
    assert report["schema"] == "tessera.nvidia.e2e-spine-comparative.v1"
    assert report["stability_policy"]["relative_fraction"] == 0.04
    assert report["method"]["discarded_end_to_end_launches_per_sample"] == 1
    assert report["method"]["end_to_end_repetitions_per_sample"] == 100
    assert report["selector_promotions"] == []
    rows = report["rows"]
    assert len(rows) == 14
    assert {row["op"] for row in rows} == {
        "softmax", "attention_forward", "moe_dispatch", "moe_combine",
        "moe_grouped_gemm",
    }
    assert all(row["stable"] for row in rows)
    for row in rows:
        assert len(row["runs"]) == 2
        assert row["resources"] is not None
        assert row["resource_fingerprint"]
        assert row["selector_eligible"] is False
        assert row["selector_changed"] is False
        assert row["selector_disposition"] == (
            "retain_existing_no_registered_materiality_threshold")
        assert row["device_repetitions_per_sample"] in {1000, 10000}


def test_dtype_matrix_records_one_explicit_non_promoting_tf32_terminal() -> None:
    report = json.loads((
        ROOT / "benchmarks/baselines/nvidia_sm120_e2e_spine_dtype_matrix.json"
    ).read_text(encoding="utf-8"))
    assert report["schema"] == "tessera.nvidia.e2e-spine-dtype-matrix.v1"
    assert report["method"]["device_repetitions_per_sample"] == 10000
    assert report["method"]["end_to_end_repetitions_per_sample"] == 50
    assert report["method"]["discarded_end_to_end_launches_per_sample"] == 1
    unstable = []
    for row in report["rows"]:
        stable = all(
            row["stability"][domain] <= row["stability"]["policy_fraction"]
            for domain in ("device_fraction", "end_to_end_fraction")
        )
        if not stable:
            unstable.append(row)
        assert row["resources"] is not None
        assert row["selector_changed"] is False
    assert len(report["rows"]) == 20
    assert len(unstable) == 1
    assert unstable[0]["storage"] == "tf32"
    assert unstable[0]["shape"] == [256, 256, 256]
