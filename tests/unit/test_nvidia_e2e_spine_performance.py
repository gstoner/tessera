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

def test_attention_benchmark_oracle_uses_bottom_right_causal_alignment() -> None:
    import numpy as np

    from benchmarks.nvidia.record_e2e_spine_attention import _attention_reference

    q = np.zeros((1, 1, 2, 1), dtype=np.float32)
    k = np.zeros((1, 1, 3, 1), dtype=np.float32)
    v = np.asarray([1.0, 2.0, 4.0], dtype=np.float32).reshape(1, 1, 3, 1)
    got = _attention_reference(q, k, v, scale=1.0, causal=True)
    np.testing.assert_allclose(got.reshape(-1), [1.5, 7.0 / 3.0], rtol=0, atol=1e-7)
