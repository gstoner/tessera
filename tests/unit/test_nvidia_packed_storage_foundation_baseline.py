import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    ROOT
    / "benchmarks"
    / "baselines"
    / "nvidia_sm120_packed_storage_foundation.json"
)


def test_packed_storage_foundation_packet_retains_two_domain_proof() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["schema"] == "tessera.nvidia.packed-storage-foundation.v1"
    assert report["arch"] == "sm_120a"
    assert report["timing_policy"] == {
        "discard_first_end_to_end": True,
        "amortized_launches": 10,
        "device_observations": 7,
        "aggregate": "median",
    }
    assert report["stability_policy"]["limit_pct"] == 4.0
    assert report["stability_policy"]["required_domains"] == [
        "device_event",
        "end_to_end",
    ]

    rows = report["rows"]
    assert {row["logical_dtype"] for row in rows} == {
        "int4",
        "nvfp4",
        "fp6_e2m3",
        "fp6_e3m2",
    }
    assert report["stability_policy"]["total_rows"] == len(rows) == 4
    assert report["stability_policy"]["stable_rows"] == sum(
        row["cross_run"]["stable_both_domains"] for row in rows
    )
    for row in rows:
        assert row["selected_route"] == "compiler_owned_tile_packed_load"
        assert row["selector_eligible"] is False
        assert row["selector_disposition"] == "retain"
        assert row["resources"]["registers_per_thread"] == 18
        assert row["resources"]["local_memory_bytes"] == 0
        assert row["cross_run"]["resource_identity"] is True
        assert row["cross_run"]["image_identity"] is True
        assert row["cross_run"]["descriptor_identity"] is True
        assert row["cross_run"]["cache_identity"] is True
        assert row["evidence_fingerprint"].startswith("sha256:")
