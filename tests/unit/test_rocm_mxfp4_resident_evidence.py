"""Bind exact-device lifecycle results to the current executor and benchmark."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "benchmarks/baselines/gfx1201_mxfp4_resident_20260923/evidence.json"


def test_resident_lifecycle_packet_matches_sources_and_route() -> None:
    packet = json.loads(EVIDENCE.read_text())
    assert packet["schema_version"] == 1
    assert len(packet["results"]) == 2
    assert {tuple(row["shape"]) for row in packet["results"]} == {
        (256, 5120, 8704), (1024, 17408, 5120),
    }
    for row in packet["results"]:
        assert row["host"] == "tajasarus"
        assert row["device_name"] == "AMD Radeon RX 9070 XT"
        assert row["target"] == "rocm_gfx1201"
        assert row["sampled_bf16_exact"] and row["full_output_equals_legacy"]
        assert row["resident_receipt"]["automatic_selection"] is False
        assert row["resident_receipt"]["module_loads"] == 1
        assert row["resident_receipt"]["weight_uploads"] == 1
        assert row["resident_receipt"]["image_sha256"] == row["hsaco_sha256"]
        assert row["resident_receipt"]["weight_sha256"] == row["weight_sha256"]
        assert row["legacy_host_wall_us_median"] > row["resident_host_wall_us_median"]
        assert row["resident_kernel_event_us_median"] > 0
        for source, digest in row["source_sha256"].items():
            assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest
