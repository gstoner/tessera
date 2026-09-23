"""Bind exact-device graph evidence to current source and a kernel-only route."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "benchmarks/baselines/gfx1201_mxfp4_graph_20260923/evidence.json"


def test_graph_packet_matches_current_sources_and_exact_device() -> None:
    packet = json.loads(EVIDENCE.read_text())
    assert packet["schema_version"] == 1
    assert {tuple(row["shape"]) for row in packet["results"]} == {
        (256, 5120, 8704), (1024, 17408, 5120),
    }
    for row in packet["results"]:
        assert row["host"] == "tajasarus"
        assert row["device_name"] == "AMD Radeon RX 9070 XT"
        assert row["target"] == "rocm_gfx1201"
        assert row["capture_nodes"] == [0]
        assert row["graph_receipt"]["graph_captures"] == 1
        assert row["graph_receipt"]["automatic_selection"] is False
        assert row["graph_receipt"]["image_sha256"] == row["hsaco_sha256"]
        assert row["sampled_bf16_exact"] and row["full_output_equals_direct"]
        assert row["graph"]["host_enqueue_us_median"] < row["direct"][
            "host_enqueue_us_median"
        ]
        for source, digest in row["source_sha256"].items():
            assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest
