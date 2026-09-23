"""Keep the immutable PR 819 graph packet bound to its recorded source."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "benchmarks/baselines/gfx1201_mxfp4_graph_20260923/evidence.json"


def test_graph_packet_matches_recorded_sources_and_exact_device() -> None:
    packet = json.loads(EVIDENCE.read_text())
    assert packet["schema_version"] == 1
    assert packet["source_commit"] == "7abdec5e592fdfb48fcd4dc9c7f60006ceed5121"
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
        assert row["source_sha256"] == {
            "python/tessera/compiler/rocm_mxfp4_graph.py": (
                "74ac87e4c1a19740d051c0aec05e419a4b180bb9351183af0ae6a5bb75310d78"
            ),
            "benchmarks/rocm/benchmark_gfx1201_mxfp4_graph.py": (
                "56eadd9a2f83507a302001eb978b63ed93fa3e45b6a84d548a4dc005df6a1bca"
            ),
        }
