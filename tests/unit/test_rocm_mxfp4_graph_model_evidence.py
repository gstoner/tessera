"""Bind the model-owned gfx1201 graph packet to exact tested sources."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tessera.compiler.rocm_mxfp4_graph_selection import assess_graph_pipeline_admission


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "benchmarks/baselines/gfx1201_mxfp4_graph_model_20260923/evidence.json"


def test_graph_model_packet_is_current_and_selector_remains_closed() -> None:
    packet = json.loads(EVIDENCE.read_text())
    assert packet["revision"] == "bf422db924231eb56e46ce656d116a1dd0349222"
    assert packet["target"] == "rocm_gfx1201"
    assert packet["host"] == "tajasarus"
    assert packet["device_name"] == "AMD Radeon RX 9070 XT"
    assert packet["rocm_release"] == "10.0.0"
    assert packet["hip_runtime"].startswith("/opt/rocm/core-10.0/")
    assert packet["tests"]["counts"] == {
        "tests": 18, "failures": 0, "errors": 0, "skipped": 0,
    }
    for source, digest in packet["source_sha256"].items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest
    assert {tuple(row["shape"]) for row in packet["benchmarks"]} == {
        (256, 5120, 8704), (1024, 17408, 5120),
    }
    for row in packet["benchmarks"]:
        assert [item["variant"] for item in row["rounds"]] == [
            "block", "wave", "wave", "block",
        ]
        assert all(item["sampled_bf16_exact"] for item in row["rounds"])
        assert row["revision"] == packet["revision"]
        for source, digest in row["source_sha256"].items():
            assert packet["source_sha256"][source] == digest
    assert packet["admission"] == assess_graph_pipeline_admission(packet)
    assert packet["admission"]["selected_producer_variant"] == "block"
    assert packet["admission"]["automatic_selection"] is False
    assert "graph_device_time_regression" in packet["admission"]["refusals"]
