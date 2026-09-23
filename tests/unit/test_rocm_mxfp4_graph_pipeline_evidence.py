"""Bind the exact-device graph-pipeline packet to the tested source files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "benchmarks/baselines/gfx1201_mxfp4_graph_pipeline_20260923/evidence.json"


def test_graph_pipeline_packet_retains_pr820_source_binding_and_device() -> None:
    packet = json.loads(EVIDENCE.read_text())
    assert packet["schema_version"] == 1
    proof = packet["proof"]
    assert proof["revision"] == "f8e754d4a7e5e411b5972aefc54880a91277f6fd"
    assert proof["host"] == "tajasarus"
    assert proof["device_name"] == "AMD Radeon RX 9070 XT"
    assert proof["target"] == "rocm_gfx1201"
    assert proof["pool_receipt"]["dynamic_m_strategy"] == "separate_exact_shape_graphs"
    assert proof["pool_receipt"]["active_m"] == [65, 129]
    assert proof["distinct_live_input_pointers"]
    assert proof["leases_invalid_after_pool_close"]
    assert proof["automatic_selection"] is False
    assert {tuple(row["shape"]) for row in proof["rows"]} == {
        (65, 48, 64), (129, 48, 64),
    }
    for row in proof["rows"]:
        assert row["capture_nodes"] == [0, 0, 0]
        assert row["bf16_oracle_exact"]
        assert len(row["gemm_hsaco_sha256"]) == len(row["aux_hsaco_sha256"]) == 64
    historical = {
        "python/tessera/compiler/rocm_mxfp4_graph_pipeline.py":
            "3ce292a4618535b708762b09a3f86eefe85f95fc552d972ce839a6167d8402b1",
        "tests/device/rocm/test_mxfp4_graph_pipeline.py":
            "e8f578bede39cf56702631d8a99b253efcc77ef975863ac49e465eabcfbde309",
    }
    for source, digest in proof["source_sha256"].items():
        if source in historical:
            assert digest == historical[source]
        else:
            assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest

    assert {tuple(row["shape"]) for row in packet["benchmarks"]} == {
        (256, 5120, 8704), (1024, 17408, 5120),
    }
    for row in packet["benchmarks"]:
        assert row["host"] == "tajasarus"
        assert row["device_name"] == "AMD Radeon RX 9070 XT"
        assert row["capture_nodes"] == [0, 0, 0]
        assert row["sampled_bf16_exact"] and row["full_output_equals_direct"]
        assert row["automatic_selection"] is False
        assert row["revision"] == proof["revision"]
        assert row["graph"]["host_enqueue_us_median"] < row["direct"][
            "host_enqueue_us_median"
        ]
        for source, digest in row["source_sha256"].items():
            if source in historical:
                assert digest == historical[source]
            else:
                assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest
