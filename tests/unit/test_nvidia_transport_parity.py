from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_committed_transport_parity_is_oracle_checked_and_resource_linked():
    data = json.loads((ROOT / "benchmarks/baselines/nvidia_sm120_transport_parity.json").read_text())
    assert data["schema"] == "tessera.nvidia.transport-parity.v1"
    assert data["device"] == "nvidia:sm_120"
    assert data["selector_changed"] is False
    assert data["noise_policy"] == 0.04
    assert len(data["rows"]) == 13
    assert {row["op"] for row in data["rows"]} == {
        "paged_kv", "moe_dispatch", "moe_combine", "grouped_gemm"}
    paged = [row for row in data["rows"] if row["op"] == "paged_kv"]
    assert {row["candidate"] for row in paged} == {"fused", "staged"}
    assert {row["boundary_relation"] for row in paged} == {"exact", "ragged"}
    assert all(row["page_mapping"] == "permuted" for row in paged)
    for row in data["rows"]:
        assert len(row["runs"]) == 2
        assert all(len(run["device_batch_medians_ms"]) >= 21
                   and len(run["end_to_end_batch_medians_ms"]) == 101
                   for run in row["runs"])
        assert max(run["max_abs_error"] for run in row["runs"]) < 1e-5
        assert row["traffic_bytes"] > 0
        assert row["traffic_formula"]
        assert row["achieved_bandwidth_gbps"] > 0
        assert row["launches_per_call"] >= 1
        assert row["resource_evidence_complete"]
        assert row["resource_fingerprints"]
        assert all(resource["spill_evidence_complete"]
                   for resource in row["resources"])
    moe_rows = [row for row in data["rows"] if row["op"] in {"moe_dispatch", "moe_combine"}]
    assert all(row["transport_descriptor"]["schema"] == "tessera.moe_transport.v1"
               for row in moe_rows)
    grouped = next(row for row in data["rows"] if row["op"] == "grouped_gemm")
    assert grouped["grouped_expert_metadata"]["ragged"] is True
