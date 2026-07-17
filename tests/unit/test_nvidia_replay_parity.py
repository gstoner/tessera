from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_committed_replay_parity_has_wide_transition_and_resource_proof():
    data = json.loads((ROOT / "benchmarks/baselines/nvidia_sm120_replay_parity.json").read_text())
    assert data["schema"] == "tessera.nvidia.replay-parity.v1"
    assert len(data["rows"]) == 10
    assert {row["shape"] for row in data["rows"]} == {
        "1x32x16", "1x64x64", "1x128x64", "1x128x128", "4x64x64"}
    assert {row["tokens"] for row in data["rows"]} == {16, 64}
    assert set(data["transition_proofs"]) == {
        "long_decode", "flush", "rollback", "speculative_rejection",
        "block_submit", "ordered_ring", "backpressure", "teardown"}
    for row in data["rows"]:
        assert len(row["runs"]) == 2
        assert all(len(run["device_batch_medians_ms_per_token"]) >= 20
                   and len(run["device_batch_medians_ms_per_token"])
                   == len(run["end_to_end_batch_medians_ms_per_token"])
                   for run in row["runs"])
        assert max(run["max_abs_error"] for run in row["runs"]) < 1e-6
        assert row["state_bytes_per_token"] > 0
        assert row["state_traffic_ratio"] > 1
        assert row["resource_evidence_complete"]
        assert row["resource_fingerprints"]
