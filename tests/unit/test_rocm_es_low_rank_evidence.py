"""Drift gate for the exact gfx1151 EGGROLL W2 packet."""

from __future__ import annotations

import json
from pathlib import Path


PACKET = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/baselines/rocm_gfx1151_es_low_rank.json"
)


def test_gfx1151_es_packet_is_correct_but_not_performance_promoted() -> None:
    payload = json.loads(PACKET.read_text())
    assert payload["schema"] == "tessera.eggroll.es_low_rank.gfx1151.v1"
    assert payload["architecture"] == "gfx1151"
    assert payload["algorithm"] == "cooperative_wave32_lds_sgmv_rank1"
    assert payload["rng"] == "splitmix64-philox4x32-boxmuller.v1"
    assert payload["dtype"] == {"storage": "fp32", "accum": "fp32"}
    assert payload["shape"]["rank"] == 1
    assert len(payload["artifact_hash"]) == 64
    assert len(payload["hsaco_sha256"]) == 64
    assert payload["numerical"]["passed"] is True
    assert payload["verdict"]["correctness"] == "retain"
    assert payload["verdict"]["selector_eligible"] is False
    assert payload["timing"]["hip_event_ns"]["valid"] is False
    assert payload["timing"]["hip_event_ns"]["median"] is None
    assert payload["timing"]["host_wall_ns"]["median"] > 0
