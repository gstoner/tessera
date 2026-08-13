"""Truthfulness gates for the exact gfx1151 Block AttnRes Phase-5 packet."""

from __future__ import annotations

import json
from pathlib import Path


PACKET = Path(__file__).resolve().parents[2] / "benchmarks/baselines/rocm_gfx1151_block_attnres_phase5.json"


def test_gfx1151_block_attnres_packet_is_correct_and_provisionally_timed() -> None:
    packet = json.loads(PACKET.read_text(encoding="utf-8"))
    assert packet["schema"] == "tessera.block_attnres.gfx1151.phase5.v1"
    assert packet["architecture"] == "gfx1151"
    assert packet["algorithm"] == "rms_key_online_softmax_stats_v1"
    assert packet["merge"] == "max_shifted_pairwise_merge_v1"
    assert packet["verdict"]["correctness"] == "retain"
    assert packet["verdict"]["selector_eligible"] is False
    assert packet["timing_provenance"]["selector_eligible"] is False
    assert len(packet["rows"]) == 3
    for row in packet["rows"]:
        assert row["correct"] is True
        assert row["maximum_absolute_error"] <= 3.0e-5
        assert len(row["schedule_digest"]) == 64
        assert len(row["tile_ir_digest"]) == 64
        assert len(row["image_digest"]) == 64
        assert row["host_wall_ns"]["median"] > 0
        assert row["host_wall_ns"]["selector_eligible"] is False
        assert row["hip_event_ns"]["valid"] is False
