from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "benchmarks/baselines/nvidia_sm120_macro_cta_2026_08_24.json"


def _packet() -> dict[str, object]:
    return json.loads(PACKET.read_text(encoding="utf-8"))


def test_sm120_macro_cta_packet_is_fail_closed_for_wsl() -> None:
    packet = _packet()
    decision = packet["decision"]

    assert packet["schema"] == "tessera.nvidia.scheduled-macro-matmul.v3"
    assert packet["device"] == "sm_120"
    assert packet["host"]["wsl"] is True
    assert decision["route"] == "sm120_scheduled_macro_cta_32x32"
    assert decision["scheduled_route_enabled"] is True
    assert decision["global_selector_changed"] is False
    assert decision["selector_eligibility"] == "pruning_only_wsl"


def test_sm120_macro_cta_eligible_rows_match_the_recorded_decision() -> None:
    packet = _packet()
    decision = packet["decision"]
    threshold = decision["minimum_flops"]
    rows = packet["rows"]
    eligible = [row for row in rows if row["traffic_model"]["scheduled"]["flops"] >= threshold]

    assert len(eligible) == decision["eligible_rows"] == 3
    assert decision["all_numerical_rows_green"] is True
    assert decision["eligible_rows_low_variance"] is True
    assert decision["eligible_rows_at_least_three_percent_faster"] is True
    for row in rows:
        assert row["max_abs_error"]["scheduled"] == row["max_abs_error"]["direct"]
        assert row["selector_changed"] is False
    for row in eligible:
        assert row["sample_cov"]["scheduled"] <= 0.03
        assert row["sample_cov"]["direct"] <= 0.03
        assert row["scheduled_over_direct"] <= 0.97
        metrics = row["compile_resources"]["scheduled"]["metrics"]
        assert metrics["spill_load_bytes"] == 0
        assert metrics["spill_store_bytes"] == 0
        assert metrics["static_shared_memory_bytes"] == 4096
