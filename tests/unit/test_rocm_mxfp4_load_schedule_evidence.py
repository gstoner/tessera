"""Content gates for the gfx1201 folded load-schedule packet (GFX1201-LANES-2026-09-27)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "benchmarks/baselines/gfx1201_mxfp4_prefill_20260927"


def _rows(packet: dict) -> list[tuple[str, dict]]:
    return [
        (case["case"], row)
        for process in packet["processes"] for case in process["cases"]
        for row in case["rows"]
    ]


@pytest.mark.parametrize("name", ["production.json", "sweep.json"])
def test_packet_is_clean_matched_and_witnessed(name: str) -> None:
    packet = json.loads((PACKET / name).read_text())
    assert packet["schema"] == "tessera.rocm.gfx1201_mxfp4_folded_load_schedule.v1"
    assert packet["sync_key"] == "GFX1201-LANES-2026-09-27"
    assert packet["source"]["worktree_dirty"] is False
    assert packet["source"]["execution_environment"] == "wsl2"
    assert packet["radiance"]["wperm"] == 1
    assert len(packet["processes"]) == 3
    for process in packet["processes"]:
        assert process["device"] == "AMD Radeon RX 9070 XT"
        assert process["architecture"] == "gfx1201"
    rows = _rows(packet)
    for case, row in rows:
        assert row["timing_source"] == "device_wall_clock_marker"
        assert row["witness"] == "hip_event"
        assert row["witness_agrees"] and row["max_device_event_disagreement"] <= 0.05
        same_case = {r["output_sha256"] for c, r in rows if c == case}
        assert len(same_case) == 1, case  # every engine produced the same bits
    for summary in packet["summary"]:
        assert all(ratio < 1.0 for ratio in summary["selected_over_v1"]), summary["case"]


def test_selected_schedule_comes_from_target_ir_and_keeps_the_wmma_mix() -> None:
    packet = json.loads((PACKET / "production.json").read_text())
    for process in packet["processes"]:
        for case in process["cases"]:
            schedule = case["selected_schedule"]
            assert schedule["raster_group_m"] == 4
            assert schedule["staging_prefetch"] == "register_next_slab"
            assert schedule["epilogue_schedule"] == "complete_tile_vector_scales"
            assert schedule["workgroup_mode"] == ("cu" if case["m"] > 256 else "wgp")
            assert case["frontend_receipt"]["numeric_policy"] == (
                "folded_row_reference_explicit_approximate"
            )
            rows = {row["engine"]: row for row in case["rows"]}
            for engine in ("tessera_folded_selected", "tessera_folded_v1"):
                metadata = rows[engine]["metadata"]
                mnemonics = metadata["selected_isa"]["mnemonics"]
                assert mnemonics["v_wmma_f32_16x16x16_fp8_fp8"] == 32
                assert mnemonics["s_barrier_signal"] == mnemonics["s_barrier_wait"] == 2
                assert metadata["resources"]["spills"] is False
                assert metadata["resources"]["scratch_bytes"] == 0


def test_v1_source_identity_record_is_well_formed() -> None:
    identity = json.loads((PACKET / "v1_source_identity.json").read_text())
    assert identity["relabel_generator_sha256"] == (
        "0f57f6231677fb72a00a973ec332c5c8333016395163d67d0c940a854d9c3f53"
    )
    assert len(identity["default_emission_sha256"]) == 4
