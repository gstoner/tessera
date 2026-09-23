"""Current-source and timed-HSACO gates for the gfx1201 staging ablation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKET = (
    ROOT / "benchmarks/baselines/gfx1201_mxfp4_packed_staging_20260923"
    / "evidence.json"
)


def test_packed_staging_packet_binds_current_source_and_timed_isa() -> None:
    packet = json.loads(PACKET.read_text())
    assert packet["schema"] == "tessera.rocm.gfx1201_mxfp4_packed_folded_benchmark.v2"
    assert packet["sync_key"] == "GFX1201-PACKED-STAGING-ABLATION-2026-09-23"
    assert packet["source_revision"] == "2c1bacd5ad51f98015ae7eb816a51ca3118c6f78"
    assert packet["device"] == "AMD Radeon RX 9070 XT"
    assert packet["architecture"] == "gfx1201"
    assert packet["radiance"]["revision"] == (
        "dfdfa3832922c9a4253133f09c1f5c0d39748fc7"
    )
    assert packet["radiance"]["wperm"] == 1
    assert packet["radiance"]["weight_layout"] == "fragment_order"
    assert packet["generator_sha256"] == (
        "d24678f845eae599511a148de70c85a0c81a07ef6525b06bb4fb001debeba34c"
    )
    # Immutable v2 control: v3 changes the generator and benchmark sources.
    assert packet["packed_abi_sha256"] == (
        "7973372af3a0dd25e28d559c29305caad62b41fd7e28a70002d4f78bdcda9fb6"
    )
    assert packet["benchmark_sha256"] == (
        "51c7c6a6a235b9bb57e4f0cb9c6b9838e161c08015f5f6597408057abde6a825"
    )
    expected_engines = {
        "tessera", "tessera_folded", "tessera_packed_table",
        "tessera_packed_integer", "tessera_packed_batched_b_integer",
        "tessera_packed_batched_a_integer", "tessera_packed_batched_ab_integer",
        "tessera_packed_pair_scale_integer", "radiance",
    }
    for case in ("prefill_256x5120x8704", "prefill_1024x17408x5120"):
        rows = {row["engine"]: row for row in packet["rows"] if row["case"] == case}
        assert set(rows) == expected_engines
        assert len({row["output_sha256"] for row in rows.values()}) == 1
        for name in expected_engines - {"tessera", "radiance", "tessera_folded"}:
            metadata = rows[name]["metadata"]
            selected = metadata["selected_isa"]
            assert selected["payload_sha256"] == metadata["image_sha256"]
            assert selected["entry_symbol"] == "tessera_mxfp4_packed_folded_prefill"
            assert selected["instruction_stream_sha256"]
            assert selected["mnemonics"]["v_wmma_f32_16x16x16_fp8_fp8"] == 32
            assert selected["mnemonics"]["s_barrier_signal"] == 2
            assert selected["mnemonics"]["s_barrier_wait"] == 2
            assert metadata["resources"]["spills"] is False
            assert metadata["route"]["execution_state"] == "manual_executable_candidate"
            assert metadata["route"]["fold_lossless"] is True
        original = rows["tessera_packed_integer"]
        batched_b = rows["tessera_packed_batched_b_integer"]
        assert original["metadata"]["selected_isa"]["mnemonics"]["s_wait_loadcnt"] == 88
        assert batched_b["metadata"]["selected_isa"]["mnemonics"]["s_wait_loadcnt"] == 87
        assert original["metadata"]["resources"]["vgpr_count"] == 117
        assert batched_b["metadata"]["resources"]["vgpr_count"] == 118
        assert batched_b["metadata"]["route"]["sync_key"] == packet["sync_key"]
        assert batched_b["median_ms"] < original["median_ms"]
        assert rows["tessera_folded"]["median_ms"] < batched_b["median_ms"]
        assert rows["radiance"]["median_ms"] < batched_b["median_ms"]


def test_packed_staging_rejects_a_and_pair_scale_for_wide_prefill() -> None:
    packet = json.loads(PACKET.read_text())
    rows = {
        row["engine"]: row for row in packet["rows"]
        if row["case"] == "prefill_1024x17408x5120"
    }
    batched_b = rows["tessera_packed_batched_b_integer"]["median_ms"]
    for name in (
        "tessera_packed_batched_a_integer",
        "tessera_packed_batched_ab_integer",
        "tessera_packed_pair_scale_integer",
    ):
        assert batched_b < rows[name]["median_ms"]
