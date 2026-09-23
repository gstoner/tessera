"""Bind the opt-in 32-bit A-offset proof to exact-device timed payloads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tessera.compiler.rocm_mxfp4_packed_folded import _MAX_A_OFFSET32_K


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "benchmarks/baselines/gfx1201_mxfp4_a_offset32_20260923/evidence.json"


def test_offset32_packet_binds_current_sources_oracle_and_selected_isa() -> None:
    packet = json.loads(PACKET.read_text())
    assert packet["schema"] == "tessera.rocm.gfx1201_mxfp4_packed_folded_benchmark.v6"
    assert packet["sync_key"] == "GFX1201-PACKED-A-OFFSET32-2026-09-23"
    assert packet["source_revision"] == "cd758719458e6650a8e7b5e8f1c73403932f51bf"
    assert packet["device"] == "AMD Radeon RX 9070 XT"
    assert packet["architecture"] == "gfx1201"
    assert packet["radiance"]["weight_layout"] == "fragment_order"
    # The v6 packet predates the safe-epilogue generator and corrected census.
    assert packet["generator_sha256"] == (
        "d24678f845eae599511a148de70c85a0c81a07ef6525b06bb4fb001debeba34c"
    )
    assert packet["benchmark_sha256"] == (
        "07bda3c180807a28a0363cc1127ee3b5671537e11694ccd2edfd4896f491ecf5"
    )
    for field, path in (
        ("packed_abi_sha256", "python/tessera/compiler/rocm_mxfp4_packed_folded.py"),
        ("isa_inspector_sha256", "benchmarks/rocm/inspect_gfx1201_folded_prefill.py"),
    ):
        assert packet[field] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    for case in ("prefill_256x5120x8704", "prefill_1024x17408x5120"):
        rows = {row["engine"]: row for row in packet["rows"] if row["case"] == case}
        control = rows["tessera_packed_batched_b_permute"]
        base = rows["tessera_packed_hoisted_a_base_permute"]
        narrow = rows["tessera_packed_a_offset32_permute"]
        radiance = rows["radiance"]
        assert len(control["samples_ms"]) == len(base["samples_ms"]) == len(narrow["samples_ms"]) == 11
        assert len({row["output_sha256"] for row in (control, base, narrow, radiance)}) == 1
        assert narrow["metadata"]["route"]["sync_key"] == packet["sync_key"]
        assert narrow["metadata"]["route"]["a_offset32_k_bound"] == _MAX_A_OFFSET32_K
        for row in (control, base, narrow):
            meta = row["metadata"]
            assert meta["selected_isa"]["payload_sha256"] == meta["image_sha256"]
            assert meta["selected_isa"]["instruction_stream_sha256"]
            assert meta["resources"]["spills"] is False
            assert meta["resources"]["vgpr_count"] == 110
            assert meta["selected_isa"]["mnemonics"]["s_wait_loadcnt"] == 75
        assert narrow["metadata"]["selected_isa"]["instruction_count"] == 4358
        assert base["metadata"]["selected_isa"]["instruction_count"] == 4392
        assert narrow["metadata"]["integer_alu_isa"]["v_mul_lo_u32"] == 130
        assert base["metadata"]["integer_alu_isa"]["v_mul_lo_u32"] == 138
        assert narrow["metadata"]["integer_alu_isa"]["v_add_co_u32"] == 153
        assert base["metadata"]["integer_alu_isa"]["v_add_co_u32"] == 157
        assert narrow["median_ms"] > radiance["median_ms"]
    wide = {row["engine"]: row for row in packet["rows"]
            if row["case"] == "prefill_1024x17408x5120"}
    assert wide["tessera_packed_a_offset32_permute"]["median_ms"] > (
        wide["tessera_packed_batched_b_permute"]["median_ms"]
    )
