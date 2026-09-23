"""Current-source and timed-HSACO checks for the gfx1201 word-decode probe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKET = (
    ROOT / "benchmarks/baselines/gfx1201_mxfp4_packed_permute_20260923"
    / "evidence.json"
)


def test_packed_permute_packet_binds_current_sources_and_timed_isa() -> None:
    packet = json.loads(PACKET.read_text())
    assert packet["schema"] == "tessera.rocm.gfx1201_mxfp4_packed_folded_benchmark.v3"
    assert packet["sync_key"] == "GFX1201-PACKED-PERMUTE-DECODE-2026-09-23"
    assert packet["source_revision"] == "f7e3fbed7f27fc8eba0dd3411b866279acf648b9"
    assert packet["device"] == "AMD Radeon RX 9070 XT"
    assert packet["architecture"] == "gfx1201"
    assert packet["radiance"]["revision"] == (
        "dfdfa3832922c9a4253133f09c1f5c0d39748fc7"
    )
    assert packet["radiance"]["wperm"] == 1
    for key, path in {
        "generator_sha256": "python/tessera/compiler/rocm_mxfp4_folded.py",
        "packed_abi_sha256": "python/tessera/compiler/rocm_mxfp4_packed_folded.py",
        "benchmark_sha256": "benchmarks/rocm/benchmark_gfx1201_mxfp4_packed_folded.py",
        "isa_inspector_sha256": "benchmarks/rocm/inspect_gfx1201_folded_prefill.py",
    }.items():
        assert packet[key] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    expected = {
        "tessera", "tessera_folded", "tessera_packed_table",
        "tessera_packed_integer", "tessera_packed_batched_b_integer",
        "tessera_packed_batched_b_permute", "radiance",
    }
    for case in ("prefill_256x5120x8704", "prefill_1024x17408x5120"):
        rows = {row["engine"]: row for row in packet["rows"] if row["case"] == case}
        assert set(rows) == expected
        assert len({row["output_sha256"] for row in rows.values()}) == 1
        candidate = rows["tessera_packed_batched_b_permute"]
        metadata = candidate["metadata"]
        selected = metadata["selected_isa"]
        assert selected["payload_sha256"] == metadata["image_sha256"]
        assert selected["instruction_stream_sha256"]
        assert selected["mnemonics"]["v_perm_b32"] == 8
        assert selected["mnemonics"]["v_wmma_f32_16x16x16_fp8_fp8"] == 32
        assert selected["mnemonics"]["s_wait_loadcnt"] == 75
        assert metadata["resources"]["vgpr_count"] == 110
        assert metadata["resources"]["spills"] is False
        assert metadata["route"]["execution_state"] == "manual_executable_candidate"
        assert metadata["route"]["sync_key"] == packet["sync_key"]
        assert metadata["route"]["decode_strategy"] == "permute_word"
        assert candidate["median_ms"] < rows["tessera_packed_integer"]["median_ms"]
        assert candidate["median_ms"] < (
            rows["tessera_packed_batched_b_integer"]["median_ms"]
        )
        assert rows["radiance"]["median_ms"] < candidate["median_ms"]
