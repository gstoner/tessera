"""Content and selected-ISA gates for the clean gfx1201 packed proof."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKET = (
    ROOT / "benchmarks/baselines/gfx1201_mxfp4_packed_folded_20260923"
    / "evidence.json"
)


def test_packed_folded_packet_binds_source_output_and_timed_isa() -> None:
    packet = json.loads(PACKET.read_text())
    assert packet["schema"] == "tessera.rocm.gfx1201_mxfp4_packed_folded_benchmark.v1"
    assert packet["source_revision"] == "eeeabc3379d4bcbb88b6e3503fca27995d671b53"
    assert packet["device"] == "AMD Radeon RX 9070 XT"
    assert packet["architecture"] == "gfx1201"
    assert packet["radiance"]["revision"] == (
        "dfdfa3832922c9a4253133f09c1f5c0d39748fc7"
    )
    assert packet["radiance"]["wperm"] == 1
    assert packet["radiance"]["weight_layout"] == "fragment_order"
    for key, source in {
        "generator_sha256": "python/tessera/compiler/rocm_mxfp4_folded.py",
        "packed_abi_sha256": "python/tessera/compiler/rocm_mxfp4_packed_folded.py",
        "benchmark_sha256": "benchmarks/rocm/benchmark_gfx1201_mxfp4_packed_folded.py",
    }.items():
        assert packet[key] == hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
    for case in ("prefill_256x5120x8704", "prefill_1024x17408x5120"):
        rows = {row["engine"]: row for row in packet["rows"] if row["case"] == case}
        assert set(rows) == {
            "tessera", "tessera_folded", "tessera_packed_table",
            "tessera_packed_integer", "radiance",
        }
        assert len({row["output_sha256"] for row in rows.values()}) == 1
        for name in ("tessera_packed_table", "tessera_packed_integer"):
            timed = rows[name]["metadata"]
            selected = timed["selected_isa"]
            assert selected["payload_sha256"] == timed["image_sha256"]
            assert selected["entry_symbol"] == "tessera_mxfp4_packed_folded_prefill"
            assert selected["instruction_stream_sha256"]
            assert selected["mnemonics"]["v_wmma_f32_16x16x16_fp8_fp8"] == 32
            assert selected["mnemonics"]["s_barrier_signal"] == 2
            assert selected["mnemonics"]["s_barrier_wait"] == 2
            assert timed["route"]["execution_state"] == "manual_executable_candidate"
            assert timed["route"]["fold_lossless"] is True
            assert timed["resources"]["spills"] is False
        assert rows["tessera_packed_integer"]["median_ms"] < (
            rows["tessera_packed_table"]["median_ms"]
        )
        assert rows["tessera_folded"]["median_ms"] < (
            rows["tessera_packed_integer"]["median_ms"]
        )
        assert rows["radiance"]["median_ms"] < rows["tessera_folded"]["median_ms"]
