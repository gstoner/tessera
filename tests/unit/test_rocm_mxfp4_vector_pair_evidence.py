"""Bind the opt-in paired-fragment ablation to its timed exact-device ISA."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "benchmarks/baselines/gfx1201_mxfp4_vector_pair_20260923/evidence.json"


def test_vector_pair_packet_is_source_bound_and_not_a_selector_win() -> None:
    packet = json.loads(PACKET.read_text())
    assert packet["schema"] == "tessera.rocm.gfx1201_mxfp4_packed_folded_benchmark.v4"
    assert packet["device"] == "AMD Radeon RX 9070 XT"
    assert packet["architecture"] == "gfx1201"
    assert packet["radiance"]["weight_layout"] == "fragment_order"
    for field, source in (
        ("generator_sha256", "python/tessera/compiler/rocm_mxfp4_folded.py"),
        ("packed_abi_sha256", "python/tessera/compiler/rocm_mxfp4_packed_folded.py"),
        ("benchmark_sha256", "benchmarks/rocm/benchmark_gfx1201_mxfp4_packed_folded.py"),
        ("isa_inspector_sha256", "benchmarks/rocm/inspect_gfx1201_folded_prefill.py"),
    ):
        assert packet[field] == hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
    for case in ("prefill_256x5120x8704", "prefill_1024x17408x5120"):
        rows = {row["engine"]: row for row in packet["rows"] if row["case"] == case}
        control = rows["tessera_packed_batched_b_permute"]
        vector = rows["tessera_packed_vector_pair_permute"]
        radiance = rows["radiance"]
        assert len(control["samples_ms"]) == len(vector["samples_ms"]) == 11
        assert control["output_sha256"] == vector["output_sha256"] == radiance["output_sha256"]
        for row in (control, vector):
            meta = row["metadata"]
            assert meta["selected_isa"]["payload_sha256"] == meta["image_sha256"]
            assert meta["resources"]["spills"] is False
            assert meta["selected_isa"]["mnemonics"]["v_wmma_f32_16x16x16_fp8_fp8"] == 32
        assert vector["metadata"]["selected_isa"]["mnemonics"]["global_load_b64"] == 1
        assert vector["median_ms"] > radiance["median_ms"]
    skinny = {row["engine"]: row for row in packet["rows"]
              if row["case"] == "prefill_256x5120x8704"}
    wide = {row["engine"]: row for row in packet["rows"]
            if row["case"] == "prefill_1024x17408x5120"}
    assert skinny["tessera_packed_vector_pair_permute"]["median_ms"] < (
        skinny["tessera_packed_batched_b_permute"]["median_ms"]
    )
    assert wide["tessera_packed_vector_pair_permute"]["median_ms"] > (
        wide["tessera_packed_batched_b_permute"]["median_ms"]
    )
