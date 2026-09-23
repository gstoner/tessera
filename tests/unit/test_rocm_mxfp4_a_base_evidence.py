"""Bind the A-base and corrected standalone vector-pair packets to tested code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "benchmarks/baselines/gfx1201_mxfp4_a_base_20260923"
SHAPES = {"prefill_256x5120x8704", "prefill_1024x17408x5120"}


@pytest.mark.parametrize("name,schema,sync_key,candidate", [
    (
        "evidence.json",
        "tessera.rocm.gfx1201_mxfp4_packed_folded_benchmark.v5",
        "GFX1201-PACKED-A-BASE-2026-09-23",
        "tessera_packed_hoisted_a_base_permute",
    ),
    (
        "vector_pair_review_evidence.json",
        "tessera.rocm.gfx1201_mxfp4_packed_folded_benchmark.v4",
        "GFX1201-PACKED-VECTOR-PAIR-2026-09-23",
        "tessera_packed_vector_pair_permute",
    ),
])
def test_corrected_packet_has_control_provenance_and_timed_isa(
    name: str, schema: str, sync_key: str, candidate: str,
) -> None:
    packet = json.loads((EVIDENCE / name).read_text())
    assert packet["schema"] == schema
    assert packet["sync_key"] == sync_key
    assert packet["source_revision"] == "fa44b468b21ed37f07d7ed2de44ae839cb671949"
    assert packet["device"] == "AMD Radeon RX 9070 XT"
    assert packet["architecture"] == "gfx1201"
    assert packet["radiance"]["weight_layout"] == "fragment_order"
    for field, path in (
        ("generator_sha256", "python/tessera/compiler/rocm_mxfp4_folded.py"),
        ("packed_abi_sha256", "python/tessera/compiler/rocm_mxfp4_packed_folded.py"),
        ("benchmark_sha256", "benchmarks/rocm/benchmark_gfx1201_mxfp4_packed_folded.py"),
        ("isa_inspector_sha256", "benchmarks/rocm/inspect_gfx1201_folded_prefill.py"),
    ):
        assert packet[field] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    assert {row["case"] for row in packet["rows"]} == SHAPES
    for case in SHAPES:
        rows = {row["engine"]: row for row in packet["rows"] if row["case"] == case}
        control = rows["tessera_packed_batched_b_permute"]
        selected = rows[candidate]
        radiance = rows["radiance"]
        assert len(control["samples_ms"]) == len(selected["samples_ms"]) == 11
        assert control["output_sha256"] == selected["output_sha256"] == radiance["output_sha256"]
        assert selected["metadata"]["route"]["sync_key"] == sync_key
        assert control["metadata"]["route"]["sync_key"] == (
            "GFX1201-PACKED-PERMUTE-DECODE-2026-09-23"
        )
        for row in (control, selected):
            meta = row["metadata"]
            assert meta["selected_isa"]["payload_sha256"] == meta["image_sha256"]
            assert meta["selected_isa"]["instruction_stream_sha256"]
            assert meta["resources"]["spills"] is False
        assert selected["median_ms"] > radiance["median_ms"]


def test_a_base_hoist_does_not_win_both_shapes_or_reduce_vgprs() -> None:
    packet = json.loads((EVIDENCE / "evidence.json").read_text())
    rows = {(row["case"], row["engine"]): row for row in packet["rows"]}
    baseline = "tessera_packed_batched_b_permute"
    candidate = "tessera_packed_hoisted_a_base_permute"
    skinny = "prefill_256x5120x8704"
    wide = "prefill_1024x17408x5120"
    assert rows[(skinny, candidate)]["median_ms"] < rows[(skinny, baseline)]["median_ms"]
    assert rows[(wide, candidate)]["median_ms"] > rows[(wide, baseline)]["median_ms"]
    for shape in SHAPES:
        assert rows[(shape, baseline)]["metadata"]["resources"]["vgpr_count"] == 110
        assert rows[(shape, candidate)]["metadata"]["resources"]["vgpr_count"] == 110
        assert rows[(shape, baseline)]["metadata"]["selected_isa"]["mnemonics"]["s_wait_loadcnt"] == 75
        assert rows[(shape, candidate)]["metadata"]["selected_isa"]["mnemonics"]["s_wait_loadcnt"] == 75
