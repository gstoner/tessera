"""Content and route gates for the clean gfx1201 K64 staging packet."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKET = (
    ROOT / "benchmarks/baselines/gfx1201_mxfp4_folded_k64_staging_20260922"
    / "evidence.json"
)
CENSUS = PACKET.with_name("staging_census.json")


def test_k64_staging_packet_binds_selected_generator_and_exact_output() -> None:
    packet = json.loads(PACKET.read_text())
    assert packet["schema"] == "tessera.rocm.gfx1201_mxfp4_folded_benchmark.v3"
    assert packet["source_revision"] == (
        "0e21ea607cc2f12dd9e34f310b77f31004eaeccd"
    )
    assert packet["device"] == "AMD Radeon RX 9070 XT"
    assert packet["architecture"] == "gfx1201"
    assert packet["radiance"]["wperm"] == 1
    assert packet["radiance"]["weight_layout"] == "fragment_order"
    for key, path in {
        "benchmark_sha256": "benchmarks/rocm/benchmark_gfx1201_mxfp4_folded.py",
        "frontend_sha256": "python/tessera/compiler/rocm_mxfp4_folded_frontend.py",
        "materializer_sha256": "python/tessera/compiler/rocm_mxfp4_folded_carrier.py",
        "folded_generator_sha256": "python/tessera/compiler/rocm_mxfp4_folded.py",
    }.items():
        assert packet[key] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    for case in ("prefill_256x5120x8704", "prefill_1024x17408x5120"):
        rows = {row["engine"]: row for row in packet["rows"] if row["case"] == case}
        assert set(rows) == {"tessera", "tessera_folded", "radiance"}
        assert len({row["output_sha256"] for row in rows.values()}) == 1
        folded = rows["tessera_folded"]
        metadata = folded["metadata"]
        assert metadata["route"]["staging_policy"] == "unconditional_k64"
        assert metadata["route"]["fold_lossless"] is True
        assert metadata["frontend_receipt"]["hsaco_sha256"] == (
            metadata["image_sha256"]
        )
        assert metadata["resources"]["vgpr_count"] == 109
        assert metadata["resources"]["spills"] is False
        assert folded["median_ms"] < rows["tessera"]["median_ms"]


def test_k64_selected_isa_census_binds_the_timed_packet() -> None:
    packet = json.loads(PACKET.read_text())
    census = json.loads(CENSUS.read_text())
    assert census["schema"] == "tessera.rocm.gfx1201_folded_staging_census.v2"
    assert census["source_revision"] == packet["source_revision"]
    assert census["matched_packet_sha256"] == hashlib.sha256(
        PACKET.read_bytes()
    ).hexdigest()
    assert census["census_sha256"] == hashlib.sha256(
        (ROOT / "benchmarks/rocm/inspect_gfx1201_folded_prefill.py").read_bytes()
    ).hexdigest()
    assert census["tessera"]["generator_sha256"] == packet["folded_generator_sha256"]
    images = census["tessera"]["timed_images"]
    assert set(images) == {
        "prefill_256x5120x8704", "prefill_1024x17408x5120",
    }
    assert len({row["image_sha256"] for row in images.values()}) == 2
    for case, image in images.items():
        timed = next(
            row for row in packet["rows"]
            if row["case"] == case and row["engine"] == "tessera_folded"
        )["metadata"]
        selected = timed["selected_isa"]
        assert image["image_sha256"] == timed["image_sha256"]
        assert image["image_sha256"] == selected["payload_sha256"]
        assert image["entry_symbol"] == timed["frontend_receipt"]["entry_symbol"]
        assert image["instruction_stream_sha256"] == (
            selected["instruction_stream_sha256"]
        )
        assert image["instruction_count"] == selected["instruction_count"]
        assert image["isa"] == selected["mnemonics"]
        assert image["isa"]["v_wmma_f32_16x16x16_fp8_fp8"] == 32
        assert image["isa"]["s_barrier_signal"] == 2
        assert image["isa"]["s_barrier_wait"] == 2
        assert image["resources"]["vgpr_count"] == 109
    assert census["not_measured_dram_or_dynamic_instructions"] is True
    for shape in ("256x5120x8704", "1024x17408x5120"):
        counts = census["requested_bytes"][shape]
        assert counts["tessera_b_folded"] == 2 * counts["radiance_b_packed"]
