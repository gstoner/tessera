"""Pure guards for the static gfx1201 folded staging comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.rocm.inspect_gfx1201_folded_prefill import (
    _mnemonics, requested_bytes,
)
from benchmarks.rocm.ablate_gfx1201_folded_b_cache import (
    NEW_LOAD, OLD_LOAD, variant_source,
)


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "benchmarks/baselines/gfx1201_mxfp4_folded_frontend_20260922"


def test_requested_bytes_distinguish_expanded_and_packed_weights() -> None:
    first = requested_bytes(256, 5120, 8704)
    assert first["tessera_b_folded"] == 5120 * 8704
    assert first["radiance_b_packed"] == 5120 * 8704 // 2
    assert first["radiance_block_scales"] == 5120 * 8704 // 32
    second = requested_bytes(1024, 17408, 5120)
    assert second["tessera_b_folded"] == 4 * 17408 * 5120
    assert second["radiance_b_packed"] == second["tessera_b_folded"] // 2
    with pytest.raises(ValueError, match="divisible by 64"):
        requested_bytes(256, 80, 96)


def test_cache_ablation_changes_only_the_single_b_vector_load() -> None:
    from tessera.compiler.rocm_mxfp4_folded import emit_mxfp4_folded_prefill_hip

    baseline = emit_mxfp4_folded_prefill_hip()
    variant = variant_source()
    assert baseline.count(OLD_LOAD) == 1
    assert variant.count(NEW_LOAD) == 1
    assert variant.replace(NEW_LOAD, OLD_LOAD) == baseline


def test_isa_census_counts_only_instruction_mnemonics() -> None:
    isa = """
      global_load_b128 v[1:4], v[5:6], off
      ds_store_2addr_b64 v1, v[2:5]
      s_wait_loadcnt 0x0
      v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[9:10], v[11:12], v[1:8]
      unrelated_label:
    """
    assert _mnemonics(isa) == {
        "ds_store_2addr_b64": 1,
        "global_load_b128": 1,
        "s_wait_loadcnt": 1,
        "v_wmma_f32_16x16x16_fp8_fp8": 1,
    }


def test_frontend_matched_packet_and_static_census_are_content_bound() -> None:
    matched_path = BASELINE / "matched.json"
    matched = json.loads(matched_path.read_text())
    census = json.loads((BASELINE / "staging_census.json").read_text())
    assert matched["schema"] == "tessera.rocm.gfx1201_mxfp4_folded_benchmark.v2"
    assert matched["source_revision"] == (
        "6f8ba84674da52a710370dc07b83ac7ec2e13637"
    )
    assert matched["architecture"] == census["architecture"] == "gfx1201"
    assert matched["device"] == census["device"] == "AMD Radeon RX 9070 XT"
    assert matched["radiance"]["wperm"] == 1
    assert matched["radiance"]["weight_layout"] == "fragment_order"
    for key, source in {
        "benchmark_sha256": "benchmarks/rocm/benchmark_gfx1201_mxfp4_folded.py",
        "frontend_sha256": "python/tessera/compiler/rocm_mxfp4_folded_frontend.py",
        "materializer_sha256": "python/tessera/compiler/rocm_mxfp4_folded_carrier.py",
        "folded_generator_sha256": "python/tessera/compiler/rocm_mxfp4_folded.py",
    }.items():
        assert matched[key] == hashlib.sha256(
            (ROOT / source).read_bytes(),
        ).hexdigest()
    assert census["schema"] == "tessera.rocm.gfx1201_folded_staging_census.v1"
    assert census["source_revision"] == matched["source_revision"]
    assert census["matched_packet_sha256"] == hashlib.sha256(
        matched_path.read_bytes(),
    ).hexdigest()
    assert census["census_sha256"] == hashlib.sha256(
        (ROOT / "benchmarks/rocm/inspect_gfx1201_folded_prefill.py").read_bytes(),
    ).hexdigest()
    assert census["not_measured_dram_or_dynamic_instructions"] is True
    assert census["radiance"]["module_sha256"] == (
        matched["radiance"]["binary_sha256"]
    )
    for case in ("prefill_256x5120x8704", "prefill_1024x17408x5120"):
        rows = {row["engine"]: row for row in matched["rows"] if row["case"] == case}
        assert set(rows) == {"tessera", "tessera_folded", "radiance"}
        assert len({row["output_sha256"] for row in rows.values()}) == 1
        assert rows["tessera"]["median_ms"] > rows["tessera_folded"]["median_ms"]
        assert rows["tessera_folded"]["median_ms"] > rows["radiance"]["median_ms"]
        receipt = rows["tessera_folded"]["metadata"]["frontend_receipt"]
        assert receipt["abi_id"] == rows["tessera_folded"]["metadata"]["abi"]
        assert receipt["schedule_hash"]
        assert len(receipt["hsaco_sha256"]) == 64
        assert receipt["hsaco_sha256"] == (
            rows["tessera_folded"]["metadata"]["image_sha256"]
        )
        assert receipt["artifact_image_digest"] != receipt["hsaco_sha256"]
        assert receipt["fold_lossless"] is True
        shape = case.removeprefix("prefill_")
        bytes_row = census["requested_bytes"][shape]
        assert bytes_row["tessera_b_folded"] == 2 * bytes_row["radiance_b_packed"]
        assert bytes_row["tessera_a"] == bytes_row["radiance_a"]


def test_single_lever_b_cache_ablation_is_refused_and_bound() -> None:
    packet = json.loads((BASELINE / "b_cache_ablation.json").read_text())
    assert packet["schema"] == "tessera.rocm.gfx1201_folded_b_cache_ablation.v1"
    assert packet["source_revision"] == (
        "679c5b6603f5abd32ddbf878929644f1c4e74e3e"
    )
    assert packet["device"] == "AMD Radeon RX 9070 XT"
    assert packet["architecture"] == "gfx1201"
    assert packet["radiance_wperm"] == 1
    assert packet["selected"] == "tessera_folded"
    assert packet["phase_attribution_admissible"] is False
    assert packet["isa_guard"] == {
        "baseline_non_temporal_load_sites": 0,
        "variant_non_temporal_load_sites": 1,
    }
    assert packet["benchmark_sha256"] == hashlib.sha256(
        (ROOT / "benchmarks/rocm/ablate_gfx1201_folded_b_cache.py").read_bytes(),
    ).hexdigest()
    assert packet["generator_sha256"] == hashlib.sha256(
        (ROOT / "python/tessera/compiler/rocm_mxfp4_folded.py").read_bytes(),
    ).hexdigest()
    assert packet["variant_source_sha256"] == hashlib.sha256(
        variant_source().encode(),
    ).hexdigest()
    for case in ("prefill_256x5120x8704", "prefill_1024x17408x5120"):
        rows = {row["engine"]: row for row in packet["rows"] if row["case"] == case}
        assert set(rows) == {
            "tessera", "tessera_folded", "tessera_b_nontemporal", "radiance",
        }
        assert len({row["output_sha256"] for row in rows.values()}) == 1
        assert rows["tessera_b_nontemporal"]["median_ms"] > (
            rows["tessera_folded"]["median_ms"]
        )
        baseline = rows["tessera_folded"]["metadata"]
        assert baseline["frontend_receipt"]["hsaco_sha256"] == baseline["image_sha256"]
        assert baseline["frontend_receipt"]["artifact_image_digest"] != (
            baseline["image_sha256"]
        )
