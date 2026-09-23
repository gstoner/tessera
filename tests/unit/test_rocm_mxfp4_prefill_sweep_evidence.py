"""Bind the six-shape no-promotion packet to timed payloads and source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "benchmarks/baselines/gfx1201_mxfp4_prefill_sweep_20260923/evidence.json"


def test_sweep_packet_binds_sources_outputs_device_and_refusal() -> None:
    packet = json.loads(PACKET.read_text())
    assert packet["schema"] == "tessera.rocm.gfx1201_mxfp4_prefill_sweep.v1"
    assert packet["sync_key"] == "GFX1201-MXFP4-PREFILL-SWEEP-2026-09-23"
    assert packet["architecture"] == "gfx1201"
    assert packet["device"] == "AMD Radeon RX 9070 XT"
    assert packet["radiance"]["weight_layout"] == "fragment_order"
    assert packet["selection_state"] == "manual_evidence_only"
    assert packet["model_budget"]["selection_state"] == "refused_no_model_budget"
    assert packet["model_budget"]["state"] == "missing_model_inventory"
    assert packet["memory_before_bytes"]["capacity_bytes"] > 0
    assert packet["memory_after_bytes"]["free_bytes"] > 0
    for key, path in (
        ("sweep_script_sha256", "benchmarks/rocm/benchmark_gfx1201_mxfp4_prefill_sweep.py"),
        ("benchmark_sha256", "benchmarks/rocm/benchmark_gfx1201_mxfp4_safe_epilogue.py"),
        ("generator_sha256", "python/tessera/compiler/rocm_mxfp4_folded.py"),
        ("tn4_generator_sha256", "python/tessera/compiler/rocm_mxfp4_tn4_experiment.py"),
        ("packed_generator_sha256", "python/tessera/compiler/rocm_mxfp4_packed_folded.py"),
    ):
        assert packet[key] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    assert len(packet["sweep_cases"]) == 6
    for case in packet["sweep_cases"]:
        rows = {row["engine"]: row for row in packet["rows"] if row["case"] == case}
        assert set(rows) == {
            "tessera", "tessera_folded", "tessera_folded_safe_epilogue",
            "tessera_folded_tn4", "tessera_packed_batched_b_permute", "radiance",
        }
        assert len({row["output_sha256"] for row in rows.values()}) == 1
        assert all(len(row["samples_ms"]) == 11 for row in rows.values())
        for name in (
            "tessera_folded_safe_epilogue", "tessera_folded_tn4",
            "tessera_packed_batched_b_permute",
        ):
            meta = rows[name]["metadata"]
            assert meta["selected_isa"]["payload_sha256"] == meta["image_sha256"]
            assert meta["resources"]["spills"] is False
