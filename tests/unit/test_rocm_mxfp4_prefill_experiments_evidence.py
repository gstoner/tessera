"""Bind the manual prefill ablations to current source and timed HSACOs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "benchmarks/baselines/gfx1201_mxfp4_prefill_experiments_20260923/evidence.json"


def test_prefill_packet_binds_sources_oracle_isa_and_budget_refusal() -> None:
    packet = json.loads(PACKET.read_text())
    assert packet["schema"] == "tessera.rocm.gfx1201_mxfp4_safe_epilogue_benchmark.v1"
    assert packet["sync_key"] == "GFX1201-MXFP4-SAFE-EPILOGUE-2026-09-23"
    assert packet["source_revision"] == "fb392b53cb00e58aa1d1ebc82a64eef2f6fdddae"
    assert packet["device"] == "AMD Radeon RX 9070 XT"
    assert packet["architecture"] == "gfx1201"
    assert packet["radiance"]["weight_layout"] == "fragment_order"
    for field, path in (
        ("tn4_generator_sha256", "python/tessera/compiler/rocm_mxfp4_tn4_experiment.py"),
        ("packed_benchmark_sha256", "benchmarks/rocm/benchmark_gfx1201_mxfp4_packed_folded.py"),
        ("benchmark_sha256", "benchmarks/rocm/benchmark_gfx1201_mxfp4_safe_epilogue.py"),
    ):
        assert packet[field] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    # Recorded before the 2026-09-24 producer relabel (pipeline_name only).
    # gfx1201_mxfp4_producer_relabel_20260924 proves the relabelled generator
    # builds the same timed kernels; do not relabel old GPU timings.
    assert packet["generator_sha256"] == "fb923edfbe4ba1c3eb4cd1da237a3ac72802e8ffa77ad3c9b7935bc071e87616"
    assert packet["packed_generator_sha256"] == "b45ad1f0b58a90199e270cffd2735d3f0fd30b216824758f40316cd5917bb090"
    assert packet["model_layers"] == [{"n": 17408, "k": 5120, "count": 32}]
    memory = packet["model_weight_residency"]
    assert memory["packed_weight_and_scale_bytes"] == 1515749376
    assert memory["expanded_weight_and_reference_bytes"] == 2852683776
    assert memory["incremental_bytes"] == 2852683776
    assert memory["selection_state"] == "refused_budget"
    for case in ("prefill_256x5120x8704", "prefill_1024x17408x5120"):
        rows = {row["engine"]: row for row in packet["rows"] if row["case"] == case}
        assert set(rows) == {
            "tessera", "tessera_folded", "tessera_folded_safe_epilogue",
            "tessera_folded_tn4", "tessera_packed_batched_b_permute", "radiance",
        }
        assert len({row["output_sha256"] for row in rows.values()}) == 1
        assert all(len(row["samples_ms"]) == 11 for row in rows.values())
        for name in (
            "tessera_folded", "tessera_folded_safe_epilogue",
            "tessera_folded_tn4", "tessera_packed_batched_b_permute",
        ):
            meta = rows[name]["metadata"]
            assert meta["selected_isa"]["payload_sha256"] == meta["image_sha256"]
            assert meta["resources"]["spills"] is False
        safe = rows["tessera_folded_safe_epilogue"]
        tn4 = rows["tessera_folded_tn4"]
        assert safe["metadata"]["selected_isa"]["instruction_count"] == 2724
        assert safe["metadata"]["resources"]["vgpr_count"] == 140
        assert safe["metadata"]["route"]["epilogue_policy"] == "host_scale_certified_manual"
        assert tn4["metadata"]["selected_isa"]["instruction_count"] == 8195
        assert tn4["metadata"]["resources"]["vgpr_count"] == 181
        assert tn4["metadata"]["route"]["execution_state"] == "manual_executable_candidate"
        assert min(row["median_ms"] for key, row in rows.items() if key != "radiance") > (
            rows["radiance"]["median_ms"]
        )
