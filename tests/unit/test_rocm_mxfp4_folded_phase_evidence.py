"""Drift and refusal gates for the gfx1201 folded phase diagnostic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKETS = ROOT / "benchmarks/baselines/gfx1201_folded_phase_diagnostic_20260922"
REVISION = "4368c825b866c3eb7b40761c5c17bbcb817e2665"


def test_phase_packets_bind_source_and_refuse_changed_isa() -> None:
    generator = (
        "14eecec2445bc2ea4a00a4958778ab927da4354dad2e3c8a48dac59952583087"
    )
    recorder = hashlib.sha256(
        (ROOT / "benchmarks/rocm/measure_gfx1201_folded_phases.py").read_bytes()
    ).hexdigest()
    for name, slots in (("small.json", 80), ("large.json", 1088)):
        packet = json.loads((PACKETS / name).read_text())
        assert packet["schema"] == "tessera.rocm.gfx1201_folded_phase_diagnostic.v1"
        assert packet["source_revision"] == REVISION
        assert packet["device"] == "AMD Radeon RX 9070 XT"
        assert packet["generator_sha256"] == generator
        assert packet["recorder_sha256"] == recorder
        assert packet["instrumentation_level"] == 2
        assert packet["promotion_eligible"] is False
        assert packet["ikf_p0_complete"] is False
        assert packet["phase_attribution_admissible"] is False
        assert packet["phase_attribution_reason"] == "instrumentation_changes_isa_structure"
        assert packet["production_isa"]["wmma_fp8_fp8"] == 32
        assert packet["production_isa"]["s_barrier"] == 4
        assert packet["trace_isa_resources"]["isa"]["wmma_fp8_fp8"] == 64
        assert packet["trace_isa_resources"]["isa"]["s_barrier"] == 24
        assert packet["phases"]["cta_slots"] == slots
        assert packet["phases"]["clock_scope"] == (
            "same_cta_delta_only_cross_cu_unvalidated"
        )
