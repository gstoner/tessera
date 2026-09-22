"""Drift and refusal gates for the gfx1201 folded phase diagnostic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKETS = ROOT / "benchmarks/baselines/gfx1201_folded_phase_diagnostic_20260922"
REVISION = "817cde29fda75c0323dba7bdbff3f673da32c2f4"


def test_phase_packets_bind_source_and_refuse_changed_isa() -> None:
    generator = hashlib.sha256(
        (ROOT / "python/tessera/compiler/rocm_mxfp4_folded.py").read_bytes()
    ).hexdigest()
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
        assert packet["trace_isa_resources"]["isa"]["wmma_fp8_fp8"] == 64
        assert packet["phases"]["cta_slots"] == slots
        assert packet["phases"]["clock_scope"] == (
            "same_cta_delta_only_cross_cu_unvalidated"
        )
