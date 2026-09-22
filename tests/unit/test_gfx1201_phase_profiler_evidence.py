"""Content-bound refusal gate for the exact-device profiler preflight."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKET = (
    ROOT / "benchmarks/baselines/gfx1201_phase_profiler_preflight_20260922/evidence.json"
)


def test_gfx1201_profiler_packet_refuses_without_kfd() -> None:
    packet = json.loads(PACKET.read_text())
    assert packet["schema"] == "tessera.rocm.gfx1201_phase_profiler_preflight.v1"
    assert packet["recorder_sha256"] == hashlib.sha256(
        (ROOT / "benchmarks/rocm/probe_gfx1201_phase_profiler.py").read_bytes()
    ).hexdigest()
    assert packet["host"] == "tajasarus"
    assert packet["architecture"] == "gfx1201"
    assert packet["device"] == "AMD Radeon RX 9070 XT"
    assert packet["refusal_reasons"] == ["kfd_device_missing"]
    assert packet["isa_preserving_phase_probe_available"] is False
    assert packet["cross_cu_clock_validated"] is False
    assert packet["clock_read_cost_validated"] is False
    assert packet["phase_attribution_admissible"] is False
    assert packet["promotion_eligible"] is False
