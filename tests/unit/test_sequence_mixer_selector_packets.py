from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKETS = (
    (
        ROOT / "benchmarks/baselines/x86_avx512_deltanet_backward_selectors.json",
        "x86_avx512",
        "parallel_c16_w8",
    ),
    (
        ROOT / "benchmarks/baselines/rocm_gfx1151_deltanet_backward_selectors.json",
        "rocm_gfx1151",
        "parallel_c16",
    ),
)


def test_modified_deltanet_backward_selector_packets_are_reproducible() -> None:
    for path, target, winner in PACKETS:
        packet = json.loads(path.read_text())
        assert packet["schema"] == "tessera.sequence_mixer.selector_packet.v1"
        assert packet["operation"] == "modified_delta_attention_backward"
        assert packet["target"] == target
        assert packet["timing_scope"] == "resident_program_wall_ms"
        assert packet["iterations"] == 21
        assert packet["winner_consensus"] is True
        assert packet["winner"] == winner
        assert len(packet["cohorts"]) == 2
        assert all(cohort["winner"] == winner for cohort in packet["cohorts"])
        assert packet["max_abs_error"] <= 3.0e-3


def test_selector_packets_preserve_exact_host_evidence_boundaries() -> None:
    x86 = json.loads(PACKETS[0][0].read_text())
    rocm = json.loads(PACKETS[1][0].read_text())

    # The AVX-512 winner repeated, but the two cohort medians differed by 12%.
    assert x86["host"]["avx512f"] is True
    assert x86["stability_fraction"] > 0.05
    assert x86["selector_eligible"] is False

    # gfx1151 timings are stable, but WSL wall-clock evidence is not a
    # production selector authority; a bare-metal packet remains required.
    assert rocm["device"] == "gfx1151"
    assert rocm["stability_fraction"] <= 0.05
    assert rocm["host"]["wsl"] is True
    assert rocm["selector_eligible"] is False
