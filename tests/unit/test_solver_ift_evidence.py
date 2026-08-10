from __future__ import annotations

import json
from pathlib import Path

import pytest

from tessera.compiler.implicit_solver import build_solver_ift_contract


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "target,filename",
    [
        ("x86", "x86_zen5_solver_ift_evidence.json"),
        ("rocm_gfx1151", "rocm_gfx1151_solver_ift_evidence.json"),
    ],
)
def test_solver_ift_packet_has_current_lineage_and_complete_backward(target: str, filename: str) -> None:
    packet = json.loads((ROOT / "benchmarks" / "baselines" / filename).read_text())
    assert packet["schema"] == "tessera.solver_ift.evidence.v1"
    assert packet["artifact_hash"] == build_solver_ift_contract(target=target, shape=packet["shape"])["artifact_hash"]
    assert packet["timing"]["complete_backward"] is True
    assert len(packet["timing"]["samples_ns"]) >= 20
    assert packet["retained_residual_bytes"] == 4 * 3 * 257
    assert packet["numerical"]["passed"] is True
    assert max(packet["numerical"]["max_abs_error_by_phase"].values()) <= 1e-6
    if target == "rocm_gfx1151":
        assert packet["promotion"]["performance_eligible"] is False
