"""Drift checks for the exact-device gfx1201 scheduled-suite closure."""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.rocm.record_gfx1201_scheduled_closure import (
    DEVICE_CASE_FAMILIES,
    HOST_CONTRACT_FAMILIES,
)
from tessera.compiler.rocm_exact_device_proofs import GFX1201_SCHEDULED_SUITE_PROOF


ROOT = Path(__file__).resolve().parents[2]


def test_gfx1201_closure_registry_counts_the_complete_suite() -> None:
    proof = GFX1201_SCHEDULED_SUITE_PROOF
    assert sum(DEVICE_CASE_FAMILIES.values()) == proof.device_dependent_cases == 90
    assert sum(HOST_CONTRACT_FAMILIES.values()) == proof.host_contract_cases == 5
    assert proof.required_passed_cases == 95
    assert proof.required_skipped_cases == 0
    assert (ROOT / proof.numerical_fixture).is_file()


def test_gfx1201_committed_closure_packet_matches_registry() -> None:
    proof = GFX1201_SCHEDULED_SUITE_PROOF
    packet = json.loads((ROOT / proof.evidence_packet).read_text())
    assert packet["schema"] == "tessera.rocm.gfx1201_scheduled_closure.v1"
    assert packet["target"] == proof.target
    assert packet["fixture"] == proof.numerical_fixture
    assert packet["proof_build"] == proof.proof_build
    assert packet["live_architecture"] == "gfx1201"
    assert packet["compiler_target"] == "gfx1201"
    assert packet["compiler"]["stale_generator_sources"] == 0
    assert packet["compiler"]["source_dirty"] is False
    assert packet["compiler"]["source_revision"] == packet["source_revision"]
    assert packet["result"]["tests"] == proof.required_passed_cases
    assert packet["result"]["failures"] == 0
    assert packet["result"]["errors"] == 0
    assert packet["result"]["skipped"] == proof.required_skipped_cases
    assert packet["device_dependent_cases"]["count"] == proof.device_dependent_cases
    assert packet["host_contract_cases"]["count"] == proof.host_contract_cases
