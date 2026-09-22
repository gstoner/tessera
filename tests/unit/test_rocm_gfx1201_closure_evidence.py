"""Drift checks for the exact-device gfx1201 scheduled-suite closure."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.rocm.record_gfx1201_scheduled_closure import (
    DEVICE_CASE_FAMILIES,
    EXPECTED_DEVICE,
    HOST_CONTRACT_FAMILIES,
    _require_toolkit_runtime,
    _rocm_release,
    _sha256,
    _validate_build_versions,
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
    assert packet["device"] == EXPECTED_DEVICE
    assert packet["live_architecture"] == "gfx1201"
    assert packet["compiler_target"] == "gfx1201"
    assert packet["compiler"]["stale_generator_sources"] == 0
    assert packet["compiler"]["source_dirty"] is False
    assert packet["compiler"]["source_revision"] == packet["source_revision"]
    toolkit = Path(packet["toolchain"]["rocm_path"])
    assert Path(packet["toolchain"]["hipcc_path"]).is_relative_to(toolkit)
    assert Path(packet["toolchain"]["hip_runtime_library"]).is_relative_to(toolkit)
    assert len(packet["toolchain"]["hip_runtime_sha256"]) == 64
    assert packet["result"]["tests"] == proof.required_passed_cases
    assert packet["result"]["failures"] == 0
    assert packet["result"]["errors"] == 0
    assert packet["result"]["skipped"] == proof.required_skipped_cases
    assert packet["device_dependent_cases"]["count"] == proof.device_dependent_cases
    assert packet["host_contract_cases"]["count"] == proof.host_contract_cases
    assert packet["fixture_sha256"] == _sha256(ROOT / proof.numerical_fixture)
    assert packet["recorder_sha256"] == _sha256(
        ROOT / "benchmarks/rocm/record_gfx1201_scheduled_closure.py"
    )


def test_gfx1201_proof_build_versions_are_parsed_and_validated() -> None:
    versions = _validate_build_versions(
        "llvm23.1.1+rocm10.0+gfx1201",
        compiler_output="LLVM version 23.1.1\nOptimized build",
        hipcc_output="HIP version: 7.15.26333-0000000",
        rocm_release="10.0.0",
    )
    assert versions["llvm"] == "23.1.1"
    assert versions["rocm"] == "10.0.0"
    assert versions["hip"] == "7.15.26333"


def test_gfx1201_proof_build_rejects_mislabeled_toolchain() -> None:
    with pytest.raises(RuntimeError, match="requires LLVM 23.1.1; observed 23.0.0"):
        _validate_build_versions(
            "llvm23.1.1+rocm10.0+gfx1201",
            compiler_output="LLVM version 23.0.0",
            hipcc_output="HIP version: 7.15.26333-0000000",
            rocm_release="10.0.0",
        )
    with pytest.raises(RuntimeError, match="requires ROCm 10.0; observed 9.9.0"):
        _validate_build_versions(
            "llvm23.1.1+rocm10.0+gfx1201",
            compiler_output="LLVM version 23.1.1",
            hipcc_output="HIP version: 7.15.26333-0000000",
            rocm_release="9.9.0",
        )
    with pytest.raises(RuntimeError, match="requires HIP 7.15; observed 7.14.0"):
        _validate_build_versions(
            "llvm23.1.1+rocm10.0+gfx1201",
            compiler_output="LLVM version 23.1.1",
            hipcc_output="HIP version: 7.14.0",
            rocm_release="10.0.0",
        )


def test_gfx1201_release_is_read_only_from_selected_toolkit(tmp_path: Path) -> None:
    selected = tmp_path / "rocm" / "core"
    (selected / ".info").mkdir(parents=True)
    (selected / ".info" / "version").write_text("10.0.0\n")
    (selected.parent / ".info").mkdir()
    (selected.parent / ".info" / "version").write_text("9.9.0\n")
    assert _rocm_release(selected) == "10.0.0"


def test_gfx1201_loaded_hip_runtime_must_belong_to_toolkit(tmp_path: Path) -> None:
    toolkit = tmp_path / "rocm" / "core"
    runtime = toolkit / "lib" / "libamdhip64.so"
    runtime.parent.mkdir(parents=True)
    runtime.touch()
    assert _require_toolkit_runtime(toolkit, runtime) == runtime.resolve()
    other = tmp_path / "stale" / "libamdhip64.so"
    other.parent.mkdir()
    other.touch()
    with pytest.raises(RuntimeError, match="outside the selected ROCm toolkit"):
        _require_toolkit_runtime(toolkit, other)
