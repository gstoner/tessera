"""Drift gates for bounded gfx1201 public proof projection."""
from __future__ import annotations

from pathlib import Path

from tessera.compiler import execution_matrix
from tessera.compiler.capabilities import runtime_status, supports_op
from tessera.compiler.rocm_exact_device_proofs import GFX1201_PUBLIC_PROOFS


ROOT = Path(__file__).resolve().parents[2]


def test_gfx1201_public_proofs_join_capability_execution_and_fixture() -> None:
    for proof in GFX1201_PUBLIC_PROOFS:
        capability = supports_op(proof.target, proof.op_name)
        assert capability.supported and capability.runtime_status == "ready"
        row = execution_matrix.lookup(proof.target, proof.compiler_path)
        assert row is not None and row.executable
        assert row.executor_id == proof.executor_id
        assert row.device_proof == "device_verified_jit"
        assert row.evidence_target == proof.target
        assert row.numerical_fixture == proof.numerical_fixture
        assert (ROOT / proof.numerical_fixture).is_file()


def test_gfx1201_public_proofs_are_a_subset_of_scheduled_launcher_admission() -> None:
    from tessera import runtime

    admitted = runtime._gfx1201_proved_scheduled_abis()
    projected = {abi for proof in GFX1201_PUBLIC_PROOFS
                 for abi in proof.scheduled_abis}
    assert projected <= admitted


def test_gfx1200_remains_fail_closed_without_owning_device_proof() -> None:
    assert runtime_status("rocm_gfx1200", "tessera.matmul") == "artifact_only"
    assert execution_matrix.lookup("rocm_gfx1200", "rocm_compiled") is None
    assert "rocm_gfx1200" in execution_matrix.unimplemented_targets()

