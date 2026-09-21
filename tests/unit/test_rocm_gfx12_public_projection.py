"""Drift gates for bounded gfx1201 public proof projection."""
from __future__ import annotations

from pathlib import Path

import pytest

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


def _artifact(compiler_path: str):
    from tessera import runtime

    return runtime.RuntimeArtifact(metadata={
        "target": "rocm_gfx1201",
        "compiler_path": compiler_path,
        "executable": True,
        "execution_kind": "native_gpu",
    })


def test_gfx1201_exact_executors_refuse_a_different_live_arch(monkeypatch) -> None:
    from tessera import runtime

    monkeypatch.setattr(runtime, "_rocm_live_arch", lambda: "gfx1151")
    monkeypatch.setattr(runtime, "_rocm_chip", lambda: "gfx1201")
    executors = runtime._executor_table()
    for proof in GFX1201_PUBLIC_PROOFS:
        assert proof.executor_id in executors
        called = False

        def delegate(_artifact, _args):
            nonlocal called
            called = True
            return object()

        with pytest.raises(
            runtime._RocmCompiledUnavailable, match="selected HIP device gfx1201"
        ):
            runtime._execute_rocm_gfx1201_exact(
                _artifact(proof.compiler_path), (), delegate
            )
        assert not called


def test_gfx1201_execution_rows_cannot_report_success_on_gfx1151(
    monkeypatch,
) -> None:
    from tessera import runtime

    monkeypatch.setattr(runtime, "_rocm_live_arch", lambda: "gfx1151")
    monkeypatch.setattr(runtime, "_rocm_chip", lambda: "gfx1201")
    for proof in GFX1201_PUBLIC_PROOFS:
        result = runtime.launch(_artifact(proof.compiler_path), ())
        assert result["ok"] is False
        assert result["runtime_status"] == "invalid_artifact"
        assert "selected HIP device gfx1201" in result["reason"]


def test_gfx1201_exact_executor_requires_matching_compiler_target(monkeypatch) -> None:
    from tessera import runtime

    monkeypatch.setattr(runtime, "_rocm_live_arch", lambda: "gfx1201")
    monkeypatch.setattr(runtime, "_rocm_chip", lambda: "gfx1151")
    with pytest.raises(
        runtime._RocmCompiledUnavailable, match="TESSERA_ROCM_CHIP=gfx1201"
    ):
        runtime._execute_rocm_gfx1201_exact(
            _artifact("rocm_compiled"), (), lambda _artifact, _args: object()
        )


def test_gfx1201_exact_executor_delegates_only_when_both_arches_match(
    monkeypatch,
) -> None:
    from tessera import runtime

    monkeypatch.setattr(runtime, "_rocm_live_arch", lambda: "gfx1201")
    monkeypatch.setattr(runtime, "_rocm_chip", lambda: "gfx1201")
    sentinel = object()
    assert runtime._execute_rocm_gfx1201_exact(
        _artifact("rocm_compiled"), (), lambda _artifact, _args: sentinel
    ) is sentinel
