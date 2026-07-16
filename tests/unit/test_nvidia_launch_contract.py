"""Host-free execution-matrix contracts for the NVIDIA MMA runtime lane."""
from __future__ import annotations

from tessera.compiler import execution_matrix as EM


def test_execution_matrix_has_nvidia_mma_row():
    row = EM.lookup("nvidia_sm120", "nvidia_mma")
    assert row is not None and row.executable
    assert row.executor_id == "nvidia_mma"
    assert row.execution_kind == "native_gpu"
    assert row.execution_mode == "cuda_runtime"
    assert "nvidia_sm120" not in EM.unimplemented_targets()
    assert "nvidia_mma" in EM.KNOWN_EXECUTORS
    for target in ("nvidia_sm80", "nvidia_sm90", "nvidia_sm100"):
        assert target in EM.unimplemented_targets()


def test_nvidia_mma_executor_registered_in_runtime_table():
    from tessera import runtime as rt

    assert "nvidia_mma" in rt._executor_table()
