"""sm_120 execution lane — @jit(target="nvidia_sm120") matmul dispatches through
the shipped libtessera_nvidia_gemm.so via runtime.launch().

The NVIDIA analog of test_rocm_launch_execute.py. Two layers:
  - host-independent matrix tests (the executable nvidia_mma row exists + its
    executor_id resolves in the runtime table) — run everywhere;
  - execute-and-compare tests gated on a usable NVIDIA GPU + the built lib.

Skip-clean: no GPU / no libtessera_nvidia_gemm.so / no NVRTC.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler import execution_matrix as EM


# ---- host-independent matrix wiring ----

def test_execution_matrix_has_nvidia_mma_row():
    """`(nvidia_sm120, nvidia_mma)` is an executable native_gpu row, and
    nvidia_sm120 is no longer on the unimplemented list."""
    row = EM.lookup("nvidia_sm120", "nvidia_mma")
    assert row is not None and row.executable
    assert row.executor_id == "nvidia_mma"
    assert row.execution_kind == "native_gpu"
    assert row.execution_mode == "cuda_runtime"
    assert "nvidia_sm120" not in EM.unimplemented_targets()
    assert "nvidia_mma" in EM.KNOWN_EXECUTORS
    # The other NVIDIA arches stay unimplemented (only sm_120 is proven).
    for t in ("nvidia_sm80", "nvidia_sm90", "nvidia_sm100"):
        assert t in EM.unimplemented_targets()


def test_nvidia_mma_executor_registered_in_runtime_table():
    """The matrix's executor_id must resolve to a real function in runtime."""
    from tessera import runtime as rt
    assert "nvidia_mma" in rt._executor_table()


# ---- execute-and-compare (hardware-gated) ----

def _nvidia_runtime_or_skip():
    from tessera import runtime as rt
    if not rt._nvidia_mma_runtime_available():
        pytest.skip("no NVIDIA GPU / libtessera_nvidia_gemm.so "
                    "(build tessera_nvidia_gemm)")
    return rt


def _matmul_artifact(rt):
    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_mma",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a", "b"], "output_name": "c",
        "ops": [{"op_name": "tessera.matmul", "result": "c",
                 "operands": ["a", "b"], "kwargs": {}}],
    })


@pytest.mark.parametrize("shape", [(16, 16, 16), (64, 48, 32), (128, 96, 64)])
def test_launch_nvidia_mma_matmul_f16_matches_numpy(shape):
    rt = _nvidia_runtime_or_skip()
    m, n, k = shape
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((m, k)) * 0.5).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.5).astype(np.float16)
    res = rt.launch(_matmul_artifact(rt), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["runtime_status"] == "success"
    assert res["compiler_path"] == "nvidia_mma"
    assert res["execution_kind"] == "native_gpu"
    ref = a.astype(np.float32) @ b.astype(np.float32)
    maxerr = float(np.max(np.abs(res["output"] - ref)))
    assert maxerr < 1e-2, f"nvidia_mma launch{shape} maxerr={maxerr}"


def test_launch_nvidia_mma_matmul_bf16_matches_numpy():
    rt = _nvidia_runtime_or_skip()
    bf16 = rt._bfloat16_dtype()
    if bf16 is None:
        pytest.skip("no bfloat16 dtype available")
    m, n, k = 64, 64, 48
    rng = np.random.default_rng(1)
    a = (rng.standard_normal((m, k)) * 0.5).astype(bf16)
    b = (rng.standard_normal((k, n)) * 0.5).astype(bf16)
    res = rt.launch(_matmul_artifact(rt), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "nvidia_mma"
    ref = a.astype(np.float32) @ b.astype(np.float32)
    assert float(np.max(np.abs(res["output"] - ref))) < 2e-1


def test_jit_nvidia_sm120_matmul_dispatches_to_shipped_symbol():
    """On a capable host, @jit(target="nvidia_sm120") matmul is executable
    through the shipped mma.sync lane (was artifact_only). Off-device it stays
    artifact_only — so this asserts the flip only where the lane can run."""
    import tessera
    rt = _nvidia_runtime_or_skip()

    @tessera.jit(target="nvidia_sm120")
    def mm(a, b):
        return tessera.ops.matmul(a, b)

    art = mm.runtime_artifact()
    md = art.metadata or {}
    assert md.get("executable") is True
    assert md.get("compiler_path") == "nvidia_mma"
    assert md.get("execution_kind") == "native_gpu"

    m, n, k = 128, 96, 64
    rng = np.random.default_rng(3)
    a = (rng.standard_normal((m, k)) * 0.4).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.4).astype(np.float16)
    res = rt.launch(art, (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "nvidia_mma"
    ref = a.astype(np.float32) @ b.astype(np.float32)
    assert float(np.max(np.abs(res["output"] - ref))) < 1e-2
