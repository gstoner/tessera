"""ROCm WMMA GEMM through ``runtime.launch()`` — the auto-registered executor.

Stage E of the Strix Halo bring-up: the shipped ``tessera_rocm_wmma_gemm_*``
symbol is now wired into ``runtime.launch()`` via the single-source execution
matrix (``compiler_path == "rocm_wmma"`` → ``rocm_wmma`` executor). Unlike the
Stage C/D harness tests (which registered their own launcher) and the
shipped-symbol fixture (which dlopens the symbol directly), this drives the
*production* runtime dispatch path: ``launch()`` reads the artifact metadata,
consults the matrix, resolves the executor, runs the kernel on the AMD GPU, and
reports ``execution_kind="native_gpu"``.

Two host-independent matrix-shape tests run everywhere (Mac CI included); the
execute tests skip-clean when there's no AMD GPU / GEMM lib.
"""

from __future__ import annotations

import pytest

from tessera.compiler import execution_matrix as EM

np = pytest.importorskip("numpy")


# ── Host-independent: the matrix is wired (runs on every host) ───────────────

def test_execution_matrix_has_rocm_wmma_row():
    """`(rocm, rocm_wmma)` is an executable native_gpu row, and `rocm` is no
    longer on the unimplemented list."""
    row = EM.lookup("rocm", "rocm_wmma")
    assert row is not None and row.executable
    assert row.executor_id == "rocm_wmma"
    assert row.execution_kind == "native_gpu"
    assert "rocm" not in EM.unimplemented_targets()
    assert "rocm_wmma" in EM.KNOWN_EXECUTORS


def test_rocm_wmma_executor_registered_in_runtime_table():
    """The matrix's executor_id must resolve to a real function in runtime."""
    from tessera import runtime as rt
    assert "rocm_wmma" in rt._executor_table()


# ── Execution: needs a live AMD GPU + the shipped GEMM lib ───────────────────

def _rocm_runtime_or_skip():
    from tessera import runtime as rt
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no AMD GPU / libtessera_rocm_gemm.so (build tessera_rocm_gemm)")
    return rt


def _matmul_artifact(rt):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_wmma", "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": ["a", "b"], "output_name": "c",
        "ops": [{"op_name": "tessera.matmul", "result": "c",
                 "operands": ["a", "b"], "kwargs": {}}],
    })


@pytest.mark.parametrize("shape", [(16, 16, 16), (64, 48, 32), (128, 96, 64)])
def test_launch_rocm_wmma_matmul_f16_matches_numpy(shape):
    rt = _rocm_runtime_or_skip()
    m, n, k = shape
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((m, k)) * 0.5).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.5).astype(np.float16)
    res = rt.launch(_matmul_artifact(rt), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["runtime_status"] == "success"
    assert res["compiler_path"] == "rocm_wmma"
    assert res["execution_kind"] == "native_gpu"
    ref = a.astype(np.float32) @ b.astype(np.float32)
    maxerr = float(np.max(np.abs(res["output"] - ref)))
    assert maxerr < 1e-2, f"rocm_wmma launch{shape} maxerr={maxerr}"


def test_launch_rocm_wmma_matmul_bf16_matches_numpy():
    rt = _rocm_runtime_or_skip()
    bf16 = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((64, 32)) * 0.5).astype(bf16)
    b = (rng.standard_normal((32, 48)) * 0.5).astype(bf16)
    res = rt.launch(_matmul_artifact(rt), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["execution_kind"] == "native_gpu"
    ref = a.astype(np.float32) @ b.astype(np.float32)
    maxerr = float(np.max(np.abs(res["output"] - ref)))
    assert maxerr < 5e-2 * 32, f"rocm_wmma bf16 launch maxerr={maxerr}"


def test_launch_rocm_wmma_rejects_non_half_dtype():
    """f32 operands aren't a WMMA storage dtype — the executor raises, and
    launch() maps the raise to a structured invalid_artifact result (never a
    fabricated output)."""
    rt = _rocm_runtime_or_skip()
    a = np.zeros((16, 16), np.float32)
    b = np.zeros((16, 16), np.float32)
    res = rt.launch(_matmul_artifact(rt), (a, b))
    assert res["ok"] is False
    assert res["runtime_status"] == "invalid_artifact"
    assert "f16/bf16" in res["reason"]
