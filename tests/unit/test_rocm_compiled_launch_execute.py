"""Stage L4 — the COMPILER-GENERATED ROCm GEMM through ``runtime.launch()``.

Stages L1–L3 made the compiler generate a correct, problem-generic, register-
blocked WMMA GEMM and serialize it to hsaco entirely in-process (no mlir-opt).
L4 wires that compiled path into the production runtime dispatch as an *opt-in*
lane: an artifact stamped ``compiler_path = "rocm_compiled"`` routes through the
execution matrix to ``_execute_rocm_compiled_gemm``, which generates + serializes
the kernel via tessera-opt and launches the hsaco on the AMD GPU.

This is the same ``runtime.launch()`` entry point the hand-written ``rocm_wmma``
lane uses — the only difference is *which kernel runs*: the one the Tessera
compiler emitted, not the hand-written HIPRTC kernel. The compiled lane is
opt-in; the hand-written lane stays the default + reference oracle (ROCM_AUDIT
L4). Here we prove the compiled lane executes through ``launch()`` and matches
both numpy AND the hand-written oracle (bit-identical).

Skip-clean: tessera-opt not built (or no in-process serialization), or no AMD GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_execution_matrix_has_rocm_compiled_row():
    from tessera.compiler import execution_matrix as em
    row = em.lookup("rocm", "rocm_compiled")
    assert row is not None
    assert row.executor_id == "rocm_compiled"
    assert row.execution_kind == "native_gpu"


def test_rocm_compiled_executor_registered():
    from tessera import runtime as rt
    assert "rocm_compiled" in rt._executor_table()


def _compiled_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no AMD GPU / libtessera_rocm_gemm.so")
    return rt


def _artifact(rt, compiler_path):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": compiler_path, "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": ["a", "b"], "output_name": "c",
        "ops": [{"op_name": "tessera.matmul", "result": "c",
                 "operands": ["a", "b"], "kwargs": {}}],
    })


@pytest.mark.parametrize("shape", [(16, 16, 16), (64, 48, 32), (256, 256, 256)])
def test_launch_rocm_compiled_matmul_f16_matches_numpy_and_oracle(shape):
    rt = _compiled_or_skip()
    m, n, k = shape
    rng = np.random.default_rng(5)
    a = (rng.standard_normal((m, k)) * 0.4).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.4).astype(np.float16)

    # The compiled lane: tessera-opt generates + serializes the kernel in-process,
    # HIP launches the hsaco.
    res = rt.launch(_artifact(rt, "rocm_compiled"), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["runtime_status"] == "success"
    assert res["compiler_path"] == "rocm_compiled"
    assert res["execution_kind"] == "native_gpu"
    out = res["output"]

    ref = a.astype(np.float32) @ b.astype(np.float32)
    assert float(np.max(np.abs(out - ref))) < 5e-2, f"vs numpy {shape}"

    # vs the hand-written oracle through the SAME launch() entry point — the two
    # lanes compute the identical GEMM, so they agree bit-for-bit.
    oracle = rt.launch(_artifact(rt, "rocm_wmma"), (a, b))
    assert oracle["ok"] is True, oracle.get("reason")
    assert float(np.max(np.abs(out - oracle["output"]))) == 0.0, \
        f"compiled lane != hand-written oracle at {shape}"


def test_launch_rocm_compiled_rejects_bf16():
    """The generated kernel is f16-storage today; bf16 is an explicit, structured
    invalid_artifact (never a silent miscompute) — use rocm_wmma for bf16."""
    rt = _compiled_or_skip()
    bf16 = pytest.importorskip("ml_dtypes").bfloat16
    a = np.zeros((32, 32), bf16)
    b = np.zeros((32, 32), bf16)
    res = rt.launch(_artifact(rt, "rocm_compiled"), (a, b))
    assert res["ok"] is False
    assert res["runtime_status"] == "invalid_artifact"
    assert "f16" in res["reason"]
