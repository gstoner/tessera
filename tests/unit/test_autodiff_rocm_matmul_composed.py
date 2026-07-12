"""Exact-target proof for the ROCm matmul paired-backward composition.

There is intentionally no matmul-backward kernel: the paired ABI launches the
compiler-generated forward GEMM twice and compares both gradients to NumPy.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


@ts.jit(target="rocm", autodiff="reverse", wrt=("a", "b"))
def _matmul(a, b):
    return ts.ops.matmul(a, b)


def test_rocm_composed_matmul_backward_matches_numpy() -> None:
    from tessera import runtime as rt

    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU / generated ROCm GEMM runtime")

    rng = np.random.default_rng(47)
    a = (rng.standard_normal((32, 48)) * 0.2).astype(np.float16)
    b = (rng.standard_normal((48, 24)) * 0.2).astype(np.float16)
    dout = (rng.standard_normal((32, 24)) * 0.2).astype(np.float16)

    da, db = _matmul.native_backward(a, b, out_cotangents=dout)
    np.testing.assert_allclose(da, dout.astype(np.float32) @ b.astype(np.float32).T,
                               atol=5e-2, rtol=5e-3)
    np.testing.assert_allclose(db, a.astype(np.float32).T @ dout.astype(np.float32),
                               atol=5e-2, rtol=5e-3)
    assert _matmul.last_backward_execution == {
        "compiler_path": "rocm_compiled+rocm_compiled",
        "execution_kind": "native_gpu",
        "execution_mode": "hip_runtime",
        "evidence_target": "rocm_gfx1151",
        "implementation": "composition",
        "residual_policy": "save_inputs",
    }
