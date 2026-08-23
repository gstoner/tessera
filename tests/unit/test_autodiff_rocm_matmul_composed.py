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
    # Assert the fields this proof depends on, not whole-dict equality. The
    # record gained six tracer-authority fields in E2E-REAL-6 (`family`,
    # `frontend_authority`, and the four `*_consumer` keys) and `implementation`
    # became `family_plugin_composition`; a `==` over the whole dict fails on
    # additive metadata that does not change what was proven, which is exactly
    # how this test went stale. Indexing keeps it fail-closed: a dropped key
    # raises KeyError rather than silently passing.
    execution = _matmul.last_backward_execution
    assert {
        key: execution[key]
        for key in (
            "compiler_path",
            "evidence_target",
            "execution_kind",
            "execution_mode",
            "family",
            "implementation",
            "residual_policy",
        )
    } == {
        # Two compiled forward GEMM launches -- the point of the composition.
        "compiler_path": "rocm_compiled+rocm_compiled",
        "evidence_target": "rocm_gfx1151",
        "execution_kind": "native_gpu",
        "execution_mode": "hip_runtime",
        "family": "matmul_backward",
        "implementation": "family_plugin_composition",
        "residual_policy": "save_inputs",
    }
