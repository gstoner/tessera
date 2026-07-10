"""Apple GPU matmul-family lane — einsum / factorized_matmul.

einsum (single-contraction spec) and factorized_matmul (GEMM + rank-r SVD
truncation) resolve on the numpy reference the x86/ROCm GEMM lanes match.
Reachable via `compiler_path="apple_gpu_matmul_family_compiled"`. Validated vs
numpy — parity with test_rocm_matmul_family_compiled. Numpy path always runs.
"""

from __future__ import annotations

import numpy as np

from tessera import runtime as rt


def _launch(op, operands, kwargs=None):
    names = [f"a{i}" for i in range(len(operands))]
    art = rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_matmul_family_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": dict(kwargs or {})}]})
    res = rt.launch(art, tuple(operands))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "apple_gpu_matmul_family_compiled"
    assert res["execution_kind"] == "native_gpu"
    return np.asarray(res["output"])


def test_einsum_single_contraction_matches_reference():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((3, 4)).astype(np.float32)
    y = rng.standard_normal((4, 5)).astype(np.float32)
    np.testing.assert_allclose(_launch("tessera.einsum", [x, y], {"spec": "ij,jk->ik"}),
                               np.einsum("ij,jk->ik", x, y), atol=1e-3, rtol=1e-4)
    # batched contraction
    xb = rng.standard_normal((2, 3, 4)).astype(np.float32)
    yb = rng.standard_normal((2, 4, 5)).astype(np.float32)
    np.testing.assert_allclose(
        _launch("tessera.einsum", [xb, yb], {"equation": "bij,bjk->bik"}),
        np.einsum("bij,bjk->bik", xb, yb), atol=1e-3, rtol=1e-4)


def test_factorized_matmul_matches_reference():
    rng = np.random.default_rng(12)
    a = rng.standard_normal((6, 4)).astype(np.float32)
    b = rng.standard_normal((4, 5)).astype(np.float32)
    for rank in (2, 3):
        got = _launch("tessera.factorized_matmul", [a, b], {"rank": rank})
        out = (a @ b).astype(np.float32)
        u, s, vh = np.linalg.svd(out, full_matrices=False)
        r = max(1, min(rank, s.shape[-1]))
        exp = (u[..., :r] * s[..., :r]) @ vh[..., :r, :]
        np.testing.assert_allclose(got, exp, atol=1e-3, rtol=1e-3)
