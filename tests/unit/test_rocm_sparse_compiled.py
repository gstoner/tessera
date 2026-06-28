"""Compiler-generated sparse linear algebra on gfx1151 (Sparse PR) — spmm_csr /
spmm_coo / sddmm / bsmm. The Tessera compiler GENERATES the sparse kernels
(generate-rocm-spmm-kernel / generate-rocm-sddmm-kernel → ROCDL → hsaco), then
HIP launches them. Reachable via `compiler_path="rocm_sparse_compiled"`.
Validated vs numpy on gfx1151. Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, op_name, n_operands):
    names = [f"a{i}" for i in range(n_operands)]
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_sparse_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": names,
                 "kwargs": {}}],
    })


def _csr(dense):
    indptr = [0]
    indices: list[int] = []
    vals: list[float] = []
    for i in range(dense.shape[0]):
        for j in range(dense.shape[1]):
            if dense[i, j] != 0:
                indices.append(j)
                vals.append(float(dense[i, j]))
        indptr.append(len(indices))
    return (np.array(indptr, np.int32), np.array(indices, np.int32),
            np.array(vals, np.float32), tuple(dense.shape))


def _coo(dense):
    rows, cols = np.nonzero(dense)
    return (np.stack([rows, cols], 1).astype(np.int32),
            dense[rows, cols].astype(np.float32), tuple(dense.shape))


def _sparse_dense(rng, m, k, density=0.5):
    d = rng.standard_normal((m, k)).astype(np.float32)
    d[rng.random((m, k)) > density] = 0.0
    return d


@pytest.mark.parametrize("shape", [(6, 5, 4), (8, 3, 7)])
def test_spmm_csr(shape):
    rt = _rocm_or_skip()
    m, k, n = shape
    rng = np.random.default_rng(1 + m + n)
    a = _sparse_dense(rng, m, k)
    b = rng.standard_normal((k, n)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.spmm_csr", 2), (_csr(a), b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_sparse_compiled"
    np.testing.assert_allclose(np.asarray(res["output"]), a @ b, atol=1e-3)


def test_spmm_coo():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(7)
    a = _sparse_dense(rng, 6, 5)
    b = rng.standard_normal((5, 4)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.spmm_coo", 2), (_coo(a), b))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]), a @ b, atol=1e-3)


def test_sddmm():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(3)
    a = rng.standard_normal((6, 5)).astype(np.float32)
    b = rng.standard_normal((5, 4)).astype(np.float32)
    mask = (rng.random((6, 4)) > 0.4).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.sddmm", 3), (a, b, mask))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]), (a @ b) * mask,
                               atol=1e-3)


def test_bsmm():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(5)
    x = rng.standard_normal((16, 16)).astype(np.float32)
    w = rng.standard_normal((16, 16)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.bsmm", 2), (x, w))
    assert res["ok"] is True, res.get("reason")
    # bsmm runs the bf16 WMMA matmul on gfx1151 — bf16 tolerance.
    np.testing.assert_allclose(np.asarray(res["output"]), x @ w,
                               atol=3e-1, rtol=3e-1)


@pytest.mark.parametrize("op", ["spmm", "sddmm"])
def test_sparse_codegen_lowers(op):
    import subprocess
    from pathlib import Path
    opt = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"
    if not opt.is_file():
        pytest.skip("build tessera-opt")
    d = f'module {{\n  "tessera_rocm.{op}"() {{name = "k"}} : () -> ()\n}}\n'
    low = subprocess.run(
        [str(opt), "-",
         f"--pass-pipeline=builtin.module(generate-rocm-{op}-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=d, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout
