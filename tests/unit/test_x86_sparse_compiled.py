"""Compiler-generated sparse linear algebra on x86 AVX-512 (Sparse PR) —
spmm_csr / spmm_coo / sddmm / bsmm. GENUINELY sparse kernels (iterate the
nonzero structure, not densify-then-GEMM). Reachable via
`compiler_path="x86_sparse_compiled"`. Validated vs numpy. Skip-clean: x86
elementwise lib not built.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op_name, n_operands):
    names = [f"a{i}" for i in range(n_operands)]
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_sparse_compiled",
        "executable": True, "execution_kind": "native_cpu",
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
    coords = np.stack([rows, cols], 1).astype(np.int32)
    return (coords, dense[rows, cols].astype(np.float32), tuple(dense.shape))


def _sparse_dense(rng, m, k, density=0.5):
    d = rng.standard_normal((m, k)).astype(np.float32)
    d[rng.random((m, k)) > density] = 0.0
    return d


@pytest.mark.parametrize("shape", [(6, 5, 4), (1, 1, 1), (8, 3, 7)])
def test_spmm_csr(shape):
    rt = _rt_or_skip()
    m, k, n = shape
    rng = np.random.default_rng(1 + m + n)
    a = _sparse_dense(rng, m, k)
    b = rng.standard_normal((k, n)).astype(np.float32)
    a_csr = _csr(a)
    res = rt.launch(_art(rt, "tessera.spmm_csr", 2), (a_csr, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_sparse_compiled"
    np.testing.assert_allclose(np.asarray(res["output"]), a @ b, atol=1e-4)


def test_spmm_coo():
    rt = _rt_or_skip()
    rng = np.random.default_rng(7)
    a = _sparse_dense(rng, 6, 5)
    b = rng.standard_normal((5, 4)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.spmm_coo", 2), (_coo(a), b))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]), a @ b, atol=1e-4)


@pytest.mark.parametrize("shape", [(6, 5, 4), (3, 8, 3)])
def test_sddmm(shape):
    rt = _rt_or_skip()
    m, k, n = shape
    rng = np.random.default_rng(3 + n)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    mask = (rng.random((m, n)) > 0.4).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.sddmm", 3), (a, b, mask))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]), (a @ b) * mask,
                               atol=1e-4)


def test_bsmm():
    rt = _rt_or_skip()
    rng = np.random.default_rng(5)
    x = rng.standard_normal((6, 5)).astype(np.float32)
    w = rng.standard_normal((5, 4)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.bsmm", 2), (x, w))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]), x @ w, atol=1e-4)
