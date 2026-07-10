"""Apple GPU sparse + MoE lane — spmm_csr / spmm_coo / sddmm / bsmm / moe.

Apple ships no device sparse/moe kernel, so these run the numpy reference the
x86/ROCm device kernels are matched against (CSR SpMM, (a@b)*mask, a@b, routed
per-token expert GEMVs). Reachable via
`compiler_path="apple_gpu_sparse_compiled"`; execution_kind=reference_cpu.
Validated vs numpy / tessera.ops — parity with test_x86_sparse_compiled /
test_x86_moe_compiled.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import ops as O
from tessera import runtime as rt


def _run(op, operands, kwargs=None):
    names = [f"a{i}" for i in range(len(operands))]
    art = rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_sparse_compiled",
        "executable": True, "execution_kind": "reference_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": dict(kwargs or {})}]})
    res = rt.launch(art, tuple(operands))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "apple_gpu_sparse_compiled"
    assert res["execution_kind"] == "reference_cpu"
    return np.asarray(res["output"])


def _sparse_dense(rng, m, k, density=0.5):
    d = rng.standard_normal((m, k)).astype(np.float32)
    d[rng.random((m, k)) > density] = 0.0
    return d


def _csr(d):
    indptr, indices, vals = [0], [], []
    for row in d:
        for j, v in enumerate(row):
            if v != 0.0:
                indices.append(j)
                vals.append(v)
        indptr.append(len(indices))
    return (np.array(indptr, np.int32), np.array(indices, np.int32),
            np.array(vals, np.float32), tuple(d.shape))


def _coo(d):
    rows, cols = np.nonzero(d)
    coords = np.stack([rows, cols], 1).astype(np.int32)
    return (coords, d[rows, cols].astype(np.float32), tuple(d.shape))


@pytest.mark.parametrize("shape", [(6, 5, 4), (1, 1, 1), (8, 3, 7)])
def test_spmm_csr(shape):
    m, k, n = shape
    rng = np.random.default_rng(1 + m + n)
    a = _sparse_dense(rng, m, k)
    b = rng.standard_normal((k, n)).astype(np.float32)
    np.testing.assert_allclose(_run("tessera.spmm_csr", [_csr(a), b]), a @ b,
                               atol=1e-4)


def test_spmm_coo():
    rng = np.random.default_rng(7)
    a = _sparse_dense(rng, 6, 5)
    b = rng.standard_normal((5, 4)).astype(np.float32)
    np.testing.assert_allclose(_run("tessera.spmm_coo", [_coo(a), b]), a @ b,
                               atol=1e-4)


@pytest.mark.parametrize("shape", [(6, 5, 4), (3, 8, 3)])
def test_sddmm(shape):
    m, k, n = shape
    rng = np.random.default_rng(2 + m)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    mask = (rng.random((m, n)) > 0.4).astype(np.float32)
    np.testing.assert_allclose(_run("tessera.sddmm", [a, b, mask]),
                               (a @ b) * mask, atol=1e-4)


def test_bsmm():
    rng = np.random.default_rng(9)
    a = rng.standard_normal((6, 5)).astype(np.float32)
    b = rng.standard_normal((5, 4)).astype(np.float32)
    np.testing.assert_allclose(_run("tessera.bsmm", [a, b]), a @ b, atol=1e-4)


def test_moe_round_robin_scores_route():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((10, 5)).astype(np.float32)
    experts = rng.standard_normal((4, 5, 6)).astype(np.float32)
    np.testing.assert_allclose(_run("tessera.moe", [x, experts], {"extras": []}),
                               np.asarray(O.moe(x, experts)), atol=1e-4)
    scores = rng.standard_normal((10, 4)).astype(np.float32)
    np.testing.assert_allclose(
        _run("tessera.moe", [x, experts, scores], {"extras": ["scores"]}),
        np.asarray(O.moe(x, experts, scores=scores)), atol=1e-4)
    route = rng.integers(0, 4, size=10).astype(np.int64)
    np.testing.assert_allclose(
        _run("tessera.moe", [x, experts, route], {"extras": ["route"]}),
        np.asarray(O.moe(x, experts, route=route)), atol=1e-4)
