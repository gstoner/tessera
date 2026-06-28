"""Compiler-generated LU + Householder QR on x86 AVX-512 (linalg PR-B) — lu / qr.
Genuinely computes the factorization (does not wrap LAPACK). Reachable via
`compiler_path="x86_linalg_compiled"`. Validated by invariants (P·A=L·U; A=Q·R,
QᵀQ=I, R upper). Skip-clean: x86 lib not built.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op, operands):
    names = [f"a{i}" for i in range(len(operands))]
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_linalg_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names, "kwargs": {}}],
    })


def _apply_piv(a, piv):
    ap = a.copy()
    for k in range(len(piv)):
        p = int(piv[k])
        if p != k:
            ap[[k, p]] = ap[[p, k]]
    return ap


@pytest.mark.parametrize("n", [4, 5])
def test_lu(n):
    rt = _rt_or_skip()
    a = np.random.default_rng(1 + n).standard_normal((n, n)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.lu", [a]), (a,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_linalg_compiled"
    lu, piv = res["output"]
    lu = np.asarray(lu)
    low = np.tril(lu, -1) + np.eye(n, dtype=np.float32)
    up = np.triu(lu)
    np.testing.assert_allclose(low @ up, _apply_piv(a, np.asarray(piv)), atol=1e-4)


def test_lu_batched():
    rt = _rt_or_skip()
    a = np.random.default_rng(2).standard_normal((3, 5, 5)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.lu", [a]), (a,))
    assert res["ok"] is True, res.get("reason")
    lu, piv = res["output"]
    assert np.asarray(lu).shape == (3, 5, 5)
    assert np.asarray(piv).shape == (3, 5)


@pytest.mark.parametrize("m,n", [(6, 4), (4, 6), (5, 5)])
def test_qr(m, n):
    rt = _rt_or_skip()
    a = np.random.default_rng(3 + m + n).standard_normal((m, n)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.qr", [a]), (a,))
    assert res["ok"] is True, res.get("reason")
    q, r = (np.asarray(x) for x in res["output"])
    k = min(m, n)
    assert q.shape == (m, k) and r.shape == (k, n)
    np.testing.assert_allclose(q @ r, a, atol=1e-3)
    np.testing.assert_allclose(q.T @ q, np.eye(k), atol=1e-3)
    np.testing.assert_allclose(np.tril(r, -1), 0.0, atol=1e-4)


def test_qr_batched():
    rt = _rt_or_skip()
    a = np.random.default_rng(9).standard_normal((2, 6, 4)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.qr", [a]), (a,))
    assert res["ok"] is True, res.get("reason")
    q, r = (np.asarray(x) for x in res["output"])
    assert q.shape == (2, 6, 4) and r.shape == (2, 4, 4)
    np.testing.assert_allclose(q @ r, a, atol=1e-3)
