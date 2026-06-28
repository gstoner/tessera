"""Compiler-generated dense linear algebra on x86 AVX-512 (linalg PR-A) —
cholesky / tri_solve / cholesky_solve. Genuinely computes the factorization /
substitution (does not wrap LAPACK). Reachable via
`compiler_path="x86_linalg_compiled"`. Validated vs numpy. Skip-clean: x86 lib
not built.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op, operands, kwargs=None):
    names = [f"a{i}" for i in range(len(operands))]
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_linalg_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": kwargs or {}}],
    })


def _spd(rng, n):
    m = rng.standard_normal((n, n)).astype(np.float32)
    return (m @ m.T + n * np.eye(n)).astype(np.float32)


@pytest.mark.parametrize("n", [4, 5])
def test_cholesky(n):
    rt = _rt_or_skip()
    a = _spd(np.random.default_rng(1 + n), n)
    res = rt.launch(_art(rt, "tessera.cholesky", [a]), (a,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_linalg_compiled"
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.linalg.cholesky(a), atol=1e-4)


def test_cholesky_batched():
    rt = _rt_or_skip()
    rng = np.random.default_rng(2)
    a = np.stack([_spd(rng, 5), _spd(rng, 5)])
    res = rt.launch(_art(rt, "tessera.cholesky", [a]), (a,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.linalg.cholesky(a), atol=1e-4)


def test_tri_solve_lower_vector():
    rt = _rt_or_skip()
    rng = np.random.default_rng(3)
    n = 5
    a = np.tril(rng.standard_normal((n, n)).astype(np.float32)) + n * np.eye(n, dtype=np.float32)
    b = rng.standard_normal((n,)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.tri_solve", [a, b], {"lower": True}), (a, b))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.linalg.solve(np.tril(a), b), atol=1e-3)


def test_tri_solve_upper_matrix():
    rt = _rt_or_skip()
    rng = np.random.default_rng(4)
    n = 6
    a = np.triu(rng.standard_normal((n, n)).astype(np.float32)) + n * np.eye(n, dtype=np.float32)
    b = rng.standard_normal((n, 3)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.tri_solve", [a, b], {"lower": False}), (a, b))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.linalg.solve(np.triu(a), b), atol=1e-3)


def test_cholesky_solve():
    rt = _rt_or_skip()
    rng = np.random.default_rng(5)
    a = _spd(rng, 5)
    ell = np.linalg.cholesky(a).astype(np.float32)
    b = rng.standard_normal((5,)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.cholesky_solve", [ell, b]), (ell, b))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.linalg.solve(a, b), atol=1e-3)
