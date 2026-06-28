"""Compiler-generated SVD on x86 AVX-512 (linalg PR-C) — one-sided Jacobi.
Genuinely computes the decomposition (does not wrap LAPACK). Reachable via
`compiler_path="x86_linalg_compiled"`. Validated by invariants (A=U·diag(S)·Vh,
S matches numpy, orthonormal U/Vh). Skip-clean: x86 lib not built.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, operands):
    names = [f"a{i}" for i in range(len(operands))]
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_linalg_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": "tessera.svd", "result": "o", "operands": names,
                 "kwargs": {}}],
    })


def _check(a, u, s, vh):
    m, n = a.shape[-2], a.shape[-1]
    k = min(m, n)
    assert u.shape == a.shape[:-2] + (m, k)
    assert s.shape == a.shape[:-2] + (k,)
    assert vh.shape == a.shape[:-2] + (k, n)
    np.testing.assert_allclose(u @ (s[..., None] * vh), a, atol=1e-3)
    ref_s = np.linalg.svd(a, full_matrices=False)[1]
    np.testing.assert_allclose(np.sort(s, axis=-1)[..., ::-1], ref_s, atol=1e-3)
    eye = np.broadcast_to(np.eye(k, dtype=np.float32), a.shape[:-2] + (k, k))
    np.testing.assert_allclose(np.swapaxes(u, -1, -2) @ u, eye, atol=1e-3)
    np.testing.assert_allclose(vh @ np.swapaxes(vh, -1, -2), eye, atol=1e-3)


@pytest.mark.parametrize("m,n", [(6, 4), (5, 5), (4, 6), (8, 3)])
def test_svd(m, n):
    rt = _rt_or_skip()
    a = np.random.default_rng(1 + m + n).standard_normal((m, n)).astype(np.float32)
    res = rt.launch(_art(rt, [a]), (a,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_linalg_compiled"
    u, s, vh = (np.asarray(x) for x in res["output"])
    _check(a, u, s, vh)


def test_svd_batched():
    rt = _rt_or_skip()
    a = np.random.default_rng(9).standard_normal((3, 6, 4)).astype(np.float32)
    res = rt.launch(_art(rt, [a]), (a,))
    assert res["ok"] is True, res.get("reason")
    u, s, vh = (np.asarray(x) for x in res["output"])
    _check(a, u, s, vh)
