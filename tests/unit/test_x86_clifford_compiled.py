"""Geometric-algebra (Clifford) lane on x86 AVX-512 (P12 of
S_SERIES_GAP_CLOSURE_PLAN) — Cl(3,0) geometric_product / wedge /
left_contraction / inner / rotor_sandwich via the table-driven bilinear kernel
(tessera_x86_clifford_bilinear_f32; blade-major [8,n]; compile-time Cayley
table). Reachable via `compiler_path="x86_clifford_compiled"`. Validated vs the
numpy GA reference (tessera._clifford_ops). Skip-clean:
libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera._clifford_ops as ref


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op, n_operands):
    names = [f"a{i}" for i in range(n_operands)]
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_clifford_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": {}}]})


def _run(rt, op, *arrs):
    res = rt.launch(_art(rt, op, len(arrs)), arrs)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_clifford_compiled"
    return np.asarray(res["output"])


_RNG = np.random.default_rng(3)
A = _RNG.standard_normal((5, 8)).astype(np.float32)
B = _RNG.standard_normal((5, 8)).astype(np.float32)


def test_geometric_product():
    rt = _rt_or_skip()
    got = _run(rt, "clifford_geometric_product", A, B)
    np.testing.assert_allclose(got, ref.clifford_geometric_product(A, B),
                               rtol=1e-5, atol=1e-5)


def test_wedge():
    rt = _rt_or_skip()
    got = _run(rt, "clifford_wedge", A, B)
    np.testing.assert_allclose(got, ref.clifford_wedge(A, B),
                               rtol=1e-5, atol=1e-5)


def test_left_contraction():
    rt = _rt_or_skip()
    got = _run(rt, "clifford_left_contraction", A, B)
    np.testing.assert_allclose(got, ref.clifford_left_contraction(A, B),
                               rtol=1e-5, atol=1e-5)


def test_inner():
    rt = _rt_or_skip()
    got = _run(rt, "clifford_inner", A, B)
    np.testing.assert_allclose(got, ref.clifford_inner(A, B),
                               rtol=1e-5, atol=1e-5)


def test_rotor_sandwich():
    rt = _rt_or_skip()
    # a unit rotor (scalar + bivector) so the sandwich is a rotation
    rotor = np.zeros((5, 8), np.float32)
    rotor[:, 0] = np.cos(0.3)
    rotor[:, 3] = np.sin(0.3)         # e12 bivector blade
    got = _run(rt, "clifford_rotor_sandwich", rotor, A)
    np.testing.assert_allclose(got, ref.clifford_rotor_sandwich(rotor, A),
                               rtol=1e-5, atol=1e-5)


def test_geometric_product_identity():
    rt = _rt_or_skip()
    # scalar 1 multivector is the geometric-product identity
    one = np.zeros((5, 8), np.float32)
    one[:, 0] = 1.0
    np.testing.assert_allclose(_run(rt, "clifford_geometric_product", one, A),
                               A, rtol=1e-6, atol=1e-6)
