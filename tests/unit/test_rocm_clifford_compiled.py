"""Geometric-algebra (Clifford) lane on AMD ROCm gfx1151 (P12 of
S_SERIES_GAP_CLOSURE_PLAN) — Cl(3,0) geometric_product / wedge /
left_contraction / inner / rotor_sandwich via the COMPILER-GENERATED
table-driven bilinear kernel (generate-rocm-clifford-kernel; one thread per
batch element; triples unrolled at generation time). Reachable via
`compiler_path="rocm_clifford_compiled"`. Validated vs the numpy GA reference
(tessera._clifford_ops). Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera._clifford_ops as ref


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, op, n_operands):
    names = [f"a{i}" for i in range(n_operands)]
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_clifford_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": {}}]})


def _run(rt, op, *arrs):
    res = rt.launch(_art(rt, op, len(arrs)), arrs)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_clifford_compiled"
    return np.asarray(res["output"])


_RNG = np.random.default_rng(3)
A = _RNG.standard_normal((5, 8)).astype(np.float32)
B = _RNG.standard_normal((5, 8)).astype(np.float32)


def test_geometric_product():
    rt = _rocm_or_skip()
    got = _run(rt, "clifford_geometric_product", A, B)
    np.testing.assert_allclose(got, ref.clifford_geometric_product(A, B),
                               rtol=1e-5, atol=1e-5)


def test_wedge():
    rt = _rocm_or_skip()
    got = _run(rt, "clifford_wedge", A, B)
    np.testing.assert_allclose(got, ref.clifford_wedge(A, B),
                               rtol=1e-5, atol=1e-5)


def test_left_contraction():
    rt = _rocm_or_skip()
    got = _run(rt, "clifford_left_contraction", A, B)
    np.testing.assert_allclose(got, ref.clifford_left_contraction(A, B),
                               rtol=1e-5, atol=1e-5)


def test_inner():
    rt = _rocm_or_skip()
    got = _run(rt, "clifford_inner", A, B)
    np.testing.assert_allclose(got, ref.clifford_inner(A, B),
                               rtol=1e-5, atol=1e-5)


def test_rotor_sandwich():
    rt = _rocm_or_skip()
    rotor = np.zeros((5, 8), np.float32)
    rotor[:, 0] = np.cos(0.3)
    rotor[:, 3] = np.sin(0.3)
    got = _run(rt, "clifford_rotor_sandwich", rotor, A)
    np.testing.assert_allclose(got, ref.clifford_rotor_sandwich(rotor, A),
                               rtol=1e-5, atol=1e-5)
