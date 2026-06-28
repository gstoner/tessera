"""Compiler-generated lgamma on gfx1151 (P2e of S_SERIES_GAP_CLOSURE_PLAN) —
ln Γ(x) via an MLIR-built Numerical-Recipes Lanczos g=5 series (no math.lgamma
op exists), with reflection for x<0.5 via math.sin. On the unary lane (kind
"lgamma"). Reachable via `compiler_path="rocm_unary_compiled"`. Validated vs
math.lgamma on gfx1151. Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_unary_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a"], "output_name": "o",
        "ops": [{"op_name": "tessera.lgamma", "result": "o", "operands": ["a"],
                 "kwargs": {}}],
    })


def _ref(x: np.ndarray) -> np.ndarray:
    return np.vectorize(math.lgamma, otypes=[np.float64])(x).astype(np.float32)


# (257,) crosses the 256-thread block, exercising the strided grid.
@pytest.mark.parametrize("shape", [(64,), (4, 9), (257,)])
def test_rocm_lgamma_positive_lanczos(shape):
    rt = _rocm_or_skip()
    x = np.random.default_rng(3).uniform(0.5, 60.0, shape).astype(np.float32)
    res = rt.launch(_art(rt), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_unary_compiled"
    out = np.asarray(res["output"]).astype(np.float32)
    np.testing.assert_allclose(out, _ref(x), rtol=2e-4, atol=2e-4)


def test_rocm_lgamma_reflection_and_small():
    rt = _rocm_or_skip()
    x = np.array([0.5, 1.0, 2.0, 0.1, 3.5, -0.5, -1.5, -2.5], np.float32)
    out = np.asarray(rt.launch(_art(rt), (x,))["output"]).astype(np.float32)
    np.testing.assert_allclose(out, _ref(x), rtol=2e-4, atol=2e-4)
    assert abs(out[0] - math.log(math.sqrt(math.pi))) < 2e-4


def test_rocm_lgamma_poles_are_inf():
    """lnΓ has poles at non-positive integers. The reflection sin(πx) with an
    f32 π is not exactly 0 there, so the explicit pole check must force +inf
    (matching std::lgamma) instead of returning finite garbage."""
    rt = _rocm_or_skip()
    x = np.array([0.0, -1.0, -2.0, -5.0], np.float32)
    out = np.asarray(rt.launch(_art(rt), (x,))["output"]).astype(np.float32)
    assert np.all(np.isinf(out)) and np.all(out > 0), out
    # a near-integer is NOT a pole — stays finite
    near = np.array([-1.00005, -2.0003], np.float32)
    out2 = np.asarray(rt.launch(_art(rt), (near,))["output"]).astype(np.float32)
    assert np.all(np.isfinite(out2)), out2
