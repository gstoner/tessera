"""Compiler-generated digamma on gfx1151 (P2e of S_SERIES_GAP_CLOSURE_PLAN) —
ψ(x) via an MLIR-built recurrence-to-x>=8 + asymptotic series (no math.digamma
op exists), with reflection (math.tan) + pole->NaN for x<=0. On the unary lane
(kind "digamma"). Reachable via `compiler_path="rocm_unary_compiled"`. Validated
vs tessera.ops.digamma on gfx1151. Skip-clean: tessera-opt not built / no GPU.
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


def _art(rt):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_unary_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a"], "output_name": "o",
        "ops": [{"op_name": "tessera.digamma", "result": "o",
                 "operands": ["a"], "kwargs": {}}],
    })


def _ref(x):
    import tessera as ts
    return np.asarray(ts.ops.digamma(x), np.float32)


@pytest.mark.parametrize("shape", [(64,), (4, 9), (257,)])
def test_rocm_digamma_positive(shape):
    rt = _rocm_or_skip()
    x = np.random.default_rng(5).uniform(0.05, 50.0, shape).astype(np.float32)
    res = rt.launch(_art(rt), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_unary_compiled"
    out = np.asarray(res["output"]).astype(np.float32)
    np.testing.assert_allclose(out, _ref(x), rtol=3e-4, atol=3e-4)


def test_rocm_digamma_reflection_and_identity():
    rt = _rocm_or_skip()
    x = np.array([1.0, 2.0, 0.5, 0.1, 5.0, -0.5, -1.5, -2.5], np.float32)
    out = np.asarray(rt.launch(_art(rt), (x,))["output"]).astype(np.float32)
    np.testing.assert_allclose(out, _ref(x), rtol=3e-4, atol=3e-4)
    assert abs(out[0] - (-0.5772156649)) < 3e-4  # ψ(1) = -γ


def test_rocm_digamma_poles_are_nan():
    rt = _rocm_or_skip()
    x = np.array([0.0, -1.0, -2.0, -5.0], np.float32)
    out = np.asarray(rt.launch(_art(rt), (x,))["output"]).astype(np.float32)
    assert np.all(np.isnan(out))
