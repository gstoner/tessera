"""Compiler-generated digamma on x86 AVX-512 (P2e of S_SERIES_GAP_CLOSURE_PLAN)
— ψ(x)=d/dx lnΓ(x) via a recurrence-to-x>=8 + asymptotic-series SIMD core
(x>0) + scalar fallback for x<=0 (reflection / poles). On the transcendental
lane (kind 20). Reachable via `compiler_path="x86_transcendental_compiled"`.
Validated vs tessera.ops.digamma. Skip-clean: x86 lib not built.
"""

from __future__ import annotations

import numpy as np
import pytest


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_transcendental_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["a"], "output_name": "o",
        "ops": [{"op_name": "tessera.digamma", "result": "o",
                 "operands": ["a"], "kwargs": {}}],
    })


def _ref(x):
    import tessera as ts
    return np.asarray(ts.ops.digamma(x), np.float32)


@pytest.mark.parametrize("shape", [(64,), (4, 9), (17,)])
def test_x86_digamma_positive(shape):
    """Positive domain — recurrence + asymptotic SIMD core."""
    rt = _x86_or_skip()
    x = np.random.default_rng(5).uniform(0.05, 50.0, shape).astype(np.float32)
    res = rt.launch(_art(rt), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_transcendental_compiled"
    out = np.asarray(res["output"]).astype(np.float32)
    np.testing.assert_allclose(out, _ref(x), rtol=1e-4, atol=1e-4)


def test_x86_digamma_reflection_and_identity():
    """Negative non-integers (reflection) + ψ(1) == -Euler-Mascheroni."""
    rt = _x86_or_skip()
    x = np.array([1.0, 2.0, 0.5, 0.1, 5.0, -0.5, -1.5, -2.5, -3.3], np.float32)
    out = np.asarray(rt.launch(_art(rt), (x,))["output"]).astype(np.float32)
    np.testing.assert_allclose(out, _ref(x), rtol=1e-4, atol=1e-4)
    assert abs(out[0] - (-0.5772156649)) < 1e-4  # ψ(1) = -γ


def test_x86_digamma_poles_are_nan():
    rt = _x86_or_skip()
    x = np.array([0.0, -1.0, -2.0, -5.0], np.float32)
    out = np.asarray(rt.launch(_art(rt), (x,))["output"]).astype(np.float32)
    assert np.all(np.isnan(out))
