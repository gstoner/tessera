"""Compiler-generated lgamma on x86 AVX-512 (P2e of S_SERIES_GAP_CLOSURE_PLAN) —
ln Γ(x) via the Numerical-Recipes Lanczos g=5 SIMD core (positive domain) +
std::lgamma fallback for the reflection lanes (x<0.5). On the transcendental
lane (kind 19). Reachable via `compiler_path="x86_transcendental_compiled"`.
Validated vs math.lgamma. Skip-clean: libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import math

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
        "ops": [{"op_name": "tessera.lgamma", "result": "o", "operands": ["a"],
                 "kwargs": {}}],
    })


def _ref(x: np.ndarray) -> np.ndarray:
    return np.vectorize(math.lgamma, otypes=[np.float64])(x).astype(np.float32)


# (17,) exercises the scalar tail past the 16-lane __m512 body.
@pytest.mark.parametrize("shape", [(64,), (4, 9), (17,)])
def test_x86_lgamma_positive_lanczos(shape):
    """Positive domain — the SIMD Lanczos core."""
    rt = _x86_or_skip()
    x = np.random.default_rng(3).uniform(0.5, 60.0, shape).astype(np.float32)
    res = rt.launch(_art(rt), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_transcendental_compiled"
    out = np.asarray(res["output"]).astype(np.float32)
    np.testing.assert_allclose(out, _ref(x), rtol=1e-4, atol=1e-4)


def test_x86_lgamma_reflection_and_small():
    """Small positive (<0.5) + negative non-integers — the reflection fallback.
    lgamma(0.5)==ln√π, lgamma(1)==lgamma(2)==0."""
    rt = _x86_or_skip()
    x = np.array([0.5, 1.0, 2.0, 0.1, 0.01, 3.5, -0.5, -1.5, -2.5, -10.3],
                 np.float32)
    out = np.asarray(rt.launch(_art(rt), (x,))["output"]).astype(np.float32)
    np.testing.assert_allclose(out, _ref(x), rtol=1e-4, atol=1e-4)
    assert abs(out[0] - math.log(math.sqrt(math.pi))) < 1e-4


def test_x86_lgamma_non_finite_full_block():
    """A full 16-lane block of +inf / nan exercises the SIMD path (not the
    scalar tail). lgamma(+inf)=+inf, lgamma(nan)=nan — must NOT degrade to NaN
    from the Lanczos `inf - inf*log(inf)`, so the result is block-independent."""
    rt = _x86_or_skip()
    inf_blk = np.full(16, np.inf, np.float32)
    out = np.asarray(rt.launch(_art(rt), (inf_blk,))["output"]).astype(np.float32)
    assert np.all(np.isinf(out)) and np.all(out > 0), out
    nan_blk = np.full(16, np.nan, np.float32)
    out2 = np.asarray(rt.launch(_art(rt), (nan_blk,))["output"]).astype(np.float32)
    assert np.all(np.isnan(out2)), out2
