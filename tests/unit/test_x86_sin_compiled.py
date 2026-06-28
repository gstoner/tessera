"""Compiler-generated sin on x86 AVX-512 (P2d of S_SERIES_GAP_CLOSURE_PLAN) — the
kSin transcendental core (reuses the kernel's sincos512). Reachable via the
x86_transcendental_compiled lane. Validated vs np.sin. Skip-clean: x86 lib not built.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, x):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_transcendental_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["a"], "output_name": "o",
        "ops": [{"op_name": "tessera.sin", "result": "o", "operands": ["a"],
                 "kwargs": {}}],
    })


def test_sin():
    rt = _rt_or_skip()
    x = (np.random.default_rng(0).standard_normal((4, 9)) * 4).astype(np.float32)
    res = rt.launch(_art(rt, x), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]), np.sin(x), atol=1e-5)


def test_sin_range():
    rt = _rt_or_skip()
    x = np.linspace(-10, 10, 101).astype(np.float32)
    res = rt.launch(_art(rt, x), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]), np.sin(x), atol=2e-5)
