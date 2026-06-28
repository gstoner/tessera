"""Compiler-generated sin on gfx1151 (P2d of S_SERIES_GAP_CLOSURE_PLAN) — math.sin
-> ROCDL in generate-rocm-unary-kernel. Reachable via the rocm_unary_compiled
lane. Validated vs np.sin on gfx1151. Skip-clean: no opt / no GPU.
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


def _art(rt, x):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_unary_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a"], "output_name": "o",
        "ops": [{"op_name": "tessera.sin", "result": "o", "operands": ["a"],
                 "kwargs": {}}],
    })


def test_sin():
    rt = _rocm_or_skip()
    x = (np.random.default_rng(0).standard_normal((4, 9)) * 4).astype(np.float32)
    res = rt.launch(_art(rt, x), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]), np.sin(x), atol=1e-4)


def test_sin_range():
    rt = _rocm_or_skip()
    x = np.linspace(-10, 10, 101).astype(np.float32)
    res = rt.launch(_art(rt, x), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]), np.sin(x), atol=1e-4)
