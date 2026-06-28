"""Compiler-generated atan2 on x86 AVX-512 (P2e of S_SERIES_GAP_CLOSURE_PLAN) —
quadrant-aware atan2(y, x), composed on the transcendental atan lane (no new
kernel; sign/quadrant logic on host). Reachable via
`compiler_path="x86_atan2_compiled"`. Validated vs np.arctan2 across all four
quadrants + the on-axis / origin special cases. Skip-clean: x86 lib not built.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_atan2_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["y", "x"], "output_name": "o",
        "ops": [{"op_name": "tessera.atan2", "result": "o",
                 "operands": ["y", "x"], "kwargs": {}}],
    })


@pytest.mark.parametrize("shape", [(64,), (4, 9), (2, 3, 5)])
def test_atan2_all_quadrants(shape):
    rt = _rt_or_skip()
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(shape) * 5).astype(np.float32)
    x = (rng.standard_normal(shape) * 5).astype(np.float32)
    res = rt.launch(_art(rt), (y, x))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_atan2_compiled"
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.arctan2(y, x).astype(np.float32), atol=2e-5)


def test_atan2_axis_and_origin():
    """The on-axis half-lines and the origin (numpy: atan2(0,0)==0)."""
    rt = _rt_or_skip()
    y = np.array([1, -1, 0, 0, 1, -1, 1, -1, 0], np.float32)
    x = np.array([0, 0, 1, -1, 1, -1, -1, 1, 0], np.float32)
    out = np.asarray(rt.launch(_art(rt), (y, x))["output"])
    np.testing.assert_allclose(out, np.arctan2(y, x).astype(np.float32), atol=2e-5)
