"""Compiler-generated atan2 on gfx1151 (P2e of S_SERIES_GAP_CLOSURE_PLAN) —
quadrant-aware atan2(y, x), composed on the COMPILER-GENERATED unary atan kernel
(no new kernel; sign/quadrant logic on host). Reachable via
`compiler_path="rocm_atan2_compiled"`. Validated vs np.arctan2 across all four
quadrants + the on-axis / origin special cases on gfx1151. Skip-clean:
tessera-opt not built / no GPU.
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
        "target": "rocm", "compiler_path": "rocm_atan2_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["y", "x"], "output_name": "o",
        "ops": [{"op_name": "tessera.atan2", "result": "o",
                 "operands": ["y", "x"], "kwargs": {}}],
    })


@pytest.mark.parametrize("shape", [(64,), (4, 9), (2, 3, 5)])
def test_atan2_all_quadrants(shape):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(shape) * 5).astype(np.float32)
    x = (rng.standard_normal(shape) * 5).astype(np.float32)
    res = rt.launch(_art(rt), (y, x))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_atan2_compiled"
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.arctan2(y, x).astype(np.float32), atol=1e-4)


def test_atan2_axis_and_origin():
    rt = _rocm_or_skip()
    y = np.array([1, -1, 0, 0, 1, -1, 1, -1, 0], np.float32)
    x = np.array([0, 0, 1, -1, 1, -1, -1, 1, 0], np.float32)
    out = np.asarray(rt.launch(_art(rt), (y, x))["output"])
    np.testing.assert_allclose(out, np.arctan2(y, x).astype(np.float32), atol=1e-4)
