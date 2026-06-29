"""Compiler-generated clamp / clip on gfx1151 (P2c of
S_SERIES_GAP_CLOSURE_PLAN) — composed on the COMPILER-GENERATED binary max/min
kernel (no new kernel; scalar bounds broadcast on host). Reachable via
`compiler_path="rocm_clamp_compiled"`. Validated vs np.clip on gfx1151.
Skip-clean: tessera-opt not built / no GPU.
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


def _art(rt, op, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_clamp_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a"], "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": ["a"],
                 "kwargs": kwargs}],
    })


_X = (np.random.default_rng(0).standard_normal((4, 9)) * 3).astype(np.float32)


@pytest.mark.parametrize("op,kwargs,lo,hi", [
    ("tessera.clamp", {"min": -1.0, "max": 1.0}, -1.0, 1.0),
    ("tessera.clamp", {"min": 0.0}, 0.0, None),
    ("tessera.clip", {"min_val": -2.0, "max_val": 2.0}, -2.0, 2.0),
    ("tessera.clip", {"max": 0.5}, None, 0.5),
])
def test_clamp(op, kwargs, lo, hi):
    rt = _rocm_or_skip()
    res = rt.launch(_art(rt, op, kwargs), (_X,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_clamp_compiled"
    np.testing.assert_allclose(np.asarray(res["output"]), np.clip(_X, lo, hi),
                               atol=1e-4)
