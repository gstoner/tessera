"""Compiler-generated clamp / clip on x86 AVX-512 (P2c of
S_SERIES_GAP_CLOSURE_PLAN) — composed on the binary max/min lane (no new kernel;
scalar bounds broadcast on host). Either bound may be None. Reachable via
`compiler_path="x86_clamp_compiled"`. Validated vs np.clip. Skip-clean: x86 lib
not built.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_clamp_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["a"], "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": ["a"],
                 "kwargs": kwargs}],
    })


_X = (np.random.default_rng(0).standard_normal((4, 9)) * 3).astype(np.float32)


@pytest.mark.parametrize("op,kwargs,lo,hi", [
    ("tessera.clamp", {"min": -1.0, "max": 1.0}, -1.0, 1.0),
    ("tessera.clamp", {"min": 0.0}, 0.0, None),
    ("tessera.clamp", {"max": 0.5}, None, 0.5),
    ("tessera.clip", {"min_val": -2.0, "max_val": 2.0}, -2.0, 2.0),
    ("tessera.clip", {"min": -0.5}, -0.5, None),
])
def test_clamp(op, kwargs, lo, hi):
    rt = _rt_or_skip()
    res = rt.launch(_art(rt, op, kwargs), (_X,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_clamp_compiled"
    np.testing.assert_allclose(np.asarray(res["output"]), np.clip(_X, lo, hi),
                               atol=1e-6)
