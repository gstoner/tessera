"""Compiler-generated unary predicates on x86 AVX-512 (P2b of
S_SERIES_GAP_CLOSURE_PLAN) — isnan / isinf / isfinite (f32 -> bool). Reachable via
`compiler_path="x86_predicate_compiled"`. Validated vs numpy. Skip-clean: x86 lib
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


def _art(rt, op):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_predicate_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["a"], "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": ["a"], "kwargs": {}}],
    })


_X = np.array([1.0, np.nan, np.inf, -np.inf, 0.0, -3.5, 1e30, -1e-30],
              np.float32)


@pytest.mark.parametrize("op,ref", [
    ("tessera.isnan", np.isnan),
    ("tessera.isinf", np.isinf),
    ("tessera.isfinite", np.isfinite),
])
def test_predicate(op, ref):
    rt = _rt_or_skip()
    res = rt.launch(_art(rt, op), (_X,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_predicate_compiled"
    out = np.asarray(res["output"])
    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out, ref(_X))


def test_predicate_shape_preserved():
    rt = _rt_or_skip()
    x = np.array([[1.0, np.nan], [np.inf, 2.0]], np.float32)
    res = rt.launch(_art(rt, "tessera.isfinite"), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_array_equal(np.asarray(res["output"]), np.isfinite(x))
