"""Compiler-generated elementwise completion on x86 AVX-512 (P2a of
S_SERIES_GAP_CLOSURE_PLAN) — binary arithmetic add / mul / mod / floor_div (new
AVX-512 binary kinds) + abs (the numeric_helper alias of the already-shipped
absolute unary kernel). Reachable via the x86_binary_compiled / x86_unary_compiled
lanes. Validated vs numpy. Skip-clean: x86 lib not built.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op, path, operands):
    names = [f"a{i}" for i in range(len(operands))]
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": path,
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names, "kwargs": {}}],
    })


@pytest.mark.parametrize("op,ref", [
    ("tessera.add", lambda a, b: a + b),
    ("tessera.mul", lambda a, b: a * b),
    ("tessera.mod", np.mod),
    ("tessera.floor_div", np.floor_divide),
])
def test_binary(op, ref):
    rt = _rt_or_skip()
    rng = np.random.default_rng(1 + hash(op) % 100)
    a = rng.standard_normal(37).astype(np.float32)
    b = (rng.standard_normal(37) + 1.5).astype(np.float32)  # avoid /0
    res = rt.launch(_art(rt, op, "x86_binary_compiled", [a, b]), (a, b))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]), ref(a, b), atol=1e-5)


def test_abs():
    rt = _rt_or_skip()
    a = np.random.default_rng(7).standard_normal(40).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.abs", "x86_unary_compiled", [a]), (a,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]), np.abs(a), atol=1e-6)
