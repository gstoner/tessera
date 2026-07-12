"""Hand-written AVX-512 logical kernel on x86 — the CPU analog of the ROCm
device_verified_jit logical lane.

`tessera_x86_avx512_logical_i8` is exported by libtessera_x86_elementwise.so; the
Python runtime ctypes-loads it and calls it from `runtime.launch()` via
`compiler_path="x86_logical_compiled"`. Covers logical_and/or/xor (binary) +
logical_not (unary) over i8 booleans; inputs normalized via `!= 0` (numpy
semantics). bool in/out.

Validated vs numpy. Skip-clean: libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import numpy as np
import pytest


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _artifact(rt, op_name, nin):
    operands = ["a", "b"][:nin]
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_logical_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": operands, "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": operands}],
    })


_BINARY = {
    "tessera.logical_and": np.logical_and,
    "tessera.logical_or": np.logical_or,
    "tessera.logical_xor": np.logical_xor,
}


@pytest.mark.parametrize("op_name", list(_BINARY))
@pytest.mark.parametrize("shape", [(8, 64), (130,), (3, 5, 7)])
def test_x86_logical_binary_matches_numpy(op_name, shape):
    rt = _x86_or_skip()
    ref = _BINARY[op_name]
    rng = np.random.default_rng(43 + len(shape) + int(np.prod(shape)))
    a = rng.random(shape) < 0.5
    b = rng.random(shape) < 0.5
    res = rt.launch(_artifact(rt, op_name, 2), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_logical_compiled"
    out = np.asarray(res["output"])
    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out, ref(a, b))


@pytest.mark.parametrize("shape", [(8, 64), (130,), (3, 5, 7)])
def test_x86_logical_not_matches_numpy(shape):
    rt = _x86_or_skip()
    a = np.random.default_rng(5 + int(np.prod(shape))).random(shape) < 0.5
    res = rt.launch(_artifact(rt, "tessera.logical_not", 1), (a,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_array_equal(
        np.asarray(res["output"]), np.logical_not(a))


def test_x86_logical_normalizes_nonzero():
    rt = _x86_or_skip()
    a = np.array([0, 1, 2, 0, 5], np.uint8)
    b = np.array([0, 0, 3, 7, 0], np.uint8)
    res = rt.launch(_artifact(rt, "tessera.logical_and", 2), (a, b))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_array_equal(
        np.asarray(res["output"]), np.logical_and(a, b))


def test_x86_logical_shape_mismatch_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), bool)
    b = np.zeros((4, 9), bool)
    with pytest.raises(ValueError, match="matching operand shapes"):
        rt._execute_x86_compiled_logical(
            _artifact(rt, "tessera.logical_or", 2), (a, b))


def test_x86_logical_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), bool)
    with pytest.raises(ValueError, match="x86_logical_compiled executor"):
        rt._execute_x86_compiled_logical(
            _artifact(rt, "tessera.softmax", 1), (a,))
