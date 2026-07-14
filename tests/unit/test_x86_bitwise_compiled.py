"""Hand-written AVX-512 bitwise kernel on x86 — the CPU analog of the ROCm
compiled bitwise lane.

`tessera_x86_avx512_bitwise_i32` is exported by libtessera_x86_elementwise.so;
the Python runtime ctypes-loads it and calls it from `runtime.launch()` via
`compiler_path="x86_bitwise_compiled"`. Covers bitwise_and/or/xor (binary) +
bitwise_not (unary) over i32 integers, acting on the full bit pattern.

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
        "target": "x86", "compiler_path": "x86_bitwise_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": operands, "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": operands}],
    })


_BINARY = {
    "tessera.bitwise_and": np.bitwise_and,
    "tessera.bitwise_or": np.bitwise_or,
    "tessera.bitwise_xor": np.bitwise_xor,
}


@pytest.mark.parametrize("op_name", list(_BINARY))
@pytest.mark.parametrize("shape", [(8, 64), (130,), (3, 5, 7)])
def test_x86_bitwise_binary_matches_numpy(op_name, shape):
    rt = _x86_or_skip()
    ref = _BINARY[op_name]
    rng = np.random.default_rng(61 + len(shape) + int(np.prod(shape)))
    a = rng.integers(-(1 << 20), 1 << 20, size=shape, dtype=np.int32)
    b = rng.integers(-(1 << 20), 1 << 20, size=shape, dtype=np.int32)
    res = rt.launch(_artifact(rt, op_name, 2), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_bitwise_compiled"
    out = np.asarray(res["output"]).astype(np.int32)
    np.testing.assert_array_equal(out, ref(a, b))


@pytest.mark.parametrize("shape", [(8, 64), (130,), (3, 5, 7)])
def test_x86_bitwise_not_matches_numpy(shape):
    rt = _x86_or_skip()
    a = np.random.default_rng(11 + int(np.prod(shape))).integers(
        -(1 << 20), 1 << 20, size=shape, dtype=np.int32)
    res = rt.launch(_artifact(rt, "tessera.bitwise_not", 1), (a,))
    assert res["ok"] is True, res.get("reason")
    out = np.asarray(res["output"]).astype(np.int32)
    np.testing.assert_array_equal(out, np.bitwise_not(a))


def test_x86_bitwise_shape_mismatch_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.int32)
    b = np.zeros((4, 9), np.int32)
    with pytest.raises(ValueError, match="matching operand shapes"):
        rt._execute_x86_compiled_bitwise(
            _artifact(rt, "tessera.bitwise_or", 2), (a, b))


def test_x86_bitwise_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.int32)
    with pytest.raises(ValueError, match="x86_bitwise_compiled executor"):
        rt._execute_x86_compiled_bitwise(
            _artifact(rt, "tessera.softmax", 1), (a,))
