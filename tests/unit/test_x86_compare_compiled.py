"""Hand-written AVX-512 comparison kernel on x86 — the CPU analog of the ROCm
compiled compare lane.

The f32/i32/u32 comparison entry points are exported by libtessera_x86_elementwise.so;
the Python runtime ctypes-loads it and calls it from `runtime.launch()` via
`compiler_path="x86_compare_compiled"`. Covers eq/ne/lt/le/gt/ge; f32/i32/u32 inputs,
bool output. NaN semantics match numpy (ordered everywhere except ne).

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


def _artifact(rt, op_name):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_compare_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["a", "b"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["a", "b"]}],
    })


_CASES = {
    "tessera.eq": np.equal, "tessera.ne": np.not_equal,
    "tessera.lt": np.less, "tessera.le": np.less_equal,
    "tessera.gt": np.greater, "tessera.ge": np.greater_equal,
}


@pytest.mark.parametrize("op_name", list(_CASES))
@pytest.mark.parametrize("shape", [(8, 64), (130,), (3, 5, 7)])
def test_x86_compare_matches_numpy(op_name, shape):
    rt = _x86_or_skip()
    ref = _CASES[op_name]
    rng = np.random.default_rng(37 + len(shape) + int(np.prod(shape)))
    a = (rng.standard_normal(shape) * 1.5).astype(np.float32)
    b = a.copy()
    mask = rng.random(shape) < 0.5
    b[mask] = (rng.standard_normal(int(mask.sum())) * 1.5).astype(np.float32)
    res = rt.launch(_artifact(rt, op_name), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_compare_compiled"
    out = np.asarray(res["output"])
    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out, ref(a, b))


def test_x86_compare_nan_semantics():
    rt = _x86_or_skip()
    a = np.array([1.0, np.nan, np.nan, 2.0], np.float32)
    b = np.array([1.0, 1.0, np.nan, 3.0], np.float32)
    for op_name, ref in _CASES.items():
        res = rt.launch(_artifact(rt, op_name), (a, b))
        assert res["ok"] is True, res.get("reason")
        np.testing.assert_array_equal(
            np.asarray(res["output"]), ref(a, b), err_msg=f"{op_name} NaN")


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
@pytest.mark.parametrize("op_name", list(_CASES))
def test_x86_integer_compare_signedness_matches_numpy(dtype, op_name):
    rt = _x86_or_skip()
    if dtype == np.int32:
        a = np.array([-2**31, -7, -1, 0, 1, 7, 2**31 - 1], dtype=dtype)
        b = np.array([0, -8, 1, 0, -1, 8, 2**31 - 1], dtype=dtype)
    else:
        a = np.array([0, 1, 7, 2**31, 2**32 - 1], dtype=dtype)
        b = np.array([1, 0, 8, 2**31 - 1, 2**32 - 1], dtype=dtype)
    out = rt.launch(_artifact(rt, op_name), (a, b))["output"]
    np.testing.assert_array_equal(out, _CASES[op_name](a, b))


def test_x86_compare_shape_mismatch_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.float32)
    b = np.zeros((4, 9), np.float32)
    with pytest.raises(ValueError, match="matching operand shapes"):
        rt._execute_x86_compiled_compare(_artifact(rt, "tessera.lt"), (a, b))


def test_x86_compare_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.float32)
    b = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="x86_compare_compiled executor"):
        rt._execute_x86_compiled_compare(_artifact(rt, "tessera.softmax"), (a, b))


def test_x86_compare_rejects_non_f32():
    rt = _x86_or_skip()
    a = np.zeros((4, 8), np.float64)
    b = np.zeros((4, 8), np.float64)
    with pytest.raises(ValueError, match="matching f32/i32/u32"):
        rt._execute_x86_compiled_compare(_artifact(rt, "tessera.eq"), (a, b))
