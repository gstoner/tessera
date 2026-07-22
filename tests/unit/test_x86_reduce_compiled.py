"""Hand-written AVX-512 row-reduction kernel on x86 — the CPU analog of the ROCm
compiled reduce lane.

`tessera_x86_avx512_reduce_f32` (sum/mean/max/min) is exported by
libtessera_x86_elementwise.so; the Python runtime ctypes-loads it and calls it
from `runtime.launch()` via `compiler_path="x86_reduce_compiled"`. An arbitrary
`axis` folds to a [outer, inner] last-axis reduction; `keepdims` supported. f32
only; max/min are NaN-propagating (numpy `np.amax`/`np.amin`).

Validated vs numpy. Skip-clean: libtessera_x86_elementwise.so not built
(cmake target `tessera_x86_elementwise`, or set TESSERA_X86_ELEMENTWISE_LIB).
"""

from __future__ import annotations

import numpy as np
import pytest


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _artifact(rt, op_name, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_reduce_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["x"],
                 "kwargs": kwargs}],
    })


_REFS = {
    "tessera.sum": np.sum, "tessera.mean": np.mean,
    "tessera.max": np.max, "tessera.amax": np.max,
    "tessera.min": np.min, "tessera.amin": np.min,
}


@pytest.mark.parametrize("op_name", list(_REFS))
@pytest.mark.parametrize("shape,axis", [
    ((8, 64), -1), ((8, 64), 0), ((130,), None),
    ((3, 5, 7), -1), ((3, 5, 7), (1, 2)), ((4, 6), None),
    ((), None),
])
def test_x86_reduce_matches_numpy(op_name, shape, axis):
    rt = _x86_or_skip()
    ref = _REFS[op_name]
    rng = np.random.default_rng(17 + len(shape) + int(np.prod(shape)))
    x = (rng.standard_normal(shape) * 2.0).astype(np.float32)
    res = rt.launch(_artifact(rt, op_name, {"axis": axis}), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_reduce_compiled"
    out = np.asarray(res["output"]).astype(np.float32)
    expect = ref(x, axis=axis).astype(np.float32)
    np.testing.assert_allclose(out, expect, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("op_name", ["tessera.sum", "tessera.max"])
def test_x86_reduce_keepdims(op_name):
    rt = _x86_or_skip()
    ref = _REFS[op_name]
    x = np.random.default_rng(3).standard_normal((4, 9)).astype(np.float32)
    res = rt.launch(_artifact(rt, op_name, {"axis": -1, "keepdims": True}), (x,))
    assert res["ok"] is True, res.get("reason")
    out = np.asarray(res["output"]).astype(np.float32)
    expect = ref(x, axis=-1, keepdims=True).astype(np.float32)
    assert out.shape == expect.shape
    np.testing.assert_allclose(out, expect, atol=2e-4, rtol=2e-4)


def test_x86_reduce_max_nan_propagates():
    rt = _x86_or_skip()
    x = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], np.float32)
    res = rt.launch(_artifact(rt, "tessera.max", {"axis": -1}), (x,))
    assert res["ok"] is True, res.get("reason")
    out = np.asarray(res["output"]).astype(np.float32)
    np.testing.assert_array_equal(out, np.max(x, axis=-1))  # row0 -> NaN


def test_x86_reduce_unknown_op_rejected():
    from tessera import runtime as rt
    x = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="x86_reduce_compiled executor"):
        rt._execute_x86_compiled_reduce(
            _artifact(rt, "tessera.softmax", {"axis": -1}), (x,))


def test_x86_reduce_rejects_non_f32():
    rt = _x86_or_skip()
    x = np.zeros((4, 8), np.float64)
    with pytest.raises(ValueError, match="f32 only"):
        rt._execute_x86_compiled_reduce(
            _artifact(rt, "tessera.sum", {"axis": -1}), (x,))


@pytest.mark.parametrize("axis", [(), (1, 1), 2, -3, (0, "1"), True])
def test_x86_dynamic_reduce_rejects_invalid_axes_before_native_load(axis):
    from tessera import runtime as rt
    from tessera.compiler.emit.executable_layout import DynamicShapeGuardError

    x = np.zeros((4, 8), np.float32)
    with pytest.raises(DynamicShapeGuardError, match="axes"):
        rt._execute_x86_compiled_reduce(
            _artifact(rt, "tessera.sum", {"axis": axis}), (x,))


def test_x86_dynamic_last_axis_reduce_rejects_empty_extent_before_native_load():
    from tessera import runtime as rt
    from tessera.compiler.emit.executable_layout import DynamicShapeGuardError

    x = np.zeros((4, 0), np.float32)
    with pytest.raises(DynamicShapeGuardError, match="must be positive"):
        rt._execute_x86_compiled_reduce(
            _artifact(rt, "tessera.sum", {"axis": -1}), (x,))
