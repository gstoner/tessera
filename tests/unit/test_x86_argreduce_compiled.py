"""x86 argmax/argmin — the CPU analog of the ROCm warp-shuffle arg-reduce lane,
loaded from libtessera_x86_elementwise.so.

Reachable through `runtime.launch()` via `compiler_path="x86_argreduce_compiled"`;
op names tessera.argmax / argmin; f32 input, i32 index output; numpy
first-occurrence tie-break.

Validated vs np.argmax/argmin. Skip-clean: libtessera_x86_elementwise.so absent.
"""

from __future__ import annotations

import numpy as np
import pytest


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _artifact(rt, op_name, axis):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_argreduce_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["x"],
                 "kwargs": {"axis": axis}}],
    })


_REFS = {"tessera.argmax": np.argmax, "tessera.argmin": np.argmin}


@pytest.mark.parametrize("op_name", list(_REFS))
@pytest.mark.parametrize("shape,axis", [
    ((8, 64), -1), ((8, 64), 0), ((130,), None), ((3, 5, 7), -1),
    ((3, 5, 7), 1), ((4, 6), None),
])
def test_x86_argreduce_matches_numpy(op_name, shape, axis):
    rt = _x86_or_skip()
    ref = _REFS[op_name]
    rng = np.random.default_rng(97 + len(shape) + int(np.prod(shape)))
    flat = rng.permutation(int(np.prod(shape))).astype(np.float32)
    x = flat.reshape(shape)
    res = rt.launch(_artifact(rt, op_name, axis), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_argreduce_compiled"
    out = np.asarray(res["output"]).astype(np.int64)
    np.testing.assert_array_equal(out, ref(x, axis=axis).astype(np.int64))


def test_x86_argreduce_first_occurrence():
    rt = _x86_or_skip()
    x = np.array([[1.0, 5.0, 5.0, 2.0, 5.0]], np.float32)
    res = rt.launch(_artifact(rt, "tessera.argmax", -1), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_array_equal(
        np.asarray(res["output"]).astype(np.int64), np.argmax(x, axis=-1))


def test_x86_argreduce_unknown_op_rejected():
    from tessera import runtime as rt
    x = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="x86_argreduce_compiled executor"):
        rt._execute_x86_compiled_argreduce(
            _artifact(rt, "tessera.softmax", -1), (x,))
