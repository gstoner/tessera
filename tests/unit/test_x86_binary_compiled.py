"""Hand-written AVX-512 2-operand binary-arithmetic kernel on x86 — the CPU
analog of the ROCm device_verified_jit binary lane, for the direct-intrinsic subset.

`tessera_x86_avx512_binary_f32` is exported by libtessera_x86_elementwise.so; the
Python runtime ctypes-loads it and calls it from `runtime.launch()` via
`compiler_path="x86_binary_compiled"`. Covers sub/div/maximum/minimum
(NaN-propagating max/min). `pow` is transcendental → numpy-reference. f32 only.

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
        "target": "x86", "compiler_path": "x86_binary_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["a", "b"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["a", "b"]}],
    })


def _s_div(rng, shp):
    a = (rng.standard_normal(shp) * 2.0).astype(np.float32)
    b = (rng.random(shp) * 3.0 + 0.25).astype(np.float32)
    b *= rng.choice([-1.0, 1.0], size=shp).astype(np.float32)
    return a, b


def _s_any(rng, shp):
    return ((rng.standard_normal(shp) * 2.0).astype(np.float32),
            (rng.standard_normal(shp) * 2.0).astype(np.float32))


_CASES = {
    "tessera.sub": (lambda a, b: a - b, _s_any),
    "tessera.div": (lambda a, b: a / b, _s_div),
    "tessera.maximum": (np.maximum, _s_any),
    "tessera.minimum": (np.minimum, _s_any),
}


@pytest.mark.parametrize("op_name", list(_CASES))
@pytest.mark.parametrize("shape", [(8, 64), (130,), (3, 5, 7)])
def test_x86_binary_matches_numpy(op_name, shape):
    rt = _x86_or_skip()
    ref, sampler = _CASES[op_name]
    rng = np.random.default_rng(29 + len(shape) + int(np.prod(shape)))
    a, b = sampler(rng, shape)
    res = rt.launch(_artifact(rt, op_name), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_binary_compiled"
    out = np.asarray(res["output"]).astype(np.float32)
    np.testing.assert_allclose(out, ref(a, b), atol=2e-5, rtol=2e-5)


def test_x86_binary_max_min_nan_propagating():
    rt = _x86_or_skip()
    a = np.array([1.0, np.nan, 3.0, -1.0], np.float32)
    b = np.array([2.0, 5.0, np.nan, -2.0], np.float32)
    mx = rt.launch(_artifact(rt, "tessera.maximum"), (a, b))
    mn = rt.launch(_artifact(rt, "tessera.minimum"), (a, b))
    assert mx["ok"] is True and mn["ok"] is True
    np.testing.assert_array_equal(
        np.asarray(mx["output"]).astype(np.float32), np.maximum(a, b))
    np.testing.assert_array_equal(
        np.asarray(mn["output"]).astype(np.float32), np.minimum(a, b))


def test_x86_binary_shape_mismatch_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.float32)
    b = np.zeros((4, 9), np.float32)
    with pytest.raises(ValueError, match="matching operand shapes"):
        rt._execute_x86_compiled_binary(_artifact(rt, "tessera.sub"), (a, b))


def test_x86_binary_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.float32)
    b = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="x86_binary_compiled executor"):
        rt._execute_x86_compiled_binary(_artifact(rt, "tessera.pow"), (a, b))


def test_x86_binary_rejects_non_f32():
    rt = _x86_or_skip()
    a = np.zeros((4, 8), np.float64)
    b = np.zeros((4, 8), np.float64)
    with pytest.raises(ValueError, match="f32 only"):
        rt._execute_x86_compiled_binary(_artifact(rt, "tessera.sub"), (a, b))
