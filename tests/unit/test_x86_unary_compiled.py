"""Hand-written AVX-512 unary-math kernel on x86 — the CPU analog of the ROCm
compiled unary lane, for the direct-intrinsic algebraic + rounding subset.

`tessera_x86_avx512_unary_f32` is exported by libtessera_x86_elementwise.so; the
Python runtime ctypes-loads it and calls it from `runtime.launch()` via
`compiler_path="x86_unary_compiled"`. Covers sqrt/rsqrt/reciprocal/abs/sign +
floor/ceil/trunc/round (numpy round-half-to-even). Transcendentals stay
numpy-reference on CPU. f32 only.

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
        "target": "x86", "compiler_path": "x86_unary_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["x"]}],
    })


_POS = lambda rng, shp: (rng.random(shp) * 4.0 + 0.05).astype(np.float32)
_ANY = lambda rng, shp: (rng.standard_normal(shp) * 3.0).astype(np.float32)

_CASES = {
    "tessera.sqrt": (np.sqrt, _POS),
    "tessera.rsqrt": (lambda x: 1.0 / np.sqrt(x), _POS),
    "tessera.reciprocal": (np.reciprocal, _POS),
    "tessera.absolute": (np.abs, _ANY),
    "tessera.sign": (np.sign, _ANY),
    "tessera.floor": (np.floor, _ANY),
    "tessera.ceil": (np.ceil, _ANY),
    "tessera.trunc": (np.trunc, _ANY),
    "tessera.round": (np.round, _ANY),   # round-half-to-even
}


@pytest.mark.parametrize("op_name", list(_CASES))
@pytest.mark.parametrize("shape", [(8, 64), (130,), (3, 5, 7)])
def test_x86_unary_matches_numpy(op_name, shape):
    rt = _x86_or_skip()
    ref, sampler = _CASES[op_name]
    rng = np.random.default_rng(19 + len(shape) + int(np.prod(shape)))
    x = sampler(rng, shape)
    res = rt.launch(_artifact(rt, op_name), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_unary_compiled"
    out = np.asarray(res["output"]).astype(np.float32)
    expect = np.asarray(ref(x)).astype(np.float32)
    np.testing.assert_allclose(out, expect, atol=2e-5, rtol=2e-5)


def test_x86_unary_abs_alias():
    rt = _x86_or_skip()
    x = np.array([-2.0, 0.0, 3.0], np.float32)
    res = rt.launch(_artifact(rt, "tessera.abs"), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_array_equal(
        np.asarray(res["output"]).astype(np.float32), np.abs(x))


def test_x86_unary_round_half_to_even():
    rt = _x86_or_skip()
    x = np.array([0.5, 1.5, 2.5, 3.5, -0.5, -1.5], np.float32)
    res = rt.launch(_artifact(rt, "tessera.round"), (x,))
    assert res["ok"] is True, res.get("reason")
    # numpy round is banker's rounding: 0.5->0, 1.5->2, 2.5->2, 3.5->4
    np.testing.assert_array_equal(
        np.asarray(res["output"]).astype(np.float32), np.round(x))


def test_x86_unary_unknown_op_rejected():
    from tessera import runtime as rt
    x = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="x86_unary_compiled executor"):
        rt._execute_x86_compiled_unary(_artifact(rt, "tessera.exp"), (x,))


def test_x86_unary_rejects_non_f32():
    rt = _x86_or_skip()
    x = np.zeros((4, 8), np.float64)
    with pytest.raises(ValueError, match="f32 only"):
        rt._execute_x86_compiled_unary(_artifact(rt, "tessera.sqrt"), (x,))
