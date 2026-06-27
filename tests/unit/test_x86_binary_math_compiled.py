"""x86 transcendental-backed binary lane — pow(a,b) (positive base) and
silu_mul(a,b)=silu(a)*b (SwiGLU gate-multiply), loaded from
libtessera_x86_elementwise.so.

Reachable through `runtime.launch()` via `compiler_path="x86_binary_math_compiled"`;
op names tessera.pow / tessera.silu_mul; f32. Share the AVX-512 exp/log/sigmoid
cores of the transcendental kernel.

Validated vs numpy at atol/rtol 2e-5. Skip-clean: lib absent.
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
        "target": "x86", "compiler_path": "x86_binary_math_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["a", "b"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["a", "b"]}],
    })


@pytest.mark.parametrize("shape", [(16,), (300,), (8, 64), (4, 3, 33), (5,)])
def test_x86_pow_matches_numpy(shape):
    rt = _x86_or_skip()
    rng = np.random.default_rng(31 + len(shape) + int(np.prod(shape)))
    a = (rng.random(shape) * 8 + 0.05).astype(np.float32)   # positive base
    b = (rng.standard_normal(shape) * 3).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.pow"), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_binary_math_compiled"
    out = np.asarray(res["output"]).astype(np.float32)
    np.testing.assert_allclose(out, np.power(a, b).astype(np.float32),
                               atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("shape", [(16,), (300,), (8, 64), (4, 3, 33), (5,)])
def test_x86_silu_mul_matches_numpy(shape):
    rt = _x86_or_skip()
    rng = np.random.default_rng(53 + len(shape) + int(np.prod(shape)))
    a = (rng.standard_normal(shape) * 6).astype(np.float32)
    b = (rng.standard_normal(shape) * 6).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.silu_mul"), (a, b))
    assert res["ok"] is True, res.get("reason")
    ref = (a / (1.0 + np.exp(-a)) * b).astype(np.float32)
    out = np.asarray(res["output"]).astype(np.float32)
    np.testing.assert_allclose(out, ref, atol=2e-5, rtol=2e-5)


def test_x86_binary_math_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.float32)
    art = rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_binary_math_compiled",
        "arg_names": ["a", "b"], "output_name": "o",
        "ops": [{"op_name": "tessera.softmax", "result": "o",
                 "operands": ["a", "b"]}],
    })
    with pytest.raises(ValueError, match="x86_binary_math_compiled executor"):
        rt._execute_x86_compiled_binary_math(art, (a, a))


def test_x86_binary_math_shape_mismatch_rejected():
    rt = _x86_or_skip()
    a = np.zeros((4, 8), np.float32)
    b = np.zeros((4, 9), np.float32)
    with pytest.raises(ValueError, match="matching operand shapes"):
        rt._execute_x86_compiled_binary_math(
            _artifact(rt, "tessera.pow"), (a, b))
