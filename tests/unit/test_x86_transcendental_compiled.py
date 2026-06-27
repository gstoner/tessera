"""x86 transcendental / activation lane — the CPU analog of the ROCm
math->ROCDL unary/activation kernels, loaded from libtessera_x86_elementwise.so.

Reachable through `runtime.launch()` via
`compiler_path="x86_transcendental_compiled"`; op names tessera.exp / log /
tanh / sigmoid / silu / gelu / erf / softplus / expm1 / log1p; f32. Cephes
exp/log minimax cores (~1 ulp) + Abramowitz-Stegun erf; activations compose.
gelu uses the tanh approximation (matching the ROCm activation reference).

Validated vs numpy at atol/rtol 2e-5. Skip-clean: lib absent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

_np_erf = np.vectorize(math.erf, otypes=[np.float64])


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _artifact(rt, op_name):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_transcendental_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["x"]}],
    })


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _gelu_tanh(x):
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def _silu(x):
    return x * _sigmoid(x)


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _erf(x):
    return _np_erf(x).astype(np.float32)


def _erfc(x):
    return (1.0 - _np_erf(x)).astype(np.float32)


def _normal(scale):
    return lambda r, s: (r.standard_normal(s) * scale).astype(np.float32)


def _pos(scale, base=0.0):
    return lambda r, s: (r.random(s) * scale + base).astype(np.float32)


def _uniform(lo, hi):
    return lambda r, s: (r.random(s) * (hi - lo) + lo).astype(np.float32)


# op -> (numpy reference, input sampler). Domains avoid singularities
# (tan near ±π/2) and pi/2-asin cancellation (acos near ±1).
_REFS = {
    "tessera.exp": (np.exp, _normal(3)),
    "tessera.log": (np.log, _pos(50, 1e-3)),
    "tessera.tanh": (np.tanh, _normal(4)),
    "tessera.sigmoid": (_sigmoid, _normal(6)),
    "tessera.silu": (_silu, _normal(6)),
    "tessera.gelu": (_gelu_tanh, _normal(3)),
    "tessera.erf": (_erf, _normal(2)),
    "tessera.softplus": (_softplus, _normal(8)),
    "tessera.expm1": (np.expm1, _normal(3)),
    "tessera.log1p": (np.log1p, _pos(50)),
    "tessera.cos": (np.cos, _normal(4)),
    "tessera.tan": (np.tan, _uniform(-1.3, 1.3)),
    "tessera.sinh": (np.sinh, _normal(4)),
    "tessera.cosh": (np.cosh, _normal(4)),
    "tessera.asin": (np.arcsin, _uniform(-0.99, 0.99)),
    "tessera.acos": (np.arccos, _uniform(-0.9, 0.9)),
    "tessera.atan": (np.arctan, _normal(20)),
    "tessera.erfc": (_erfc, _normal(2)),
}


@pytest.mark.parametrize("op_name", list(_REFS))
@pytest.mark.parametrize("shape", [(16,), (300,), (8, 64), (4, 3, 33), (5,)])
def test_x86_transcendental_matches_numpy(op_name, shape):
    rt = _x86_or_skip()
    ref, sampler = _REFS[op_name]
    rng = np.random.default_rng(11 + len(shape) + int(np.prod(shape)))
    x = sampler(rng, shape)
    res = rt.launch(_artifact(rt, op_name), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_transcendental_compiled"
    out = np.asarray(res["output"]).astype(np.float32)
    np.testing.assert_allclose(out, ref(x).astype(np.float32),
                               atol=2e-5, rtol=2e-5)


def test_x86_transcendental_unknown_op_rejected():
    from tessera import runtime as rt
    x = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="x86_transcendental_compiled executor"):
        rt._execute_x86_compiled_transcendental(
            _artifact(rt, "tessera.softmax"), (x,))


def test_x86_transcendental_rejects_non_f32():
    rt = _x86_or_skip()
    x = np.zeros((4, 8), np.float64)
    with pytest.raises(ValueError, match="f32 only"):
        rt._execute_x86_compiled_transcendental(
            _artifact(rt, "tessera.exp"), (x,))
