"""x86 row-reduction norm/softmax lanes — the CPU analog of the ROCm
warp-shuffle norm/softmax kernels, loaded from libtessera_x86_elementwise.so.

Unweighted (no γ/β) rmsnorm / layer_norm and numerically-stable softmax over the
last axis, matching the ROCm lanes' op signatures. Reachable through
`runtime.launch()` via `compiler_path="x86_norm_compiled"` (rmsnorm /
rmsnorm_safe / layer_norm) and `"x86_softmax_compiled"` (softmax / softmax_safe).

Validated vs numpy at atol/rtol 2e-5. Skip-clean: lib absent.
"""

from __future__ import annotations

import numpy as np
import pytest

_EPS = 1e-5


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _artifact(rt, op_name, path, kwargs=None):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": path,
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["x"],
                 "kwargs": kwargs or {}}],
    })


def _rmsnorm(x):
    return x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + _EPS)


def _layernorm(x):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + _EPS)


def _softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


@pytest.mark.parametrize("shape", [(4, 16), (8, 64), (3, 5, 33), (2, 257),
                                   (1, 5)])
def test_x86_rmsnorm_matches_numpy(shape):
    rt = _x86_or_skip()
    rng = np.random.default_rng(17 + len(shape) + int(np.prod(shape)))
    x = (rng.standard_normal(shape) * 3).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.rmsnorm", "x86_norm_compiled"), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_norm_compiled"
    out = np.asarray(res["output"]).astype(np.float32)
    np.testing.assert_allclose(out, _rmsnorm(x).astype(np.float32),
                               atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("shape", [(4, 16), (8, 64), (3, 5, 33), (2, 257),
                                   (1, 5)])
def test_x86_layernorm_matches_numpy(shape):
    rt = _x86_or_skip()
    rng = np.random.default_rng(29 + len(shape) + int(np.prod(shape)))
    x = (rng.standard_normal(shape) * 3).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.layer_norm", "x86_norm_compiled"),
                    (x,))
    assert res["ok"] is True, res.get("reason")
    out = np.asarray(res["output"]).astype(np.float32)
    np.testing.assert_allclose(out, _layernorm(x).astype(np.float32),
                               atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("shape", [(4, 16), (8, 64), (3, 5, 33), (2, 257),
                                   (1, 5)])
def test_x86_softmax_matches_numpy(shape):
    rt = _x86_or_skip()
    rng = np.random.default_rng(41 + len(shape) + int(np.prod(shape)))
    x = (rng.standard_normal(shape) * 6).astype(np.float32)
    res = rt.launch(
        _artifact(rt, "tessera.softmax", "x86_softmax_compiled"), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_softmax_compiled"
    out = np.asarray(res["output"]).astype(np.float32)
    np.testing.assert_allclose(out, _softmax(x).astype(np.float32),
                               atol=2e-5, rtol=2e-5)
    # rows sum to 1
    np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-5)


def test_x86_norm_custom_eps():
    rt = _x86_or_skip()
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((4, 32)) * 3).astype(np.float32)
    res = rt.launch(
        _artifact(rt, "tessera.rmsnorm", "x86_norm_compiled", {"eps": 1e-3}),
        (x,))
    assert res["ok"] is True, res.get("reason")
    ref = x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-3)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               ref.astype(np.float32), atol=2e-5, rtol=2e-5)


def test_x86_norm_unknown_op_rejected():
    from tessera import runtime as rt
    x = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="x86_norm_compiled executor"):
        rt._execute_x86_compiled_norm(
            _artifact(rt, "tessera.softmax", "x86_norm_compiled"), (x,))


def test_x86_softmax_axis_rejected():
    rt = _x86_or_skip()
    x = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="axis=-1"):
        rt._execute_x86_compiled_softmax(
            _artifact(rt, "tessera.softmax", "x86_softmax_compiled",
                      {"axis": 0}), (x,))


def test_x86_dynamic_softmax_materializes_noncontiguous_input():
    rt = _x86_or_skip()
    base = np.random.default_rng(67).standard_normal((4, 16)).astype(np.float32)
    x = base[:, ::2]
    assert not x.flags.c_contiguous
    out = rt._execute_x86_compiled_softmax(
        _artifact(rt, "tessera.softmax", "x86_softmax_compiled"), (x,)
    )
    np.testing.assert_allclose(out, _softmax(x), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("shape", [(), (2, 0), (0, 4)])
def test_x86_dynamic_softmax_rejects_invalid_shape_before_native_load(shape):
    from tessera import runtime as rt
    from tessera.compiler.emit.executable_layout import DynamicShapeGuardError

    x = np.empty(shape, np.float32)
    with pytest.raises(DynamicShapeGuardError):
        rt._execute_x86_compiled_softmax(
            _artifact(rt, "tessera.softmax", "x86_softmax_compiled"), (x,)
        )
