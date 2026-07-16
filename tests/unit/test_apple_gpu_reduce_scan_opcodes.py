"""Batch 2 — reduce/scan opcode completions on Apple GPU (metal_runtime).

Closes the remaining MPSGraph reduce/scan opcode slots: logsumexp (reduce op 7,
stable log-sum-exp) and cummax/cummin (scan ops 2/3 via cumulativeMaximum/
Minimum). Kernel numerics via the dispatcher; metal_runtime classification via
literal @jit calls.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import runtime as _runtime
from tessera.compiler import driver as _driver

O = ts.ops
_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.hardware_apple_gpu


def test_in_envelope():
    for op in ("tessera.logsumexp", "tessera.cummax", "tessera.cummin"):
        assert op in _driver._APPLE_GPU_RUNTIME_OPS, op
        assert op in _runtime._APPLE_GPU_RUNTIME_OPS, op


@gpu
@pytest.mark.parametrize("axis", [-1, 0, 1])
def test_logsumexp_kernel(axis):
    x = np.random.default_rng(axis + 1).standard_normal((4, 6)).astype(np.float32)
    out = _runtime._apple_gpu_dispatch_reduce("tessera.logsumexp", [x], {"axis": axis}, np)
    m = x.max(axis=axis, keepdims=True)
    ref = (np.log(np.sum(np.exp(x - m), axis=axis)) + np.squeeze(m, axis=axis))
    np.testing.assert_allclose(np.asarray(out).ravel(), ref.ravel(), atol=1e-4)


@gpu
@pytest.mark.parametrize("name,fn", [("cummax", np.maximum.accumulate),
                                     ("cummin", np.minimum.accumulate)])
@pytest.mark.parametrize("axis", [-1, 0])
def test_cumscan_kernel(name, fn, axis):
    x = np.random.default_rng(hash(name) % 99 + axis).standard_normal((4, 6)).astype(np.float32)
    out = _runtime._apple_gpu_dispatch_reduce(f"tessera.{name}", [x], {"axis": axis}, np)
    np.testing.assert_allclose(np.asarray(out), fn(x, axis=axis), atol=1e-5)


@gpu
def test_logsumexp_jit_metal_runtime():
    x = np.random.default_rng(7).standard_normal((4, 6)).astype(np.float32)

    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.logsumexp(x, axis=-1)

    m = x.max(-1, keepdims=True)
    ref = np.log(np.sum(np.exp(x - m), -1)) + np.squeeze(m, -1)
    np.testing.assert_allclose(np.asarray(f(x)).ravel(), ref, atol=1e-4)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


@gpu
def test_cummax_jit_metal_runtime():
    x = np.random.default_rng(8).standard_normal((4, 6)).astype(np.float32)

    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.cummax(x, axis=-1)

    np.testing.assert_allclose(np.asarray(f(x)), np.maximum.accumulate(x, axis=-1), atol=1e-5)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
