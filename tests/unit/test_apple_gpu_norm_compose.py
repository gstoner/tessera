"""Batch 3c — group / instance / weight norm on Apple GPU via composition.

These three norms have no dedicated kernel: each folds its normalized axes to
the last axis and runs the MPSGraph layer_norm row-op (mean/var reduction) on
the GPU, with the optional per-channel affine as a GPU mul/add and weight_norm's
||w|| via the GPU sum-reduce lane.  metal_runtime.
"""

import numpy as np
import pytest

import tessera as ts
import tessera.nn.functional as F
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import runtime as _runtime
from tessera.compiler import driver as _driver

R = _runtime
_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.hardware_apple_gpu


def test_in_envelope():
    for op in ("tessera.group_norm", "tessera.instance_norm", "tessera.weight_norm"):
        assert op in _driver._APPLE_GPU_RUNTIME_OPS, op
        assert op in _runtime._APPLE_GPU_RUNTIME_OPS, op
        assert op in _driver._APPLE_GPU_NORM_COMPOSE_OPS, op


def test_driver_runtime_norm_envelopes_agree():
    assert _driver._APPLE_GPU_NORM_COMPOSE_OPS == _runtime._APPLE_GPU_NORM_COMPOSE_OPS


@gpu
@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (1, 4, 8), (3, 6, 2, 2)])
def test_instance_norm(shape):
    x = np.random.default_rng(hash(shape) % 99).standard_normal(shape).astype(np.float32)
    out = R._apple_gpu_dispatch_norm("tessera.instance_norm", [x], {}, np)
    np.testing.assert_allclose(np.asarray(out), F.instance_norm(x), atol=1e-4)


@gpu
def test_instance_norm_affine():
    x = np.random.default_rng(1).standard_normal((2, 3, 4, 5)).astype(np.float32)
    w = np.random.default_rng(2).standard_normal(3).astype(np.float32)
    b = np.random.default_rng(3).standard_normal(3).astype(np.float32)
    out = R._apple_gpu_dispatch_norm("tessera.instance_norm", [x, w, b], {}, np)
    np.testing.assert_allclose(np.asarray(out), F.instance_norm(x, weight=w, bias=b), atol=1e-4)


@gpu
@pytest.mark.parametrize("groups", [1, 3, 6])
def test_group_norm(groups):
    x = np.random.default_rng(groups).standard_normal((2, 6, 4, 4)).astype(np.float32)
    out = R._apple_gpu_dispatch_norm("tessera.group_norm", [x], {"num_groups": groups}, np)
    np.testing.assert_allclose(np.asarray(out), F.group_norm(x, groups), atol=1e-4)


@gpu
def test_group_norm_affine():
    x = np.random.default_rng(4).standard_normal((2, 6, 4, 4)).astype(np.float32)
    w = np.random.default_rng(5).standard_normal(6).astype(np.float32)
    b = np.random.default_rng(6).standard_normal(6).astype(np.float32)
    out = R._apple_gpu_dispatch_norm("tessera.group_norm", [x, w, b], {"num_groups": 3}, np)
    np.testing.assert_allclose(np.asarray(out), F.group_norm(x, 3, weight=w, bias=b), atol=1e-4)


@gpu
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_weight_norm(axis):
    w = np.random.default_rng(abs(axis) + 7).standard_normal((4, 6, 3)).astype(np.float32)
    out = R._apple_gpu_dispatch_norm("tessera.weight_norm", [w], {"axis": axis}, np)
    np.testing.assert_allclose(np.asarray(out), F.weight_norm(w, axis=axis), atol=1e-4)


@gpu
def test_instance_norm_jit_metal_runtime():
    x = np.random.default_rng(8).standard_normal((2, 3, 4, 5)).astype(np.float32)

    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.instance_norm(x)

    np.testing.assert_allclose(np.asarray(f(x)), F.instance_norm(x), atol=1e-4)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


@gpu
def test_weight_norm_jit_metal_runtime():
    w = np.random.default_rng(9).standard_normal((4, 6)).astype(np.float32)

    @ts.jit(target="apple_gpu")
    def f(w):
        return ts.ops.weight_norm(w)

    np.testing.assert_allclose(np.asarray(f(w)), F.weight_norm(w, axis=0), atol=1e-4)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
