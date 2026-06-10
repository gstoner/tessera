"""Batch 2 remainder — predicate/logical/bitwise/compose ops on Apple GPU.

Closes the remaining numeric_helper/logical/reduction eligible ops:
  * unary predicates  isfinite/isinf/isnan + logical_not + bitwise_not (→ f32 mask)
  * binary logical     logical_and/or/xor  + bitwise and/or/xor (int32)
  * reduce             max/min (reduce-max/min aliases)
  * compose            clamp/clip = max(min(x,hi),lo); where = c*a + (1-c)*b
All execute on the GPU (metal_runtime) — predicates/logical via the MPSGraph
opcode lane, compose via chained GPU binary ops.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import runtime as _runtime
from tessera.compiler import driver as _driver

R = _runtime
_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")


def test_in_envelope():
    for op in ("isfinite", "isinf", "isnan", "logical_not", "bitwise_not",
               "logical_and", "logical_or", "logical_xor",
               "bitwise_and", "bitwise_or", "bitwise_xor",
               "max", "min", "clamp", "clip", "where"):
        g = f"tessera.{op}"
        assert g in _driver._APPLE_GPU_RUNTIME_OPS, op
        assert g in _runtime._APPLE_GPU_RUNTIME_OPS, op


@gpu
@pytest.mark.parametrize("name,ref", [
    ("isfinite", np.isfinite), ("isinf", np.isinf), ("isnan", np.isnan)])
def test_predicates(name, ref):
    x = np.array([[1.0, np.inf, np.nan, -2.0, 0.0, -np.inf]], np.float32)
    out = R._apple_gpu_dispatch_unary(f"tessera.{name}", [x], np)
    np.testing.assert_array_equal(np.asarray(out), ref(x).astype(np.float32))


@gpu
def test_logical_not_and_bitwise_not():
    x = np.array([0.0, 1.0, 2.0, 0.0], np.float32)
    np.testing.assert_array_equal(
        np.asarray(R._apple_gpu_dispatch_unary("tessera.logical_not", [x], np)),
        (x == 0).astype(np.float32))
    xi = np.array([0.0, 1.0, 5.0, 255.0], np.float32)
    np.testing.assert_array_equal(
        np.asarray(R._apple_gpu_dispatch_unary("tessera.bitwise_not", [xi], np)),
        (~xi.astype(np.int32)).astype(np.float32))


@gpu
@pytest.mark.parametrize("name,ref", [
    ("logical_and", lambda a, b: (a != 0) & (b != 0)),
    ("logical_or", lambda a, b: (a != 0) | (b != 0)),
    ("logical_xor", lambda a, b: (a != 0) ^ (b != 0)),
    ("bitwise_and", lambda a, b: a.astype(np.int32) & b.astype(np.int32)),
    ("bitwise_or", lambda a, b: a.astype(np.int32) | b.astype(np.int32)),
    ("bitwise_xor", lambda a, b: a.astype(np.int32) ^ b.astype(np.int32))])
def test_binary_logical_bitwise(name, ref):
    a = np.array([0.0, 1.0, 5.0, 3.0, 12.0], np.float32)
    b = np.array([1.0, 1.0, 0.0, 6.0, 10.0], np.float32)
    out = R._apple_gpu_dispatch_mpsgraph_binary(f"tessera.{name}", [a, b], {}, np)
    np.testing.assert_array_equal(np.asarray(out).astype(np.float32),
                                  np.asarray(ref(a, b)).astype(np.float32))


@gpu
@pytest.mark.parametrize("name,fn", [("max", np.max), ("min", np.min)])
@pytest.mark.parametrize("axis", [-1, 0])
def test_reduce_max_min(name, fn, axis):
    # hash() is per-process randomized and axis can be -1 — keep the seed
    # deterministic and non-negative.
    x = np.random.default_rng(len(name) + axis + 1).standard_normal((4, 6)).astype(np.float32)
    out = R._apple_gpu_dispatch_reduce(f"tessera.{name}", [x], {"axis": axis}, np)
    np.testing.assert_allclose(np.asarray(out).ravel(), fn(x, axis=axis).ravel(), atol=1e-5)


@gpu
def test_clamp_and_clip():
    x = np.random.default_rng(1).standard_normal((4, 6)).astype(np.float32)
    np.testing.assert_allclose(
        np.asarray(R._apple_gpu_dispatch_clamp("tessera.clamp", [x], {"min": -0.5, "max": 0.5}, np)),
        np.clip(x, -0.5, 0.5), atol=1e-5)
    # one-sided
    np.testing.assert_allclose(
        np.asarray(R._apple_gpu_dispatch_clamp("tessera.clip", [x], {"min": 0.0}, np)),
        np.clip(x, 0.0, None), atol=1e-5)


@gpu
def test_where():
    rng = np.random.default_rng(2)
    c = (rng.random((4, 6)) > 0.5).astype(np.float32)
    a = rng.standard_normal((4, 6)).astype(np.float32)
    b = rng.standard_normal((4, 6)).astype(np.float32)
    out = R._apple_gpu_dispatch_where([c, a, b], np)
    np.testing.assert_allclose(np.asarray(out), np.where(c != 0, a, b), atol=1e-5)


@gpu
def test_jit_metal_runtime():
    x = np.array([[1.0, np.nan, 3.0]], np.float32)

    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.isnan(x)

    np.testing.assert_array_equal(np.asarray(f(x)), np.isnan(x).astype(np.float32))
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"

    rng = np.random.default_rng(3)
    c = (rng.random((4, 6)) > 0.5).astype(np.float32)
    a = rng.standard_normal((4, 6)).astype(np.float32)
    b = rng.standard_normal((4, 6)).astype(np.float32)

    @ts.jit(target="apple_gpu")
    def w(c, a, b):
        return ts.ops.where(c, a, b)

    np.testing.assert_allclose(np.asarray(w(c, a, b)), np.where(c != 0, a, b), atol=1e-5)
    assert w.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
