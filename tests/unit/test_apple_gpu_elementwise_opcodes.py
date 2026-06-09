"""Batch 1 — elementwise opcode lanes on Apple GPU (metal_runtime).

Routes the float-output elementwise math + comparison primitives through the
MPSGraph unary/binary opcode lane so a `@jit(target="apple_gpu")` call executes
them natively (execution_mode == "metal_runtime"). Covers the 18 new unary math
ops, the binary math ops, and 6 comparison ops (→ f32 0/1 mask).

Kernel numerics are checked via the runtime dispatcher directly (the GPU path);
end-to-end metal_runtime classification is checked with literal `ts.ops.<op>`
@jit functions (a closure `fn(x)` would defeat the AST extractor).
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
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")

_UNARY = ["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "erf",
          "erfc", "expm1", "log1p", "reciprocal", "sign", "floor", "ceil",
          "round", "trunc"]
_BINARY = ["add", "sub", "mul", "div", "maximum", "minimum", "pow", "atan2",
           "mod", "floor_div"]
_COMPARE = ["eq", "ne", "lt", "le", "gt", "ge"]


# ── envelope membership (⇒ classified metal_runtime) ─────────────────────────
def test_all_opcodes_in_envelope():
    for n in _UNARY + _BINARY + _COMPARE:
        op = f"tessera.{n}"
        assert op in _driver._APPLE_GPU_RUNTIME_OPS, op
        assert op in _runtime._APPLE_GPU_RUNTIME_OPS, op


def test_driver_runtime_mpsgraph_envelopes_agree():
    assert _driver._APPLE_GPU_MPSGRAPH_OPS == _runtime._APPLE_GPU_MPSGRAPH_OPS


# ── kernel numerics via the GPU dispatcher ───────────────────────────────────
def _domain_x(name, seed=0):
    rng = np.random.default_rng(seed)
    if name in ("asin", "acos"):
        return rng.random((4, 5)).astype(np.float32) * 1.6 - 0.8   # in (-1, 1)
    if name in ("log1p", "reciprocal"):
        return rng.random((4, 5)).astype(np.float32) + 0.5         # > 0
    return rng.standard_normal((4, 5)).astype(np.float32)


@gpu
@pytest.mark.parametrize("name", _UNARY)
def test_unary_kernel_matches_reference(name):
    x = _domain_x(name, hash(name) % 99)
    out = _runtime._apple_gpu_dispatch_unary(f"tessera.{name}", [x], np)
    np.testing.assert_allclose(np.asarray(out), np.asarray(getattr(O, name)(x)), atol=1e-4)


@gpu
@pytest.mark.parametrize("name", _BINARY)
def test_binary_kernel_matches_reference(name):
    rng = np.random.default_rng(hash(name) % 97)
    a = rng.standard_normal((4, 5)).astype(np.float32)
    b = np.abs(rng.standard_normal((4, 5)).astype(np.float32)) + 0.5
    out = _runtime._apple_gpu_dispatch_mpsgraph_binary(f"tessera.{name}", [a, b], {}, np)
    np.testing.assert_allclose(np.asarray(out), np.asarray(getattr(O, name)(a, b)), atol=1e-4)


@gpu
@pytest.mark.parametrize("name", _COMPARE)
def test_comparison_kernel_returns_f32_mask(name):
    rng = np.random.default_rng(hash(name) % 91)
    a = rng.integers(0, 3, (4, 5)).astype(np.float32)
    b = rng.integers(0, 3, (4, 5)).astype(np.float32)
    out = _runtime._apple_gpu_dispatch_mpsgraph_binary(f"tessera.{name}", [a, b], {}, np)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(getattr(O, name)(a, b)).astype(np.float32))


@gpu
def test_mod_is_floor_mod_for_negatives():
    a = np.array([-3.0, -1.0, 5.0, -7.0], np.float32)
    b = np.array([3.0, 4.0, 3.0, 4.0], np.float32)
    out = _runtime._apple_gpu_dispatch_mpsgraph_binary("tessera.mod", [a, b], {}, np)
    np.testing.assert_allclose(np.asarray(out), np.mod(a, b), atol=1e-5)


@gpu
def test_binary_scalar_form():
    x = np.random.default_rng(1).standard_normal((3, 4)).astype(np.float32)
    out = _runtime._apple_gpu_dispatch_mpsgraph_binary("tessera.add", [x], {"scalar": 2.5}, np)
    np.testing.assert_allclose(np.asarray(out), x + 2.5, atol=1e-5)


# ── end-to-end metal_runtime classification (literal @jit calls) ─────────────
@gpu
def test_unary_jit_metal_runtime():
    x = np.random.default_rng(2).standard_normal((4, 5)).astype(np.float32)

    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.cos(x)

    np.testing.assert_allclose(np.asarray(f(x)), np.cos(x), atol=1e-4)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


@gpu
def test_binary_jit_metal_runtime():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((4, 5)).astype(np.float32)
    b = np.abs(rng.standard_normal((4, 5)).astype(np.float32)) + 0.5

    @ts.jit(target="apple_gpu")
    def f(a, b):
        return ts.ops.atan2(a, b)

    np.testing.assert_allclose(np.asarray(f(a, b)), np.arctan2(a, b), atol=1e-4)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


@gpu
def test_comparison_jit_metal_runtime():
    rng = np.random.default_rng(4)
    a = rng.integers(0, 3, (4, 5)).astype(np.float32)
    b = rng.integers(0, 3, (4, 5)).astype(np.float32)

    @ts.jit(target="apple_gpu")
    def f(a, b):
        return ts.ops.gt(a, b)

    np.testing.assert_array_equal(np.asarray(f(a, b)), (a > b).astype(np.float32))
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
