"""Owning-device math proof for the native mixed scalar Tile materializer.

Raw fixture execution is not public Graph/Schedule/ABI integration evidence.
"""
import ctypes as ct
from functools import lru_cache

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.unit.test_sm120_scalar_contraction import source

pytestmark = pytest.mark.hardware_nvidia


@lru_cache(maxsize=64)
def image(a, b, output, transpose_a, transpose_b):
    from tessera.compiler.nvidia_native import _compile_tile_ir
    _, ptx, *_ = _compile_tile_ir(
        source(a, b, output, transpose_a, transpose_b), "mixed_adjoint")
    return ptx


def driver():
    lib = ct.CDLL("libcuda.so.1")
    signatures = {
        "cuModuleLoadData": [ct.POINTER(ct.c_void_p), ct.c_void_p],
        "cuModuleGetFunction": [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p],
        "cuModuleUnload": [ct.c_void_p],
        "cuLaunchKernel": [ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_uint,
                           ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint,
                           ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p],
    }
    for name, arguments in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = arguments, ct.c_int
    return lib


@pytest.mark.parametrize("half", ["f16", "bf16"])
@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("output", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("orientation", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("shape", [(1, 1, 1), (17, 19, 35)])
def test_mixed_scalar_native_values_and_store_bounds(half, side, output, orientation, shape):
    import ml_dtypes
    assert rt._nvidia_device_name() == "sm_120"
    dtype = {"f32": np.dtype("float32"), "f16": np.dtype("float16"),
             "bf16": np.dtype(ml_dtypes.bfloat16)}
    a_type, b_type = ("f32", half) if side == "a" else (half, "f32")
    ta, tb = orientation
    m, n, k = shape
    ptx = image(a_type, b_type, output, ta, tb)
    lib, module, function = driver(), ct.c_void_p(), ct.c_void_p()
    with NvidiaDeviceSession() as session:
        encoded = ct.create_string_buffer(ptx.encode())
        assert lib.cuModuleLoadData(ct.byref(module), encoded) == 0
        try:
            assert lib.cuModuleGetFunction(ct.byref(function), module, b"mixed_adjoint") == 0
            for seed in ((71, 93, 0) if k > 1 else (71, 93)):
                rng = np.random.default_rng(seed)
                a = rng.normal(0, .25, (m, k)).astype(dtype[a_type])
                b = rng.normal(0, .25, (k, n)).astype(dtype[b_type])
                if seed == 0:
                    # A dyadic cancellation witnesses forbidden input narrowing:
                    # converting the wide operand to the half format yields zero.
                    a.fill(0)
                    b.fill(0)
                    delta = 2.0 ** (-12 if half == "f16" else -9)
                    if side == "a":
                        a[:, 0], a[:, 1], b[:2, :] = 1 + delta, -1, 1
                    else:
                        a[:, :2], b[0, :], b[1, :] = 1, 1 + delta, -1
                expected = (a.astype(np.float64) @ b.astype(np.float64)).astype(dtype[output])
                if seed == 0:
                    narrowed_a = a.astype(dtype[half]).astype(np.float64)
                    narrowed_b = b.astype(dtype[half]).astype(np.float64)
                    changed = (narrowed_a @ narrowed_b).astype(dtype[output])
                    assert np.all(expected != changed)
                magnitude = np.abs(a.astype(np.float64)) @ np.abs(b.astype(np.float64))
                gamma = k * np.finfo(np.float32).eps / (1 - k * np.finfo(np.float32).eps)
                adjacent = np.nextafter(expected, dtype[output].type(np.inf)).astype(np.float64)
                budget = gamma * magnitude + np.abs(adjacent - expected.astype(np.float64)) + 1e-8
                aa = session.upload(np.ascontiguousarray(a.T if ta else a))
                bb = session.upload(np.ascontiguousarray(b.T if tb else b))
                sentinel = dtype[output].type(123)
                out = session.upload(np.full(m * n + 16, sentinel, dtype=dtype[output]))
                arguments = [ct.c_uint64(aa.ptr), ct.c_uint64(bb.ptr), ct.c_uint64(out.ptr),
                             ct.c_int64(m), ct.c_int64(n), ct.c_int64(k)]
                parameters = (ct.c_void_p * 6)(
                    *(ct.cast(ct.pointer(value), ct.c_void_p) for value in arguments))
                assert lib.cuLaunchKernel(function, (n + 7) // 8, (m + 15) // 16, 1,
                                          32, 1, 1, 0, ct.c_void_p(session.stream),
                                          parameters, None) == 0
                assert session.synchronize() == 0
                actual = session.download(out)
                np.testing.assert_array_equal(actual[m * n:], sentinel)
                actual = actual[:m * n].reshape(m, n)
                assert np.isfinite(actual).all()
                assert np.all(np.abs(actual.astype(np.float64) - expected.astype(np.float64)) <= budget)
        finally:
            assert session.synchronize() == 0
            if module.value:
                assert lib.cuModuleUnload(module) == 0
