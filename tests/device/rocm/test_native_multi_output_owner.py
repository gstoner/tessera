"""Diagnostic ABI proof for a native stage with two genuine SSA outputs.

This tests native ownership, not a compiled producer-to-matmul program.
"""
import ctypes as C
import os

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.native_scaled_program import _Buffer
from tessera.compiler.nvfp4_expansion import nvfp4_expansion_policy, reference_nvfp4_expand
from tessera.compiler.rocm_nvfp4_expand_native import (
    build_nvfp4_expansion_graph, package_nvfp4_expansion,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="requires the gfx1201 owning device")


class MultiOutputStep(C.Structure):
    _fields_ = [
        ("image", C.c_void_p), ("image_bytes", C.c_uint64), ("entry", C.c_char_p),
        ("input_count", C.c_uint32), ("scalar_count", C.c_uint32),
        ("inputs", C.c_uint32*6), ("output_count", C.c_uint32),
        ("outputs", C.c_uint32*7), ("geometry", C.c_uint32*6),
        ("scalars", C.c_int64*8),
    ]


@pytest.fixture(scope="module")
def lib():
    assert rt._rocm_live_arch() == "gfx1201"
    path = os.environ.get("TESSERA_ROCM_MULTI_OUTPUT_LIB")
    if not path:
        pytest.fail("owning-device multi-output provider is required")
    library = C.CDLL(path)
    library.tessera_rocm_program_prepare_multi_output.argtypes = [
        C.c_char_p, C.c_uint32, C.c_uint32, C.POINTER(_Buffer), C.c_uint32,
        C.POINTER(MultiOutputStep), C.POINTER(C.c_void_p),
        C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]
    library.tessera_rocm_program_invoke.argtypes = [
        C.c_uint64, C.c_uint32, C.POINTER(C.c_uint64), C.POINTER(C.c_float)]
    library.tessera_rocm_program_read_many.argtypes = [
        C.c_uint64, C.c_uint64, C.c_uint32, C.POINTER(C.c_uint32),
        C.POINTER(C.c_void_p), C.POINTER(C.c_uint64)]
    library.tessera_rocm_program_close.argtypes = [C.c_uint64]
    for name in ("prepare_multi_output", "invoke", "read_many", "close"):
        getattr(library, "tessera_rocm_program_" + name).restype = C.c_int
    return library


@pytest.fixture(scope="module")
def package():
    return package_nvfp4_expansion(build_nvfp4_expansion_graph(
        7, 48, [0, 3, 7], numeric_policy=nvfp4_expansion_policy()))


def frame(package):
    codes = np.resize(np.arange(256, dtype=np.uint8), (7, 24))
    scales = np.resize(np.arange(127, dtype=np.uint8), (7, 3)).view(ml_dtypes.float8_e4m3fn)
    scales[0, 0] = -0.
    globals_ = np.array([.000123456789, .007654321], np.float64)
    roots = [codes, scales, globals_]
    expected = reference_nvfp4_expand(*roots, row_offsets=[0, 3, 7],
                                      numeric_policy=nvfp4_expansion_policy())
    arrays = roots + [np.full_like(expected[0], -1), np.full_like(expected[1], -1)]
    buffers = (_Buffer*5)(*[
        _Buffer(a.nbytes, a.size, -1 if i < 3 else 0, 0 if i < 3 else 1,
                0 if i < 3 else 2, 0) for i, a in enumerate(arrays)])
    blob = C.create_string_buffer(package.native.image.payload)
    entry = package.native.descriptor.entry_symbol.encode()
    step = MultiOutputStep()
    step.image = C.cast(blob, C.c_void_p)
    step.image_bytes = len(package.native.image.payload)
    step.entry = entry
    step.input_count, step.output_count = 3, 2
    step.inputs[:3], step.outputs[:2] = (0, 1, 2), (3, 4)
    geometry = package.native.descriptor.geometry
    step.geometry[:] = (*geometry.grid, *geometry.workgroup)
    pointers = (C.c_void_p*3)(*[a.ctypes.data for a in roots])
    sizes = (C.c_uint64*3)(*[a.nbytes for a in roots])
    # Keep every borrowed host/image object alive through native admission.
    return arrays, expected, buffers, step, pointers, sizes, blob, entry


def prepare(lib, frame_):
    _, _, buffers, step, pointers, sizes, _, _ = frame_
    handle = C.c_uint64()
    status = lib.tessera_rocm_program_prepare_multi_output(
        b"gfx1201", 3, 5, buffers, 1, C.byref(step), pointers, sizes, C.byref(handle))
    return status, handle


def test_native_two_output_storage_and_transactional_read(lib, package):
    data = frame(package)
    arrays, expected = data[:2]
    status, handle = prepare(lib, data)
    try:
        assert status == 0 and handle.value
        generation = C.c_uint64()
        assert lib.tessera_rocm_program_invoke(handle, 1, C.byref(generation), None) == 0
        outputs = (C.c_void_p*2)(*[a.ctypes.data for a in arrays[3:]])
        sizes = (C.c_uint64*2)(*[a.nbytes for a in arrays[3:]])
        slots = (C.c_uint32*2)(3, 4)
        assert lib.tessera_rocm_program_read_many(
            handle, generation, 2, slots, outputs, sizes) == 0
        np.testing.assert_array_equal(arrays[3].view(np.uint8), expected[0].view(np.uint8))
        np.testing.assert_array_equal(arrays[4].view(np.uint32), expected[1].view(np.uint32))
        # A failed batched read must not partially publish either host output.
        arrays[3].view(np.uint8).fill(31)
        arrays[4].view(np.uint32).fill(31)
        duplicate = (C.c_uint32*2)(3, 3)
        assert lib.tessera_rocm_program_read_many(
            handle, generation, 2, duplicate, outputs, sizes) != 0
        assert np.all(arrays[3].view(np.uint8) == 31)
        assert np.all(arrays[4].view(np.uint32) == 31)
    finally:
        if handle.value:
            assert lib.tessera_rocm_program_close(handle) == 0


@pytest.mark.parametrize("bad", [
    "zero_results", "too_many_results", "duplicate_result", "reordered_results",
    "omitted_result", "future_input", "wrong_write", "wrong_last_read", "too_many_scalars",
])
def test_multi_output_ssa_admission(lib, package, bad):
    data = frame(package)
    buffers, step = data[2:4]
    if bad == "zero_results":
        step.output_count = 0
    elif bad == "too_many_results":
        step.output_count = 7
    elif bad == "duplicate_result":
        step.outputs[1] = 3
    elif bad == "reordered_results":
        step.outputs[:2] = (4, 3)
    elif bad == "omitted_result":
        step.output_count = 1
    elif bad == "future_input":
        step.inputs[0] = 3
    elif bad == "wrong_write":
        buffers[4].first_write = 1
    elif bad == "wrong_last_read":
        buffers[4].last_read = 0
    else:
        step.scalar_count = 9
    status, handle = prepare(lib, data)
    try:
        assert status == 1
        assert handle.value == 0
    finally:
        if handle.value:
            assert lib.tessera_rocm_program_close(handle) == 0
