"""Bit-exact native packed E2M1 decoding before any MMA can hide signed zero."""
import ctypes as c
import subprocess

import ml_dtypes
import numpy as np
import pytest

from benchmarks.rocm.benchmark_gfx1201_fp8_blockscale import Hip, Launch, memref
from tessera import runtime
from tessera.compiler.rocm_native import _extract_hsaco
from tests.unit.test_rocm_bounded_fragment_pack import lower_fragment
from tests.unit.test_rocm_mxfp4_fragment_decode import CASES, fragment_byte_source

pytestmark = pytest.mark.hardware_rocm


def expected_registers(codes, k, n, row, col):
    # Independent value-domain oracle; NumPy performs the E4M3 encoding.
    levels = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.], np.float32)
    nibbles = np.arange(16, dtype=np.uint8)
    values = np.copysign(levels[nibbles & 7], np.where(nibbles & 8, -1., 1.))
    table = values.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    assert table[8] == 128  # Negative zero is a bit contract, not float equality.
    result = np.zeros((32, 8), np.uint8)
    for lane in range(32):
        for j in range(8):
            r, column = row + (lane // 16) * 8 + j, col + lane % 16
            if r < k and column < n:
                result[lane, j] = table[codes[r, column]]
    return result.view(np.uint32).reshape(32, 2)


@pytest.mark.parametrize("pitch", [16, 17])
@pytest.mark.parametrize("shape", CASES)
def test_packed_fragment_all_codes_and_bounds(compiler_toolchain, pitch, shape):
    assert runtime._rocm_live_arch() == "gfx1201"
    k, n, row, col = shape
    lowered = lower_fragment(compiler_toolchain, fragment_byte_source(pitch),
                             "gfx1201", rocdl=True)
    pipeline = ("builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
                "reconcile-unrealized-casts),rocdl-attach-target{chip=gfx1201},"
                "gpu-module-to-binary)")
    result = subprocess.run([str(compiler_toolchain.require_mlir_opt()),
        "--pass-pipeline=" + pipeline], input=lowered, text=True,
        capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    image = _extract_hsaco(result.stdout)
    codes = np.random.default_rng(834 + k + row).integers(
        0, 16, (2 * ((k + 1) // 2), pitch), dtype=np.uint8)
    codes[0, :16] = np.arange(16)
    codes[1, :16] = np.arange(16)[::-1]
    if n < 16:
        codes[0, 0] = 15  # A live nonzero even in the 1x1 boundary case.
    packed = np.ascontiguousarray(codes[::2] | (codes[1::2] << np.uint8(4)))
    expected = expected_registers(codes, k, n, row, col)
    initial = np.full((33, 2), 0xDEADBEEF, np.uint32)
    hip = Hip()
    pointers = [hip.upload(value) for value in (packed, initial)]
    launch = None
    try:
        arguments = (memref(pointers[0], packed.size) + memref(pointers[1], 64)
                     + [c.c_int64(value) for value in shape])
        launch = Launch(hip, image, "packed_fragment_bytes", arguments,
                        (1, 1, 1), (32, 1, 1))
        launch()
        actual = hip.download(pointers[1], initial)
        np.testing.assert_array_equal(actual[:32], expected)
        np.testing.assert_array_equal(actual[32:], initial[32:])
        np.testing.assert_array_equal(hip.download(pointers[0], packed), packed)
    finally:
        hip.check(hip.lib.hipDeviceSynchronize())
        if launch is not None:
            hip.check(hip.lib.hipModuleUnload(launch.module))
        for pointer in pointers:
            hip.check(hip.lib.hipFree(pointer))
