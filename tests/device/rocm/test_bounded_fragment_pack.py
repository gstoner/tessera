"""gfx1201 bounded fragment values are valid without an output store mask."""
import ctypes as ct
import subprocess

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.rocm_native import _extract_hsaco
from benchmarks.rocm.benchmark_gfx1201_fp8_blockscale import Hip, Launch, memref
from tests.unit.test_rocm_bounded_fragment_pack import PAIRS, SHAPES, fragment_source, lower_fragment

pytestmark = pytest.mark.hardware_rocm


@pytest.mark.parametrize("a_type,b_type", PAIRS)
@pytest.mark.parametrize("orders",
    [("row_major", "col_major"), ("col_major", "col_major"),
     ("row_major", "row_major"), ("col_major", "row_major")])
def test_bounded_pack_zero_fill_and_canaries(
        compiler_toolchain, a_type, b_type, orders):
    assert rt._rocm_live_arch() == "gfx1201"
    lowered = lower_fragment(compiler_toolchain,
        fragment_source(a_type, b_type, orders=orders), "gfx1201", rocdl=True)
    pipeline = ("builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
                "reconcile-unrealized-casts),rocdl-attach-target{chip=gfx1201},"
                "gpu-module-to-binary)")
    result = subprocess.run([str(compiler_toolchain.require_mlir_opt()),
        "--pass-pipeline=" + pipeline], input=lowered, text=True,
        capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    image = _extract_hsaco(result.stdout)
    types = {"f16": np.float16, "bf16": ml_dtypes.bfloat16,
             "e4m3": ml_dtypes.float8_e4m3fn, "e5m2": ml_dtypes.float8_e5m2,
             "int8": np.int8, "int4": np.int8}
    output_type = np.int32 if a_type.startswith("int") else np.float32
    hip = Hip()
    for m, n, k in SHAPES:
        rng = np.random.default_rng(m * 1307 + n * 31 + k)
        logical_a = rng.integers(-3, 4, (16, 16)).astype(types[a_type])
        logical_b = rng.integers(-3, 4, (16, 16)).astype(types[b_type])
        # Padding stays nonzero to detect incorrectly treating partial K as full.
        physical_a = np.ascontiguousarray(
            logical_a[:m] if orders[0] == "row_major" else logical_a.T[:k])
        physical_b = np.ascontiguousarray(
            logical_b[:k] if orders[1] == "row_major" else logical_b.T[:n])
        initial = np.full((17, 16), -1729, dtype=output_type)
        expected = np.zeros((16, 16), dtype=output_type)
        expected[:m, :n] = (logical_a[:m, :k].astype(np.int64)
                            @ logical_b[:k, :n].astype(np.int64))
        pointers = [hip.upload(value) for value in (physical_a, physical_b, initial)]
        launch = None
        try:
            arguments = (memref(pointers[0], physical_a.size) + memref(pointers[1], physical_b.size)
                         + memref(pointers[2], 256)
                         + [ct.c_int64(value) for value in (m, n, k)])
            launch = Launch(hip, image, "bounded_fragment_values",
                            arguments, (1, 1, 1), (32, 1, 1))
            launch()
            actual = hip.download(pointers[2], initial)
            np.testing.assert_array_equal(actual[:16], expected)
            np.testing.assert_array_equal(actual[16:], initial[16:])
            for pointer, original in zip(pointers[:2], (physical_a, physical_b), strict=True):
                np.testing.assert_array_equal(
                    hip.download(pointer, original).view(np.uint8), original.view(np.uint8))
        finally:
            hip.check(hip.lib.hipDeviceSynchronize())
            if launch is not None:
                hip.check(hip.lib.hipModuleUnload(launch.module))
            for pointer in pointers:
                hip.check(hip.lib.hipFree(pointer))
