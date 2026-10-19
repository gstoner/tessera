"""Observe all generic folded-scale lanes, including masked logical columns."""
import ctypes as ct
import os

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import rocm_native
from benchmarks.rocm.benchmark_gfx1201_fp8_blockscale import Hip, Launch, memref
from tests.device.rocm.test_folded_scale_native import SOURCE as BASE_SOURCE

pytestmark = [pytest.mark.hardware_rocm, pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 device gate")]

SOURCE = BASE_SOURCE.replace(
    "bounds(%m, %n) : !fc", "bounds(%m, %n) {vector_scales = true} : !fc")
SOURCE = SOURCE.replace(
    "tile.store %v, %o, %c0, %c0, %m, %n, %n",
    "tile.store %v, %o, %c0, %c0, %c16").replace(
    "!tile.tile, memref<?xbf16>, index, index, index, index, index",
    "!tile.tile, memref<?xbf16>, index, index, index")


@pytest.fixture(scope="module")
def image():
    assert rt._rocm_live_arch() == "gfx1201"
    _, _, image, *_ = rocm_native._compile_native_tile_ir(
        SOURCE, directive="gpu.module", family="matmul",
        architecture="gfx1201", schedule_kernel=True)
    return image


@pytest.mark.parametrize("m,n", [(16,16), (16,11), (9,16), (9,11), (1,1)])
@pytest.mark.parametrize("misaligned", [False, True])
@pytest.mark.parametrize("probe", ["ordinary", "mixed_overflow", "mixed_underflow", "classification"])
def test_vector_row_scale_preserves_unmasked_result_lanes(image, m, n, misaligned, probe):
    assert rt._rocm_live_arch() == "gfx1201"
    k = 64
    value = 2.0**-9 if probe == "mixed_overflow" else 448.0 if probe == "mixed_underflow" else 1.0
    a = np.full((16,k), value, dtype=ml_dtypes.float8_e4m3fn)
    b = a.copy()
    scales = (np.arange(m, dtype=np.float32) + 4) / 16
    refs = (120 + np.arange(n) % 13).astype(np.uint8)
    if probe != "ordinary":
        values = [0.5, 0.0, -0.0, 2.0**20, 2.0**-30, -0.5, 2.0**-149, -2.0**-149]
        if probe == "classification":
            values[3:6] = [np.nan, np.inf, -np.inf]
        scales = np.resize(np.array(values, np.float32), m)
        refs = np.resize(np.array([127,240,1,0], np.uint8), n)
    scale_payload = np.concatenate((np.array([-77], np.float32), scales)) if misaligned else scales
    hip = Hip()
    pointers = [hip.upload(a), hip.upload(b), hip.upload(scale_payload),
                hip.upload(refs), hip.upload(np.full((16,16), -101, ml_dtypes.bfloat16))]
    launch = None
    try:
        args = []
        for i, (pointer, size) in enumerate(zip(pointers, (16*k,16*k,m,n,256), strict=True)):
            if i == 2 and misaligned:
                assert pointer.value is not None
                pointer = ct.c_void_p(pointer.value + 4)
            args += memref(pointer, size)
        args += [ct.c_int64(m), ct.c_int64(n), ct.c_int64(k)]
        launch = Launch(hip, image, "folded_scale_native", args, (1,1,1), (32,1,1))
        launch()
        got = hip.download(pointers[-1], np.zeros((16,16), ml_dtypes.bfloat16))
        rows = np.where(np.arange(16) < m, np.arange(16), 0)
        cols = np.where(np.arange(16) < n, np.arange(16), 0)
        partial = a.astype(np.float64) @ b.astype(np.float64).T
        activation = scales[rows,None]
        reference = np.where(refs[cols] == 0, 0.0,
                             np.exp2(refs[cols].astype(np.int64)-127))[None,:].astype(np.float32)
        # Mirror the specified rounding points independently with NumPy, not
        # a uniformly wide oracle that would change regular-lane semantics.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            combined = reference * activation
            regular = np.isfinite(combined) & (combined != 0)
            normal = partial.astype(np.float32) * combined
            wide = (partial * reference.astype(np.float64)
                    * activation.astype(np.float64)).astype(np.float32)
            fallback = np.where((partial == 0) & np.isfinite(activation),
                                np.float32(0), wide)
            want = np.where(regular, normal, fallback).astype(ml_dtypes.bfloat16)
        nan = np.isnan(want)
        np.testing.assert_array_equal(np.isnan(got), nan)
        np.testing.assert_array_equal(got.view(np.uint16)[~nan], want.view(np.uint16)[~nan])
    finally:
        if launch is not None:
            hip.check(hip.lib.hipModuleUnload(launch.module))
        for pointer in pointers:
            hip.check(hip.lib.hipFree(pointer))
