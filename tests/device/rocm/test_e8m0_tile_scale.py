"""Exact gfx1201 numerical checks for the native Tile E8M0 scale consumer.

This fixture enters at Tile IR. It proves no Graph/Schedule package route.
A pair of FP8 WMMA instructions computes the isolated K32 partial.
"""
from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path
import subprocess

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.rocm_native import _extract_hsaco
from tessera.compiler.scheduled_matmul import find_tessera_opt

pytestmark = [
    pytest.mark.hardware_rocm,
    pytest.mark.skipif(
        os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
        reason="explicit gfx1201 owning-device gate",
    ),
]


@lru_cache(maxsize=1)
def _image() -> bytes:
    assert rt._rocm_live_arch() == "gfx1201"
    tool = find_tessera_opt()
    assert tool is not None
    fixture = Path(__file__).resolve().parent / "fixtures/e8m0_tile_scale.mlir"
    compiled = subprocess.run(
        [str(tool), str(fixture),
         "--pass-pipeline=builtin.module(tessera-rocm-executable{family=matmul input=tile output=binary arch=gfx1201})"],
        capture_output=True, text=True, check=True,
    )
    return _extract_hsaco(compiled.stdout)


def _launch(sa_codes: np.ndarray, sb_codes: np.ndarray, *, lhs_value: float) -> np.ndarray:
    hip = rt._load_hip_for_launch()
    assert hip is not None and hip.hipInit(0) == 0
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    pointers = []
    image = _image()
    assert hip.hipModuleLoadData(ctypes.byref(module), image) == 0
    try:
        assert hip.hipModuleGetFunction(
            ctypes.byref(function), module, b"e8m0_consumer_fixture"
        ) == 0
        m, n, k = len(sa_codes), len(sb_codes), 32
        a = np.full((m, k), lhs_value, dtype=ml_dtypes.float8_e4m3fn)
        b = np.ones((k, n), dtype=ml_dtypes.float8_e4m3fn)
        output = np.full((m, n), np.nan, np.float32)
        hosts = [a, b, sa_codes, sb_codes, output]
        values = []
        for host in hosts:
            ptr = ctypes.c_void_p()
            assert hip.hipMalloc(ctypes.byref(ptr), host.nbytes) == 0
            pointers.append(ptr)
            assert hip.hipMemcpy(ptr, host.ctypes.data_as(ctypes.c_void_p), host.nbytes, 1) == 0
            values.extend([ctypes.c_void_p(ptr.value), ctypes.c_void_p(ptr.value),
                           ctypes.c_int64(0), ctypes.c_int64(host.size), ctypes.c_int64(1)])
        values.extend([ctypes.c_int64(m), ctypes.c_int64(n), ctypes.c_int64(k)])
        args = (ctypes.c_void_p * len(values))(
            *[ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values]
        )
        assert hip.hipModuleLaunchKernel(
            function, (n + 15) // 16, (m + 15) // 16, 1,
            32, 1, 1, 0, None, args, None,
        ) == 0
        assert hip.hipDeviceSynchronize() == 0
        assert hip.hipMemcpy(output.ctypes.data_as(ctypes.c_void_p),
                             pointers[-1], output.nbytes, 2) == 0
        return output
    finally:
        for ptr in pointers:
            hip.hipFree(ptr)
        hip.hipModuleUnload(module)


@pytest.mark.parametrize("lhs_value", [-1.375, 0.0, 1.375])
@pytest.mark.parametrize("rows,cols", [(255, 15), (255, 16), (256, 15), (256, 16), (256, 256)])
def test_e8m0_all_codes_and_extreme_scale_pairs(lhs_value, rows, cols):
    # Independent ml_dtypes E8M0 oracle covers every code, including code
    # zero's 2^-127 and code 255's NaN. Ragged 255/256 edges exercise masked
    # scale loads as well as the reciprocal and overflow scale combinations.
    sa = np.arange(rows, dtype=np.uint8)
    sb = (np.arange(256, dtype=np.uint8) if cols == 256 else
          np.array([0, 1, 126, 127, 128, 253, 254, 255] * 2, dtype=np.uint8)[:cols].copy())
    actual = _launch(sa, sb, lhs_value=lhs_value)
    a_scale = sa.view(ml_dtypes.float8_e8m0fnu).astype(np.float64)
    b_scale = sb.view(ml_dtypes.float8_e8m0fnu).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        expected = ((32.0 * lhs_value) * a_scale[:, None] *
                    b_scale[None, :]).astype(np.float32)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(actual[~np.isnan(expected)], expected[~np.isnan(expected)])
