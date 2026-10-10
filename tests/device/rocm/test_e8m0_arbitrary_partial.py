"""Generic Tile scale proof over arbitrary f32 bits and every E8M0 code pair."""
from __future__ import annotations

import ctypes as C
from pathlib import Path
import subprocess

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.rocm_native import _extract_hsaco
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests.device.rocm.test_e8m0_tile_scale import pytestmark


@pytest.mark.parametrize("acc_bits", [0, 0x80000000, 0x3f800001, 0xff7fffff])
def test_e8m0_arbitrary_f32_partials_all_code_pairs(acc_bits):
    assert rt._rocm_live_arch() == "gfx1201"
    tool = find_tessera_opt()
    assert tool is not None
    fixture = Path(__file__).parent/"fixtures/e8m0_arbitrary_partial.mlir"
    result = subprocess.run([str(tool), str(fixture),
        "--pass-pipeline=builtin.module(tessera-rocm-executable{family=matmul input=tile output=binary arch=gfx1201})"],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    image = _extract_hsaco(result.stdout)
    edge = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
        0x00800000, 0x80800000, 0x3f800000, 0xbf800000, 0x3f800001, 0xbf800001,
        0x3fffffff, 0xbfffffff, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
        0x7fc00000, 0x7fa00001], np.uint32)
    random = np.random.default_rng(718).integers(0, 2**32, size=96, dtype=np.uint32)
    partials = np.concatenate([edge, random]).view(np.float32)
    sa = np.tile(np.arange(256, dtype=np.uint8), len(partials))
    sb = np.arange(256, dtype=np.uint8)
    output = np.full((len(sa), 256), np.nan, np.float32)
    hip = rt._load_hip_for_launch()
    assert hip is not None and hip.hipInit(0) == 0
    module, function = C.c_void_p(), C.c_void_p()
    pointers, values = [], []
    assert hip.hipModuleLoadData(C.byref(module), image) == 0
    try:
        assert hip.hipModuleGetFunction(C.byref(function), module, b"e8m0_arbitrary") == 0
        for host in [partials, sa, sb, output]:
            ptr = C.c_void_p()
            assert hip.hipMalloc(C.byref(ptr), host.nbytes) == 0
            pointers.append(ptr)
            assert hip.hipMemcpy(ptr, C.c_void_p(host.ctypes.data), host.nbytes, 1) == 0
            values.extend([C.c_void_p(ptr.value), C.c_void_p(ptr.value),
                           C.c_int64(0), C.c_int64(host.size), C.c_int64(1)])
        values.extend([C.c_int64(len(sa)), C.c_int64(256), C.c_uint32(acc_bits)])
        argv = (C.c_void_p*len(values))(*[C.cast(C.byref(v), C.c_void_p) for v in values])
        assert hip.hipModuleLaunchKernel(function, 16, len(sa)//16, 1,
            32, 1, 1, 0, None, argv, None) == 0
        assert hip.hipDeviceSynchronize() == 0
        assert hip.hipMemcpy(C.c_void_p(output.ctypes.data), pointers[-1], output.nbytes, 2) == 0
    finally:
        for ptr in pointers:
            hip.hipFree(ptr)
        hip.hipModuleUnload(module)
    lhs = sa.view(ml_dtypes.float8_e8m0fnu).astype(np.float64)
    rhs = sb.view(ml_dtypes.float8_e8m0fnu).astype(np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        p = np.repeat(partials.astype(np.float64), 256)
        expected = ((p[:, None]*lhs[:, None])*rhs[None, :]).astype(np.float32)
        acc = np.array([acc_bits], np.uint32).view(np.float32)[0]
        expected = (acc + expected).astype(np.float32)
    np.testing.assert_array_equal(np.isnan(output), np.isnan(expected))
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(output[valid].view(np.uint32),
                                  expected[valid].view(np.uint32))
