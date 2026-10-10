"""Graph/Schedule/Tile native MXFP8 proof; raw HIP is diagnostic ABI evidence.

Checked production packaging and runtime.launch admission remain separate.
"""
from __future__ import annotations

import ctypes
import os
import subprocess

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape
from tessera.compiler.rocm_mxfp8_blockscale import author_mxfp8_graph
from tessera.compiler.rocm_native import _directive_symbol, _extract_hsaco
from tessera.compiler.scheduled_matmul import find_tessera_opt

pytestmark = [
    pytest.mark.hardware_rocm,
    pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
                       reason="explicit gfx1201 owning-device gate"),
]


def _compile(shape):
    assert rt._rocm_live_arch() == "gfx1201"
    tool = find_tessera_opt()
    assert tool is not None
    graph = author_mxfp8_graph(shape, entry="mxfp8")
    # All compilation/selection/fragment construction occurs in MLIR passes.
    def run(source, passes):
        return subprocess.run([str(tool), "-", *passes], input=source,
                              capture_output=True, text=True, check=True).stdout
    schedule = run(graph, ["--tessera-graph-to-schedule"])
    tile = run(schedule, ["--tessera-schedule-to-tile"])
    assert 'physical_contract = "rocm_mxfp8_e4m3_e8m0_k32' in schedule
    assert 'scale_format = "e8m0"' in schedule
    def pipeline(output):
        return "--pass-pipeline=builtin.module(tessera-rocm-executable{family=matmul input=tile output=" + output + " arch=gfx1201})"
    target = run(tile, [pipeline("target")])
    assert "tessera.rocm.mxfp8_e4m3_e8m0_k32." in target
    binary = run(tile, [pipeline("binary")])
    return graph, schedule, tile, target, _extract_hsaco(binary)


def _inputs(shape):
    rng = np.random.default_rng(71)
    # Integer operands make each K32 WMMA partial exact in f32, allowing a
    # bitwise comparison that catches group order and scale layout mistakes.
    a = rng.integers(-3, 4, (shape.m, shape.k)).astype(ml_dtypes.float8_e4m3fn)
    b = rng.integers(-3, 4, (shape.k, shape.n)).astype(ml_dtypes.float8_e4m3fn)
    sa = rng.integers(123, 132, (shape.m, shape.groups), dtype=np.uint8)
    sb = rng.integers(123, 132, (shape.groups, shape.n), dtype=np.uint8)
    sa[0, 0] = 0
    sb[0, 0] = 254
    return a, b, sa, sb


def _reference(a, b, sa, sb):
    aa, bb = a.astype(np.float64), b.astype(np.float64)
    lhs = sa.view(ml_dtypes.float8_e8m0fnu).astype(np.float64)
    rhs = sb.view(ml_dtypes.float8_e8m0fnu).astype(np.float64)
    out = np.zeros((a.shape[0], b.shape[1]), np.float32)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for group in range(a.shape[1] // 32):
            partial = (aa[:, group*32:(group+1)*32] @ bb[group*32:(group+1)*32]).astype(np.float32)
            scaled = (partial.astype(np.float64) * lhs[:, group, None] * rhs[group, None, :]).astype(np.float32)
            out = (out + scaled).astype(np.float32)
    return out


def _launch(image, entry, shape, a, b, sa, sb):
    hip = rt._load_hip_for_launch()
    assert hip is not None and hip.hipInit(0) == 0
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    pointers = []
    assert hip.hipModuleLoadData(ctypes.byref(module), image) == 0
    try:
        assert hip.hipModuleGetFunction(ctypes.byref(function), module, entry.encode()) == 0
        output = np.full((shape.m, shape.n),
                         np.nan, dtype=ml_dtypes.bfloat16 if shape.output == "bf16" else np.float32)
        weight = np.ascontiguousarray(b.T) if shape.weight_layout == "nk" else b
        values = []
        for host in [a, weight, sa, sb, output]:
            ptr = ctypes.c_void_p()
            assert hip.hipMalloc(ctypes.byref(ptr), host.nbytes) == 0
            pointers.append(ptr)
            assert hip.hipMemcpy(ptr, host.ctypes.data_as(ctypes.c_void_p), host.nbytes, 1) == 0
            values.extend([ctypes.c_void_p(ptr.value), ctypes.c_void_p(ptr.value),
                           ctypes.c_int64(0), ctypes.c_int64(host.size), ctypes.c_int64(1)])
        values.extend([ctypes.c_int64(shape.m), ctypes.c_int64(shape.n), ctypes.c_int64(shape.k)])
        args = (ctypes.c_void_p * len(values))(
            *[ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values])
        assert hip.hipModuleLaunchKernel(
            function, (shape.n+15)//16, (shape.m+15)//16, 1,
            32, 1, 1, 0, None, args, None) == 0
        assert hip.hipDeviceSynchronize() == 0
        assert hip.hipMemcpy(output.ctypes.data_as(ctypes.c_void_p),
                             pointers[-1], output.nbytes, 2) == 0
        return output
    finally:
        for ptr in pointers:
            hip.hipFree(ptr)
        hip.hipModuleUnload(module)


@pytest.mark.parametrize("mnk", [(17, 19, 64), (16, 16, 128), (16, 2, 64)])
@pytest.mark.parametrize("layout", ["kn", "nk"])
@pytest.mark.parametrize("output", ["f32", "bf16"])
def test_mxfp8_graph_native_multigroup(mnk, layout, output):
    shape = BlockScaleShape(*mnk, scale_k=32, scale_n=1, weight_layout=layout, output=output)
    graph, schedule, tile, target, image = _compile(shape)
    a, b, sa, sb = _inputs(shape)
    expected = _reference(a, b, sa, sb)
    actual = _launch(image, _directive_symbol(target, "tessera_rocm.scaled_wmma_gemm"),
                     shape, a, b, sa, sb)
    if output == "bf16":
        np.testing.assert_array_equal(actual.view(np.uint16),
                                      expected.astype(ml_dtypes.bfloat16).view(np.uint16))
    else:
        np.testing.assert_array_equal(actual, expected)
