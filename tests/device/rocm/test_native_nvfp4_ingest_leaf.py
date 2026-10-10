"""Exact-device proof of the physical ingest leaf; Graph integration is pending."""
import ctypes as C
import hashlib
import time
import os
import re
import subprocess

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler import rocm_nvfp4_ingest as ingest
from tessera.compiler.rocm_native import _compile_native_tile_ir
from tessera.compiler.rocm_pipeline import ROCMInputLevel

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="requires the gfx1201 owning device")


def directive(n, k, offsets):
    return ('module attributes {tessera.arch = "gfx1201"} {\n'
            'tessera_rocm.nvfp4_requantize {name = "native_ingest", arch = "gfx1201", '
            f'n = {n} : i64, k = {k} : i64, row_offsets = ['
            + ', '.join(f'{v} : i64' for v in offsets)
            + '], execution_mode = "explicit_scale_requantization", '
            'source_layout = "e2m1_row_k_e4m3_k16_projection_global", '
            'destination_layout = "e2m1_row_k_e8m0_k32_group_n"}\n}\n')


def run_leaf(projections, *, timing=False, graph_route=False):
    assert rt._rocm_live_arch() == "gfx1201"
    started = time.perf_counter_ns()
    reference = ingest.ingest_nvfp4_projections(projections)
    host_oracle_ms = (time.perf_counter_ns() - started) / 1e6
    n, k = reference.shape
    started = time.perf_counter_ns()
    source = directive(n, k, reference.row_offsets)
    if graph_route:
        from tessera.compiler.scheduled_matmul import find_tessera_opt
        from tests.unit.test_rocm_graph_nvfp4_ingest import graph
        source = graph().replace("tensor<7x32xui8>", f"tensor<{n}x{k//2}xui8>")
        source = source.replace("tensor<7x4xf8E4M3FN>", f"tensor<{n}x{k//16}xf8E4M3FN>")
        source = source.replace("tensor<2x7xui8>", f"tensor<{k//32}x{n}xui8>")
        source = source.replace("tensor<7x2x2xf64>", f"tensor<{n}x{k//32}x2xf64>")
        source = source.replace("tensor<2xf64>", f"tensor<{len(projections)}xf64>")
        source = source.replace("[0 : i64, 3 : i64, 7 : i64]",
            "[" + ", ".join(f"{v} : i64" for v in reference.row_offsets) + "]")
        lowered = subprocess.run([str(find_tessera_opt()),
            "--tessera-graph-to-schedule", "--tessera-schedule-to-tile",
            "--lower-tile-to-rocm=arch=gfx1201"],
            input=source,text=True,capture_output=True)
        assert lowered.returncode == 0, lowered.stderr
        source = lowered.stdout
    target, backend, binary, compiler, toolchain, _, _ = _compile_native_tile_ir(
        source,
        directive="tessera_rocm.nvfp4_requantize", family="quant_fp",
        architecture="gfx1201", input_level=ROCMInputLevel.DIRECTIVE)
    compile_ms = (time.perf_counter_ns() - started) / 1e6
    entry = re.search(r'name = "([^"]+)"', target).group(1)
    record = dict(shape_nk=[n,k], row_offsets=list(reference.row_offsets),
        projection_globals=[p.global_scale for p in projections],
        host_oracle_ms=host_oracle_ms, compile_ms=compile_ms,
        compiler_fingerprint=compiler, toolchain_fingerprint=toolchain,
        target_sha256=hashlib.sha256(target.encode()).hexdigest(),
        hsaco_sha256=hashlib.sha256(binary).hexdigest(),
        route=("GraphIR->ScheduleIR->TileIR->ROCm Target IR->GPU MLIR->ROCDL/LLVM->HSACO"
               if graph_route else "native ROCm Target IR->GPU MLIR->ROCDL/LLVM->HSACO"),
        frontend_schedule_tile_integrated=graph_route, selector_promotion=False)
    assert "tessera_rocm.nvfp4_requantize" in target
    assert "gpu.binary" in backend
    arrays = [
        np.ascontiguousarray(np.concatenate([p.packed_codes for p in projections])),
        np.ascontiguousarray(np.concatenate([p.e4m3_scales.view(np.uint8) for p in projections])),
        np.asarray([p.global_scale for p in projections], np.float64),
        np.full_like(reference.packed_codes, 0xFF),
        np.full_like(reference.scale_exponents, 0xFF),
        np.full((n, k // 32, 2), np.nan, np.float64),
    ]
    hip = rt._load_hip_for_launch()
    assert hip is not None and hip.hipInit(0) == 0
    blob = C.create_string_buffer(binary)
    module, function = C.c_void_p(), C.c_void_p()
    pointers, values = [], []
    def check(rc):
        if rc:
            raise RuntimeError(f"HIP native ingest rc={rc}")
    check(hip.hipModuleLoadData(C.byref(module), blob))
    try:
        check(hip.hipModuleGetFunction(C.byref(function), module, entry.encode()))
        for array in arrays:
            pointer = C.c_void_p()
            check(hip.hipMalloc(C.byref(pointer), array.nbytes))
            pointers.append(pointer)
            check(hip.hipMemcpy(pointer, C.c_void_p(array.ctypes.data), array.nbytes, 1))
            values += [C.c_void_p(pointer.value), C.c_void_p(pointer.value),
                       C.c_int64(0), C.c_int64(array.size), C.c_int64(1)]
        args = (C.c_void_p * len(values))(
            *[C.cast(C.byref(value), C.c_void_p) for value in values])
        check(hip.hipModuleLaunchKernel(function, (n * (k // 32) + 255) // 256,
            1, 1, 256, 1, 1, 0, None, args, None))
        check(hip.hipDeviceSynchronize())
        if timing:
            signatures = {
                "hipEventCreate": [C.POINTER(C.c_void_p)],
                "hipEventRecord": [C.c_void_p, C.c_void_p],
                "hipEventSynchronize": [C.c_void_p],
                "hipEventElapsedTime": [C.POINTER(C.c_float), C.c_void_p, C.c_void_p],
                "hipEventDestroy": [C.c_void_p],
            }
            for symbol, types in signatures.items():
                getattr(hip,symbol).argtypes = types
                getattr(hip,symbol).restype = C.c_int
            events = [C.c_void_p(), C.c_void_p()]
            samples = []
            for event in events:
                check(hip.hipEventCreate(C.byref(event)))
            try:
                for _ in range(3):
                    check(hip.hipEventRecord(events[0], None))
                    for _ in range(10):
                        check(hip.hipModuleLaunchKernel(function,
                            (n * (k // 32) + 255) // 256,
                            1, 1, 256, 1, 1, 0, None, args, None))
                    check(hip.hipEventRecord(events[1], None))
                    check(hip.hipEventSynchronize(events[1]))
                    elapsed = C.c_float()
                    check(hip.hipEventElapsedTime(C.byref(elapsed), *events))
                    samples.append(elapsed.value / 10)
            finally:
                for event in events:
                    hip.hipEventDestroy(event)
            record["resident_event_ms"] = samples
            record["timing_scope"] = "HIP event window over ten resident launches; includes device dispatch"
        for index in (3, 4, 5):
            check(hip.hipMemcpy(C.c_void_p(arrays[index].ctypes.data),
                                pointers[index], arrays[index].nbytes, 2))
    finally:
        hip.hipDeviceSynchronize()
        for pointer in pointers:
            hip.hipFree(pointer)
        hip.hipModuleUnload(module)
    np.testing.assert_array_equal(arrays[3], reference.packed_codes)
    np.testing.assert_array_equal(arrays[4], reference.scale_exponents)
    assert np.isfinite(arrays[5]).all()
    for p, start, end in zip(projections, reference.row_offsets[:-1],
                             reference.row_offsets[1:], strict=True):
        codes = mx.unpack_e2m1_codes(p.packed_codes)
        levels = np.asarray([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
        source = levels[codes] * (p.e4m3_scales.astype(np.float64) * p.global_scale).repeat(16, axis=1)
        decoded = mx.exact_weights(
            mx.unpack_e2m1_codes(arrays[3][start:end]), arrays[4][:, start:end]).astype(np.float64)
        stats = arrays[5][start:end]
        np.testing.assert_allclose(stats[..., 0].sum(), np.square(source).sum(), rtol=1e-13)
        np.testing.assert_allclose(stats[..., 1].sum(), np.square(source-decoded).sum(), rtol=1e-13, atol=1e-30)
    record["correctness"] = "packed codes/exponents bitwise and independent f64 signal/error verified after timing"
    return record


@pytest.mark.parametrize("n,k", [(1, 32), (3, 64), (17, 128), (33, 256)])
def test_native_joint_sse_preserves_projection_globals_and_codes(n, k):
    rng = np.random.default_rng(1201 + n + k)
    projections = []
    for name, rows, global_scale in (("gate", n, .5), ("up", n + 1, 2.)):
        codes = rng.integers(0, 16, (rows, k), dtype=np.uint8)
        codes[0, :16] = np.arange(16, dtype=np.uint8)
        scales = rng.choice(np.asarray([0., .125, .5, 1., 1.5, 2., 3., 6.]), (rows, k // 16))
        projections.append(ingest.NVFP4Projection(
            name, mx.pack_e2m1_codes(codes),
            scales.astype(ml_dtypes.float8_e4m3fn), global_scale))
    run_leaf(projections)


@pytest.mark.parametrize("global_scale", [.7, 1.3, 1e-20])
def test_raw_e4m3_scale_bytes_and_zero_code_blocks(global_scale):
    rng = np.random.default_rng(32)
    n, k = 65, 256
    codes = rng.integers(0, 16, (n, k), dtype=np.uint8)
    codes[0] = 0
    codes[1] = 8
    scale_bytes = rng.integers(0, 127, (n, k // 16), dtype=np.uint8)
    scale_bytes[2] = 0
    scales = scale_bytes.view(ml_dtypes.float8_e4m3fn)
    run_leaf([ingest.NVFP4Projection(
        "projection", mx.pack_e2m1_codes(codes), scales, global_scale)])

def test_graph_owned_ingest_executes_with_distinct_output_buffers():
    rng = np.random.default_rng(120132)
    projections = [
        ingest.NVFP4Projection(name,
            mx.pack_e2m1_codes(rng.integers(0,16,(rows,64),dtype=np.uint8)),
            rng.choice(np.array([0,.125,.5,1,2,6]),(rows,4)).astype(ml_dtypes.float8_e4m3fn),
            scale)
        for name,rows,scale in (("gate",3,.5),("up",4,2.))]
    result = run_leaf(projections, graph_route=True, timing=True)
    assert result["frontend_schedule_tile_integrated"]
    assert len(result["resident_event_ms"]) == 3


@pytest.mark.parametrize("global_scale", [
    2.0**-124, 2.0**-123, 2.0**-100, .125, .25, .5, 1.25,
    2.0**120, 2.0**127,
])
def test_candidate_power_of_two_normalization_keeps_extreme_scales(global_scale):
    # Include signed zero, every E2M1 value, unequal K16 scales and exact
    # quantizer midpoint ratios. Check bytes, selected exponent and f64 SSE.
    codes=np.tile(np.arange(16,dtype=np.uint8),(3,4))
    scales=np.array([[.5,1.,2.,4.], [.125,.25,.5,1.], [0.,0.,1.5,3.]])
    if global_scale == 2.0**127:
        # Exercise candidates at the maximum exponent while keeping decoded
        # destination values finite in the existing fp32 weight oracle.
        codes = codes & np.uint8(11)
        scales = scales / 8
    run_leaf([ingest.NVFP4Projection(
        "extreme", mx.pack_e2m1_codes(codes),
        scales.astype(ml_dtypes.float8_e4m3fn), global_scale)])
