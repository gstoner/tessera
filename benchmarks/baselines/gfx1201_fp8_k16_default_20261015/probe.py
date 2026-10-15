"""Synthetic source-preserving K16 control; host expansion is diagnostic only."""
from __future__ import annotations

import ctypes as C
import hashlib
import json
import lzma
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
import ml_dtypes
import numpy as np
from tessera import runtime as rt
from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape, compile_blockscale
from benchmarks.rocm.benchmark_gfx1201_three_formats import quantize_fp8
from benchmarks.rocm.benchmark_gfx1201_mxfp8_package import Resident
from benchmarks.rocm.folded_graph_windows import FoldedGraphWindows
from benchmarks.rocm.record_gfx1201_mxfp4_folded_load_schedule import DeviceClock
from tests._support import rocm_isa


def digest(value):
    return hashlib.sha256(value).hexdigest()


def run(output_path):
    compiler = Path(os.environ["TESSERA_OPT"])
    llvm = Path(os.environ["LLVM_BIN"])
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("requires owning gfx1201")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):
        raise RuntimeError("HIP unavailable")
    ordinal, name = C.c_int(), C.create_string_buffer(256)
    Resident.check(hip.hipGetDevice(C.byref(ordinal)))
    Resident.check(hip.hipDeviceGetName(name, len(name), ordinal.value))
    if not name.value:
        raise RuntimeError("missing GPU identity")
    levels = np.asarray([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    rng = np.random.default_rng(151091)
    packet = {
        "schema": "tessera.gfx1201.nvfp4_k16_control.v1",
        "architecture": rt._rocm_live_arch(),
        "device": name.value.decode(),
        "compiler_sha256": digest(compiler.read_bytes()),
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in [
                Path(__file__), ROOT / "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAGemmKernel.cpp",
                ROOT / "python/tessera/compiler/rocm_fp8_blockscale.py",
            ]
        },
        "scope": "synthetic host NVFP4 value expansion into existing native FP8 packages; not native NVFP4 ingest or model-quality acceptance",
        "selector_promotion": False,
        "rows": [],
    }
    clock = DeviceClock(hip, compiler, llvm)
    try:
        for m, n, k in [(7, 19, 32), (128, 128, 128), (129, 144, 128)]:
            a = rng.normal(size=(m, k)).astype(np.float32)
            codes = rng.integers(0, 16, size=(n, k), dtype=np.uint8)
            values = levels[codes]
            scales = rng.choice(np.array([.125, .5, .75, 1., 3.5, 7.5], np.float32), size=(n, k // 16))
            e4m3 = scales.astype(ml_dtypes.float8_e4m3fn)
            globals_ = np.where(np.arange(n) < n // 2, np.float32(.03125), np.float32(.21875)).astype(np.float32)
            sb = np.ascontiguousarray((e4m3.astype(np.float32) * globals_[:, None]).T)
            qb = np.ascontiguousarray(values.astype(ml_dtypes.float8_e4m3fn))
            np.testing.assert_array_equal(qb.astype(np.float32).view(np.uint32), values.view(np.uint32))
            qa, _, scales_a, da, _ = quantize_fp8(a, np.zeros((k, n), np.float32), 16, 1)
            expected_b = (levels[codes].astype(np.float64) *
                          np.repeat(e4m3.astype(np.float64), 16, axis=1) * globals_.astype(np.float64)[:, None]).T
            represented = qb.astype(np.float64).T * np.repeat(sb.astype(np.float64), 16, axis=0)
            np.testing.assert_array_equal(represented, expected_b)
            shape = BlockScaleShape(m, n, k, 16, 1, "nk", "bf16")
            started = time.perf_counter_ns()
            package = compile_blockscale(shape)
            compile_ms = (time.perf_counter_ns() - started) / 1e6
            output = np.full((m, n), -101, dtype=ml_dtypes.bfloat16)
            buffers = dict(a=qa, b=qb, a_scale=scales_a[0], b_scale=sb, o=output)
            artifact = rt.RuntimeArtifact(
                metadata={"target": package.image.target}, native_image=package.image,
                launch_descriptor=package.descriptor, tile_ir=package.tile_ir, target_ir=package.target_ir,
            )
            binding = dict(buffers=buffers, scalars=dict(M=m, N=n, K=k))
            ideal, absolute = da @ expected_b, np.abs(da) @ np.abs(expected_b)
            gamma = (k + 8 * (k // 16) + 16) * 2.**-24
            gamma /= 1 - gamma
            bound = gamma * absolute + 2.**-8 * (np.abs(ideal) + gamma * absolute) + 2.**-126

            def check(value):
                wide = value.astype(np.float64)
                if not np.isfinite(wide).all() or np.any(np.abs(wide - ideal) > bound):
                    raise AssertionError("native output exceeds elementwise f32/BF16 bound")

            def launch():
                receipt = rt.launch(artifact, binding)
                if not receipt.get("ok") or receipt.get("execution_kind") != "native_gpu":
                    raise RuntimeError(receipt)

            launch()
            check(output)
            isa = rocm_isa.disassemble(package.image.payload, chip="gfx1201")
            if "v_wmma_f32_16x16x16_fp8_fp8" not in isa:
                raise AssertionError("missing owning FP8 WMMA instruction")
            isa_file = f"{package.image.payload_digest}.s.xz"
            if (output_path.parent / isa_file).exists():
                isa = lzma.decompress((output_path.parent / isa_file).read_bytes()).decode()
            else:
                (output_path.parent / isa_file).write_bytes(lzma.compress(isa.encode()))
            engine = Resident(hip, package, buffers, shape)
            graphs = FoldedGraphWindows(clock)
            try:
                count = 32
                while graphs.window(engine, count, bracketed=True)["device_window_ms"] < 5:
                    count *= 2
                    if count > 16384:
                        raise RuntimeError("control calibration did not reach 5ms")
                samples = [graphs.window(engine, count, bracketed=True) for _ in range(3)]
                if any(sample["device_event_disagreement"] > .05 for sample in samples):
                    raise RuntimeError("device timer disagreement")
                check(engine.result(output))
                host = []
                for _ in range(3):
                    started = time.perf_counter_ns()
                    for _ in range(3):
                        launch()
                    host.append((time.perf_counter_ns() - started) / 1e6 / 3)
                check(output)
                packet["rows"].append({
                    "shape_mnk": [m, n, k], "scale_k": 16,
                    "input_sha256": {"a": digest(a.tobytes()), "codes": digest(codes.tobytes()),
                                     "e4m3_scales": digest(e4m3.tobytes()), "globals": digest(globals_.tobytes())},
                    "source_preservation": "all E2M1 values/signed zeros and independent K16 scale planes; binary-exact control globals",
                    "correctness": "elementwise f32 forward bound plus BF16 store", "violations": 0,
                    "max_abs_error": float(np.max(np.abs(output.astype(np.float64) - ideal))),
                    "route": package.descriptor.provenance["route"], "abi": package.descriptor.abi_id,
                    "provenance": dict(package.descriptor.provenance),
                    "image_sha256": package.image.payload_digest, "isa_file": isa_file,
                    "isa_sha256": digest(isa.encode()),
                    "tile_sha256": digest(package.tile_ir.encode()), "target_sha256": digest(package.target_ir.encode()),
                    "compile_ms": compile_ms, "device_graph_samples": samples,
                    "device_execution_dispatch_median_ms": statistics.median(s["device_window_ms"] / s["launches"] for s in samples),
                    "completed_host_samples_ms": host, "completed_host_median_ms": statistics.median(host),
                })
                output_path.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
                print(json.dumps(packet["rows"][-1]), flush=True)
            finally:
                graphs.close()
                engine.close()
    finally:
        clock.close()


if __name__ == "__main__":
    output_path = Path(sys.argv[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(output_path)
