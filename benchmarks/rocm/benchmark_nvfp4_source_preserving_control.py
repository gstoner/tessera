"""Diagnostic checkpoint NVFP4 expansion into native FP8 K16 consumers.

Host expansion is an attribution control, not a native packed NVFP4 route.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import lzma
import os
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

import ml_dtypes
import numpy as np
from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx, rocm_nvfp4_ingest as ingest
from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape, compile_blockscale
from benchmarks.rocm import benchmark_rocm_nvfp4_checkpoint as checkpoint
from benchmarks.rocm import benchmark_gfx1201_three_formats as formats
from benchmarks.rocm.checkpoint_activations import load_capture
from benchmarks.rocm.benchmark_gfx1201_mxfp8_package import Resident
from benchmarks.rocm.folded_graph_windows import FoldedGraphWindows
from benchmarks.rocm.record_gfx1201_mxfp4_folded_load_schedule import DeviceClock
from tests._support import rocm_isa

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(value).hexdigest()


def expand_projection_control(projections: Sequence[ingest.NVFP4Projection]) -> tuple[np.ndarray, np.ndarray]:
    """Reference-only expansion; retain K16 blocks and each projection global."""
    if not projections or any(not p.name for p in projections) or len({p.name for p in projections}) != len(projections):
        raise ValueError("control needs nonempty distinct projections")
    values, scales = [], []
    width = None
    for projection in projections:
        packed = np.asarray(projection.packed_codes)
        if packed.dtype != np.uint8 or packed.ndim != 2:
            raise TypeError("NVFP4 source codes must be uint8 [N,K/2]")
        n, packed_k = packed.shape
        k = packed_k * 2
        if n <= 0 or k <= 0 or k % 16:
            raise ValueError("NVFP4 source needs positive N/K16 extents")
        source_scales = np.asarray(projection.e4m3_scales)
        if source_scales.dtype != np.dtype(ml_dtypes.float8_e4m3fn):
            raise TypeError("NVFP4 source scales require decoded E4M3 storage")
        if source_scales.shape != (n, k // 16):
            raise ValueError("NVFP4 source scales require [N,K/16]")
        global_scale = float(projection.global_scale)
        if not np.isfinite(global_scale) or global_scale <= 0:
            raise ValueError("NVFP4 source global must be finite and positive")
        scaled = source_scales.astype(np.float64) * global_scale
        if not np.isfinite(scaled).all() or np.any(scaled < 0):
            raise ValueError("NVFP4 source scales must be finite and non-negative")
        if width is not None and width != k:
            raise ValueError("control projection K dimensions differ")
        width = k
        decoded = ingest._E2M1[mx.unpack_e2m1_codes(packed)]
        encoded = np.ascontiguousarray(decoded.astype(ml_dtypes.float8_e4m3fn))
        np.testing.assert_array_equal(encoded.astype(np.float32).view(np.uint32), decoded.view(np.uint32))
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            fp32 = scaled.astype(np.float32)
        if not np.isfinite(fp32).all() or np.any((scaled > 0) & (fp32 == 0)):
            raise ValueError("source scale product cannot be represented by this FP32 control")
        values.append(encoded)
        scales.append(fp32)
    return np.ascontiguousarray(np.concatenate(values)), np.ascontiguousarray(np.concatenate(scales).T)


def measure(args):
    if args.windows < 3 or not args.m or any(type(m) is not int or m <= 0 for m in args.m):
        raise ValueError("positive row counts and at least three windows required")
    pool, activation_metadata = load_capture(args.activations, args.m)
    if pool.shape[1] != 4096:
        raise ValueError("pinned Qwen3 control requires hidden width 4096")
    os.environ["TESSERA_OPT"] = str(args.compiler.resolve())
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("requires owning gfx1201")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):
        raise RuntimeError("HIP unavailable")
    ordinal, name = C.c_int(), C.create_string_buffer(256)
    Resident.check(hip.hipGetDevice(C.byref(ordinal)))
    Resident.check(hip.hipDeviceGetName(name, len(name), ordinal.value))
    if not name.value:
        raise RuntimeError("missing device identity")
    loaded = [checkpoint._load_projection("model.layers.0.mlp." + role + ".weight") for role in ("gate_proj", "up_proj")]
    sources = [item["source"] for item in loaded]
    projections = [item["projection"] for item in loaded]
    base = np.concatenate([np.asarray(item["bf16"], np.float32) for item in loaded])
    del loaded
    qb, sb = expand_projection_control(projections)
    # Independent shipped-weight decode from the original packed nibbles.
    levels = np.asarray([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float64)
    reference_parts = []
    for projection in projections:
        packed = projection.packed_codes
        decoded = np.empty((packed.shape[0], packed.shape[1] * 2), np.float64)
        decoded[:, 0::2] = levels[packed & 15]
        decoded[:, 1::2] = levels[packed >> 4]
        decoded *= np.repeat(projection.e4m3_scales.astype(np.float64) * projection.global_scale, 16, axis=1)
        reference_parts.append(decoded)
    shipped = np.concatenate(reference_parts).T
    del reference_parts, projections
    represented = qb.astype(np.float64).T * np.repeat(sb.astype(np.float64), 16, axis=0)
    delta = np.abs(represented - shipped)
    if np.any(delta > np.abs(shipped) * 2.**-24 + 6 * 2.**-150):
        raise AssertionError("expansion changes source weights beyond FP32 scale rounding")
    packet: dict[str, Any] = dict(
        schema="tessera.gfx1201.nvfp4_checkpoint_k16_control.v1",
        architecture=rt._rocm_live_arch(), device=name.value.decode(),
        compiler_sha256=digest(args.compiler.read_bytes()),
        source_checkpoints=sources, activation_source=activation_metadata,
        source_scale_rounding_error=formats.errors(represented, shipped),
        native_nvfp4_expansion=False, packed_transport=False, model_quality_acceptance=False,
        selector_promotion=False, physical_storage="expanded_e4m3_codes_plus_fp32_k16_scales",
        rows=[],
    )
    source_files = [
        Path(__file__), Path(formats.__file__), Path(checkpoint.__file__),
        ROOT / "python/tessera/compiler/rocm_fp8_blockscale.py",
        ROOT / "src/compiler/programming_model/lib/PMPasses.cpp",
        ROOT / "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAGemmKernel.cpp",
        ROOT / "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp",
    ]
    packet["source_sha256"] = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in source_files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clock = DeviceClock(hip, args.compiler, args.llvm_bin)
    try:
        for m in args.m:
            a = np.ascontiguousarray(pool[:m], dtype=np.float32)
            k, n = shipped.shape
            qa, _, scales, da, _ = formats.quantize_fp8(a, np.zeros((k, 1), np.float32), 16, 1)
            original = a.astype(np.float64) @ base.astype(np.float64).T
            shipped_reference = a.astype(np.float64) @ shipped
            represented_reference = a.astype(np.float64) @ represented
            ideal = da @ represented
            magnitude = np.abs(da) @ np.abs(represented)
            gamma = (k + 8 * (k // 16) + 16) * 2.**-24
            gamma /= 1 - gamma
            bound = gamma * magnitude + 2.**-8 * (np.abs(ideal) + gamma * magnitude) + 2.**-126

            def verify(value):
                wide = value.astype(np.float64)
                if wide.shape != ideal.shape or not np.isfinite(wide).all() or np.any(np.abs(wide - ideal) > bound):
                    raise AssertionError("native arithmetic exceeds independent f32/BF16 bound")

            shape = BlockScaleShape(m, n, k, 16, 1, "nk", "bf16")
            started = time.perf_counter_ns()
            package = compile_blockscale(shape)
            compile_ms = (time.perf_counter_ns() - started) / 1e6
            output = np.full((m, n), -101, dtype=ml_dtypes.bfloat16)
            buffers = dict(a=qa, b=qb, a_scale=scales[0], b_scale=sb, o=output)
            artifact = rt.RuntimeArtifact(
                metadata={"target": package.image.target}, native_image=package.image,
                launch_descriptor=package.descriptor, tile_ir=package.tile_ir, target_ir=package.target_ir,
            )
            binding = dict(buffers=buffers, scalars=dict(M=m, N=n, K=k))

            def launch():
                receipt = rt.launch(artifact, binding)
                if not receipt.get("ok") or receipt.get("execution_kind") != "native_gpu":
                    raise RuntimeError(receipt)

            launch()
            verify(output)
            isa_file = args.output.parent / (package.image.payload_digest + ".s.xz")
            if isa_file.exists():
                isa = lzma.decompress(isa_file.read_bytes()).decode()
            else:
                isa = rocm_isa.disassemble(package.image.payload, chip="gfx1201")
                isa_file.write_bytes(lzma.compress(isa.encode()))
            if "v_wmma_f32_16x16x16_fp8_fp8" not in isa:
                raise AssertionError("native image lacks owning FP8 WMMA")
            engine = Resident(hip, package, buffers, shape)
            graphs = FoldedGraphWindows(clock)
            try:
                count = 32
                while graphs.window(engine, count, bracketed=True)["device_window_ms"] < 20:
                    count *= 2
                    if count > 16384:
                        raise RuntimeError("control timing calibration failed")
                samples = [graphs.window(engine, count, bracketed=True) for _ in range(args.windows)]
                if any(sample["device_event_disagreement"] > .05 for sample in samples):
                    raise RuntimeError("device timer disagreement")
                verify(engine.result(output))
                host = []
                for _ in range(args.windows):
                    started = time.perf_counter_ns()
                    for _ in range(3):
                        launch()
                    host.append((time.perf_counter_ns() - started) / 1e6 / 3)
                verify(output)
                row = dict(
                    shape_mnk=[m, n, k], activation_sha256=digest(a.tobytes()),
                    source_projection_error=formats.errors(shipped_reference, original),
                    represented_projection_error=formats.errors(represented_reference, original),
                    scale_rounding_projection_error=formats.errors(represented_reference, shipped_reference),
                    native_output_error=formats.errors(output.astype(np.float64), original),
                    native_vs_shipped_error=formats.errors(output.astype(np.float64), shipped_reference),
                    correctness="elementwise_f32_forward_bound_plus_bf16_store", violations=0,
                    physical_input_bytes=sum(v.nbytes for v in buffers.values() if v is not output),
                    prepare_compile_ms=compile_ms, provenance=dict(package.descriptor.provenance),
                    abi=package.descriptor.abi_id, image_sha256=package.image.payload_digest,
                    tile_sha256=digest(package.tile_ir.encode()), target_sha256=digest(package.target_ir.encode()),
                    isa_file=isa_file.name, isa_sha256=digest(isa.encode()),
                    device_execution_dispatch_median_ms=statistics.median(s["device_window_ms"] / s["launches"] for s in samples),
                    device_graph_samples=samples, completed_host_samples_ms=host,
                    completed_host_median_ms=statistics.median(host),
                )
                packet["rows"].append(row)
                args.output.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
                print(json.dumps({key: row[key] for key in (
                    "shape_mnk", "source_projection_error", "native_output_error",
                    "device_execution_dispatch_median_ms", "completed_host_median_ms",
                )}), flush=True)
            finally:
                graphs.close()
                engine.close()
    finally:
        clock.close()
    return packet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--windows", type=int, default=3)
    measure(parser.parse_args())
