"""Exact gfx1201 FP8 image-cache identity and separated timing domains."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler import rocm_native as native, scheduled_matmul
from tessera.compiler.graph_ir import GraphIRModule, GraphIRFunction, IRArg, IROp, IRType
from benchmarks.rocm.benchmark_rocm_gemm_pipeline_vs_direct import _load_hip
from benchmarks.rocm.benchmark_rocm_gemm_schedule_matrix import Case, DeviceCase


def graph(shape, lhs, rhs):
    m, k, n = shape
    elements = {"fp8_e4m3": "f8E4M3FN", "fp8_e5m2": "f8E5M2", "fp16": "f16", "bf16": "bf16"}
    a = IRType(f"tensor<{m}x{k}x{elements[lhs]}>", (str(m), str(k)), lhs)
    b = IRType(f"tensor<{k}x{n}x{elements[rhs]}>", (str(k), str(n)), rhs)
    out = IRType(f"tensor<{m}x{n}xf32>", (str(m), str(n)), "fp32")
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="fp8_cache_matmul",
                args=[IRArg("a", a), IRArg("b", b)],
                result_types=[out],
                body=[
                    IROp(
                        result="o",
                        op_name="tessera.matmul",
                        operands=["%a", "%b"],
                        operand_types=[str(a), str(b)],
                        result_type=str(out),
                        kwargs={"activation": "none"},
                    )
                ],
                return_values=["%o"],
            )
        ]
    )


def export_gfx1151_lds_packages(output: Path):
    """Cross-compile packages only; this function makes no device claim."""
    images, rows = {}, []
    for dtype in ("fp16", "bf16"):
        for shape in ((16, 32, 16), (17, 31, 19), (48, 33, 16)):
            scheduled = scheduled_matmul.lower_scheduled_matmul(graph(shape, dtype, dtype), target="rocm_gfx1151")
            package = native.package_scheduled_matmul(
                scheduled, pipeline_name="tessera-lower-to-rocm", staging="lds", k_unroll=1
            )
            images[package.image.image_digest] = package.image.to_dict()
            rows.append(
                {
                    "dtype": dtype,
                    "shape_mkn": list(shape),
                    "image_digest": package.image.image_digest,
                    "descriptor": package.descriptor.to_dict(),
                    "tile_ir": package.tile_ir,
                    "target_ir": package.target_ir,
                }
            )
    files = (
        "python/tessera/runtime.py",
        "python/tessera/compiler/jit.py",
        "python/tessera/compiler/driver.py",
        "python/tessera/compiler/rocm_native.py",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAGemmKernel.cpp",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/IR/TesseraROCMOps.cpp",
        "src/compiler/codegen/Tessera_ROCM_Backend/include/TesseraROCM/IR/TesseraROCMOps.td",
        "benchmarks/rocm/benchmark_rocm_fp8_shape_cache.py",
    )
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    packet = {
        "schema": "tessera.rocm.lds_sibling_packages.v1",
        "status": "cross_compiled_not_executed",
        "architecture": "gfx1151",
        "images": images,
        "rows": rows,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {path: sha(path) for path in files},
        "binary_sha256": {name: sha(os.environ[name]) for name in ("TESSERA_OPT", "TESSERA_ROCM_OPT")},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, indent=2) + "\n")


class DescriptorDeviceCase(DeviceCase):
    def launch(self):
        gx, gy = self.grid
        return self.hip.hipModuleLaunchKernel(
            self.fn, gx, gy, 1, self.block_threads, 1, 1, 0, None, self.arg_array, None
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    if args.samples <= 0 or args.repetitions <= 0:
        parser.error("samples and repetitions must be positive")

    assert rt._rocm_live_arch() == "gfx1201"
    hip = _load_hip()
    rows = []
    types = {"fp8_e4m3": ml_dtypes.float8_e4m3fn, "fp8_e5m2": ml_dtypes.float8_e5m2}
    public = ts.from_text(
        'def fp8_product(a,b):\n    return tessera.ops.matmul(a,b,output_dtype="fp32")\n', target="rocm_gfx1201"
    )
    for staging, unroll, lhs, rhs in (
        (stage, u, a, b)
        for stage, u in (("register", 1), ("register", 2), ("register", 4), ("register", 8), ("lds", 1))
        for a in types
        for b in types
    ):
        images = []
        for m, k, n in ((16, 32, 16), (17, 31, 19), (48, 33, 16)):
            scheduled = scheduled_matmul.lower_scheduled_matmul(graph((m, k, n), lhs, rhs), target="rocm_gfx1201")
            begin = time.perf_counter()
            package = native.package_scheduled_matmul(
                scheduled, pipeline_name="tessera-lower-to-rocm", k_unroll=unroll, staging=staging
            )
            package_ms = (time.perf_counter() - begin) * 1000
            rng = np.random.default_rng(47032 + m)
            a = rng.uniform(-0.5, 0.5, (m, k)).astype(types[lhs])
            b = rng.uniform(-0.5, 0.5, (k, n)).astype(types[rhs])
            expected = a.astype(np.float32) @ b.astype(np.float32)
            macro = package.descriptor.provenance["macro_tile"]
            device = DescriptorDeviceCase(
                hip,
                package.image.payload,
                Case("fp8-cache", m, n, k, dtype=lhs),
                (macro[0] // 16, macro[1] // 16),
                a,
                b,
                None,
                entry_symbol=package.descriptor.entry_symbol,
            )
            device.block_threads = package.descriptor.provenance["workgroup"][0]
            try:
                actual = device.download()
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-3)
                event = device.measure(trials=args.samples, iterations=args.repetitions, warmup=10)
            finally:
                device.close()
            output = np.zeros((m, n), np.float32)
            artifact = rt.RuntimeArtifact(
                metadata={"target": package.image.target},
                native_image=package.image,
                launch_descriptor=package.descriptor,
                tile_ir=scheduled.tile_ir,
                target_ir=package.target_ir,
            )
            launch_args = {"buffers": {"a": a, "b": b, "o": output}, "scalars": {"M": m, "N": n, "K": k}}
            result = rt.launch(artifact, launch_args)
            assert result["ok"] and result["execution_kind"] == "native_gpu"
            np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-3)
            wall = []
            for _ in range(args.samples):
                begin = time.perf_counter()
                for _ in range(10):
                    assert rt.launch(artifact, launch_args)["ok"]
                wall.append((time.perf_counter() - begin) * 1000 / 10)
            public_row = {}
            if staging == "register" and unroll == 1:
                begin = time.perf_counter()
                public_output = public(a, b)
                public_first_ms = (time.perf_counter() - begin) * 1000
                np.testing.assert_allclose(public_output, expected, rtol=1e-4, atol=1e-3)
                assert public.execution_kind == "native_gpu"
                public_samples = []
                for _ in range(args.samples):
                    begin = time.perf_counter()
                    for _ in range(10):
                        public(a, b)
                    public_samples.append((time.perf_counter() - begin) * 1000 / 10)
                public_row = {
                    "public_jit_first_call_ms": public_first_ms,
                    "public_jit_completed_samples_ms": public_samples,
                    "public_jit_completed_median_ms": statistics.median(public_samples),
                    "public_jit_execution_kind": public.execution_kind,
                    "public_jit_image_digest": public.compile_result.native_image.image_digest,
                    "public_jit_compiler_path": public._native_descriptor_last_receipt["compiler_path"],
                }
            images.append(package.image.image_digest)
            rows.append(
                {
                    **public_row,
                    "staging": staging,
                    "k_unroll": unroll,
                    "lhs": lhs,
                    "rhs": rhs,
                    "shape_mkn": [m, k, n],
                    "package_wall_ms": package_ms,
                    "compile_state": package.image.compile_state,
                    "image_digest": package.image.image_digest,
                    "entry": package.descriptor.entry_symbol,
                    "abi_id": package.descriptor.abi_id,
                    "workgroup": package.descriptor.provenance["workgroup"],
                    "grid": list(device.grid),
                    "correctness": "passed_before_timing",
                    "target_ir_sha256": hashlib.sha256(package.target_ir.encode()).hexdigest(),
                    "tile_ir_sha256": hashlib.sha256(package.tile_ir.encode()).hexdigest(),
                    "hsaco_sha256": hashlib.sha256(package.image.payload).hexdigest(),
                    "device_event_samples_ms": event,
                    "device_event_median_ms": statistics.median(event),
                    "completed_launch_wall_samples_ms": wall,
                    "completed_launch_wall_median_ms": statistics.median(wall),
                    "max_abs_error": float(np.max(np.abs(actual - expected))),
                }
            )
            print(rows[-1], flush=True)
        assert len(set(images)) == 1
    sources = [
        "python/tessera/compiler/rocm_native.py",
        "python/tessera/runtime.py",
        "src/compiler/codegen/Tessera_ROCM_Backend/include/TesseraROCM/IR/TesseraROCMOps.td",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAGemmKernel.cpp",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/IR/TesseraROCMOps.cpp",
        "benchmarks/rocm/benchmark_rocm_fp8_shape_cache.py",
    ]
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    cmake = (Path(os.environ["TESSERA_OPT"]).parents[2] / "CMakeCache.txt").read_text()
    configuration = {
        name: f"{name}:BOOL=ON" in cmake
        for name in (
            "TESSERA_BUILD_NVIDIA_BACKEND",
            "TESSERA_BUILD_ROCM_BACKEND",
            "TESSERA_BUILD_X86_BACKEND",
            "TESSERA_BUILD_EBM_BACKEND",
            "TESSERA_BUILD_CLIFFORD_BACKEND",
        )
    }
    packet = {
        "schema": "tessera.rocm.fp8_shape_cache.v1",
        "status": "source_pinned_characterization",
        "architecture": rt._rocm_live_arch(),
        "compiler_configuration": configuration,
        "gpu_report": subprocess.check_output(["/opt/rocm/bin/rocminfo"], text=True),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {p: sha(p) for p in sources},
        "binary_sha256": {
            name: sha(os.environ[name]) for name in ("TESSERA_OPT", "TESSERA_ROCM_OPT", "TESSERA_ROCM_NATIVE_IMAGE_LIB")
        },
        "method": {
            "event": "resident HIP-event batches; CPU enqueue gaps possible",
            "samples": args.samples,
            "repetitions": args.repetitions,
            "wall": "completed rt.launch; uploads/readback included",
            "host_repetitions": 10,
            "warmup": 10,
            "package": "first package wall per shape, new shapes must reuse the physical image",
            "public_jit": "completed frontend call; first call follows same-case image priming, not a cold compiler process; warm calls separate, default register recipe only",
            "performance_comparison_claim": False,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
