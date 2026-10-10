"""Correctness-gated exact-device measurements for the typed SM120 MMA route."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
SHAPES_MKN = (
    (16, 16, 8), (16, 32, 8), (16, 32, 32), (32, 32, 16),
    (48, 64, 24), (48, 67, 16), (64, 256, 64),
    (1, 1, 1), (17, 19, 23), (31, 33, 9), (48, 67, 17),
    (257, 513, 257),
)
SOURCE_PATHS = (
    "src/compiler/programming_model/lib/PMPasses.cpp",
    "src/transforms/lib/TileIRLoweringPass.cpp",
    "src/transforms/lib/TilingPass.cpp",
    "src/compiler/programming_model/include/tessera/ProgrammingModel/PMPasses.h",
    "python/tessera/compiler/pass_metadata.py",
    "tests/unit/test_sm120_legacy_scheduled_producer.py",
    "src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
    "python/tessera/compiler/nvidia_native.py",
    "python/tessera/compiler/native_artifact.py",
    "python/tessera/runtime.py",
    "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp",
    "python/tessera/compiler/emit/nvidia_cuda.py",
    "tests/device/nvidia/test_scheduled_matmul_consumers.py",
    "python/tessera/compiler/scheduled_matmul.py",
    "python/tessera/compiler/driver.py",
    "tests/unit/test_scheduled_matmul_consumers.py",
    "benchmarks/nvidia/benchmark_scheduled_typed_matmul.py",
)


def record(*, samples: int, device_reps: int, e2e_reps: int, warmup: int, legacy_entry: bool = False, via_tiling: bool = False) -> dict:
    from tessera import runtime as rt
    from tessera.compiler.canonical_compile import compile_result_from_bundle
    from tessera.compiler.driver import compile_graph_module
    from tests._support.nvidia import nvidia_cuda_host_ready
    from tests.unit.test_scheduled_matmul_consumers import _module

    if not nvidia_cuda_host_ready():
        raise RuntimeError("requires the owning NVIDIA SM120 host and toolchain")
    device = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap",
         "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if not any(field.strip() == "12.0" for field in device.split(",")):
        raise RuntimeError(f"requires sm_120 hardware, got {device}")

    if via_tiling and not legacy_entry:
        raise ValueError("via_tiling requires legacy_entry")
    rows = []
    for m, k, n in SHAPES_MKN:
        module = _module(target="nvidia_sm120", shape=(m, k, n))
        started = time.perf_counter()
        bundle = compile_graph_module(
            module, source_origin="NVIDIA-W1.1-TYPED-MATMUL",
            target="nvidia_sm120", options={"package_native": True},
            enable_tool_validation=False,
        )
        package_build_ms = (time.perf_counter() - started) * 1e3
        descriptor, image = bundle.launch_descriptor, bundle.native_image
        legacy_package = None
        if legacy_entry:
            from tessera.compiler import scheduled_matmul, nvidia_native
            scheduled = scheduled_matmul.lower_scheduled_matmul(module, target="nvidia_sm120")
            import re
            graph_name = re.search(r"func.func @([^ (]+)", scheduled.graph_ir)[1]
            legacy_source = scheduled.graph_ir.replace(
                "@"+graph_name+"(", "@"+module.functions[0].name+"(", 1)
            tile_input = legacy_source
            if via_tiling:
                tile_input = scheduled_matmul.run_tessera_opt(
                    scheduled_matmul.find_tessera_opt(), legacy_source,
                    "--tessera-tiling")
                if tile_input != scheduled.schedule_ir:
                    raise RuntimeError("native tiling does not retain canonical Schedule replay")
            delegated = scheduled_matmul.run_tessera_opt(
                scheduled_matmul.find_tessera_opt(), tile_input,
                "--tessera-tile-ir-lowering=sm=120")
            if delegated != scheduled.tile_ir or delegated != bundle.tile.text:
                raise RuntimeError("legacy entry does not retain canonical Schedule replay")
            legacy_package = nvidia_native.package_scheduled_matmul(
                replace(scheduled, tile_ir=delegated),
                pipeline_name=bundle.native_image.pipeline_name)
            image = legacy_package.image
            descriptor = replace(
                legacy_package.descriptor,
                provenance={**legacy_package.descriptor.provenance,
                            "graph_ir_digest": bundle.graph.output_digest,
                            "schedule_ir_digest": bundle.schedule.output_digest})
        package_build_ms = (time.perf_counter() - started) * 1e3
        if descriptor is None or image is None or bundle.tile is None or bundle.schedule is None:
            raise RuntimeError("typed scheduled matmul did not produce a native package")
        provenance = descriptor.provenance
        lineage = {
            "graph_ir_digest": bundle.graph.output_digest,
            "schedule_digest": provenance.get("schedule_digest", ""),
            "schedule_ir_digest": bundle.schedule.output_digest,
            "tile_ir_digest": bundle.tile.output_digest,
        }
        if any(len(value) != 64 for value in lineage.values()):
            raise RuntimeError(f"incomplete compiler lineage: {lineage}")
        for key, value in lineage.items():
            if provenance.get(key) != value:
                raise RuntimeError(f"descriptor {key} does not bind its artifact")
        if descriptor.geometry.policy != "sm120_scheduled_typed_16x8_mn":
            raise RuntimeError("typed producer package has incompatible CTA geometry")
        if "_macro_kernel" in descriptor.entry_symbol:
            raise RuntimeError("typed producer retained a macro-CTA launch symbol")
        if "tile.fragment_pack" not in bundle.tile.text or "tile.view" not in bundle.tile.text:
            raise RuntimeError("package did not traverse the typed fragment producer")
        if k > 16 and "scf.for" not in bundle.tile.text:
            raise RuntimeError("multi-panel K package has no Tile loop")
        if "nvvm.mma.sync" not in bundle.target_ir.text:
            raise RuntimeError("typed Tile route did not lower to NVIDIA MMA Target IR")

        rng = np.random.default_rng(0x1201 + m * 10000 + k * 100 + n)
        a = np.ascontiguousarray(rng.normal(size=(m, k)).astype(np.float16) * 0.25)
        b = np.asfortranarray(rng.normal(size=(k, n)).astype(np.float16) * 0.25)
        output = np.full((m, n), np.nan, np.float32)
        bindings = {"a": a, "b": b, "o": output, "M": m, "N": n, "K": k}
        artifact = (
            rt.RuntimeArtifact(metadata={"target": image.target}, native_image=image,
                               launch_descriptor=descriptor, tile_ir=legacy_package.tile_ir,
                               target_ir=legacy_package.target_ir)
            if legacy_package is not None else
            compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
        )
        result = rt.launch(artifact, bindings)
        if not result.get("ok") or result.get("execution_kind") != "native_gpu":
            raise RuntimeError(f"native correctness launch failed: {result}")
        expected = a.astype(np.float32) @ b.astype(np.float32)
        if not np.allclose(output, expected, rtol=2e-4, atol=2e-4):
            raise RuntimeError(
                f"oracle mismatch for M/K/N={m}/{k}/{n}: "
                f"max_abs={float(np.max(np.abs(output - expected))):.8g}"
            )
        max_abs_error = float(np.max(np.abs(output - expected)))

        device_samples = [
            rt._nvidia_native_descriptor_device_latency(
                image, descriptor, bindings, reps=device_reps, warmup=warmup,
            )
            for _ in range(samples)
        ]
        e2e_samples = []
        for _ in range(samples):
            started = time.perf_counter()
            for _ in range(e2e_reps):
                result = rt.launch(artifact, bindings)
                if not result.get("ok"):
                    raise RuntimeError(f"end-to-end launch failed: {result}")
            e2e_samples.append((time.perf_counter() - started) * 1e3 / e2e_reps)
        rows.append({
            "shape_mkn": [m, k, n],
            "k_panels": (k + 15) // 16,
            "route": "GraphIR->ScheduleIR->TileIR->NVIDIA Target IR->PTX",
            "legacy_input_graph_digest": (hashlib.sha256(legacy_source.encode()).hexdigest()
                                          if legacy_entry else None),
            "entry_pipeline": ("tessera-tiling; native Schedule; tessera-tile-ir-lowering=sm=120"
                               if via_tiling else "tessera-tile-ir-lowering=sm=120; native Schedule delegation"
                               if legacy_entry else "canonical SM120 pipeline"),
            "entry": descriptor.entry_symbol,
            "abi_id": descriptor.abi_id,
            "launch_geometry_policy": descriptor.geometry.policy,
            "correctness": "passed_before_timing",
            "max_abs_error": max_abs_error,
            "tolerance_rtol_atol": 2e-4,
            "package_build_ms": package_build_ms,
            "package_build_note": ("includes canonical comparison and delegated package build"
                                   if legacy_entry else "canonical package build"),
            "compiler_lineage": lineage,
            "target_ir_digest": image.target_ir_digest,
            "image_digest": image.image_digest,
            "device_event_samples_ms": device_samples,
            "device_event_median_ms": statistics.median(device_samples),
            "end_to_end_samples_ms": e2e_samples,
            "end_to_end_median_ms": statistics.median(e2e_samples),
            "device_event_cv_percent": 100 * statistics.pstdev(device_samples) /
                statistics.mean(device_samples) if len(device_samples) > 1 else 0.0,
            "end_to_end_cv_percent": 100 * statistics.pstdev(e2e_samples) /
                statistics.mean(e2e_samples) if len(e2e_samples) > 1 else 0.0,
        })

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    source_sha256 = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_PATHS
    }
    return {
        "schema": "tessera.nvidia.sm120.typed_matmul_benchmark.v2",
        "work_item": "W1.1",
        "device": device,
        "target": "nvidia_sm120",
        "source_revision": revision,
        "source_worktree_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()),
        "source_sha256": source_sha256,
        "compiler_sha256": hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        "native_launch_library_sha256": hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        "method": {
            "timing_domains": ["cuda_event", "end_to_end"],
            "samples": samples,
            "device_repetitions": device_reps,
            "end_to_end_repetitions": e2e_reps,
            "warmup": warmup,
            "selector_changed": False,
            "legacy_entry": legacy_entry,
            "via_tiling": via_tiling,
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--device-reps", type=int, default=100)
    parser.add_argument("--e2e-reps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--legacy-entry", action="store_true")
    parser.add_argument("--via-tiling", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = record(
        samples=args.samples, device_reps=args.device_reps,
        e2e_reps=args.e2e_reps, warmup=args.warmup, legacy_entry=args.legacy_entry,
        via_tiling=args.via_tiling,
    )
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
