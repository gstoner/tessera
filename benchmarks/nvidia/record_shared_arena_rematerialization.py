#!/usr/bin/env python3
"""Record SM120 NVPTX shared-arena and rematerialization evidence.

The compiler-owned Tile pipeline supplies the arena size and reuse offsets.
This recorder then proves the matching address-space-3 object through
LLVM->NVPTX->ptxas, retains resources/occupancy/spills, and compares an exact
device retained-arena kernel with a register-rematerialized equivalent.  The
record is evidence-only and never promotes a selector.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_shared_arena_rematerialization.json"
FIXTURE = (
    ROOT / "src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia" / "sm120_shared_arena_materialization.mlir"
)
ARENA_RE = re.compile(r"tile\.smem_arena_bytes = (?P<bytes>\d+) : i64")
OFFSET_RE = re.compile(r"arith\.constant (?P<offset>\d+) : index")


def _run(command: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        input=input_text,
        check=True,
        capture_output=True,
        text=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stability(values: list[float]) -> float:
    return (max(values) - min(values)) / min(values) * 100.0


def _cohort_medians(call: Callable[[], float], samples: int) -> list[float]:
    from benchmarks.nvidia._clock_conditioning import condition_sm120

    cohorts: tuple[list[float], list[float]] = ([], [])
    for sample in range(samples):
        for cohort in (0, 1) if sample % 2 == 0 else (1, 0):
            condition_sm120(reps=20)
            cohorts[cohort].append(float(call()))
    return [float(statistics.median(values)) for values in cohorts]


def _arena_plan(tessera_nvidia_opt: Path) -> tuple[str, int, tuple[int, ...]]:
    lowered = _run(
        [
            str(tessera_nvidia_opt),
            "--tessera-lower-to-nvidia-sm120",
            "--allow-unregistered-dialect",
            str(FIXTURE),
        ]
    ).stdout
    sizes = {int(match.group("bytes")) for match in ARENA_RE.finditer(lowered)}
    if sizes != {1024}:
        raise RuntimeError(f"SM120 arena fixture produced unexpected sizes {sorted(sizes)}")
    offsets = tuple(int(match.group("offset")) for match in OFFSET_RE.finditer(lowered))
    if offsets != (0, 512, 0):
        raise RuntimeError(f"SM120 arena fixture produced unexpected offsets {offsets}")
    if "memref<1024xi8, 3>" not in lowered or "tile.alloc_shared" in lowered:
        raise RuntimeError("SM120 arena did not become one address-space-3 object")
    return lowered, 1024, offsets


def _llvm_contract(arena_bytes: int) -> str:
    return f"""target triple = "nvptx64-nvidia-cuda"

@__tessera_smem_arena_contract = internal addrspace(3) global
    [{arena_bytes} x i8] undef, align 16

define void @tessera_nvptx_shared_arena_contract(ptr addrspace(1) %out) {{
entry:
  %first = getelementptr inbounds [{arena_bytes} x i8],
      ptr addrspace(3) @__tessera_smem_arena_contract, i64 0, i64 0
  %second = getelementptr inbounds [{arena_bytes} x i8],
      ptr addrspace(3) @__tessera_smem_arena_contract, i64 0, i64 512
  store volatile i8 11, ptr addrspace(3) %first, align 1
  store volatile i8 31, ptr addrspace(3) %second, align 1
  %a = load volatile i8, ptr addrspace(3) %first, align 1
  %b = load volatile i8, ptr addrspace(3) %second, align 1
  %a32 = zext i8 %a to i32
  %b32 = zext i8 %b to i32
  %sum = add i32 %a32, %b32
  store i32 %sum, ptr addrspace(1) %out, align 4
  ret void
}}

!nvvm.annotations = !{{!0}}
!0 = !{{ptr @tessera_nvptx_shared_arena_contract, !"kernel", i32 1}}
"""


def _cuda_source(arena_bytes: int) -> str:
    return f"""
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void tessera_retained_shared_arena(
    const float* x, float* out) {{
  __shared__ __align__(16) unsigned char arena[{arena_bytes}];
  if (threadIdx.x == 0) {{
    volatile float* first = reinterpret_cast<volatile float*>(arena);
    volatile float* second = reinterpret_cast<volatile float*>(arena + 512);
    *first = x[0] * x[1];
    *second = x[2];
    out[0] = *first + *second;
  }}
}}

extern "C" __global__ void tessera_rematerialized_register_expression(
    const float* x, float* out) {{
  if (threadIdx.x == 0)
    out[0] = x[0] * x[1] + x[2];
}}

static int launch_route(int route, const float* x, float* out) {{
  if (route == 0)
    tessera_retained_shared_arena<<<1, 128>>>(x, out);
  else if (route == 1)
    tessera_rematerialized_register_expression<<<1, 128>>>(x, out);
  else
    return 2;
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}}

extern "C" int tessera_arena_run(int route, float* host_out) {{
  const float host_x[3] = {{6.0f, 6.0f, 6.0f}};
  float *x = nullptr, *out = nullptr;
  if (!host_out || cudaMalloc(&x, sizeof(host_x)) != cudaSuccess ||
      cudaMalloc(&out, sizeof(float)) != cudaSuccess)
    return 3;
  int rc = 0;
  if (cudaMemcpy(x, host_x, sizeof(host_x), cudaMemcpyHostToDevice) != cudaSuccess ||
      launch_route(route, x, out) != 0 ||
      cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(host_out, out, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
    rc = 3;
  cudaFree(out);
  cudaFree(x);
  return rc;
}}

extern "C" int tessera_arena_measure_device(
    int route, int warmup, int repetitions, float* latency_ms) {{
  const float host_x[3] = {{6.0f, 6.0f, 6.0f}};
  float *x = nullptr, *out = nullptr;
  cudaEvent_t begin = nullptr, end = nullptr;
  if (!latency_ms || warmup < 1 || repetitions < 1 ||
      cudaMalloc(&x, sizeof(host_x)) != cudaSuccess ||
      cudaMalloc(&out, sizeof(float)) != cudaSuccess ||
      cudaMemcpy(x, host_x, sizeof(host_x), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaEventCreate(&begin) != cudaSuccess ||
      cudaEventCreate(&end) != cudaSuccess)
    return 3;
  int rc = 0;
  for (int i = 0; i < warmup && rc == 0; ++i)
    rc = launch_route(route, x, out);
  if (rc == 0 && cudaDeviceSynchronize() != cudaSuccess)
    rc = 3;
  if (rc == 0 && cudaEventRecord(begin) != cudaSuccess)
    rc = 3;
  for (int i = 0; i < repetitions && rc == 0; ++i)
    rc = launch_route(route, x, out);
  if (rc == 0 && (cudaEventRecord(end) != cudaSuccess ||
                  cudaEventSynchronize(end) != cudaSuccess))
    rc = 3;
  float total = 0.0f;
  if (rc == 0 && cudaEventElapsedTime(&total, begin, end) != cudaSuccess)
    rc = 3;
  if (rc == 0)
    *latency_ms = total / repetitions;
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  cudaFree(out);
  cudaFree(x);
  return rc;
}}
"""


def _bind_library(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.tessera_arena_run.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    lib.tessera_arena_run.restype = ctypes.c_int
    lib.tessera_arena_measure_device.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.tessera_arena_measure_device.restype = ctypes.c_int
    return lib


def record(
    *,
    samples: int,
    device_repetitions: int,
    e2e_repetitions: int,
    warmup: int,
    stability_limit: float,
) -> dict[str, Any]:
    from benchmarks.nvidia.record_tile_fragment_resources import (
        _CudaOccupancy,
        _artifact_row,
    )

    build = ROOT / "build-nvidia-cuda"
    tnv = build / "src/compiler/codegen/tessera_gpu_backend_NVIDIA/tools" / "tessera-nvidia-opt"
    lowered, arena_bytes, offsets = _arena_plan(tnv)
    device = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,compute_cap",
            "--format=csv,noheader",
        ]
    ).stdout.strip()
    if not device.endswith("12.0"):
        raise RuntimeError(f"SM120 proof requires compute capability 12.0, found {device}")

    with tempfile.TemporaryDirectory(prefix="tessera-sm120-arena-") as temporary:
        work = Path(temporary)
        llvm_ir = work / "arena.ll"
        ptx = work / "arena.ptx"
        llvm_cubin = work / "arena-llvm.cubin"
        source = work / "arena.cu"
        cuda_cubin = work / "arena-cuda.cubin"
        library = work / "libarena.so"
        llvm_ir.write_text(_llvm_contract(arena_bytes), encoding="utf-8")
        source.write_text(_cuda_source(arena_bytes), encoding="utf-8")
        _run(
            [
                "/usr/lib/llvm-23/bin/llc",
                "-march=nvptx64",
                "-mcpu=sm_120",
                "-O3",
                str(llvm_ir),
                "-o",
                str(ptx),
            ]
        )
        ptxas = _run(
            [
                "/usr/local/cuda/bin/ptxas",
                "-arch=sm_120a",
                "-v",
                str(ptx),
                "-o",
                str(llvm_cubin),
            ]
        )
        _run(
            [
                "/usr/local/cuda/bin/nvcc",
                "-arch=sm_120",
                "-O3",
                "-cubin",
                str(source),
                "-o",
                str(cuda_cubin),
            ]
        )
        _run(
            [
                "/usr/local/cuda/bin/nvcc",
                "-arch=sm_120",
                "-O3",
                "-shared",
                "-Xcompiler",
                "-fPIC",
                str(source),
                "-o",
                str(library),
            ]
        )

        occupancy = _CudaOccupancy()
        try:
            llvm_resource = _artifact_row(
                llvm_cubin,
                "tessera_nvptx_shared_arena_contract",
                128,
                occupancy,
                route="compiler_address_space_3_contract",
            )
            resources = {
                "retained_shared_arena": _artifact_row(
                    cuda_cubin,
                    "tessera_retained_shared_arena",
                    128,
                    occupancy,
                    route="retained_shared_arena",
                ),
                "rematerialized_register_expression": _artifact_row(
                    cuda_cubin,
                    "tessera_rematerialized_register_expression",
                    128,
                    occupancy,
                    route="rematerialized_register_expression",
                ),
            }
        finally:
            occupancy.close()

        declared_shared = f".b8 __tessera_smem_arena_contract[{arena_bytes}]"
        if declared_shared not in ptx.read_text(encoding="utf-8"):
            raise RuntimeError("LLVM/NVPTX did not preserve the exact Tile arena declaration")
        accounted_shared = int(llvm_resource["static_shared_memory_bytes"])
        if accounted_shared < arena_bytes:
            raise RuntimeError("ptxas accounted less shared memory than the Tile arena")
        if resources["retained_shared_arena"]["static_shared_memory_bytes"] != accounted_shared:
            raise RuntimeError("executed retained route and LLVM contract disagree on ptxas shared-memory accounting")
        if resources["rematerialized_register_expression"]["static_shared_memory_bytes"] != 0:
            raise RuntimeError("rematerialized route unexpectedly reserves shared memory")

        lib = _bind_library(library)
        rows: list[dict[str, Any]] = []
        for route_index, route in enumerate(resources):
            output = ctypes.c_float()
            if lib.tessera_arena_run(route_index, ctypes.byref(output)) != 0:
                raise RuntimeError(f"{route} numerical launch failed")
            if output.value != 42.0:
                raise RuntimeError(f"{route} returned {output.value}, expected 42")

            def device_call() -> float:
                latency = ctypes.c_float()
                if (
                    lib.tessera_arena_measure_device(
                        route_index,
                        warmup,
                        device_repetitions,
                        ctypes.byref(latency),
                    )
                    != 0
                ):
                    raise RuntimeError(f"{route} CUDA-event timing failed")
                return float(latency.value) * 1_000_000.0

            def e2e_call() -> float:
                # Discard one lifecycle result, then amortize the next N launches.
                discarded = ctypes.c_float()
                if lib.tessera_arena_run(route_index, ctypes.byref(discarded)) != 0:
                    raise RuntimeError(f"{route} first-use discard failed")
                started = time.perf_counter_ns()
                for _ in range(e2e_repetitions):
                    value = ctypes.c_float()
                    if lib.tessera_arena_run(route_index, ctypes.byref(value)) != 0:
                        raise RuntimeError(f"{route} end-to-end launch failed")
                return (time.perf_counter_ns() - started) / e2e_repetitions

            timings = {
                "device_event": _cohort_medians(device_call, samples),
                "end_to_end": _cohort_medians(e2e_call, samples),
            }
            row = {
                "route": route,
                "numerical_output": output.value,
                "resources": resources[route],
                "timings": {
                    domain: {
                        "run_medians_ns": values,
                        "median_ns": statistics.median(values),
                        "stability_pct": _stability(values),
                        "stability_limit_pct": stability_limit,
                        "stable": _stability(values) <= stability_limit,
                    }
                    for domain, values in timings.items()
                },
                "device_repetitions": device_repetitions,
                "end_to_end_repetitions": e2e_repetitions,
                "warmup": warmup,
                "discard_first": True,
                "selector_eligible": False,
                "selector_changed": False,
            }
            print(
                f"{route}: "
                + " ".join(f"{domain}={value['stability_pct']:.3f}%" for domain, value in row["timings"].items()),
                flush=True,
            )
            if not all(value["stable"] for value in row["timings"].values()):
                raise RuntimeError(f"{route} did not meet the stability policy")
            rows.append(row)

        return {
            "schema": "tessera.nvidia.shared-arena-rematerialization.v1",
            "work_item": "E2E-SPINE-3",
            "device": device,
            "source_commit": _run(
                ["git", "rev-parse", "HEAD"],
            ).stdout.strip(),
            "method": {
                "timing_domains": ["device_event", "end_to_end"],
                "samples_per_cohort": samples,
                "discard_first": True,
                "stability_limit_pct": stability_limit,
                "selector_policy": "evidence_only_no_promotion",
            },
            "tile_plan": {
                "fixture": FIXTURE.relative_to(ROOT).as_posix(),
                "fixture_sha256": _sha256(FIXTURE),
                "lowered_sha256": hashlib.sha256(lowered.encode()).hexdigest(),
                "bytes_before": 1536,
                "arena_bytes": arena_bytes,
                "reuse_offsets": list(offsets),
            },
            "llvm_nvptx": {
                "llvm_ir_sha256": _sha256(llvm_ir),
                "ptx_sha256": _sha256(ptx),
                "cubin_sha256": _sha256(llvm_cubin),
                "declared_shared_memory_bytes": arena_bytes,
                "resource_accounting_overhead_bytes": (accounted_shared - arena_bytes),
                "ptxas_stderr": ptxas.stderr,
                "resources": llvm_resource,
            },
            "rows": rows,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--device-repetitions", type=int, default=1000)
    parser.add_argument("--e2e-repetitions", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--stability-limit", type=float, default=4.0)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    payload = record(
        samples=args.samples,
        device_repetitions=args.device_repetitions,
        e2e_repetitions=args.e2e_repetitions,
        warmup=args.warmup,
        stability_limit=args.stability_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({len(payload['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
