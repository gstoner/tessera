"""Compiler-owned NVPTX dynamic/path-max shared-memory package."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .native_artifact import (
    BufferBinding,
    LaunchDescriptor,
    LaunchGeometry,
    NativeEntryPoint,
    NativeImageArtifact,
    OrderingSemantics,
    ResourceRecord,
    ScalarArgument,
    ShapeGuard,
)
from .dynamic_local_memory import align_up, evaluate_launch_expression


SM120_DYNAMIC_SMEM_ABI = (
    "tessera.nvidia.dynamic_smem.path_max."
    "output_then_else_branch.f32_i64.v1"
)
SM120_DYNAMIC_EXPR_SMEM_ABI = (
    "tessera.nvidia.dynamic_smem.expression."
    "output_base_factor_bias_fallback_branch.f32_i64.v1"
)
_cache: dict[str, tuple[str, dict[str, object], str, str]] = {}


@dataclass(frozen=True)
class NVIDIADynamicSharedPackage:
    cuda_source: str
    ptx: str
    image: NativeImageArtifact
    descriptor: LaunchDescriptor


def path_max_launch_bytes(paths: tuple[tuple[int, ...], ...]) -> int:
    if not paths:
        return 0
    return max(
        align_up(sum(align_up(size) for size in path))
        for path in paths
    )


def _tool(name: str) -> Path:
    found = shutil.which(name) or f"/usr/local/cuda/bin/{name}"
    path = Path(found)
    if not path.is_file():
        raise RuntimeError(f"CUDA dynamic shared-memory packaging requires {name}")
    return path


def _version(path: Path) -> str:
    result = subprocess.run(
        [str(path), "--version"], capture_output=True, text=True, check=False
    )
    return hashlib.sha256((result.stdout + result.stderr).encode()).hexdigest()


def _compile(source: str, entry: str) -> tuple[str, dict[str, object], str, str, str]:
    nvcc, ptxas = _tool("nvcc"), _tool("ptxas")
    compiler_fp = "nvidia-dynamic-smem-v1:" + hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    toolchain_fp = f"nvcc={_version(nvcc)};ptxas={_version(ptxas)};arch=sm_120a"
    key = hashlib.sha256(
        f"{compiler_fp}\n{toolchain_fp}\n{source}".encode()
    ).hexdigest()
    cached = _cache.get(key)
    state = "warm_cache" if cached is not None else "cold"
    if cached is None:
        with tempfile.TemporaryDirectory(prefix="tessera-sm120-dynamic-smem-") as tmp:
            root = Path(tmp)
            cu = root / "kernel.cu"
            ptx_path = root / "kernel.ptx"
            cubin = root / "kernel.cubin"
            cu.write_text(source, encoding="utf-8")
            emitted = subprocess.run(
                [
                    str(nvcc), "-std=c++17", "-arch=sm_120", "-O3", "--ptx",
                    str(cu), "-o", str(ptx_path),
                ],
                capture_output=True,
                text=True,
            )
            if emitted.returncode:
                raise RuntimeError(
                    "CUDA dynamic shared-memory PTX generation failed: "
                    + emitted.stderr.strip()
                )
            ptx = ptx_path.read_text(encoding="ascii")
            if f".visible .entry {entry}" not in ptx:
                raise RuntimeError(f"dynamic shared-memory PTX lacks {entry}")
            if ".extern .shared" not in ptx or "arena" not in ptx:
                raise RuntimeError(
                    "NVPTX did not preserve the external dynamic shared arena"
                )
            assembled = subprocess.run(
                [
                    str(ptxas), "-arch=sm_120a", "-v", str(ptx_path),
                    "-o", str(cubin),
                ],
                capture_output=True,
                text=True,
            )
            if assembled.returncode:
                raise RuntimeError(
                    "CUDA dynamic shared-memory assembly failed: "
                    + assembled.stderr.strip()
                )

            def metric(pattern: str) -> int:
                match = re.search(pattern, assembled.stderr)
                return int(match.group(1)) if match else 0

            metrics: dict[str, object] = {
                "registers_per_thread": metric(r"Used\s+(\d+)\s+registers"),
                "static_shared_memory_bytes": metric(r"(\d+)\s+bytes smem"),
                "spill_store_bytes": metric(r"(\d+)\s+bytes spill stores"),
                "spill_load_bytes": metric(r"(\d+)\s+bytes spill loads"),
                "dynamic_shared_symbol": "__tessera_dynamic_smem_probe_arena",
            }
            cached = (ptx, metrics, compiler_fp, toolchain_fp)
            _cache[key] = cached
    ptx, metrics, compiler_fp, toolchain_fp = cached
    return ptx, metrics, compiler_fp, toolchain_fp, state


def package_path_max_probe(
    *, then_bytes: int, else_bytes: int, branch: int
) -> NVIDIADynamicSharedPackage:
    if then_bytes < 2 or else_bytes < 2:
        raise ValueError("path-max probe paths require at least two bytes")
    if branch not in {0, 1}:
        raise ValueError("path-max probe branch must be zero or one")
    dynamic_bytes = path_max_launch_bytes(((then_bytes,), (else_bytes,)))
    digest = hashlib.sha256(
        f"{then_bytes}:{else_bytes}".encode()
    ).hexdigest()[:8]
    entry = f"tessera_cuda_dynamic_smem_path_max_{digest}"
    source = f"""#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void {entry}(
    float* output, int64_t then_bytes, int64_t else_bytes, int64_t branch) {{
  extern __shared__ __align__(16) unsigned char arena[];
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int64_t selected = branch ? then_bytes : else_bytes;
  unsigned char first = branch ? 17 : 29;
  unsigned char last = branch ? 31 : 43;
  volatile unsigned char* storage = arena;
  storage[0] = first;
  storage[selected - 1] = last;
  output[0] = (float)(storage[0] + storage[selected - 1]);
  output[1] = (float)selected;
}}
"""
    ptx, metrics, compiler_fp, toolchain_fp, state = _compile(source, entry)
    image = NativeImageArtifact(
        target="nvidia_sm120",
        architecture="sm_120a",
        pipeline_name="tessera-nvidia-pipeline-sm120",
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(source.encode()).hexdigest(),
        binary_format="ptx",
        payload=ptx.encode("ascii"),
        entry_points=(NativeEntryPoint(entry, SM120_DYNAMIC_SMEM_ABI),),
        compile_state=state,
        resource_record=ResourceRecord(
            provenance="nvcc --ptx sm_120 + ptxas --arch=sm_120a -v",
            metrics={**metrics, "dynamic_shared_memory_bytes": dynamic_bytes},
        ),
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=SM120_DYNAMIC_SMEM_ABI,
        buffers=(
            BufferBinding(0, "output", "output", "fp32", 1, "row_major", 4),
        ),
        scalars=(
            ScalarArgument(1, "ThenBytes", "int64"),
            ScalarArgument(2, "ElseBytes", "int64"),
            ScalarArgument(3, "Branch", "int64"),
        ),
        shape_guards=(ShapeGuard("output", 0, "min", 2),),
        geometry=LaunchGeometry(policy="sm120_dynamic_shared_probe_thread_1"),
        dynamic_local_memory_bytes=dynamic_bytes,
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "E2E-SPINE3-SM120-DYNAMIC-MEMORY",
            "launch_reduction": "aligned_sum_of_slot_maxima",
            "paths": [[then_bytes], [else_bytes]],
            "alignment": 16,
            "branch": branch,
            "selector_changed": False,
        },
    )
    return NVIDIADynamicSharedPackage(source, ptx, image, descriptor)


def package_local_expression_probe(
    *,
    base: int,
    factor: int,
    bias: int,
    fallback: int,
    branch: int,
) -> NVIDIADynamicSharedPackage:
    if min(base, factor, bias, fallback) < 0 or fallback < 2:
        raise ValueError("local dynamic shared expression inputs must be non-negative")
    if branch not in {0, 1}:
        raise ValueError("local expression branch must be zero or one")
    expression: dict[str, object] = {
        "kind": "path_max",
        "operands": [
            {
                "kind": "add",
                "operands": [
                    {
                        "kind": "multiply",
                        "operands": [
                            {"kind": "argument", "name": "Base"},
                            {"kind": "argument", "name": "Factor"},
                        ],
                    },
                    {"kind": "argument", "name": "Bias"},
                ],
            },
            {"kind": "argument", "name": "Fallback"},
        ],
    }
    arguments = {
        "Base": base,
        "Factor": factor,
        "Bias": bias,
        "Fallback": fallback,
    }
    launch_expression: dict[str, object] = {
        "kind": "align_up",
        "alignment": 16,
        "operands": [expression],
    }
    dynamic_bytes = evaluate_launch_expression(launch_expression, arguments)
    selected = base * factor + bias if branch else fallback
    if selected < 2:
        raise ValueError("selected local expression path requires at least two bytes")
    digest = hashlib.sha256(repr(arguments).encode()).hexdigest()[:8]
    entry = f"tessera_cuda_dynamic_smem_local_expr_{digest}"
    source = f"""#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void {entry}(
    float* output, int64_t base, int64_t factor, int64_t bias,
    int64_t fallback, int64_t branch) {{
  extern __shared__ __align__(16) unsigned char arena[];
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int64_t computed = base * factor + bias;
  int64_t selected = branch ? computed : fallback;
  volatile unsigned char* storage = arena;
  storage[0] = branch ? 47 : 53;
  storage[selected - 1] = branch ? 59 : 61;
  output[0] = (float)(storage[0] + storage[selected - 1]);
  output[1] = (float)selected;
}}
"""
    ptx, metrics, compiler_fp, toolchain_fp, state = _compile(source, entry)
    image = NativeImageArtifact(
        target="nvidia_sm120",
        architecture="sm_120a",
        pipeline_name="tessera-nvidia-pipeline-sm120",
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(source.encode()).hexdigest(),
        binary_format="ptx",
        payload=ptx.encode("ascii"),
        entry_points=(NativeEntryPoint(entry, SM120_DYNAMIC_EXPR_SMEM_ABI),),
        compile_state=state,
        resource_record=ResourceRecord(
            provenance="nvcc --ptx sm_120 + ptxas --arch=sm_120a -v",
            metrics={**metrics, "dynamic_shared_memory_bytes": dynamic_bytes},
        ),
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=SM120_DYNAMIC_EXPR_SMEM_ABI,
        buffers=(
            BufferBinding(0, "output", "output", "fp32", 1, "row_major", 4),
        ),
        scalars=tuple(
            ScalarArgument(index, name, "int64")
            for index, name in enumerate(
                ("Base", "Factor", "Bias", "Fallback", "Branch"), start=1
            )
        ),
        shape_guards=(ShapeGuard("output", 0, "min", 2),),
        geometry=LaunchGeometry(policy="sm120_dynamic_shared_probe_thread_1"),
        dynamic_local_memory_bytes=dynamic_bytes,
        dynamic_local_memory_expression=launch_expression,
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "E2E-SPINE3-SM120-DYNAMIC-MEMORY",
            "launch_expression": launch_expression,
            "launch_reduction": "serialized_path_max_expression",
            "alignment": 16,
            "branch": branch,
            "selector_changed": False,
        },
    )
    return NVIDIADynamicSharedPackage(source, ptx, image, descriptor)


__all__ = [
    "NVIDIADynamicSharedPackage",
    "SM120_DYNAMIC_EXPR_SMEM_ABI",
    "SM120_DYNAMIC_SMEM_ABI",
    "align_up",
    "package_path_max_probe",
    "package_local_expression_probe",
    "path_max_launch_bytes",
    "evaluate_launch_expression",
]
