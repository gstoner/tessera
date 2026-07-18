"""Record CUDA-native physical fragment and schedule resources on sm_120a.

The logical Tile fixtures are shared with the exact-device compiler-path tests;
this recorder never accepts authored physical registers.  It lowers those
fixtures with Tessera/LLVM, assembles the resulting PTX, then inspects the
cubin selected by NVIDIA's toolchain. NVFP4 uses the same logical path: Tile
IR carries packed logical matrices plus UE4M3 scale tiles, while NVIDIA owns
the per-lane register and scale-selector mapping.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

OUT = ROOT / "benchmarks/baselines/nvidia_sm120_tile_fragment_resources.json"

_RESOURCE_RE = re.compile(
    r"^ Function (?P<kernel>.+):\n"
    r"  REG:(?P<registers>\d+) STACK:(?P<stack>\d+) "
    r"SHARED:(?P<shared>\d+) LOCAL:(?P<local>\d+)", re.MULTILINE)
_FUNCTION_RE = re.compile(r"Function\s*:\s*(\S+)")
_MMA_RE = re.compile(r"\b(?:H|I|O|Q)MMA(?:\.[A-Z0-9]+)+")


def parse_resource_usage(text: str) -> dict[str, dict[str, int]]:
    return {
        match.group("kernel"): {
            "registers_per_thread": int(match.group("registers")),
            "stack_bytes": int(match.group("stack")),
            "static_shared_memory_bytes": int(match.group("shared")),
            "local_bytes": int(match.group("local")),
        }
        for match in _RESOURCE_RE.finditer(text)
    }


def parse_sass_instruction_families(text: str) -> dict[str, list[str]]:
    result: dict[str, set[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        function = _FUNCTION_RE.search(line)
        if function:
            current = function.group(1)
            result.setdefault(current, set())
            continue
        instruction = _MMA_RE.search(line)
        if current and instruction:
            result[current].add(instruction.group(0))
    return {kernel: sorted(values) for kernel, values in result.items()}


def _run(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True,
                          text=True).stdout


class _CudaOccupancy:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so.1")
        void_p = ctypes.c_void_p
        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int),
                                                  ctypes.c_int, ctypes.c_int]
        self.lib.cuCtxCreate_v2.argtypes = [ctypes.POINTER(void_p), ctypes.c_uint,
                                            ctypes.c_int]
        self.lib.cuModuleLoad.argtypes = [ctypes.POINTER(void_p), ctypes.c_char_p]
        self.lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(void_p), void_p,
                                                 ctypes.c_char_p]
        self.lib.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
            ctypes.POINTER(ctypes.c_int), void_p, ctypes.c_int, ctypes.c_size_t]
        self.lib.cuModuleUnload.argtypes = [void_p]
        self.lib.cuCtxDestroy_v2.argtypes = [void_p]
        self._check(self.lib.cuInit(0), "cuInit")
        self.device = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(self.device), 0),
                    "cuDeviceGet")
        self.context = void_p()
        self._check(self.lib.cuCtxCreate_v2(
            ctypes.byref(self.context), 0, self.device), "cuCtxCreate")
        self.max_threads_per_sm = self.attribute(39)
        major, minor = self.attribute(75), self.attribute(76)
        if (major, minor) != (12, 0):
            raise RuntimeError(f"resource record requires sm_120, found sm_{major}{minor}")

    def _check(self, result: int, operation: str) -> None:
        if result:
            raise RuntimeError(f"{operation} failed with CUDA error {result}")

    def attribute(self, attribute: int) -> int:
        value = ctypes.c_int()
        self._check(self.lib.cuDeviceGetAttribute(
            ctypes.byref(value), attribute, self.device), "cuDeviceGetAttribute")
        return value.value

    def row(self, cubin: Path, kernel: str, block_threads: int,
            dynamic_shared_memory_bytes: int = 0) -> dict[str, Any]:
        module, function = ctypes.c_void_p(), ctypes.c_void_p()
        self._check(self.lib.cuModuleLoad(ctypes.byref(module), os.fsencode(cubin)),
                    "cuModuleLoad")
        try:
            self._check(self.lib.cuModuleGetFunction(
                ctypes.byref(function), module, kernel.encode()),
                "cuModuleGetFunction")
            blocks = ctypes.c_int()
            self._check(self.lib.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                ctypes.byref(blocks), function, block_threads,
                dynamic_shared_memory_bytes),
                "cuOccupancyMaxActiveBlocksPerMultiprocessor")
            return {
                "block_threads": block_threads,
                "max_active_blocks_per_sm": blocks.value,
                "theoretical_occupancy_pct": round(
                    100 * blocks.value * block_threads / self.max_threads_per_sm, 4),
            }
        finally:
            self.lib.cuModuleUnload(module)

    def close(self) -> None:
        if self.context:
            self.lib.cuCtxDestroy_v2(self.context)
            self.context = None


def _inspect(cubin: Path) -> tuple[dict[str, dict[str, int]],
                                   dict[str, list[str]]]:
    cuobjdump = "/usr/local/cuda/bin/cuobjdump"
    return (
        parse_resource_usage(_run([cuobjdump, "--dump-resource-usage", str(cubin)])),
        parse_sass_instruction_families(_run(
            [cuobjdump, "--dump-sass", str(cubin)])),
    )


def _artifact_row(cubin: Path, kernel: str, block_threads: int,
                  occupancy: _CudaOccupancy, **metadata: Any) -> dict[str, Any]:
    resources, sass = _inspect(cubin)
    if kernel not in resources:
        raise RuntimeError(f"no resource record found for {kernel}")
    row: dict[str, Any] = {"kernel": kernel, **metadata,
                          **resources[kernel],
                          **occupancy.row(cubin, kernel, block_threads),
                          "sass_instruction_families": sass.get(kernel, [])}
    row["spill_evidence_complete"] = True
    row["spills_detected"] = bool(row["stack_bytes"] or row["local_bytes"])
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
    row["resource_fingerprint"] = "sha256:" + hashlib.sha256(
        canonical.encode()).hexdigest()
    row["cubin_sha256"] = hashlib.sha256(cubin.read_bytes()).hexdigest()
    return row


def record() -> dict[str, Any]:
    # These helpers are the canonical logical Tile fixtures used by the device
    # tests.  Physical fragments are selected only inside NVIDIA lowering.
    from tests.device.nvidia.test_tile_fragment_compiler_path import (
        KERNEL_FIXTURE, NVFP4_FIXTURE, _compile_cubin, _write_dtype_fixture)
    from tessera.compiler.nvidia_fragment import select_sm120_fragment_layout

    occupancy = _CudaOccupancy()
    rows: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="tessera-tile-resources-") as tmp:
            work = Path(tmp)
            schedule_entries = {
                "tile_matmul_f32_direct": ("f16", "direct", 32),
                "tile_matmul_f32": ("f16", "shared", 128),
                "tile_matmul_bias_relu_f16": ("f16", "shared_bias_relu", 128),
                "tile_matmul_bf16": ("bf16", "shared", 128),
                "tile_matmul_gelu": ("f16", "direct_gelu", 32),
                "tile_matmul_silu": ("f16", "direct_silu", 32),
            }
            cubin = _compile_cubin(
                work, KERNEL_FIXTURE, tuple(schedule_entries))
            for kernel, (dtype, schedule, block) in schedule_entries.items():
                descriptor = select_sm120_fragment_layout(
                    dtype, (16, 8, 16)).as_metadata_dict()
                rows.append(_artifact_row(
                    cubin, kernel, block, occupancy, row_kind="schedule",
                    schedule=schedule, fragment=descriptor))

            typed = (
                ("f16", 16, "f32"), ("bf16", 16, "f32"),
                ("tf32", 8, "f32"), ("e4m3", 32, "f32"),
                ("e5m2", 32, "f32"), ("s8", 32, "s32"),
            )
            for dtype, k, accumulator in typed:
                entry = f"pointer_fragment_store_{dtype}_resources"
                dtype_work = work / dtype
                dtype_work.mkdir()
                fixture = _write_dtype_fixture(
                    dtype_work, entry=entry, dtype=dtype, k=k,
                    accumulator=accumulator)
                expected = select_sm120_fragment_layout(
                    dtype, (16, 8, k), accumulator).instruction_family
                dtype_cubin = _compile_cubin(
                    dtype_work, fixture, (entry,), expected)
                rows.append(_artifact_row(
                    dtype_cubin, entry, 32, occupancy, row_kind="fragment",
                    schedule="direct", fragment=select_sm120_fragment_layout(
                        dtype, (16, 8, k), accumulator).as_metadata_dict()))

            nvfp4_work = work / "nvfp4"
            nvfp4_work.mkdir()
            nvfp4_desc = select_sm120_fragment_layout(
                "nvfp4", (16, 8, 64), "f32")
            nvfp4 = _compile_cubin(
                nvfp4_work, NVFP4_FIXTURE, ("nvfp4_fragment_store",),
                nvfp4_desc.instruction_family, arch="sm_120a")
            rows.append(_artifact_row(
                nvfp4, "nvfp4_fragment_store", 32, occupancy,
                row_kind="block_scaled_fragment", schedule="direct",
                fragment=nvfp4_desc.as_metadata_dict()))
    finally:
        occupancy.close()

    device = _run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
                   "--format=csv,noheader"]).strip()
    return {
        "schema": "tessera.nvidia.tile-fragment-resources.v1",
        "device": device,
        "cuda_toolchain": _run(["/usr/local/cuda/bin/ptxas", "--version"]).strip(),
        "logical_fixture_sha256": hashlib.sha256(
            (ROOT / "src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia"
             / "sm120_matmul_kernel.mlir").read_bytes()).hexdigest(),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    result = record()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output} ({len(result['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
