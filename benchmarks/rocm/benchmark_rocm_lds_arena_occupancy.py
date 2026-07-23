#!/usr/bin/env python3
"""Measure the gfx1151 occupancy effect of executable LDS arena sizes.

The benchmark emits the same address-space-3 byte allocation used by
TileBufferArenaPass, serializes it to an HSACO, and asks the live HIP driver for
the kernel's static shared-memory bytes and active blocks per compute unit.  The
paired rows model two simultaneously-live 16 KiB buffers before reuse (32 KiB
peak) and one reused 16 KiB arena after planning.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from tessera import runtime as rt


ROOT = Path(__file__).resolve().parents[2]
TESSERA_OPT = Path(
    os.environ.get(
        "TESSERA_OPT",
        ROOT / "build-rocm-7.14-llvm23-clean/tools/tessera-opt/tessera-opt",
    )
)


def _arena_module(arena_bytes: int) -> str:
    # Round-trip one byte per thread through a visible output so optimization
    # cannot remove the LDS object as a dead local store.
    return f"""module attributes {{gpu.container_module}} {{
  gpu.module @kernels {{
    memref.global "private" @arena_storage_{arena_bytes}
        : memref<{arena_bytes}xi8, 3> = uninitialized {{alignment = 16}}
    gpu.func @arena_{arena_bytes}(%out: memref<?xi8>) kernel {{
      %arena = memref.get_global @arena_storage_{arena_bytes}
          : memref<{arena_bytes}xi8, 3>
      %tid = gpu.thread_id x
      %zero = arith.constant 0 : i8
      memref.store %zero, %arena[%tid] : memref<{arena_bytes}xi8, 3>
      gpu.barrier
      %value = memref.load %arena[%tid] : memref<{arena_bytes}xi8, 3>
      memref.store %value, %out[%tid] : memref<?xi8>
      gpu.return
    }}
  }}
}}
"""


def _dynamic_arena_module() -> str:
    """One runtime-sized Tile-arena lowering shape.

    The ROCm post-ROCDL pass rewrites the addrspace(3) alloca to the external
    dynamic-LDS symbol. The host supplies ``n`` both as the descriptor extent
    and as ``sharedMemBytes`` at launch.
    """
    return """module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @arena_dynamic(%out: memref<?xi8>, %n: index) kernel {
      %arena = memref.alloca(%n) : memref<?xi8, 3>
      %tid = gpu.thread_id x
      %zero = arith.constant 0 : i8
      memref.store %zero, %arena[%tid] : memref<?xi8, 3>
      gpu.barrier
      %value = memref.load %arena[%tid] : memref<?xi8, 3>
      memref.store %value, %out[%tid] : memref<?xi8>
      gpu.return
    }
  }
}
"""


def _build_hsaco(arena_bytes: int, chip: str) -> bytes:
    pipeline = (
        "builtin.module("
        "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        "reconcile-unrealized-casts),"
        f"rocdl-attach-target{{chip={chip}}},gpu-module-to-binary)"
    )
    result = subprocess.run(
        [str(TESSERA_OPT), "-", f"--pass-pipeline={pipeline}"],
        input=_arena_module(arena_bytes),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or "gpu.binary" not in result.stdout:
        raise RuntimeError(
            f"failed to serialize {arena_bytes}-byte LDS arena: "
            f"{result.stderr[:800]}"
        )
    return rt._extract_hsaco_blob(result.stdout)


def _build_dynamic_hsaco(chip: str) -> bytes:
    pipeline = (
        "builtin.module("
        "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        "reconcile-unrealized-casts,rocm-materialize-dynamic-lds),"
        f"rocdl-attach-target{{chip={chip}}},gpu-module-to-binary)"
    )
    result = subprocess.run(
        [str(TESSERA_OPT), "-", f"--pass-pipeline={pipeline}"],
        input=_dynamic_arena_module(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or "gpu.binary" not in result.stdout:
        raise RuntimeError(
            "failed to serialize launch-sized LDS arena: "
            f"{result.stderr[:800]}"
        )
    return rt._extract_hsaco_blob(result.stdout)


def _measure(hip: ctypes.CDLL, arena_bytes: int, chip: str, threads: int) -> dict[str, Any]:
    hsaco = _build_hsaco(arena_bytes, chip)
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(module), hsaco) != 0:
        raise RuntimeError("HIP could not load the generated LDS arena HSACO")
    try:
        symbol = f"arena_{arena_bytes}".encode()
        if hip.hipModuleGetFunction(ctypes.byref(function), module, symbol) != 0:
            raise RuntimeError(f"HIP did not find {symbol.decode()}")
        shared_bytes = ctypes.c_int()
        registers = ctypes.c_int()
        active_blocks = ctypes.c_int()
        # hipFunction_attribute values are stable enum ordinals in
        # /opt/rocm/include/hip/driver_types.h.
        if hip.hipFuncGetAttribute(ctypes.byref(shared_bytes), 1, function) != 0:
            raise RuntimeError("hipFuncGetAttribute(shared size) failed")
        if hip.hipFuncGetAttribute(ctypes.byref(registers), 4, function) != 0:
            raise RuntimeError("hipFuncGetAttribute(register count) failed")
        if (
            hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                ctypes.byref(active_blocks), function, threads, 0
            )
            != 0
        ):
            raise RuntimeError("HIP occupancy query failed")
        return {
            "arena_bytes_requested": arena_bytes,
            "lds_bytes_driver": shared_bytes.value,
            "registers_per_thread": registers.value,
            "block_threads": threads,
            "active_blocks_per_cu": active_blocks.value,
            "resident_wave32_workgroups_per_cu": (
                active_blocks.value * ((threads + 31) // 32)
            ),
            "hsaco_bytes": len(hsaco),
        }
    finally:
        hip.hipModuleUnload(module)


def _measure_dynamic(
    hip: ctypes.CDLL, arena_bytes: int, chip: str, threads: int
) -> dict[str, Any]:
    hsaco = _build_dynamic_hsaco(chip)
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    output = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(module), hsaco) != 0:
        raise RuntimeError("HIP could not load launch-sized LDS HSACO")
    try:
        if hip.hipModuleGetFunction(
            ctypes.byref(function), module, b"arena_dynamic"
        ) != 0:
            raise RuntimeError("HIP did not find arena_dynamic")
        if hip.hipMalloc(ctypes.byref(output), threads) != 0:
            raise RuntimeError("HIP output allocation failed")
        cv = ctypes.c_void_p
        args = [
            cv(output.value), cv(output.value), ctypes.c_int64(0),
            ctypes.c_int64(threads), ctypes.c_int64(1),
            ctypes.c_int64(arena_bytes),
        ]
        packed = (cv * len(args))()
        for index, value in enumerate(args):
            packed[index] = ctypes.cast(ctypes.byref(value), cv)
        rc = hip.hipModuleLaunchKernel(
            function, 1, 1, 1, threads, 1, 1, arena_bytes, None, packed, None
        )
        if rc != 0 or hip.hipDeviceSynchronize() != 0:
            raise RuntimeError(f"launch-sized LDS kernel failed rc={rc}")
        host = (ctypes.c_uint8 * threads)()
        if hip.hipMemcpy(host, output, threads, 2) != 0:
            raise RuntimeError("launch-sized LDS result copy failed")
        active_blocks = ctypes.c_int()
        if hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(active_blocks), function, threads, arena_bytes
        ) != 0:
            raise RuntimeError("dynamic LDS occupancy query failed")
        return {
            "arena_bytes_requested": arena_bytes,
            "launch_shared_bytes": arena_bytes,
            "active_blocks_per_cu": active_blocks.value,
            "execution_verified": all(value == 0 for value in host),
            "hsaco_bytes": len(hsaco),
        }
    finally:
        if output.value:
            hip.hipFree(output)
        hip.hipModuleUnload(module)


def run(chip: str = "gfx1151", threads: int = 256) -> dict[str, Any]:
    if not TESSERA_OPT.is_file():
        raise RuntimeError(f"tessera-opt not found: {TESSERA_OPT}")
    live = rt._rocm_live_arch()
    if live != chip:
        raise RuntimeError(f"exact {chip} host required; live architecture is {live!r}")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("live ROCm HIP device required")
    hip.hipFuncGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    hip.hipModuleUnload.argtypes = [ctypes.c_void_p]

    sizes = (16_384, 32_768)
    rows = [_measure(hip, size, chip, threads) for size in sizes]
    dynamic_rows = [
        _measure_dynamic(hip, size, chip, threads) for size in sizes
    ]
    return {
        "schema": "tessera.rocm.lds-arena-occupancy.v1",
        "device": chip,
        "evidence": "exact_device_hip_driver",
        "planner_case": {
            "buffers": 2,
            "buffer_bytes": 16_384,
            "bytes_before_reuse": 32_768,
            "bytes_after_reuse": 16_384,
            "peak_reduction_ratio": 2.0,
        },
        "rows": rows,
        "dynamic_rows": dynamic_rows,
        "occupancy_effect": {
            "active_blocks_per_cu_before": rows[1]["active_blocks_per_cu"],
            "active_blocks_per_cu_after": rows[0]["active_blocks_per_cu"],
            "active_block_ratio": (
                rows[0]["active_blocks_per_cu"]
                / max(rows[1]["active_blocks_per_cu"], 1)
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", default="gfx1151")
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run(args.chip, args.threads)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
