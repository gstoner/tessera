#!/usr/bin/env python3
"""Exact-device packet for the canonical ROCm GEMM, built through every IR level.

Each row starts from one Graph IR ``tessera.matmul`` and goes Graph -> Schedule
-> Tile (``scheduled_matmul.lower_scheduled_matmul``, which replays
Schedule->Tile and refuses a mismatch) -> ``tessera_rocm`` Target IR -> HSACO
(``rocm_native.package_scheduled_matmul``), via
``runtime.build_canonical_gemm_hsaco``. The kernel is launched from the
package's own launch descriptor -- entry symbol, bindings, ``macro_tile`` grid
and ``workgroup`` -- never from a harness-side copy of the ABI.

Rebuilt 2026-09-26 when the Graph->Tile shortcut that skipped Schedule IR
(Lane B, ``runtime._build_canonical_gemm_hsaco``) was retired. Its committed
packet, ``benchmarks/baselines/rocm_gfx1151_canonical_gemm_kloop.json``,
measured THAT route and stays as its record; it is not a baseline for this
one. Rows here carry the route and schedule digest they measured (Decisions
#11/#12) and carry no ratchet until a packet on this route is recorded.

Timing: modules and device buffers stay resident; each sample is
``runtime._hip_resident_launch_latency`` (a HIP-event reading admitted only
inside a two-sided band around the wall clock, else the wall clock).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from tessera import runtime as rt  # noqa: E402

from benchmarks.rocm.benchmark_rocm_gemm_schedule_matrix import (  # noqa: E402
    Case,
    _correct,
    _error_metrics,
    _inputs,
    _reference,
    code_object_resources,
)

SCHEMA = "tessera.rocm.canonical_gemm.v2"
ROUTE = "graph_schedule_tile_target"
SUPPORTED_CHIPS = ("gfx1151", "gfx1201")
CASES = (
    Case("canonical_aligned", 256, 256, 256, "f16"),
    Case("canonical_ragged", 131, 127, 65, "f16"),
    Case("canonical_aligned", 256, 256, 256, "bf16"),
    Case("canonical_ragged", 131, 127, 65, "bf16"),
    Case("canonical_aligned", 256, 256, 256, "int8"),
    Case("canonical_ragged", 131, 127, 65, "int8"),
)


def _memref(pointer: ctypes.c_void_p, size: int) -> list[Any]:
    return [ctypes.c_void_p(pointer.value), ctypes.c_void_p(pointer.value),
            ctypes.c_int64(0), ctypes.c_int64(size), ctypes.c_int64(1)]


class ResidentPackage:
    """One loaded canonical package with resident device buffers.

    The argument list mirrors ``runtime.launch``'s native ROCm matmul ABI
    (one 1-D memref descriptor per binding in ordinal order, then M, N, K);
    the grid and block come from the descriptor's provenance.
    """

    def __init__(self, hip: Any, package: Any, case: Case,
                 a: np.ndarray, b: np.ndarray) -> None:
        descriptor = package.descriptor
        if descriptor.provenance.get("bias") or descriptor.provenance.get("activation") != "none":
            raise ValueError("this packet measures the plain GEMM ABI only")
        macro = descriptor.provenance.get("macro_tile")
        workgroup = descriptor.provenance.get("workgroup")
        if (not isinstance(macro, list) or len(macro) != 2
                or not isinstance(workgroup, list) or not workgroup):
            raise RuntimeError("canonical GEMM descriptor lacks macro_tile/workgroup")
        self.hip, self.case = hip, case
        self.out_dtype = np.int32 if case.dtype == "int8" else np.float32
        self.mod = ctypes.c_void_p()
        self.dev: list[ctypes.c_void_p] = []
        if hip.hipModuleLoadData(ctypes.byref(self.mod), package.image.payload) != 0:
            raise RuntimeError("hipModuleLoadData refused the canonical image")
        try:
            self.fn = ctypes.c_void_p()
            symbol = descriptor.entry_symbol.encode()
            if hip.hipModuleGetFunction(ctypes.byref(self.fn), self.mod, symbol) != 0:
                raise RuntimeError(f"entry symbol {descriptor.entry_symbol!r} not found")
            m, n, k = case.m, case.n, case.k
            out_bytes = m * n * np.dtype(self.out_dtype).itemsize
            for host in (np.ascontiguousarray(a), np.ascontiguousarray(b)):
                self.dev.append(self._upload(host))
            self.dev.append(self._alloc(out_bytes))
            args = (_memref(self.dev[0], m * k) + _memref(self.dev[1], k * n)
                    + _memref(self.dev[2], m * n)
                    + [ctypes.c_int64(m), ctypes.c_int64(n), ctypes.c_int64(k)])
            self._keep = args
            self._arr = (ctypes.c_void_p * len(args))()
            for index, value in enumerate(args):
                self._arr[index] = ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
            macro_m, macro_n = macro
            self.grid = ((n + macro_n - 1) // macro_n, (m + macro_m - 1) // macro_m)
            self.block = int(workgroup[0])
        except Exception:
            self.close()
            raise

    def _alloc(self, nbytes: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        if self.hip.hipMalloc(ctypes.byref(pointer), nbytes) != 0:
            raise RuntimeError("hipMalloc failed")
        return pointer

    def _upload(self, host: np.ndarray) -> ctypes.c_void_p:
        pointer = self._alloc(host.nbytes)
        if self.hip.hipMemcpy(pointer, host.ctypes.data_as(ctypes.c_void_p), host.nbytes, 1) != 0:
            self.hip.hipFree(pointer)
            raise RuntimeError("host-to-device copy failed")
        return pointer

    def launch(self) -> int:
        return self.hip.hipModuleLaunchKernel(
            self.fn, self.grid[0], self.grid[1], 1, self.block, 1, 1, 0, None, self._arr, None)

    def download(self) -> np.ndarray:
        if self.launch() != 0 or self.hip.hipDeviceSynchronize() != 0:
            raise RuntimeError("canonical GEMM launch failed")
        out = np.zeros((self.case.m, self.case.n), self.out_dtype)
        if self.hip.hipMemcpy(out.ctypes.data_as(ctypes.c_void_p), self.dev[2], out.nbytes, 2) != 0:
            raise RuntimeError("device-to-host copy failed")
        return out

    def close(self) -> None:
        for pointer in self.dev:
            self.hip.hipFree(pointer)
        self.dev = []
        if self.mod.value:
            self.hip.hipModuleUnload(self.mod)
            self.mod = ctypes.c_void_p()


def run(*, warmup: int, rounds: int, iterations: int, staging: str = "register") -> dict[str, Any]:
    if staging not in {"register", "lds"}:
        raise ValueError("staging must be register or lds")
    chip = os.environ.get("TESSERA_ROCM_CHIP")
    if chip not in SUPPORTED_CHIPS:
        raise RuntimeError(
            f"set TESSERA_ROCM_CHIP to the exact device ({' or '.join(SUPPORTED_CHIPS)}); "
            f"got {chip!r} -- a default would label one part's result as another's")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError(f"a live {chip} HIP device is required")
    rows = []
    for case in CASES:
        a, b, _ = _inputs(case)
        reference = _reference(case, a, b, None)
        assert reference is not None
        compile_started = time.perf_counter_ns()
        package = rt.build_canonical_gemm_hsaco(
            case.m, case.n, case.k, case.dtype, chip=chip, staging=staging)
        compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0
        provenance = dict(package.descriptor.provenance)
        device = ResidentPackage(hip, package, case, a, b)
        try:
            metrics = _error_metrics(device.download(), reference)
            samples, clocks = [], set()
            for _ in range(rounds):
                per_launch_ms, clock = rt._hip_resident_launch_latency(
                    hip, device.launch, iters=iterations, warmup=warmup,
                    what=f"canonical GEMM {case.key}")
                samples.append(per_launch_ms)
                clocks.add(clock)
        finally:
            device.close()
        rows.append({
            "case": case.key,
            "shape": [case.m, case.n, case.k],
            "storage": case.dtype,
            "accumulate": "i32" if case.dtype == "int8" else "f32",
            "route": ROUTE,
            "provenance_route": provenance.get("route"),
            "physical_route": provenance.get("physical_route"),
            "schedule_digest": provenance.get("schedule_digest"),
            "tile_ir_digest": provenance.get("tile_ir_digest"),
            "image_digest": package.image.image_digest,
            "macro_tile": provenance.get("macro_tile"),
            "workgroup": provenance.get("workgroup"),
            "k_unroll": provenance.get("k_unroll"),
            "correct": _correct(case, metrics),
            "numerics": metrics,
            "artifact": {
                "binary_format": package.image.binary_format,
                "bytes": len(package.image.payload),
                "resources": code_object_resources(package.image.payload, chip),
            },
            "timing": {
                "compiler_ms": compile_ms,
                "kernel": {
                    "method": "runtime._hip_resident_launch_latency",
                    "clocks": sorted(clocks),
                    "resident_module": True,
                    "resident_buffers": True,
                    "warmup_launches_per_round": warmup,
                    "launches_per_round": iterations,
                    "rounds": rounds,
                    "median_ms": statistics.median(samples),
                    "round_ms": samples,
                },
            },
        })
    return {
        "schema": SCHEMA,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": chip,
        "route": ROUTE,
        "physical_staging": staging,
        "rows": rows,
        "all_correct": all(row["correct"] for row in rows),
        "ratchet": "none: first packet on the scheduled route; Lane B's numbers are not its baseline",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--staging", choices=("register", "lds"), default="register")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.warmup < 0 or args.rounds <= 0 or args.iterations <= 0:
        parser.error("warmup must be nonnegative; rounds and iterations positive")
    packet = run(warmup=args.warmup, rounds=args.rounds, iterations=args.iterations,
                 staging=args.staging)
    text = json.dumps(packet, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0 if packet["all_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
