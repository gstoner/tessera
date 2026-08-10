#!/usr/bin/env python3
"""Record exact-gfx1151 EGGROLL low-rank correction evidence."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tessera import rng
from tessera.stdlib.es import low_rank_correction
from tests._support.compiler_tool import require_tessera_opt, run_tessera_opt
from tests.unit.test_rocm_es_low_rank_exec import _compile, _hip, _source


DEFAULT_OUTPUT = ROOT / "benchmarks/baselines/rocm_gfx1151_es_low_rank.json"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact_hash(source: str) -> str:
    scheduled = run_tessera_opt(source, "--tessera-graph-to-schedule")
    if scheduled.returncode:
        raise RuntimeError(scheduled.stderr)
    match = re.search(r'artifact_hash = "([0-9a-f]{64})"', scheduled.stdout)
    if match is None:
        raise RuntimeError("scheduled ES artifact did not expose its content hash")
    return match.group(1)


class _ResidentKernel:
    def __init__(self, hip: ctypes.CDLL, hsaco: bytes, arrays: list[np.ndarray]):
        self.hip = hip
        self.arrays = arrays
        self.module = ctypes.c_void_p()
        if hip.hipModuleLoadData(ctypes.byref(self.module), hsaco):
            raise RuntimeError("hipModuleLoadData failed")
        self.function = ctypes.c_void_p()
        if hip.hipModuleGetFunction(
            ctypes.byref(self.function), self.module, b"es_rank1"
        ):
            raise RuntimeError("hipModuleGetFunction(es_rank1) failed")
        self.devices: list[ctypes.c_void_p] = []
        for array in arrays:
            pointer = ctypes.c_void_p()
            if hip.hipMalloc(ctypes.byref(pointer), array.nbytes):
                raise RuntimeError("hipMalloc failed")
            self.devices.append(pointer)
            if hip.hipMemcpy(
                pointer, array.ctypes.data_as(ctypes.c_void_p), array.nbytes, 1
            ):
                raise RuntimeError("H2D copy failed")
        self.values: list[ctypes._SimpleCData] = []
        for pointer, array in zip(self.devices, arrays):
            self.values.extend(
                (
                    ctypes.c_void_p(pointer.value),
                    ctypes.c_void_p(pointer.value),
                    ctypes.c_int64(0),
                    ctypes.c_int64(array.size),
                    ctypes.c_int64(1),
                )
            )
        self.arguments = (ctypes.c_void_p * len(self.values))()
        for index, value in enumerate(self.values):
            self.arguments[index] = ctypes.cast(
                ctypes.byref(value), ctypes.c_void_p
            )
        self.blocks = arrays[0].shape[0] * arrays[0].shape[1]
        hip.hipModuleLaunchKernel.argtypes = (
            [ctypes.c_void_p]
            + [ctypes.c_uint] * 6
            + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        )

    def launch(self) -> None:
        if self.hip.hipModuleLaunchKernel(
            self.function,
            self.blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            None,
            self.arguments,
            None,
        ):
            raise RuntimeError("hipModuleLaunchKernel failed")

    def synchronize(self) -> None:
        if self.hip.hipDeviceSynchronize():
            raise RuntimeError("hipDeviceSynchronize failed")

    def read(self) -> np.ndarray:
        self.synchronize()
        result = np.empty_like(self.arrays[-1])
        if self.hip.hipMemcpy(
            result.ctypes.data_as(ctypes.c_void_p),
            self.devices[-1],
            result.nbytes,
            2,
        ):
            raise RuntimeError("D2H copy failed")
        return result

    def close(self) -> None:
        for pointer in reversed(self.devices):
            self.hip.hipFree(pointer)
        if self.module.value:
            self.hip.hipModuleUnload(self.module)


def _hip_event_ns(hip: ctypes.CDLL, resident: _ResidentKernel) -> int | None:
    start, stop = ctypes.c_void_p(), ctypes.c_void_p()
    if hip.hipEventCreate(ctypes.byref(start)) or hip.hipEventCreate(
        ctypes.byref(stop)
    ):
        return None
    try:
        hip.hipEventRecord(start, None)
        resident.launch()
        hip.hipEventRecord(stop, None)
        hip.hipEventSynchronize(stop)
        elapsed = ctypes.c_float()
        if hip.hipEventElapsedTime(ctypes.byref(elapsed), start, stop):
            return None
        value = round(float(elapsed.value) * 1.0e6)
        return value if value > 0 else None
    finally:
        hip.hipEventDestroy(stop)
        hip.hipEventDestroy(start)


def run(*, warmup: int, samples: int) -> dict[str, object]:
    if os.environ.get("TESSERA_ROCM_CHIP", "gfx1151") != "gfx1151":
        raise RuntimeError("EGGROLL evidence is fail-closed outside gfx1151")
    tool = require_tessera_opt("--generate-rocm-es-low-rank-kernel")
    hip = _hip()
    if hip is None or hip.hipInit(0):
        raise RuntimeError("usable gfx1151 HIP runtime required")

    population, rows, in_dim, out_dim = 16, 32, 1024, 1024
    source = _source(population, rows, in_dim, out_dim)
    hsaco = _compile(source)
    generator = np.random.default_rng(709)
    values = generator.standard_normal(
        (population, rows, in_dim), dtype=np.float32
    )
    members = np.arange(64, 64 + population, dtype=np.int64)
    key = rng.RNGKey(seed_high=0x123456789ABCDEF, seed_low=0x234567890ABCDEF)
    key_storage = np.array([key.seed_high, key.seed_low], dtype=np.int64)
    output = np.zeros((population, rows, out_dim), dtype=np.float32)
    expected = low_rank_correction(
        values, members, key, out_dim=out_dim, rank=1, sigma=0.02, epoch=3
    )
    resident = _ResidentKernel(
        hip, hsaco, [values, members, key_storage, output]
    )
    try:
        for _ in range(warmup):
            resident.launch()
        resident.synchronize()
        wall_samples: list[int] = []
        event_samples: list[int] = []
        for _ in range(samples):
            start = time.perf_counter_ns()
            resident.launch()
            resident.synchronize()
            wall_samples.append(time.perf_counter_ns() - start)
            event = _hip_event_ns(hip, resident)
            if event is not None:
                event_samples.append(event)
        actual = resident.read()
    finally:
        resident.close()

    max_abs = float(np.max(np.abs(actual - expected)))
    max_rel = float(
        np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), 1.0e-7))
    )
    wsl = "microsoft" in platform.release().lower()
    return {
        "schema": "tessera.eggroll.es_low_rank.gfx1151.v1",
        "work_item": "EGGROLL-W2-ROCM",
        "target": "rocm_gfx1151",
        "architecture": "gfx1151",
        "algorithm": "cooperative_wave32_lds_sgmv_rank1",
        "rng": "splitmix64-philox4x32-boxmuller.v1",
        "shape": {
            "population": population,
            "rows_per_member": rows,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "rank": 1,
        },
        "dtype": {"storage": "fp32", "accum": "fp32"},
        "artifact_hash": _artifact_hash(source),
        "hsaco_sha256": _sha256(hsaco),
        "tessera_opt_sha256": _sha256(tool.read_bytes()),
        "numerical": {
            "oracle": "tessera.stdlib.es.low_rank_correction",
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "passed": bool(np.allclose(actual, expected, rtol=2.0e-5, atol=2.0e-6)),
        },
        "timing": {
            "warmup": warmup,
            "samples": samples,
            "host_wall_ns": {
                "source": "synchronized_perf_counter",
                "samples": wall_samples,
                "median": statistics.median(wall_samples),
            },
            "hip_event_ns": {
                "source": "hip_event",
                "valid": len(event_samples) == samples,
                "samples": event_samples,
                "median": (
                    statistics.median(event_samples) if event_samples else None
                ),
                "reason": None if event_samples else "HIP_EVENT_ZERO_OR_UNAVAILABLE",
            },
        },
        "environment": {
            "host": platform.platform(),
            "execution": "wsl" if wsl else "bare_metal",
            "toolchain": subprocess.run(
                [str(tool), "--tessera-build-info"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
        },
        "verdict": {
            "correctness": "retain" if np.allclose(
                actual, expected, rtol=2.0e-5, atol=2.0e-6
            ) else "reject",
            "performance": "pending_direct_and_packed_candidate_comparison",
            "selector_eligible": False,
            "reason": (
                "WSL evidence is regression-only; no scalar/direct or packed-WMMA "
                "comparison is bound to this packet"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    packet = run(warmup=args.warmup, samples=args.samples)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps(packet, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
