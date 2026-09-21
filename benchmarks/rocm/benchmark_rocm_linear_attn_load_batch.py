#!/usr/bin/env python3
"""Paired exact-device gate for the D=128 linear-attention load batch.

The two HSACO inputs keep the comparison honest: they are loaded into the same
HIP context, receive the same resident buffers, and alternate timing order.
Static wait/resource counts are collected from the exact images being timed.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from tessera import runtime as rt  # noqa: E402


class _Session:
    def __init__(self, hip, image: bytes, q, k, v):
        self.hip = hip
        self.module = ctypes.c_void_p()
        blob = ctypes.create_string_buffer(image)
        self._blob = blob
        if hip.hipModuleLoadData(ctypes.byref(self.module), ctypes.cast(blob, ctypes.c_void_p)):
            raise RuntimeError("linear-attention module load failed")
        self.function = ctypes.c_void_p()
        if hip.hipModuleGetFunction(ctypes.byref(self.function), self.module, b"la"):
            raise RuntimeError("linear-attention symbol 'la' missing")
        self.shape = q.shape
        self.sq, self.sk, self.d = q.shape[-2], k.shape[-2], q.shape[-1]
        count_q = q.size
        count_kv = k.size
        self.output = np.zeros(q.shape, np.float32)
        self.devices = [ctypes.c_void_p() for _ in range(4)]
        sizes = (q.nbytes, k.nbytes, v.nbytes, self.output.nbytes)
        for ptr, size in zip(self.devices, sizes, strict=True):
            if hip.hipMalloc(ctypes.byref(ptr), size):
                raise RuntimeError("linear-attention hipMalloc failed")
        for ptr, host in zip(self.devices[:3], (q, k, v), strict=True):
            if hip.hipMemcpy(ptr, host.ctypes.data_as(ctypes.c_void_p), host.nbytes, 1):
                raise RuntimeError("linear-attention host-to-device copy failed")

        def mr(ptr, count):
            return [ctypes.c_void_p(ptr.value), ctypes.c_void_p(ptr.value),
                    ctypes.c_int64(0), ctypes.c_int64(count), ctypes.c_int64(1)]

        values = (mr(self.devices[0], count_q) + mr(self.devices[1], count_kv) +
                  mr(self.devices[2], count_kv) + mr(self.devices[3], count_q) +
                  [ctypes.c_int64(self.sq), ctypes.c_int64(self.sk), ctypes.c_int64(1)])
        self._values = values
        self.arguments = (ctypes.c_void_p * len(values))()
        for index, value in enumerate(values):
            self.arguments[index] = ctypes.cast(ctypes.byref(value), ctypes.c_void_p)

    def launch(self) -> None:
        rc = self.hip.hipModuleLaunchKernel(
            self.function, (self.sq + 15) // 16, 1, 1, 32, 1, 1,
            0, None, self.arguments, None)
        if rc:
            raise RuntimeError(f"linear-attention launch failed: {rc}")

    def download(self):
        self.launch()
        if self.hip.hipDeviceSynchronize():
            raise RuntimeError("linear-attention synchronization failed")
        if self.hip.hipMemcpy(self.output.ctypes.data_as(ctypes.c_void_p),
                              self.devices[3], self.output.nbytes, 2):
            raise RuntimeError("linear-attention device-to-host copy failed")
        return self.output.copy()

    def close(self) -> None:
        for ptr in self.devices:
            self.hip.hipFree(ptr)
        self.hip.hipModuleUnload(self.module)


def _device_ms(hip, session: _Session, iterations: int) -> float:
    start, stop = ctypes.c_void_p(), ctypes.c_void_p()
    if hip.hipEventCreate(ctypes.byref(start)) or hip.hipEventCreate(ctypes.byref(stop)):
        raise RuntimeError("HIP event creation failed")
    try:
        hip.hipEventRecord(start, None)
        for _ in range(iterations):
            session.launch()
        hip.hipEventRecord(stop, None)
        hip.hipEventSynchronize(stop)
        elapsed = ctypes.c_float()
        if hip.hipEventElapsedTime(ctypes.byref(elapsed), start, stop):
            raise RuntimeError("HIP event timing failed")
        sample = float(elapsed.value) / iterations
        if not math.isfinite(sample) or sample <= 0.0:
            raise RuntimeError(f"invalid HIP event sample: {sample} ms")
        return sample
    finally:
        hip.hipEventDestroy(start)
        hip.hipEventDestroy(stop)


def _static_metrics(image_path: Path, objdump: str, readobj: str) -> dict[str, object]:
    asm = subprocess.run([objdump, "-d", "--mcpu=gfx1201", str(image_path)],
                         check=True, capture_output=True, text=True).stdout
    notes = subprocess.run([readobj, "--notes", str(image_path)],
                           check=True, capture_output=True, text=True).stdout
    waits = re.findall(r"s_wait_loadcnt\s+0x([0-9a-f]+)", asm)
    loads = len(re.findall(r"\b(?:global|buffer|flat)_load\w*\b", asm))
    result: dict[str, object] = {
        "instruction_count": len(re.findall(r"//\s+[0-9A-F]+:", asm)),
        "global_load_count": loads,
        "load_wait_count": len(waits),
        "full_load_drain_count": sum(value == "0" for value in waits),
        "full_load_drain_pct": 100.0 * sum(value == "0" for value in waits) / len(waits),
        "loads_per_wait": loads / len(waits),
        "load_wait_immediates": {value: waits.count(value) for value in sorted(set(waits))},
    }
    for field in ("vgpr_count", "sgpr_count", "group_segment_fixed_size",
                  "private_segment_fixed_size"):
        match = re.search(rf"\.{field}:\s+(\d+)", notes)
        result[field] = int(match.group(1)) if match else None
    return result


def run(args) -> dict[str, object]:
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):
        raise RuntimeError("live ROCm device required")
    rng = np.random.default_rng(12832)
    shape = (1, 1, 32, 128)
    q = (rng.standard_normal(shape) * 0.20).astype(np.float16)
    k = (rng.standard_normal(shape) * 0.20).astype(np.float16)
    v = (rng.standard_normal(shape) * 0.20).astype(np.float16)
    images = {"baseline": args.baseline, "candidate": args.candidate}
    sessions = {name: _Session(hip, path.read_bytes(), q, k, v)
                for name, path in images.items()}
    try:
        outputs = {name: session.download() for name, session in sessions.items()}
        max_abs = float(np.max(np.abs(outputs["candidate"] - outputs["baseline"])))
        if max_abs > 1e-5:
            raise AssertionError(f"candidate differs from baseline: max_abs={max_abs}")
        for session in sessions.values():
            for _ in range(5):
                session.launch()
        hip.hipDeviceSynchronize()
        samples = {name: [] for name in sessions}
        for trial in range(args.trials):
            order = list(sessions)
            if trial & 1:
                order.reverse()
            for name in order:
                samples[name].append(_device_ms(hip, sessions[name], args.iterations))
        arms = {}
        for name, path in images.items():
            arms[name] = {
                "image": str(path),
                "sha256": subprocess.run(["sha256sum", str(path)], check=True,
                                            capture_output=True, text=True).stdout.split()[0],
                "device_trials_ms": samples[name],
                "device_median_ms": statistics.median(samples[name]),
                "static": _static_metrics(path, args.objdump, args.readobj),
            }
        return {
            "chip": "gfx1201",
            "shape": list(shape),
            "causal": True,
            "dtype": "fp16",
            "timing": "HIP events; resident buffers; alternating arm order",
            "trials": args.trials,
            "iterations_per_trial": args.iterations,
            "max_abs_candidate_vs_baseline": max_abs,
            "arms": arms,
            "candidate_speedup": (arms["baseline"]["device_median_ms"] /
                                  arms["candidate"]["device_median_ms"]),
        }
    finally:
        for session in sessions.values():
            session.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--objdump", default="llvm-objdump")
    parser.add_argument("--readobj", default="llvm-readobj")
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
