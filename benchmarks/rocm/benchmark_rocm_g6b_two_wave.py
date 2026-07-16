#!/usr/bin/env python3
"""G6-B correctness/resource/performance ratchet for two-wave D=128 FA."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from tessera import runtime as rt  # noqa: E402

CASES = ((1, 8, 512, 64, False), (1, 8, 1024, 64, False),
         (1, 16, 1024, 128, False), (1, 16, 1009, 128, True))


def _mr(p, size):
    return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
            ctypes.c_int64(0), ctypes.c_int64(size), ctypes.c_int64(1)]


def _resources(blob):
    tool = next((p for p in (Path("/opt/rocm/llvm/bin/llvm-readobj"),
                              Path("/usr/lib/llvm-23/bin/llvm-readobj"))
                 if p.is_file()), None)
    result = {"vgpr_count": None, "sgpr_count": None, "lds_bytes": None,
              "scratch_bytes": None, "vgpr_spill_count": None,
              "sgpr_spill_count": None}
    if tool is None: return result
    with tempfile.NamedTemporaryFile(suffix=".hsaco") as f:
        f.write(blob); f.flush()
        text = subprocess.run([str(tool), "--notes", f.name],
                              capture_output=True, text=True, check=True).stdout
    patterns = {"vgpr_count": r"\.vgpr_count:\s*(\d+)",
                "sgpr_count": r"\.sgpr_count:\s*(\d+)",
                "lds_bytes": r"\.group_segment_fixed_size:\s*(\d+)",
                "scratch_bytes": r"\.private_segment_fixed_size:\s*(\d+)",
                "vgpr_spill_count": r"\.vgpr_spill_count:\s*(\d+)",
                "sgpr_spill_count": r"\.sgpr_spill_count:\s*(\d+)"}
    for key, pattern in patterns.items():
        if match := re.search(pattern, text): result[key] = int(match.group(1))
    result["spills"] = bool((result["scratch_bytes"] or 0)
                            or (result["vgpr_spill_count"] or 0)
                            or (result["sgpr_spill_count"] or 0))
    result["vgpr_limited_waves_per_simd"] = (
        min(16, 1536 // result["vgpr_count"]) if result["vgpr_count"] else None)
    return result


class _Session:
    def __init__(self, hip, q, k, v, causal, two_wave):
        self.hip, self.q = hip, q
        self.b, self.h, self.s, self.d = map(int, q.shape)
        self.two_wave = two_wave
        self.hsaco = rt._build_compiled_flash_attn_hsaco(
            self.d, "f16", two_wave=two_wave)
        self.mod, self.fn = ctypes.c_void_p(), ctypes.c_void_p()
        if hip.hipModuleLoadData(ctypes.byref(self.mod), self.hsaco):
            raise RuntimeError("FA module load failed")
        if hip.hipModuleGetFunction(ctypes.byref(self.fn), self.mod, b"fa"):
            raise RuntimeError("FA symbol missing")
        self.out = np.zeros(q.shape, np.float32)
        self.devs = []
        for host in (q, k, v, self.out):
            ptr = ctypes.c_void_p()
            if hip.hipMalloc(ctypes.byref(ptr), host.nbytes):
                raise RuntimeError("FA hipMalloc failed")
            hip.hipMemcpy(ptr, host.ctypes.data_as(ctypes.c_void_p), host.nbytes, 1)
            self.devs.append(ptr)
        dq, dk, dv, do = self.devs
        n = q.size
        args = (_mr(dq, n) + _mr(dk, n) + _mr(dv, n) + _mr(do, n)
                + [ctypes.c_int64(self.s), ctypes.c_int64(self.s),
                   ctypes.c_float(1 / np.sqrt(self.d)), ctypes.c_int64(causal)])
        self.args = args
        self.array = (ctypes.c_void_p * len(args))()
        for i, value in enumerate(args):
            self.array[i] = ctypes.cast(ctypes.byref(value), ctypes.c_void_p)

    def launch(self):
        return self.hip.hipModuleLaunchKernel(
            self.fn, (self.s + 15) // 16, self.b * self.h, 1,
            64 if self.two_wave else 32, 1, 1, 0, None, self.array, None)

    def download(self):
        if self.launch() or self.hip.hipDeviceSynchronize():
            raise RuntimeError("FA correctness launch failed")
        self.hip.hipMemcpy(self.out.ctypes.data_as(ctypes.c_void_p), self.devs[3],
                           self.out.nbytes, 2)
        return self.out.copy()

    def close(self):
        for ptr in self.devs: self.hip.hipFree(ptr)
        self.hip.hipModuleUnload(self.mod)


def _device_ms(hip, session, iterations):
    start, stop = ctypes.c_void_p(), ctypes.c_void_p()
    if hip.hipEventCreate(ctypes.byref(start)):
        raise RuntimeError("G6-B start-event creation failed")
    if hip.hipEventCreate(ctypes.byref(stop)):
        hip.hipEventDestroy(start)
        raise RuntimeError("G6-B stop-event creation failed")
    try:
        if hip.hipEventRecord(start, None):
            raise RuntimeError("G6-B start-event record failed")
        for _ in range(iterations):
            if session.launch(): raise RuntimeError("FA timed launch failed")
        if hip.hipEventRecord(stop, None) or hip.hipEventSynchronize(stop):
            raise RuntimeError("G6-B stop-event synchronization failed")
        value = ctypes.c_float()
        if hip.hipEventElapsedTime(ctypes.byref(value), start, stop):
            raise RuntimeError("G6-B HIP event timing failed")
        sample = float(value.value) / iterations
        if not math.isfinite(sample) or sample <= 0.0:
            raise RuntimeError(f"invalid G6-B timing sample: {sample} ms")
        return sample
    finally:
        hip.hipEventDestroy(start); hip.hipEventDestroy(stop)


def _e2e(hip, q, k, v, causal, two_wave):
    start = time.perf_counter_ns()
    session = _Session(hip, q, k, v, causal, two_wave)
    try: session.download()
    finally: session.close()
    return (time.perf_counter_ns() - start) / 1e6


def run(trials, iterations, correctness_only=False):
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0): raise RuntimeError("live ROCm device required")
    rows = []
    for b, h, s, d, causal in CASES:
        rng = np.random.default_rng(b + h + s + d + causal)
        q = (rng.standard_normal((b, h, s, d), dtype=np.float32) * .2).astype(np.float16)
        k = (rng.standard_normal(q.shape, dtype=np.float32) * .2).astype(np.float16)
        v = (rng.standard_normal(q.shape, dtype=np.float32) * .2).astype(np.float16)
        base = _Session(hip, q, k, v, causal, False)
        candidate = _Session(hip, q, k, v, causal, True) if d == 128 else None
        expected = base.download()
        actual = candidate.download() if candidate else expected
        max_error = float(np.max(np.abs(actual - expected)))
        if max_error > 3e-3: raise AssertionError(f"G6-B mismatch {max_error}")
        sessions = (("one_wave", base), ("two_wave", candidate)) if candidate else (("one_wave", base),)
        if correctness_only:
            for name, session in sessions:
                rows.append({"shape": [b, h, s, d], "causal": causal,
                             "schedule": name,
                             "resources": _resources(session.hsaco),
                             "max_abs_vs_one_wave": max_error})
            base.close()
            if candidate: candidate.close()
            continue
        samples = {name: {"device": [], "e2e": []} for name, _ in sessions}
        for session in [x[1] for x in sessions]:
            for _ in range(3): session.launch()
        hip.hipDeviceSynchronize()
        for trial in range(trials):
            order = list(sessions)
            if trial & 1: order.reverse()
            for name, session in order:
                samples[name]["device"].append(_device_ms(hip, session, iterations))
                samples[name]["e2e"].append(_e2e(hip, q, k, v, causal,
                                                  name == "two_wave"))
        for name, session in sessions:
            dmed = statistics.median(samples[name]["device"])
            emed = statistics.median(samples[name]["e2e"])
            row = {"shape": [b, h, s, d], "causal": causal, "schedule": name,
                   "device_trials_ms": samples[name]["device"],
                   "device_median_ms": dmed, "e2e_trials_ms": samples[name]["e2e"],
                   "e2e_median_ms": emed, "tflops": 4*b*h*s*s*d/(dmed*1e9),
                   "resources": _resources(session.hsaco), "max_abs_vs_one_wave": max_error}
            if candidate:
                ratios = [samples["one_wave"]["device"][i] /
                          samples["two_wave"]["device"][i] for i in range(trials)]
                row["paired_speedup"] = statistics.median(ratios) if name == "two_wave" else 1.0
                row["paired_win_rate"] = (sum(x > 1 for x in ratios) / trials
                                           if name == "two_wave" else 0.0)
            rows.append(row)
            print(f"{b}x{h}x{s}x{d} causal={causal} {name:8s} "
                  f"{dmed:.3f} ms {row['tflops']:.2f} TF/s", flush=True)
        base.close()
        if candidate: candidate.close()
    return {"schema": "tessera.rocm.g6b.v1", "evidence_arch": "gfx1151",
            "trials": trials, "iterations": iterations, "rows": rows,
            "performance_status": ("not_run" if correctness_only else "measured"),
            "all_correct": True}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--trials", type=int, default=9)
    p.add_argument("--iterations", type=int, default=20); p.add_argument("--output", required=True)
    p.add_argument("--correctness-only", action="store_true")
    a = p.parse_args(); result = run(a.trials, a.iterations, a.correctness_only)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__": main()
