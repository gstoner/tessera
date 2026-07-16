#!/usr/bin/env python3
"""G6-C repeated-median split/reduced dK/dV backward ratchet."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from tessera import runtime as rt  # noqa: E402

# B, query heads, KV heads, sequence, D, causal.
CASES = ((1, 8, 8, 512, 64, False), (1, 8, 8, 512, 64, True),
         (1, 16, 16, 1024, 128, False), (1, 16, 16, 1024, 128, True),
         (1, 8, 2, 512, 64, False), (1, 16, 4, 1024, 128, True))


def _artifact(causal):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_flash_attn_bwd_compiled",
        "executable": True, "arg_names": ["do", "q", "k", "v"],
        "ops": [{"op_name": "tessera.flash_attn_bwd",
                 "operands": ["do", "q", "k", "v"],
                 "kwargs": {"causal": causal}}],
    })


def _call(artifact, values, split, reps=0):
    return rt._execute_rocm_compiled_flash_attn_bwd(
        artifact, values, _split_reduced=split, _timed_reps=reps)


def run(trials, iterations, correctness_only=False):
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0): raise RuntimeError("live ROCm device required")
    rows = []
    for b, h, g, s, d, causal in CASES:
        rng = np.random.default_rng(b + h + g + s + d + causal)
        qshape, kvshape = (b, h, s, d), (b, g, s, d)
        q = (rng.standard_normal(qshape, dtype=np.float32) * .15).astype(np.float16)
        k = (rng.standard_normal(kvshape, dtype=np.float32) * .15).astype(np.float16)
        v = (rng.standard_normal(kvshape, dtype=np.float32) * .15).astype(np.float16)
        do = (rng.standard_normal(qshape, dtype=np.float32) * .15).astype(np.float16)
        artifact, values = _artifact(causal), (do, q, k, v)
        expected = _call(artifact, values, False)
        actual = _call(artifact, values, True)
        errors = [float(np.max(np.abs(a - e)))
                  for a, e in zip(actual, expected, strict=True)]
        for a, e in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, e, rtol=4e-3, atol=4e-4)
        if correctness_only:
            for split in (False, True):
                rows.append({
                    "shape": [b, h, g, s, d], "causal": causal,
                    "schedule": "split_reduced" if split else "serial_dkdv",
                    "max_abs_vs_serial": errors,
                    "temporary_bytes": 2 * b * g * s * d * 4 if split else 0,
                    "gradient_kv_bytes": 2 * b * g * s * d * 4,
                })
            continue
        samples = {False: {"device": [], "e2e": []},
                   True: {"device": [], "e2e": []}}
        for trial in range(trials):
            order = (False, True) if not (trial & 1) else (True, False)
            for split in order:
                _, device_ms = _call(artifact, values, split, iterations)
                if (device_ms is None or not math.isfinite(device_ms)
                        or device_ms <= 0.0):
                    raise RuntimeError(
                        f"invalid G6-C timing sample: {device_ms} ms")
                samples[split]["device"].append(device_ms)
                start = time.perf_counter_ns(); _call(artifact, values, split)
                samples[split]["e2e"].append(
                    (time.perf_counter_ns() - start) / 1e6)
        ratios = [samples[False]["device"][i] / samples[True]["device"][i]
                  for i in range(trials)]
        eratio = [samples[False]["e2e"][i] / samples[True]["e2e"][i]
                  for i in range(trials)]
        for split in (False, True):
            dmed = statistics.median(samples[split]["device"])
            emed = statistics.median(samples[split]["e2e"])
            rows.append({
                "shape": [b, h, g, s, d], "causal": causal,
                "schedule": "split_reduced" if split else "serial_dkdv",
                "device_trials_ms": samples[split]["device"],
                "device_median_ms": dmed, "e2e_trials_ms": samples[split]["e2e"],
                "e2e_median_ms": emed,
                "tflops": 10*b*h*s*s*d/(dmed*1e9),
                "paired_device_speedup": statistics.median(ratios) if split else 1.0,
                "paired_device_win_rate": (sum(x > 1 for x in ratios)/trials
                                             if split else 0.0),
                "paired_e2e_speedup": statistics.median(eratio) if split else 1.0,
                "paired_e2e_win_rate": (sum(x > 1 for x in eratio)/trials
                                         if split else 0.0),
                "max_abs_vs_serial": errors,
                "temporary_bytes": 2 * b * g * s * d * 4 if split else 0,
                "gradient_kv_bytes": 2 * b * g * s * d * 4,
            })
            print(f"{b}xH{h}/G{g}x{s}x{d} causal={causal} "
                  f"{'split' if split else 'serial':6s} {dmed:.3f} ms "
                  f"e2e={emed:.2f} ms", flush=True)
    return {"schema": "tessera.rocm.g6c.v1", "evidence_arch": "gfx1151",
            "trials": trials, "iterations": iterations, "rows": rows,
            "performance_status": ("not_run" if correctness_only else "measured"),
            "all_correct": True}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--trials", type=int, default=9)
    p.add_argument("--iterations", type=int, default=10); p.add_argument("--output", required=True)
    p.add_argument("--correctness-only", action="store_true")
    a = p.parse_args(); result = run(a.trials, a.iterations, a.correctness_only)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__": main()
