"""Record the exact-case NVIDIA GEMM/fused schedule ratchet on sm_120.

Unlike the dispatch corpus, this audit artifact never buckets distinct shapes.
Each timing domain is measured twice with a fresh MeasureCache and joined to
retained Nsight resource fingerprints.  It records evidence only; it does not
change a production selector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_gemm_schedule_matrix.json"
RESOURCES = ROOT / "benchmarks/baselines/nvidia_sm120_test5_route_resources.json"

MATMUL_CASES = (
    ("square", (256, 256, 256)),
    ("square", (1024, 1024, 1024)),
    ("rectangular", (128, 512, 64)),
    ("rectangular", (512, 128, 256)),
    ("ragged", (127, 259, 63)),
    ("ragged", (513, 255, 129)),
)
FUSED_CASES = tuple(
    (activation, (256, 256, 256))
    for activation in ("none", "relu", "gelu", "silu")) + (
    ("gelu", (127, 259, 63)),
)


def near_winner_consensus(first: dict[str, float], second: dict[str, float],
                          noise_fraction: float) -> list[str]:
    def near(values: dict[str, float]) -> set[str]:
        floor = min(values.values())
        return {name for name, value in values.items()
                if value <= floor * (1 + noise_fraction)}
    return sorted(near(first) & near(second))


def _fingerprint(command: list[str]) -> str:
    text = subprocess.run(command, check=True, capture_output=True,
                          text=True).stdout
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _measure(region: Any, op: str, inputs: tuple[np.ndarray, ...],
             dims: tuple[int, ...], dtype: str, timing: str, *, reps: int,
             warmup: int) -> dict[str, Any]:
    from tessera.compiler.emit import autotune as at
    from tessera.compiler.emit.candidate import candidates_for, verify_candidate
    from tessera.compiler.emit.kernel_emitter import REFERENCE_EXECUTIONS
    if timing == at.TIMING_END_TO_END:
        candidates = [candidate for candidate in candidates_for("nvidia", op)
                      if candidate.available() and candidate.applies_to(region)
                      and verify_candidate(candidate, region)]
        if not candidates:
            raise RuntimeError(f"no F4-verified end-to-end candidate for {op} {dims}")
        active = []
        for candidate in candidates:
            _, tag = candidate.run(region, *inputs)
            if tag not in REFERENCE_EXECUTIONS:
                active.append(candidate)
        candidates = active
        if not candidates:
            raise RuntimeError(f"all candidates declined exact audit inputs for {op} {dims}")
        samples: dict[str, list[float]] = {candidate.name: []
                                          for candidate in candidates}
        for candidate in candidates:
            for _ in range(max(0, warmup - 1)):
                candidate.run(region, *inputs)
        for iteration in range(reps):
            rotated = candidates[iteration % len(candidates):] + candidates[:iteration % len(candidates)]
            for candidate in rotated:
                start = time.perf_counter()
                _, tag = candidate.run(region, *inputs)
                elapsed = (time.perf_counter() - start) * 1e3
                if tag in REFERENCE_EXECUTIONS:
                    raise RuntimeError(f"{candidate.name} changed to reference during timing")
                samples[candidate.name].append(elapsed)
        medians = {name: statistics.median(values)
                   for name, values in samples.items()}
        winner = min(medians, key=medians.__getitem__)
        return {"winner": winner, "latency_ms": medians[winner],
                "candidates": medians,
                "sampling": "rotated_interleaved_wall_clock"}
    cache = at.MeasureCache()
    winner = at.measured_arbitrate(
        region, op, "nvidia", *inputs, dims=dims, dtype=dtype, cache=cache,
        reps=reps, warmup=warmup, timing=timing)
    if winner is None or cache.size != 1:
        raise RuntimeError(f"no measured {timing} candidate for {op} {dims}")
    record = next(iter(cache._store.values()))
    finite = {name: value for name, value in record.candidates.items()
              if np.isfinite(value)}
    if not finite:
        raise RuntimeError(f"no finite {timing} measurements for {op} {dims}")
    return {"winner": record.winner, "latency_ms": record.latency_ms,
            "candidates": finite, "sampling": "candidate_cuda_event_median"}


def record(*, reps: int, warmup: int, device_reps: int,
           device_warmup: int, noise_fraction: float) -> dict[str, Any]:
    from tessera import runtime as rt
    from tessera.compiler.emit import nvidia_cuda  # noqa: F401
    from tessera.compiler.emit.autotune import TIMING_DEVICE, TIMING_END_TO_END
    from tessera.compiler.emit.candidate import OP_FUSED_REGION, OP_MATMUL
    from tessera.compiler.fusion import FusedRegion, MatmulRegion

    if rt._nvidia_device_name() != "sm_120":
        raise RuntimeError("schedule matrix requires the sm_120 NVIDIA host")
    route_resources = json.loads(RESOURCES.read_text()).get("routes", {})
    rng = np.random.default_rng(20260720)
    cases: list[tuple[str, str, tuple[int, int, int], str, Any,
                      tuple[np.ndarray, ...]]] = []
    for category, dims in MATMUL_CASES:
        m, n, k = dims
        for dtype in ("float16", "bfloat16"):
            a = (rng.standard_normal((m, k)) * .2).astype(np.float32)
            b = (rng.standard_normal((k, n)) * .2).astype(np.float32)
            cases.append((f"matmul:{category}:{m}x{n}x{k}:{dtype}",
                          OP_MATMUL, dims, dtype, MatmulRegion(dtype=dtype),
                          (a, b)))
    for activation, dims in FUSED_CASES:
        m, n, k = dims
        a = (rng.standard_normal((m, k)) * .2).astype(np.float32)
        b = (rng.standard_normal((k, n)) * .2).astype(np.float32)
        bias = (rng.standard_normal(n) * .1).astype(np.float32)
        epilogue = ("bias",) if activation == "none" else ("bias", activation)
        cases.append((f"fused:{activation}:{m}x{n}x{k}:f16",
                      OP_FUSED_REGION, dims, "f16",
                      FusedRegion(epilogue=epilogue), (a, b, bias)))

    rows: list[dict[str, Any]] = []
    for case_id, op, dims, dtype, region, inputs in cases:
        for timing in (TIMING_DEVICE, TIMING_END_TO_END):
            timing_reps = device_reps if timing == TIMING_DEVICE else reps
            timing_warmup = device_warmup if timing == TIMING_DEVICE else warmup
            runs = [_measure(region, op, inputs, dims, dtype, timing,
                             reps=timing_reps, warmup=timing_warmup)
                    for _ in range(2)]
            consensus = near_winner_consensus(
                runs[0]["candidates"], runs[1]["candidates"], noise_fraction)
            winner = (min(consensus, key=lambda name: (
                runs[0]["candidates"][name] + runs[1]["candidates"][name], name))
                if consensus else None)
            candidate_names = sorted(set(runs[0]["candidates"]) |
                                     set(runs[1]["candidates"]))
            resource_map = {name: list(route_resources.get(name, ()))
                            for name in candidate_names}
            rows.append({
                "case": case_id, "op": op, "shape": list(dims),
                "dtype": dtype, "timing": timing, "runs": runs,
                "noise_fraction": noise_fraction,
                "near_winner_consensus": consensus,
                "stable": bool(consensus), "selected_winner": winner,
                "candidate_resource_fingerprints": resource_map,
                "winner_resource_fingerprints": (
                    resource_map.get(winner, []) if winner else []),
                "selector_eligible": bool(
                    winner and resource_map.get(winner) and
                    all(resource_map.get(name) for name in candidate_names)),
            })
            print(case_id, timing, winner or "UNSTABLE")

    return {
        "schema": "tessera.nvidia.gemm-schedule-matrix.v1",
        "device": subprocess.run([
            "nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version",
            "--format=csv,noheader"], check=True, capture_output=True,
            text=True).stdout.strip(),
        "compiler_fingerprints": {
            "nvcc": _fingerprint(["/usr/local/cuda/bin/nvcc", "--version"]),
            "llvm": _fingerprint(["/usr/lib/llvm-23/bin/llvm-config", "--version"]),
        },
        "method": {
            "runs": 2, "compile_state": "warm_after_correctness_gate",
            "measure_cache": "fresh_per_case_and_timing_domain",
            "reps": reps, "warmup": warmup,
            "device_reps": device_reps, "device_warmup": device_warmup,
            "noise_fraction": noise_fraction,
            "correctness": "Candidate F4 oracle before every timing set",
            "end_to_end_sampling": "rotated interleaving removes order drift",
            "selector_changed": False,
        },
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device-reps", type=int, default=100)
    parser.add_argument("--device-warmup", type=int, default=10)
    parser.add_argument("--noise-fraction", type=float, default=.03)
    args = parser.parse_args(argv)
    payload = record(reps=args.reps, warmup=args.warmup,
                     device_reps=args.device_reps,
                     device_warmup=args.device_warmup,
                     noise_fraction=args.noise_fraction)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output} ({len(payload['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
