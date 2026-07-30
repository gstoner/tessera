#!/usr/bin/env python3
"""Measure saved versus recomputed LSE on the exact x86 AVX-512 host."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import statistics
import time

import numpy as np

from tessera import runtime


def _artifact(sequence: int, checkpoint: str) -> runtime.RuntimeArtifact:
    return runtime.RuntimeArtifact(
        metadata={
            "target": "x86",
            "compiler_path": "x86_flash_attn_bwd_compiled",
            "executable": True,
            "execution_kind": "native_cpu",
            "arg_names": ["do", "q", "k", "v"],
            "output_name": "grads",
            "ops": [
                {
                    "op_name": "tessera.flash_attn",
                    "operands": ["q", "k", "v"],
                    "kwargs": {
                        "scale": 0.25,
                        "causal": True,
                        "lse_checkpoint": checkpoint,
                        "sequence": sequence,
                    },
                }
            ],
        }
    )


def _forward_artifact() -> runtime.RuntimeArtifact:
    return runtime.RuntimeArtifact(
        metadata={
            "target": "x86",
            "compiler_path": "x86_flash_attn_compiled",
            "executable": True,
            "execution_kind": "native_cpu",
            "arg_names": ["q", "k", "v"],
            "output_name": "o",
            "ops": [
                {
                    "op_name": "tessera.flash_attn",
                    "operands": ["q", "k", "v"],
                    "kwargs": {"scale": 0.25, "causal": True},
                }
            ],
        }
    )


def _measure(sequence: int, checkpoint: str, warmup: int, reps: int) -> float:
    rng = np.random.default_rng(sequence)
    shape = (1, 2, sequence, 16)
    args = tuple(
        np.ascontiguousarray(rng.standard_normal(shape), dtype=np.float32)
        for _ in range(4)
    )
    artifact = _artifact(sequence, checkpoint)
    forward = _forward_artifact()

    def run_once():
        # Recompute policy pays the established forward plus backward row-stat
        # recomputation. Saved policy's backward artifact executes the
        # forward+LSE entry and then consumes that checkpoint.
        if checkpoint == "recompute":
            result = runtime.launch(forward, args[1:])
            if not result.get("ok"):
                return result
        return runtime.launch(artifact, args)

    for _ in range(warmup):
        result = run_once()
        if not result.get("ok"):
            raise RuntimeError(str(result.get("reason")))
    samples = []
    for _ in range(reps):
        start = time.perf_counter_ns()
        result = run_once()
        elapsed = time.perf_counter_ns() - start
        if not result.get("ok"):
            raise RuntimeError(str(result.get("reason")))
        samples.append(elapsed / 1_000_000.0)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=21)
    parser.add_argument("--sequences", type=int, nargs="+", default=[32, 64, 128])
    args = parser.parse_args()
    rows = []
    for sequence in args.sequences:
        recompute = _measure(sequence, "recompute", args.warmup, args.reps)
        saved = _measure(sequence, "saved", args.warmup, args.reps)
        rows.append(
            {
                "sequence": sequence,
                "recompute_ms": recompute,
                "saved_ms": saved,
                "saved_speedup": recompute / saved,
                "winner": "saved" if saved < recompute else "recompute",
            }
        )
    cpu_name = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor() or platform.uname().processor,
    )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "work_item": "X86-LSE-1",
                "host": cpu_name,
                "platform": platform.platform(),
                "timer": "time.perf_counter_ns",
                "warmup": args.warmup,
                "reps": args.reps,
                "rows": rows,
                "verdict": (
                    "saved"
                    if all(row["winner"] == "saved" for row in rows)
                    else "recompute"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
