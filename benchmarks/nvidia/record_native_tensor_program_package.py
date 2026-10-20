"""Measure source compilation, native binding, completed calls and device time."""
from __future__ import annotations

import argparse
import ctypes as ct
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess
import time
from unittest.mock import patch

import numpy as np

from tessera import runtime as rt
from tests.device.nvidia.test_native_tensor_program_package_execution import (
    CASES, compile_case, oracle, roots, values,
)


def run(provider, *, repetitions=32, rounds=7, artifacts=None):
    if rt._nvidia_device_name() != "sm_120":
        raise RuntimeError("native tensor source recorder requires exact SM120")
    if repetitions < 1 or rounds < 1:
        raise ValueError("positive repetitions and rounds required")
    lib = ct.CDLL(str(provider))
    records = {}
    for kind, storage in CASES:
        a, b = values(storage, 117)
        started = time.perf_counter()
        program = compile_case(kind, storage)
        compile_ms = (time.perf_counter() - started) * 1e3
        if artifacts is not None:
            artifacts.mkdir(parents=True, exist_ok=True)
            for stage, text in (("source", program.source_ir), ("target", program.target_ir)):
                (artifacts / f"{kind}-{storage}.{stage}.mlir.gz").write_bytes(
                    gzip.compress(text.encode(), mtime=0))
        expected = oracle(kind, a, b)
        started = time.perf_counter()
        owner = program.prepare(provider=lib)
        bind_ms = (time.perf_counter() - started) * 1e3
        with owner:
            result = owner(roots(kind, a, b))
            np.testing.assert_allclose(result, expected, rtol=.004, atol=2e-5)
            initial = owner.stats()
            host_samples, device_samples, stage_samples = [], [], []
            with patch.object(subprocess, "run", side_effect=AssertionError("warm call invoked compiler")), \
                    patch.object(subprocess, "Popen", side_effect=AssertionError("warm call started tool")):
                for _ in range(rounds):
                    started = time.perf_counter()
                    for _ in range(repetitions):
                        result = owner(roots(kind, a, b))
                    host_samples.append((time.perf_counter() - started) * 1e3 / repetitions)
                    measured = owner.profile(repetitions, captured=True)
                    device_samples.append(measured["program_ms"])
                    stage_samples.append(measured["stage_ms"])
            np.testing.assert_allclose(result, expected, rtol=.004, atol=2e-5)
            final = owner.stats()
            assert final["allocations"] == initial["allocations"] == 1
            assert final["capacity"] == initial["capacity"]
        records[kind + "-" + storage] = {
            "source_compile_ms": compile_ms, "native_bind_ms": bind_ms,
            "completed_call_samples_ms": host_samples, "completed_call_median_ms": median(host_samples),
            "captured_program_samples_ms": device_samples, "captured_program_median_ms": median(device_samples),
            "captured_stage_samples_ms": stage_samples,
            "source_sha256": hashlib.sha256(program.source_ir.encode()).hexdigest(),
            "target_sha256": hashlib.sha256(program.target_ir.encode()).hexdigest(),
            "image_sha256": [hashlib.sha256(image).hexdigest() for image in program.images],
            "compiler_identity": program.compiler_identity, "frame": program.frame,
            "device_allocations": final["allocations"], "device_capacity_bytes": final["capacity"],
            "max_abs_error": float(np.max(np.abs(result - expected))),
            "correctness": "passed_before_and_after_timing",
        }
        print(kind + "-" + storage, "compile_ms", round(compile_ms, 3), "bind_ms", round(bind_ms, 3),
              "host_ms", round(median(host_samples), 6), "device_ms", round(median(device_samples), 6),
              flush=True)
    device = subprocess.check_output([
        "/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=name,uuid,driver_version",
        "--format=csv,noheader"], text=True).strip()
    return {
        "scope": "named original-source native package API; ordinary JIT/portable integration remains open",
        "shape_mnk": [17, 11, 32], "device": device, "architecture": "sm_120",
        "provider_sha256": hashlib.sha256(provider.read_bytes()).hexdigest(),
        "rounds": rounds, "repetitions": repetitions,
        "timing_contract": {
            "source_compile": "frontend trace, admission, full native lowering, image packaging and source replay; existing process/tool caches",
            "native_bind": "driver context/module JIT, ABI checks and arena preparation; first bind may include lazy context initialization",
            "completed_call": "validated roots, H2D, all kernels, D2H, completion and host result allocation; external tools forbidden",
            "device": "captured native full-program/stage event windows; excludes transfers/readback/capture setup",
            "stage_windows": "independent, nonadditive",
        },
        "cases": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--artifacts", type=Path, help="Retain exact compressed source/Target witnesses")
    args = parser.parse_args()
    data = run(args.provider.resolve(), repetitions=args.repetitions, rounds=args.rounds, artifacts=args.artifacts)
    args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
