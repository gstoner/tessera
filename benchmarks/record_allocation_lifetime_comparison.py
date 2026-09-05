#!/usr/bin/env python3
"""Exact-host baseline/candidate compiler comparison; never updates selectors.

This verifier does not change kernels. Image equality separates analysis changes
from code generation; timings are observations, not evidence of a queue speedup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]


def record(args):
    compilers = {name: Path(getattr(args, name)).resolve() for name in ("before", "after")}
    for compiler in compilers.values():
        if not compiler.is_file():
            raise RuntimeError(f"missing compiler: {compiler}")
    result = {
        "schema": "tessera.allocation-lifetime-comparison.v1",
        "sync_key": "IR-NATIVE-FOUNDATION-1",
        "backend": args.backend,
        "compiler_sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in compilers.items()},
        "selector_changed": False,
        "scope": "existing native attention/GEMM regression; no new queue schedule or performance promotion",
    }
    if args.backend == "nvidia":
        import numpy as np
        from benchmarks.nvidia.record_lse_checkpoint import _compile, _sample
        from tests.device.nvidia.test_lse_checkpoint_native import _forward_module, _backward_module, _reference
        from tessera.compiler import nvidia_native
        import subprocess
        result["device"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True
        ).strip()
        shape = (1, 2, 1, 8, 8, 8, 8)
        rng = np.random.default_rng(12005)
        q = (rng.normal(size=(1, 2, 8, 8)) * .2).astype(np.float32)
        k = (rng.normal(size=(1, 1, 8, 8)) * .2).astype(np.float32)
        v = (rng.normal(size=(1, 1, 8, 8)) * .2).astype(np.float32)
        do = (rng.normal(size=(1, 2, 8, 8)) * .2).astype(np.float32)
        expected, expected_lse, grads = _reference(q, k, v, do)
        packets = {}
        for variant, compiler in compilers.items():
            os.environ["TESSERA_OPT"] = str(compiler)
            nvidia_native._cache.clear()
            output, lse = np.empty_like(q), np.empty((1, 2, 8), np.float32)
            dq, dk, dv = np.empty_like(q), np.empty_like(k), np.empty_like(v)
            scalars = dict(zip(("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv"), shape, strict=True))
            forward = {"q": q, "k": k, "v": v, "o": output, "row_lse": lse, **scalars}
            backward = {"q": q, "k": k, "v": v, "do": do, "row_lse": lse,
                        "dq": dq, "dk": dk, "dv": dv, **scalars}
            rows = []
            for stage, make_module, bindings in (("forward", _forward_module, forward),
                                                  ("backward", _backward_module, backward)):
                module = make_module(saved=True, shape=shape)
                bundle = _compile(module)
                row = _sample(bundle, module, bindings, samples=args.samples, reps=args.reps, warmup=10)
                row.update(stage=stage, image_digest=bundle.native_image.image_digest)
                rows.append(row)
            for actual, reference in zip((output, lse, dq, dk, dv), (expected, expected_lse, *grads), strict=True):
                np.testing.assert_allclose(actual, reference, rtol=3e-4, atol=3e-5)
            packets[variant] = {"rows": rows, "oracle": "forward/LSE/dQ/dK/dV allclose", "shape": shape}
        result["measurements"] = packets
        result["images_equal"] = all(
            a["image_digest"] == b["image_digest"]
            for a, b in zip(packets["before"]["rows"], packets["after"]["rows"], strict=True)
        )
        result["sampling_limit"] = "before then after; no performance inference from ordering-sensitive deltas"

    else:
        from benchmarks.rocm import benchmark_rocm_gemm_pipeline_vs_direct as bench
        tool = bench._find_mlir_opt()
        hip = bench._load_hip()
        if tool is None or hip is None or hip.hipInit(0) != 0:
            raise RuntimeError("ROCm device/toolchain unavailable")
        builds = {}
        for variant, compiler in compilers.items():
            bench.TESSERA_OPT = compiler
            for mode in ("direct", "fork_a"):
                builds[(variant, mode)] = bench._build(tool, mode, 2, 4)
        samples = {key: [] for key in builds}
        errors = {}
        for trial in range(args.samples):
            keys = list(builds)
            if trial % 2: keys.reverse()
            for key in keys:
                measured = bench._run(hip, builds[key], 512, 512, 512, 2, 4, args.reps, check=(trial == 0))
                if measured is None:
                    raise RuntimeError(f"no execution for {key}")
                ms, error = measured
                if error is not None:
                    if error > 1e-2:
                        raise RuntimeError(f"oracle mismatch for {key}: {error}")
                    errors[key] = error
                samples[key].append(ms)
        result["device"] = bench.CHIP
        result["clock"] = "synchronized host wall time; not HIP device events"
        result["measurements"] = [
            {"variant": key[0], "mode": key[1], "shape": [512]*3,
             "image_sha256": hashlib.sha256(builds[key]).hexdigest(),
             "samples_ms": values, "median_ms": statistics.median(values),
             "oracle_relative_error": errors[key], "repetitions": args.reps}
            for key, values in samples.items()
        ]
        result["images_equal"] = all(builds[("before", mode)] == builds[("after", mode)]
                                     for mode in ("direct", "fork_a"))
    if not result["images_equal"]:
        raise RuntimeError("verifier-only change unexpectedly changed native images")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nvidia", "rocm"), required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    args = parser.parse_args()
    if args.samples < 3 or args.reps < 1:
        parser.error("require at least three samples and one repetition")
    result = record(args)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}; native images unchanged")


if __name__ == "__main__":
    main()
