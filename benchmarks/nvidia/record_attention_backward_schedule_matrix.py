"""Record NVIDIA atomic versus deterministic split/reduced attention backward.

The recorder keeps two rotated repeated-median runs, CUDA-event and full-call
timing, determinism policy, the exact workspace bound, numerical error, and
per-kernel resource fingerprints.  It never changes the production selector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_attention_backward_schedules.json"
NOISE = 0.03
ROUTES = ("atomic", "split_reduced")


def _cases():
    return (
        ("mha_d64", (1, 8, 8, 128, 128, 64), {"causal": False}),
        ("causal_mha_d128", (1, 8, 8, 64, 64, 128), {"causal": True}),
        ("ragged_gqa", (1, 8, 2, 129, 131, 64),
         {"causal": True, "window_left": 63, "window_right": 3}),
    )


def _inputs(name, shape):
    b, hq, hkv, sq, sk, d = shape
    rng = np.random.default_rng(20260801 + sum(map(ord, name)))
    q = rng.standard_normal((b, hq, sq, d), dtype=np.float32) * 0.2
    k = rng.standard_normal((b, hkv, sk, d), dtype=np.float32) * 0.2
    v = rng.standard_normal((b, hkv, sk, d), dtype=np.float32) * 0.2
    do = rng.standard_normal((b, hq, sq, d), dtype=np.float32) * 0.2
    return do, q, k, v


def _wall(call):
    start = time.perf_counter(); value = call()
    return (time.perf_counter() - start) * 1e3, value


def _resources():
    from benchmarks.nvidia.record_tile_fragment_resources import (
        _CudaOccupancy, _artifact_row, _inspect,
    )
    from tessera.compiler.emit.nvidia_cuda import _synthesize_flash_bwd_cuda

    entries = {
        "atomic": (("_Z13tsr_flash_bwd", "atomic_query_owned", 128),),
        "split_reduced": (
            ("_Z16tsr_flash_bwd_dq", "split_dq", 128),
            ("_Z19tsr_flash_bwd_split", "split_dkdv", 128),
            ("_Z20tsr_flash_bwd_reduce", "fixed_order_reduce", 128),
        ),
    }
    with tempfile.TemporaryDirectory(prefix="tessera-attn-bwd-resource-") as tmp:
        source = Path(tmp) / "attention_backward.cu"
        cubin = Path(tmp) / "attention_backward.cubin"
        source.write_text(_synthesize_flash_bwd_cuda(), encoding="utf-8")
        subprocess.run(
            ["/usr/local/cuda/bin/nvcc", "-arch=sm_120a", "-O3", "-cubin",
             str(source), "-o", str(cubin)], check=True)
        occupancy = _CudaOccupancy()
        try:
            names = tuple(_inspect(cubin)[0])
            resolved = {
                prefix: next(name for name in names if name.startswith(prefix))
                for specs in entries.values() for prefix, _, _ in specs
            }
            return {
                route: [
                    _artifact_row(cubin, resolved[entry], threads, occupancy,
                                  schedule=schedule,
                                  row_kind="attention_backward_candidate")
                    for entry, schedule, threads in specs
                ]
                for route, specs in entries.items()
            }
        finally:
            occupancy.close()


def record(*, e2e_reps=10, e2e_batches=10, device_reps=10,
           device_batches=5, warmup=5):
    from tessera.compiler.emit.nvidia_cuda import (
        flash_attention_backward_workspace_bytes,
        measure_flash_attention_backward_device,
        run_flash_attention_backward,
    )

    resources = _resources()
    rows = []
    for name, shape, options in _cases():
        do, q, k, v = _inputs(name, shape)
        kwargs = {**options, "scale": shape[-1] ** -0.5}
        outputs = {
            route: run_flash_attention_backward(
                do, q, k, v, route=route, **kwargs)
            for route in ROUTES
        }
        # Both routes consume one forward-derived gradient contract.  The
        # incumbent is already oracle-proven; compare the new route directly
        # and separately retain its repeatability result.
        errors = {
            route: max(float(np.max(np.abs(a - b)))
                       for a, b in zip(outputs[route], outputs["atomic"]))
            for route in ROUTES
        }
        repeated = run_flash_attention_backward(
            do, q, k, v, route="split_reduced", deterministic=True, **kwargs)
        deterministic_equal = all(
            np.array_equal(a, b)
            for a, b in zip(outputs["split_reduced"], repeated))
        runs = {route: [] for route in ROUTES}
        for _ in range(warmup):
            for route in ROUTES:
                run_flash_attention_backward(
                    do, q, k, v, route=route, **kwargs)
        device_samples = {
            run_id: {route: [] for route in ROUTES}
            for run_id in (1, 2)}
        for batch in range(device_batches):
            run_order = (1, 2) if batch % 2 == 0 else (2, 1)
            route_order = ROUTES if batch % 2 == 0 else ROUTES[::-1]
            for run_id in run_order:
                for route in route_order:
                    device_samples[run_id][route].append(
                        measure_flash_attention_backward_device(
                            do, q, k, v, route=route, reps=device_reps,
                            **kwargs))
        e2e_batches_by_run = {
            run_id: {route: [] for route in ROUTES}
            for run_id in (1, 2)}
        cohort_order = (1, 2, 2, 1)
        for batch in range(e2e_batches):
            samples = {
                run_id: {route: [] for route in ROUTES}
                for run_id in (1, 2)}
            for sample in range(e2e_reps * 2):
                run_id = cohort_order[sample % len(cohort_order)]
                route_order = (ROUTES if (sample // 4 + batch) % 2 == 0
                               else ROUTES[::-1])
                for route in route_order:
                    elapsed, _ = _wall(lambda route=route:
                        run_flash_attention_backward(
                            do, q, k, v, route=route, **kwargs))
                    samples[run_id][route].append(elapsed)
            for run_id in (1, 2):
                for route in ROUTES:
                    e2e_batches_by_run[run_id][route].append(
                        statistics.median(samples[run_id][route]))
        for run_id in (1, 2):
            for route in ROUTES:
                e2e_batches_for_route = e2e_batches_by_run[run_id][route]
                runs[route].append({
                    "run": run_id,
                    "device_event_ms": statistics.median(
                        device_samples[run_id][route]),
                    "device_batch_medians_ms": device_samples[run_id][route],
                    "end_to_end_ms": statistics.median(
                        e2e_batches_for_route),
                    "end_to_end_batch_medians_ms": e2e_batches_for_route,
                    "max_abs_error_vs_atomic": errors[route],
                })
        winners = {
            domain: [min(ROUTES, key=lambda route: runs[route][i][domain])
                     for i in range(2)]
            for domain in ("device_event_ms", "end_to_end_ms")
        }
        for route in ROUTES:
            domain_stable = {
                domain: abs(runs[route][0][domain] - runs[route][1][domain]) /
                max(runs[route][0][domain], runs[route][1][domain]) <= NOISE
                for domain in ("device_event_ms", "end_to_end_ms")
            }
            route_resources = resources[route]
            workspace = flash_attention_backward_workspace_bytes(
                k, v, route=route)
            rows.append({
                "case": name, "shape": list(shape), "dtype": "f32",
                "candidate": route, "runs": runs[route],
                "device_stable": domain_stable["device_event_ms"],
                "end_to_end_stable": domain_stable["end_to_end_ms"],
                "stable": all(domain_stable.values()),
                "device_winner_consensus": winners["device_event_ms"] == [route] * 2,
                "end_to_end_winner_consensus": winners["end_to_end_ms"] == [route] * 2,
                "determinism_policy": (
                    "fixed_order_bitwise" if route == "split_reduced"
                    else "floating_atomic_nondeterministic"),
                "observed_bitwise_repeatable": (
                    deterministic_equal if route == "split_reduced" else None),
                "workspace_bytes": workspace,
                "workspace_limit_contract": (
                    "one_extra_dk_plus_dv_f32" if route == "split_reduced"
                    else "none"),
                "traffic_bytes": int(
                    do.nbytes + q.nbytes + k.nbytes + v.nbytes +
                    sum(x.nbytes for x in outputs[route]) + workspace),
                "resources": route_resources,
                "resource_fingerprints": [
                    row["resource_fingerprint"] for row in route_resources],
                "resource_evidence_complete": bool(route_resources),
                "sampling": {
                    "device_batches": device_batches,
                    "device_reps_per_batch": device_reps,
                    "end_to_end_batches": e2e_batches,
                    "end_to_end_reps_per_batch": e2e_reps,
                    "run_cohorts": "balanced_abba_disjoint_samples",
                    "candidate_order": "rotated_interleaved",
                },
            })
    device = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version",
         "--format=csv,noheader"], check=True, capture_output=True,
        text=True).stdout.strip()
    return {
        "schema": "tessera.nvidia.attention-backward-schedules.v1",
        "device": device, "noise_policy": NOISE,
        "production_route": "atomic",
        "selector_changed": False,
        "rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e-reps", type=int, default=10)
    parser.add_argument("--e2e-batches", type=int, default=10)
    parser.add_argument("--device-reps", type=int, default=10)
    parser.add_argument("--device-batches", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    result = record(e2e_reps=args.e2e_reps,
                    e2e_batches=args.e2e_batches,
                    device_reps=args.device_reps,
                    device_batches=args.device_batches, warmup=args.warmup)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(result['rows'])} candidate rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
