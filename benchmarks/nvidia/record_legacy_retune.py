"""Consolidated NVIDIA legacy retune for GEMM, grouped work, and transport.

Rows retain one numerical oracle, two rotated repeated-median runs, resident
CUDA-event and end-to-end timing, launch counts, byte/bandwidth evidence, and
the exact retained resource fingerprints available for each route.
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
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_legacy_retune.json"
RESOURCE = ROOT / "benchmarks/baselines/nvidia_sm120_test5_route_resources.json"
SERVING = ROOT / "benchmarks/baselines/nvidia_sm120_serving.json"
NOISE = 0.03


def _wall(call):
    start = time.perf_counter(); value = call()
    return (time.perf_counter() - start) * 1e3, value


def _resources(route):
    data = json.loads(RESOURCE.read_text())
    rows = data["details"].get(route, [])
    return rows, [row["resource_fingerprint"] for row in rows]


def _gated_epilogue_resource(activation):
    """Compile and inspect the exact generated gate used by grouped SwiGLU."""
    from benchmarks.nvidia.record_tile_fragment_resources import (
        _CudaOccupancy, _artifact_row)
    from tessera.compiler.emit.nvidia_cuda import _synthesize_gated_epilogue_cuda
    _, kernel, source = _synthesize_gated_epilogue_cuda(activation)
    with tempfile.TemporaryDirectory(prefix="tessera-gated-resource-") as tmp:
        work = Path(tmp)
        cuda, cubin = work / "gate.cu", work / "gate.cubin"
        cuda.write_text(source)
        subprocess.run(["/usr/local/cuda/bin/nvcc", "-arch=sm_120", "-O3",
                        "-cubin", str(cuda), "-o", str(cubin)], check=True)
        occupancy = _CudaOccupancy()
        try:
            return _artifact_row(cubin, kernel, 256, occupancy,
                                 row_kind="generated_epilogue",
                                 schedule=f"gated_{activation}")
        finally:
            occupancy.close()


def _tf32_run(a, b):
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    with NvidiaDeviceSession() as session:
        da, db = session.upload(a), session.upload(b)
        out = session.empty((a.shape[0], b.shape[1]), np.float32)
        session.gemm(da, db, out, "float32")
        return out.numpy()


def _tf32_device(a, b, reps, warmup):
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    with NvidiaDeviceSession() as session:
        da, db = session.upload(a), session.upload(b)
        out = session.empty((a.shape[0], b.shape[1]), np.float32)
        return session.measure(lambda: session.gemm(da, db, out, "float32"),
                               reps=reps, warmup=warmup)


def _legacy_grouped(x, weights, groups):
    from tessera.compiler.emit.nvidia_cuda import run_grouped_gemm_f32
    parts = []; start = 0
    for expert, count in enumerate(groups):
        stop = start + int(count)
        if stop > start:
            parts.append(run_grouped_gemm_f32(
                x[start:stop], weights[expert:expert + 1],
                np.array([stop - start], np.int64)))
        start = stop
    return np.concatenate(parts) if parts else np.empty((0, weights.shape[-1]), np.float32)


def _legacy_grouped_device(x, weights, groups, reps):
    from tessera.compiler.emit.nvidia_cuda import measure_grouped_gemm_device
    elapsed = 0.0; start = 0
    for expert, count in enumerate(groups):
        stop = start + int(count)
        if stop > start:
            elapsed += measure_grouped_gemm_device(
                x[start:stop], weights[expert:expert + 1],
                np.array([stop - start], np.int64), reps=reps)
        start = stop
    return elapsed


def _candidate_case(name, shape, dtype, reference, candidates, *,
                    e2e_reps, e2e_batches, device_reps, device_batches,
                    warmup, traffic_bytes):
    from benchmarks.nvidia._clock_conditioning import condition_sm120
    errors = {}
    for candidate, spec in candidates.items():
        errors[candidate] = float(np.max(np.abs(spec["run"]() - reference)))
    runs = {candidate: [] for candidate in candidates}
    names = tuple(candidates)
    clock_condition_ms = condition_sm120(reps=100)
    device_samples = {
        run_id: {candidate: [] for candidate in candidates}
        for run_id in (1, 2)}
    for batch in range(device_batches):
        run_order = (1, 2) if batch % 2 == 0 else (2, 1)
        order = names if batch % 2 == 0 else names[::-1]
        for run_id in run_order:
            for candidate in order:
                device_samples[run_id][candidate].append(
                    candidates[candidate]["device"](device_reps, warmup))
    batch_medians = {
        run_id: {candidate: [] for candidate in candidates}
        for run_id in (1, 2)}
    for batch in range(e2e_batches):
        samples = {
            run_id: {candidate: [] for candidate in candidates}
            for run_id in (1, 2)}
        cohort_order = (1, 2, 2, 1)
        for sample in range(e2e_reps * 2):
            run_id = cohort_order[sample % len(cohort_order)]
            order = (names if (sample // 4 + batch) % 2 == 0
                     else names[::-1])
            for candidate in order:
                elapsed, _ = _wall(candidates[candidate]["run"])
                samples[run_id][candidate].append(elapsed)
        for run_id in (1, 2):
            for candidate in candidates:
                batch_medians[run_id][candidate].append(
                    statistics.median(samples[run_id][candidate]))
    for run_id in (1, 2):
        for candidate in candidates:
            candidate_batches = batch_medians[run_id][candidate]
            runs[candidate].append({
                "run": run_id,
                "device_event_ms": statistics.median(
                    device_samples[run_id][candidate]),
                "device_batch_medians_ms":
                    device_samples[run_id][candidate],
                "end_to_end_ms": statistics.median(candidate_batches),
                "end_to_end_batch_medians_ms": candidate_batches,
                "clock_condition_ms": clock_condition_ms,
                "max_abs_error": errors[candidate],
            })
    winners = {domain: [min(names, key=lambda n: runs[n][i][domain])
                        for i in range(2)]
               for domain in ("device_event_ms", "end_to_end_ms")}
    rows = []
    for candidate, spec in candidates.items():
        resource_rows, fingerprints = _resources(spec["resource_route"])
        resource_rows = resource_rows + spec.get("additional_resources", [])
        fingerprints = [row["resource_fingerprint"] for row in resource_rows]
        domain_stable = {
            domain: abs(runs[candidate][0][domain] - runs[candidate][1][domain]) /
            max(runs[candidate][0][domain], runs[candidate][1][domain]) <= NOISE
            for domain in ("device_event_ms", "end_to_end_ms")}
        stable = all(domain_stable.values())
        device_median = statistics.median(r["device_event_ms"] for r in runs[candidate])
        rows.append({
            "case": name, "shape": shape, "dtype": dtype,
            "candidate": candidate, "runs": runs[candidate], "stable": stable,
            "device_stable": domain_stable["device_event_ms"],
            "end_to_end_stable": domain_stable["end_to_end_ms"],
            "device_winner_consensus": winners["device_event_ms"] == [candidate] * 2,
            "end_to_end_winner_consensus": winners["end_to_end_ms"] == [candidate] * 2,
            "launches_per_call": spec["launches"], "traffic_bytes": traffic_bytes,
            "achieved_bandwidth_gbps": traffic_bytes / (device_median * 1e6),
            "resources": resource_rows, "resource_fingerprints": fingerprints,
            "resource_evidence_complete": bool(resource_rows),
            "sampling": {
                "device_batches": device_batches,
                "device_reps_per_batch": device_reps,
                "end_to_end_batches": e2e_batches,
                "end_to_end_reps_per_batch": e2e_reps,
                "candidate_order": "rotated_interleaved",
                "run_cohorts": "disjoint_interleaved_samples",
                "clock_conditioning": "resident_tf32_gemm",
            },
        })
    return rows


def record(*, e2e_reps=40, e2e_batches=20, device_reps=1000,
           device_batches=10, warmup=1000):
    from tessera.compiler.emit import nvidia_cuda as nv
    rng = np.random.default_rng(20260723)
    gate_resource = _gated_epilogue_resource("silu")
    rows = []
    for name, (M, N, K) in (("f32_square", (512, 512, 512)),
                            ("f32_ragged", (509, 773, 257))):
        a = (rng.standard_normal((M, K)) * .2).astype(np.float32)
        b = (rng.standard_normal((K, N)) * .2).astype(np.float32)
        reference = a @ b; weights = b[None, :, :]; groups = np.array([M], np.int64)
        candidates = {
            "compiled_exact_f32": {
                "run": lambda a=a, weights=weights, groups=groups:
                    nv.run_grouped_gemm_f32(a, weights, groups),
                "device": lambda reps, warmup, a=a, weights=weights, groups=groups:
                    nv.measure_grouped_gemm_device(a, weights, groups, reps=reps),
                "launches": 1, "resource_route": "generated_grouped"},
            "shipped_tf32": {
                "run": lambda a=a, b=b: _tf32_run(a, b),
                "device": lambda reps, warmup, a=a, b=b:
                    _tf32_device(a, b, reps, warmup),
                "launches": 1, "resource_route": "nvidia_mma_fused_composed_tf32"},
        }
        rows.extend(_candidate_case(name, f"{M}x{N}x{K}", "f32/tf32",
            reference, candidates, e2e_reps=e2e_reps,
            e2e_batches=e2e_batches, device_reps=device_reps,
            device_batches=device_batches, warmup=warmup,
            traffic_bytes=a.nbytes + b.nbytes + reference.nbytes))

    T, K, N, E = 1024, 384, 256, 5
    groups = np.array([203, 0, 291, 244, 286], np.int64)
    x = (rng.standard_normal((T, K)) * .2).astype(np.float32)
    weights = (rng.standard_normal((E, K, N)) * .1).astype(np.float32)
    reference = _legacy_grouped(x, weights, groups)
    rows.extend(_candidate_case("grouped_gemm", f"{T}x{K}x{N}x{E}", "f32",
        reference, {
            "single_grouped_launch": {"run": lambda: nv.run_grouped_gemm_f32(x, weights, groups),
                "device": lambda reps, warmup: nv.measure_grouped_gemm_device(x, weights, groups, reps=reps),
                "launches": 1, "resource_route": "generated_grouped"},
            "legacy_per_expert": {"run": lambda: _legacy_grouped(x, weights, groups),
                "device": lambda reps, warmup: _legacy_grouped_device(x, weights, groups, reps),
                "launches": int(np.count_nonzero(groups)), "resource_route": "generated_grouped"}},
        e2e_reps=e2e_reps, e2e_batches=e2e_batches,
        device_reps=device_reps, device_batches=device_batches, warmup=warmup,
        traffic_bytes=x.nbytes + weights.nbytes + reference.nbytes))

    T, H, F, E = 512, 256, 384, 8; groups = np.full(E, T // E, np.int64)
    x = (rng.standard_normal((T, H)) * .15).astype(np.float32)
    wg = (rng.standard_normal((E, H, F)) * .1).astype(np.float32)
    wu = (rng.standard_normal((E, H, F)) * .1).astype(np.float32)
    wd = (rng.standard_normal((E, F, H)) * .1).astype(np.float32)
    reference = nv.run_grouped_swiglu_legacy_f32(x, wg, wu, wd, groups)
    rows.extend(_candidate_case("grouped_swiglu", f"{T}x{H}x{F}x{E}", "f32",
        reference, {
            "collapsed_grouped": {"run": lambda: nv.run_grouped_swiglu_f32(x, wg, wu, wd, groups),
                "device": lambda reps, warmup: nv.measure_grouped_swiglu_device(x, wg, wu, wd, groups, reps=reps),
                "launches": 4, "resource_route": "generated_grouped",
                "additional_resources": [gate_resource]},
            "legacy_per_expert": {"run": lambda: nv.run_grouped_swiglu_legacy_f32(x, wg, wu, wd, groups),
                "device": lambda reps, warmup: nv.measure_grouped_swiglu_device(x, wg, wu, wd, groups, legacy=True, reps=reps),
                "launches": 4 * E, "resource_route": "generated_grouped",
                "additional_resources": [gate_resource]}},
        e2e_reps=e2e_reps, e2e_batches=e2e_batches,
        device_reps=device_reps, device_batches=device_batches, warmup=warmup,
        traffic_bytes=x.nbytes + wg.nbytes + wu.nbytes + wd.nbytes + reference.nbytes))

    # Transport is already exact-device timed; retain its production rows and
    # explicit dependency on PARITY-TRANSPORT before any legacy selector change.
    reduction = json.loads((ROOT / "benchmarks/baselines/nvidia_sm120_reduction_transport.json").read_text())
    transport = [row for row in reduction["rows"] if row["op"] in
                 {"moe_dispatch", "moe_combine"}]
    serving = json.loads(SERVING.read_text())["runs"]
    device = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version",
        "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()
    return {"schema": "tessera.nvidia.legacy-retune.v1", "device": device,
            "noise_policy": NOISE, "rows": rows,
            "retained_transport_rows": transport,
            "retained_kv_rows": [row for row in serving if "paged" in row["op"]],
            "transport_dependency": "NVIDIA-PARITY-TRANSPORT"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--e2e-reps", type=int, default=40)
    p.add_argument("--e2e-batches", type=int, default=20)
    p.add_argument("--device-reps", type=int, default=1000)
    p.add_argument("--device-batches", type=int, default=10)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--output", type=Path, default=OUT)
    a = p.parse_args(argv); result = record(
        e2e_reps=a.e2e_reps, e2e_batches=a.e2e_batches,
        device_reps=a.device_reps, device_batches=a.device_batches,
        warmup=a.warmup)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {a.output} ({len(result['rows'])} candidate rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
