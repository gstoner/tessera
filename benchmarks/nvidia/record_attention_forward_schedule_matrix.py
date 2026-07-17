"""Record CUDA-native D=128 forward-attention schedule evidence on sm_120.

The candidates are architecture-owned 4-/8-warp CTAs. Each warp owns one
query and maintains distributed online-softmax state. The recorder retains two
independent repeated-median runs, CUDA-event and end-to-end timing, numerical
error, traffic, and cubin resource fingerprints before any selector change.
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
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_attention_forward_schedules.json"
NOISE = 0.03


def _reference(q, k, v, *, scale, causal=False, window_left=None,
               window_right=None, bias=None, softcap=None):
    B, Hq, Sq, _ = q.shape; Hkv, Sk = k.shape[1:3]
    out = np.empty((B, Hq, Sq, v.shape[-1]), np.float32)
    for b in range(B):
        for hq in range(Hq):
            hk = hq // (Hq // Hkv)
            for m in range(Sq):
                score = q[b, hq, m] @ k[b, hk].T * scale
                if bias is not None:
                    score += bias[b, hq, m]
                if softcap is not None:
                    score = softcap * np.tanh(score / softcap)
                keep = np.ones(Sk, bool)
                if causal:
                    keep &= np.arange(Sk) <= m
                if window_left is not None:
                    keep &= np.arange(Sk) >= m - window_left
                if window_right is not None:
                    keep &= np.arange(Sk) <= m + window_right
                score = np.where(keep, score, -np.inf)
                p = np.exp(score - np.max(score)); p /= p.sum()
                out[b, hq, m] = p @ v[b, hk]
    return out


def _cases():
    return (
        ("mha_512", (1, 8, 512, 512, 128), {}),
        ("causal_ragged_1009", (1, 8, 1009, 1009, 128), {"causal": True}),
        ("gqa_window_ragged", (1, 8, 2, 257, 509, 128),
         {"window_left": 127, "window_right": 3}),
        ("mqa_bias_softcap", (1, 8, 1, 129, 257, 128),
         {"bias": True, "softcap": 1.7}),
    )


def _inputs(name, shape, options):
    rng = np.random.default_rng(20260731 + sum(map(ord, name)))
    if len(shape) == 5:
        B, Hq, Sq, Sk, D = shape; Hkv = Hq
    else:
        B, Hq, Hkv, Sq, Sk, D = shape
    q = rng.standard_normal((B, Hq, Sq, D), dtype=np.float32) * 0.2
    k = rng.standard_normal((B, Hkv, Sk, D), dtype=np.float32) * 0.2
    v = rng.standard_normal((B, Hkv, Sk, D), dtype=np.float32) * 0.2
    kwargs = {key: value for key, value in options.items() if key != "bias"}
    if options.get("bias"):
        kwargs["bias"] = rng.standard_normal((B, Hq, Sq, Sk),
                                               dtype=np.float32) * 0.05
    kwargs["scale"] = D ** -0.5
    return q, k, v, kwargs


def _median_ms(call, reps):
    samples = []
    for _ in range(reps):
        start = time.perf_counter(); call()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def _resources(warps):
    from benchmarks.nvidia.record_tile_fragment_resources import (
        _CudaOccupancy, _artifact_row,
    )
    from tessera.compiler.emit.nvidia_cuda import (
        _synthesize_flash_fwd_multiwarp_cuda,
    )
    entry = f"tessera_nvidia_flash_attn_fwd_w{warps}_kernel"
    with tempfile.TemporaryDirectory(prefix=f"tessera-attn-w{warps}-") as tmp:
        source = Path(tmp) / "attention.cu"; cubin = Path(tmp) / "attention.cubin"
        source.write_text(_synthesize_flash_fwd_multiwarp_cuda(warps))
        subprocess.run(["/usr/local/cuda/bin/nvcc", "-arch=sm_120", "-O3",
                        "-cubin", str(source), "-o", str(cubin)], check=True)
        occupancy = _CudaOccupancy()
        try:
            return _artifact_row(cubin, entry, warps * 32, occupancy,
                                 schedule=f"warp_per_query_w{warps}")
        finally:
            occupancy.close()


def record(*, e2e_reps=40, e2e_batches=10, device_reps=100,
           device_batches=5, warmup=20):
    from benchmarks.nvidia._clock_conditioning import condition_sm120
    from tessera.compiler.emit.nvidia_cuda import (
        measure_flash_attention_forward_schedule_device,
        run_flash_attention_forward_schedule,
    )
    resources = {warps: _resources(warps) for warps in (4, 8)}
    rows = []
    for name, shape, options in _cases():
        q, k, v, kwargs = _inputs(name, shape, options)
        reference = _reference(q, k, v, **kwargs)
        candidate_runs = {4: [], 8: []}
        errors = {}
        for warps in (4, 8):
            actual = run_flash_attention_forward_schedule(
                q, k, v, warps_per_cta=warps, **kwargs)
            errors[warps] = float(np.max(np.abs(actual - reference)))
        clock_condition_ms = condition_sm120(reps=100)
        device_samples = {run_id: {4: [], 8: []} for run_id in (1, 2)}
        for epoch in range(device_batches * 2):
            run_id = epoch % 2 + 1
            order = (4, 8) if epoch % 2 == 0 else (8, 4)
            for warps in order:
                device_samples[run_id][warps].append(
                    measure_flash_attention_forward_schedule_device(
                        q, k, v, warps_per_cta=warps, warmup=warmup,
                        reps=device_reps, **kwargs))
        # End-to-end includes CUDA allocation/copy. Prime that lifecycle once,
        # then assign alternating batches to two disjoint run cohorts. This
        # preserves independent evidence while cancelling monotonic clock and
        # allocator drift that otherwise aliases with run order.
        for _ in range(warmup):
            for warps in (4, 8):
                run_flash_attention_forward_schedule(
                    q, k, v, warps_per_cta=warps, **kwargs)
        e2e_condition_ms = condition_sm120(reps=100)
        e2e_batches_by_run = {
            run_id: {4: [], 8: []} for run_id in (1, 2)}
        for batch in range(e2e_batches):
            samples = {
                run_id: {4: [], 8: []} for run_id in (1, 2)}
            for sample in range(e2e_reps * 2):
                run_id = sample % 2 + 1
                order = ((4, 8) if (sample // 2 + batch) % 2 == 0
                         else (8, 4))
                for warps in order:
                    start = time.perf_counter()
                    run_flash_attention_forward_schedule(
                        q, k, v, warps_per_cta=warps, **kwargs)
                    samples[run_id][warps].append(
                        (time.perf_counter() - start) * 1e3)
            for run_id in (1, 2):
                for warps in (4, 8):
                    e2e_batches_by_run[run_id][warps].append(
                        statistics.median(samples[run_id][warps]))
        for run_id in (1, 2):
            for warps in (4, 8):
                batch_medians = e2e_batches_by_run[run_id][warps]
                candidate_runs[warps].append({
                    "run": run_id,
                    "device_event_ms": statistics.median(
                        device_samples[run_id][warps]),
                    "end_to_end_ms": statistics.median(batch_medians),
                    "end_to_end_batch_medians_ms": batch_medians,
                    "clock_condition_ms": clock_condition_ms,
                    "end_to_end_clock_condition_ms": e2e_condition_ms,
                    "max_abs_error": errors[warps],
                })
        winners = {}
        for domain in ("device_event_ms", "end_to_end_ms"):
            winners[domain] = [min((4, 8), key=lambda w: candidate_runs[w][i][domain])
                               for i in range(2)]
        for warps in (4, 8):
            runs = candidate_runs[warps]
            domain_stable = {
                key: abs(runs[0][key] - runs[1][key]) /
                max(runs[0][key], runs[1][key]) <= NOISE
                for key in ("device_event_ms", "end_to_end_ms")}
            stable = all(domain_stable.values())
            rows.append({
                "case": name, "shape": list(q.shape[:-1]) + [k.shape[2], q.shape[-1]],
                "dtype": "f32", "candidate": f"warp_per_query_w{warps}",
                "warps_per_cta": warps, "runs": runs, "stable": stable,
                "device_stable": domain_stable["device_event_ms"],
                "end_to_end_stable": domain_stable["end_to_end_ms"],
                "device_winner_consensus": winners["device_event_ms"] == [warps, warps],
                "end_to_end_winner_consensus": winners["end_to_end_ms"] == [warps, warps],
                "traffic_bytes": int(q.nbytes + k.nbytes + v.nbytes + reference.nbytes +
                                     (0 if kwargs.get("bias") is None else kwargs["bias"].nbytes)),
                "resource": resources[warps],
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
    device = subprocess.run([
        "nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version",
        "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()
    return {"schema": "tessera.nvidia.attention-forward-schedules.v1",
            "device": device, "noise_policy": NOISE, "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e-reps", type=int, default=40)
    parser.add_argument("--e2e-batches", type=int, default=10)
    parser.add_argument("--device-reps", type=int, default=100)
    parser.add_argument("--device-batches", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    result = record(e2e_reps=args.e2e_reps,
                    e2e_batches=args.e2e_batches,
                    device_reps=args.device_reps,
                    device_batches=args.device_batches,
                    warmup=args.warmup)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output} ({len(result['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
