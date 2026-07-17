"""Record consolidated sm_120 paged-KV and MoE transport parity evidence.

Every row is oracle-checked and retains two rotated repeated-median runs,
CUDA-event and allocation/transfer-inclusive timing, auditable byte counts,
launch amortization, achieved bandwidth, and the exact TEST-5 resource rows.
This recorder is evidence-only and never changes a selector.
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
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_transport_parity.json"
RESOURCES = ROOT / "benchmarks/baselines/nvidia_sm120_test5_route_resources.json"
NOISE = .03


def _resource_rows(route: str) -> list[dict[str, Any]]:
    return json.loads(RESOURCES.read_text())["details"][route]


def _wall_ms(call) -> float:
    start = time.perf_counter_ns()
    call()
    return (time.perf_counter_ns() - start) / 1e6


def _moe_device_samples(measure, *args, batches: int = 202) -> list[float]:
    samples: list[float] = []
    measure(*args, reps=10000, batches=batches, batch_medians=samples)
    return samples


def _record_case(case: dict[str, Any], *, reps: int, warmup: int) -> dict[str, Any]:
    from benchmarks.nvidia._clock_conditioning import condition_sm120

    actual = case["run"]()
    np.testing.assert_allclose(actual, case["reference"], rtol=2e-5, atol=2e-5)
    max_abs_error = float(np.max(np.abs(actual - case["reference"])))
    device_condition_ms = condition_sm120()
    device_samples = [float(sample) for sample in case["device_samples"]()]
    if len(device_samples) < 42 or len(device_samples) % 2:
        raise RuntimeError("transport device sampler requires an even 42+ batches")
    device_by_run = {
        1: device_samples[0::2],
        2: device_samples[1::2],
    }
    e2e_condition_ms = condition_sm120()
    for _ in range(warmup):
        case["run"]()
    e2e_by_run = {1: [], 2: []}
    for batch in range(202):
        run_id = 1 + batch % 2
        e2e_by_run[run_id].append(statistics.median(
            float(_wall_ms(case["run"])) for _ in range(reps)))
    runs = []
    for run_id in (1, 2):
        device = device_by_run[run_id]
        e2e_batches = e2e_by_run[run_id]
        runs.append({
            "run": run_id,
            "device_event_ms": statistics.median(device),
            "end_to_end_ms": statistics.median(e2e_batches),
            "max_abs_error": max_abs_error,
            "clock_condition_device_ms": device_condition_ms,
            "clock_condition_end_to_end_ms": e2e_condition_ms,
            "device_batch_medians_ms": device,
            "end_to_end_batch_medians_ms": e2e_batches,
            "sampling_order": "resident_odd_even_interleaved",
        })
    stable = {}
    for domain in ("device_event_ms", "end_to_end_ms"):
        a, b = (run[domain] for run in runs)
        stable[domain] = abs(a - b) / max(a, b) <= NOISE
    device_ms = statistics.median(run["device_event_ms"] for run in runs)
    resources = _resource_rows(case["resource_route"])
    return {
        "op": case["op"], "shape": case["shape"], "dtype": "f32",
        "candidate": case["candidate"], "runs": runs,
        "device_stable": stable["device_event_ms"],
        "end_to_end_stable": stable["end_to_end_ms"],
        "stable": all(stable.values()),
        "traffic_bytes": case["traffic_bytes"],
        "traffic_formula": case["traffic_formula"],
        "achieved_bandwidth_gbps": case["traffic_bytes"] / (device_ms * 1e6),
        "launches_per_call": case["launches"],
        "launch_amortization_key": case["amortization"],
        "resource_fingerprints": [r["resource_fingerprint"] for r in resources],
        "resources": resources, "resource_evidence_complete": bool(resources),
        **case.get("metadata", {}),
    }


def _paged_cases() -> list[dict[str, Any]]:
    from tessera.cache.paged_kv import _reference_attention
    from tessera.compiler.emit import nvidia_cuda as nv

    cases = []
    for tokens in (127, 128, 129, 511):
        rng = np.random.default_rng(6000 + tokens)
        page_size, heads, dim, q_len = 16, 8, 64, 3
        pages = (tokens + page_size - 1) // page_size
        logical_k = (rng.standard_normal(
            (pages, page_size, heads, dim)) * .1).astype(np.float32)
        logical_v = (rng.standard_normal(logical_k.shape) * .1).astype(np.float32)
        table = np.roll(np.arange(pages, dtype=np.int32), 1)
        k, v = np.empty_like(logical_k), np.empty_like(logical_v)
        for logical, physical in enumerate(table):
            k[physical], v[physical] = logical_k[logical], logical_v[logical]
        indices = np.arange(tokens, dtype=np.int64)
        q = (rng.standard_normal((heads, q_len, dim)) * .1).astype(np.float32)
        reference = _reference_attention(
            q, np.transpose(logical_k.reshape(-1, heads, dim)[:tokens], (1, 0, 2)),
            np.transpose(logical_v.reshape(-1, heads, dim)[:tokens], (1, 0, 2)),
            dim ** -.5, True)
        traffic = q.nbytes + indices.nbytes + table.nbytes + (
            2 * tokens * heads * dim * 4) + reference.nbytes
        for route in ("fused", "staged"):
            def run(route=route, q=q, k=k, v=v, table=table, indices=indices):
                return nv.run_paged_attention_resident_f32(
                    q, k, v, table, indices, scale=dim ** -.5,
                    causal=True, route=route, measure_device=False)[0]

            def device_samples(route=route, q=q, k=k, v=v, table=table,
                               indices=indices):
                samples = []
                nv.run_paged_attention_resident_f32(
                    q, k, v, table, indices, scale=dim ** -.5,
                    causal=True, route=route, device_reps=1000,
                    device_warmup=100, device_batches=42,
                    device_batch_medians=samples)
                return samples

            cases.append({
                "op": "paged_kv", "shape": f"{heads}x{q_len}x{tokens}x{dim}",
                "candidate": route, "run": run,
                "device_samples": device_samples,
                "reference": reference, "traffic_bytes": traffic,
                "traffic_formula": "Q + logical K/V + page table + indices + O",
                "launches": 1 if route == "fused" else 2,
                "amortization": f"tokens_per_launch:{tokens}",
                "resource_route": f"{route}_paged_attention",
                "metadata": {"page_mapping": "permuted", "page_size": page_size,
                             "causal_offset": tokens - q_len,
                             "boundary_relation": "exact" if tokens % page_size == 0 else "ragged"},
            })
    return cases


def _moe_cases() -> list[dict[str, Any]]:
    from tessera.compiler.emit import nvidia_cuda as nv

    cases = []
    for tokens, slots, hidden in ((17, 23, 31), (257, 389, 193)):
        rng = np.random.default_rng(tokens + slots + hidden)
        x = (rng.standard_normal((tokens, hidden)) * .2).astype(np.float32)
        token_ids = rng.integers(0, tokens, size=slots, dtype=np.int32)
        packed = x[token_ids]
        weights = rng.random(slots, dtype=np.float32)
        combined = np.zeros_like(x)
        np.add.at(combined, token_ids, packed * weights[:, None])
        dispatch_bytes = packed.nbytes * 2 + token_ids.nbytes
        combine_bytes = packed.nbytes + weights.nbytes + token_ids.nbytes + 2 * combined.nbytes
        cases.extend(({
            "op": "moe_dispatch", "shape": f"{tokens}x{slots}x{hidden}",
            "candidate": "generated_gather",
            "run": lambda x=x, token_ids=token_ids: nv.run_moe_dispatch_f32(x, token_ids),
            "device_samples": lambda x=x, token_ids=token_ids:
                _moe_device_samples(nv.measure_moe_dispatch_device, x, token_ids),
            "reference": packed, "traffic_bytes": dispatch_bytes,
            "traffic_formula": "slot index + gathered input read + packed output write",
            "launches": 1, "amortization": f"slots_per_launch:{slots}",
            "resource_route": "generated_gather",
        }, {
            "op": "moe_combine", "shape": f"{tokens}x{slots}x{hidden}",
            "candidate": "generated_combine",
            "run": lambda packed=packed, token_ids=token_ids, weights=weights, tokens=tokens:
                nv.run_moe_combine_f32(packed, token_ids, weights, tokens),
            "device_samples": lambda packed=packed, token_ids=token_ids,
                weights=weights, tokens=tokens: _moe_device_samples(
                    nv.measure_moe_combine_device, packed, token_ids,
                    weights, tokens),
            "reference": combined, "traffic_bytes": combine_bytes,
            "traffic_formula": "partials + weights + indices + atomic output read/write",
            "launches": 1, "amortization": f"slots_per_launch:{slots}",
            "resource_route": "generated_combine",
        }))
    rng = np.random.default_rng(7711)
    tokens, kdim, ndim, experts = 257, 193, 127, 5
    groups = np.array([51, 0, 73, 61, 72], np.int64)
    x = (rng.standard_normal((tokens, kdim)) * .2).astype(np.float32)
    weights = (rng.standard_normal((experts, kdim, ndim)) * .1).astype(np.float32)
    reference = np.concatenate([
        x[sum(groups[:e]):sum(groups[:e + 1])] @ weights[e]
        for e in range(experts) if groups[e]])
    traffic = x.nbytes + weights.nbytes + groups.nbytes + reference.nbytes
    cases.append({
        "op": "grouped_gemm", "shape": f"{tokens}x{kdim}x{ndim}x{experts}",
        "candidate": "generated_grouped",
        "run": lambda: nv.run_grouped_gemm_f32(x, weights, groups),
        "device_samples": lambda: _moe_device_samples(
            nv.measure_grouped_gemm_device, x, weights, groups),
        "reference": reference, "traffic_bytes": traffic,
        "traffic_formula": "packed X + expert weights + group offsets + output",
        "launches": 1, "amortization": f"nonempty_groups_per_launch:{np.count_nonzero(groups)}",
        "resource_route": "generated_grouped",
        "metadata": {"group_sizes": groups.tolist(), "ragged_groups": True},
    })
    return cases


def record(*, reps: int = 20, warmup: int = 3) -> dict[str, Any]:
    from tessera import runtime as rt
    if rt._nvidia_device_name() != "sm_120":
        raise RuntimeError("NVIDIA transport parity requires exact sm_120")
    version = subprocess.run(["/usr/local/cuda/bin/nvcc", "--version"],
                             check=True, capture_output=True, text=True).stdout
    rows = [_record_case(case, reps=reps, warmup=warmup)
            for case in (*_paged_cases(), *_moe_cases())]
    return {
        "schema": "tessera.nvidia.transport-parity.v1",
        "device": "nvidia:sm_120", "noise_policy": NOISE,
        "compiler_fingerprint": "sha256:" + hashlib.sha256(version.encode()).hexdigest(),
        "selector_changed": False, "rows": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    result = record(reps=args.reps, warmup=args.warmup)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output} ({len(result['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
