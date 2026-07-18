"""Paired APPLE-PAGED-KV-1 staged-vs-direct resident attention corpus.

Both candidates consume the same non-identity physical page layout. ``direct``
follows the resident table inside one MSL dispatch. ``staged`` performs two
on-GPU page gathers followed by dense resident attention. The report keeps GPU
execution time separate from end-to-end latency and records two independent
runs by default so selector evidence cannot be inferred from one noisy sweep.
"""
from __future__ import annotations

import argparse
import json
import statistics
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from tessera._apple_gpu_dispatch import (
    clear_dispatch_telemetry, set_dispatch_telemetry_enabled)
from tessera.cache import ResidentBlockPagedKVCache
from tessera.compiler.apple_route_selector import live_apple_device_tag


def _shape(text: str) -> tuple[int, int, int, int]:
    values = tuple(int(v) for v in text.lower().split("x"))
    if len(values) != 4 or min(values) <= 0:
        raise ValueError(f"shape must be TOKENSxLATENTxROPExQ, got {text!r}")
    return values  # type: ignore[return-value]


def _median(values: list[int]) -> int:
    return int(statistics.median(values))


def _make_case(spec: str, block_size: int, seed: int):
    tokens, latent_dim, rope_dim, q_len = _shape(spec)
    blocks_per_seq = (tokens + block_size - 1) // block_size
    cache = ResidentBlockPagedKVCache(
        latent_dim=latent_dim, rope_dim=rope_dim,
        num_blocks=blocks_per_seq * 2, block_size=block_size)
    rng = np.random.default_rng(seed)
    cache.add_sequence("measured"); cache.add_sequence("interleave")
    remaining = tokens
    while remaining:
        width = min(block_size, remaining)
        cache.append(
            "measured", rng.standard_normal((width, latent_dim)).astype(np.float32),
            rng.standard_normal((width, rope_dim)).astype(np.float32))
        cache.append(
            "interleave", rng.standard_normal((width, latent_dim)).astype(np.float32),
            rng.standard_normal((width, rope_dim)).astype(np.float32))
        remaining -= width
    query = rng.standard_normal((q_len, rope_dim)).astype(np.float32)
    resident = cache._resident
    cache._resident = False
    reference = cache.attention("measured", query, causal=True, window=tokens)
    cache._resident = resident
    return cache, query, np.asarray(reference).copy()


def run_benchmark(
    shapes: list[str], *, block_size: int, warmup: int, reps: int, runs: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    device_tag = live_apple_device_tag()
    set_dispatch_telemetry_enabled(True)
    try:
        for run in range(runs):
            for case_index, spec in enumerate(shapes):
                cache, query, reference = _make_case(
                    spec, block_size, seed=4100 + run * 101 + case_index)
                try:
                    for route in ("staged", "direct"):
                        for _ in range(warmup):
                            out = cache.attention(
                                "measured", query, causal=True, route=route)
                            if hasattr(out, "free"):
                                out.free()
                        wall_times: list[int] = []
                        device_times: list[int] = []
                        valid = True
                        native = True
                        telemetry: dict[str, Any] = {}
                        for _ in range(reps):
                            clear_dispatch_telemetry()
                            start = time.perf_counter_ns()
                            out = cache.attention(
                                "measured", query, causal=True, route=route)
                            got = (out.copy_to_host() if hasattr(out, "copy_to_host")
                                   else np.asarray(out).copy())
                            wall_times.append(time.perf_counter_ns() - start)
                            if hasattr(out, "free"):
                                out.free()
                            valid = valid and bool(np.allclose(
                                got, reference, rtol=2e-4, atol=2e-4))
                            native = native and cache.last_attention_execution == "native_gpu"
                            telemetry = cache.last_attention_telemetry or {}
                            dt = telemetry.get("device_time_ns")
                            if isinstance(dt, int):
                                device_times.append(dt)
                        rows.append({
                            "run": run + 1,
                            "device": device_tag,
                            "shape": spec,
                            "route": route,
                            "timing_domain_end_to_end_ns": _median(wall_times),
                            "timing_domain_device_ns": (
                                _median(device_times)
                                if len(device_times) == reps else None),
                            "device_time_coverage": len(device_times) / reps,
                            "native_proof": native,
                            "correctness": valid,
                            "page_table": cache.block_table("measured"),
                            "non_identity_page_table": cache.block_table("measured")
                            != list(range(len(cache.block_table("measured")))),
                            "subdispatches": telemetry.get("subdispatches"),
                            "device_time_scope": telemetry.get("device_time_scope"),
                            "resources": telemetry.get("resources"),
                            "lifecycle": cache.lifecycle_telemetry(),
                        })
                finally:
                    cache.free()
    finally:
        set_dispatch_telemetry_enabled(False)

    decisions: list[dict[str, Any]] = []
    for spec in shapes:
        for domain, field in (
            ("end_to_end", "timing_domain_end_to_end_ns"),
            ("device", "timing_domain_device_ns"),
        ):
            winners = []
            for run in range(1, runs + 1):
                candidates = [r for r in rows if r["shape"] == spec and r["run"] == run
                              and r["native_proof"] and r["correctness"]
                              and r[field] is not None]
                winners.append(min(candidates, key=lambda r: r[field])["route"]
                               if len(candidates) == 2 else None)
            decisions.append({
                "shape": spec, "timing_domain": domain,
                "run_winners": winners,
                "stable_winner": winners[0] if winners and len(set(winners)) == 1
                and winners[0] is not None else None,
            })
    return {
        "schema": "tessera.apple.resident_paged_kv.v1",
        "device": device_tag, "os": platform.platform(),
        "runs": runs, "warmup": warmup, "reps": reps,
        "block_size": block_size, "rows": rows, "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", default=[
        "127x64x32x1", "512x128x64x1", "1025x128x64x4"])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(
        args.shapes, block_size=args.block_size, warmup=args.warmup,
        reps=args.reps, runs=args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["decisions"], indent=2))


if __name__ == "__main__":
    main()
