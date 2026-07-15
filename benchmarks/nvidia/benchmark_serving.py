"""Live sm_120 serving benchmarks for ReplaySSM and paged-KV decode.

The output separates host wall time from CUDA-event time and carries the
analytical ReplaySSM state-traffic model used by the architecture review.
Nothing is emitted when a live NVIDIA sm_120 device is unavailable.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def parse_shape(spec: str) -> tuple[int, int, int]:
    dims = tuple(int(v) for v in spec.lower().split("x"))
    if len(dims) != 3 or any(v <= 0 for v in dims):
        raise ValueError(f"expected positive BxDxN shape, got {spec!r}")
    return dims  # type: ignore[return-value]


def summary_state_bytes_per_token(d: int, n: int, elem: int = 4) -> float:
    return float(2 * elem * d * n)


def replay_state_bytes_per_token(d: int, n: int, capacity: int,
                                 elem: int = 4) -> float:
    return float(elem * d * n) / capacity + elem * (2 * d + n)


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _replay_row(shape: str, tokens: int, chunk: int, slots: int,
                reps: int) -> dict[str, Any]:
    from tessera import runtime as rt

    bsz, d, n = parse_shape(shape)
    if tokens % chunk:
        raise ValueError("ReplaySSM tokens must be divisible by chunk")
    chunks = tokens // chunk
    if chunks > slots:
        raise ValueError("one benchmark wave must fit in the async slot ring")
    rng = np.random.default_rng(1200 + bsz + d + n)
    wall_samples: list[float] = []
    device_samples: list[float] = []
    for _ in range(reps):
        a = -np.abs(rng.standard_normal(d)).astype(np.float32)
        delta = (np.abs(rng.standard_normal((tokens, bsz, d))) * .1).astype(np.float32)
        x = (rng.standard_normal((tokens, bsz, d)) * .1).astype(np.float32)
        bb = (rng.standard_normal((tokens, bsz, n)) * .1).astype(np.float32)
        cc = (rng.standard_normal((tokens, bsz, n)) * .1).astype(np.float32)
        handle = rt.nvidia_ssm_replay_state_handle(
            bsz, d, n, a, capacity=tokens + 1, async_slots=slots)
        if handle._device is None:
            raise RuntimeError("live CUDA ReplaySSM device state unavailable")
        t0 = time.perf_counter_ns()
        futures = [handle.submit_block_async(
            delta[i:i + chunk], x[i:i + chunk], bb[i:i + chunk], cc[i:i + chunk])
            for i in range(0, tokens, chunk)]
        # Read device timing before wait() retires each leased slot.
        device_samples.append(sum(f.event.elapsed_ms() for f in futures))
        for future in futures:
            future.wait()
        wall_samples.append((time.perf_counter_ns() - t0) / 1e6)
        handle._device.close()
    replay_bytes = replay_state_bytes_per_token(d, n, tokens + 1)
    summary_bytes = summary_state_bytes_per_token(d, n)
    return {
        "backend": "nvidia", "device": "sm_120", "op": "ssm_replay_decode",
        "shape": shape, "dtype": "f32", "mode": "async_ring",
        "tokens": tokens, "chunk": chunk, "async_slots": slots, "reps": reps,
        "latency_ms": _median(wall_samples) / tokens,
        "device_latency_ms": _median(device_samples) / tokens,
        "state_bytes_per_token": replay_bytes,
        "summary_state_bytes_per_token": summary_bytes,
        "state_traffic_ratio": summary_bytes / replay_bytes,
    }


def _paged_kv_row(tokens: int, heads: int, dim: int, page_size: int,
                  reps: int, route: str) -> dict[str, Any]:
    from tessera.compiler.emit.nvidia_cuda import run_paged_attention_resident_f32

    rng = np.random.default_rng(5070 + tokens)
    pages = (tokens + page_size - 1) // page_size
    q = (rng.standard_normal((heads, 1, dim)) * .1).astype(np.float32)
    k = (rng.standard_normal((pages, page_size, heads, dim)) * .1).astype(np.float32)
    v = (rng.standard_normal(k.shape) * .1).astype(np.float32)
    table = np.arange(pages, dtype=np.int32)
    indices = np.arange(tokens, dtype=np.int64)
    wall: list[float] = []
    device: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        _, latency = run_paged_attention_resident_f32(
            q, k, v, table, indices, scale=dim ** -.5, causal=True,
            route=route)
        wall.append((time.perf_counter_ns() - t0) / 1e6)
        device.append(latency)
    return {
        "backend": "nvidia", "device": "sm_120", "op": "paged_kv_decode",
        "shape": f"1x{heads}x{tokens}x{dim}", "dtype": "f32",
        "mode": f"{route}_paged_attention", "tokens": tokens,
        "page_size": page_size, "reps": reps, "latency_ms": _median(wall),
        "device_latency_ms": _median(device),
    }


def run_benchmark(shapes: list[str], *, tokens: int, chunk: int, slots: int,
                  kv_tokens: list[int], heads: int, dim: int, page_size: int,
                  reps: int) -> list[dict[str, Any]]:
    from tessera import runtime as rt
    if rt._nvidia_device_name() != "sm_120":
        return []
    rows = [_replay_row(s, tokens, chunk, slots, reps) for s in shapes]
    rows.extend(_paged_kv_row(t, heads, dim, page_size, reps, route)
                for t in kv_tokens for route in ("fused", "staged"))
    return rows


def update_d2_corpus(rows: list[dict[str, Any]]) -> Path:
    """Persist serving route verdicts in the shared device-timed D2 corpus."""
    from tessera.compiler.emit import autotune as at
    from tessera.compiler.emit.kernel_emitter import SpecPolicy, bucket_key

    cache = at.MeasureCache()
    at.load_corpus(cache=cache)
    groups: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        key = (str(row["op"]), str(row["shape"]), str(row["dtype"]))
        groups.setdefault(key, {})[str(row["mode"])] = float(
            row["device_latency_ms"])
    for (op, shape, dtype), candidates in groups.items():
        dims = tuple(int(v) for v in shape.split("x"))
        winner = min(candidates, key=candidates.__getitem__)
        cache.put(("nvidia:sm_120", "nvidia", op,
                   bucket_key(dims, SpecPolicy.BUCKET), dtype, "device"),
                  at.MeasureRecord(winner, candidates[winner], candidates))
    return at.save_corpus(cache=cache)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapes", nargs="+", default=["1x128x64", "1x256x128"])
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--slots", type=int, default=4)
    p.add_argument("--kv-tokens", nargs="+", type=int, default=[128, 512, 2048])
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--update-corpus", action="store_true")
    args = p.parse_args(argv)
    rows = run_benchmark(args.shapes, tokens=args.tokens, chunk=args.chunk,
                         slots=args.slots, kv_tokens=args.kv_tokens,
                         heads=args.heads, dim=args.dim,
                         page_size=args.page_size, reps=args.reps)
    output = json.dumps({"runs": rows}, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")
    if args.update_corpus:
        update_d2_corpus(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
