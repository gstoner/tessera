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


def replay_wave_offsets(tokens: int, chunk: int, slots: int) -> tuple[tuple[int, ...], ...]:
    """Chunk offsets grouped into waves that fit the leased async-slot ring."""
    if tokens <= 0 or chunk <= 0 or slots <= 0 or tokens % chunk:
        raise ValueError("ReplaySSM requires positive divisible tokens/chunk and slots")
    width = chunk * slots
    return tuple(tuple(range(start, min(tokens, start + width), chunk))
                 for start in range(0, tokens, width))


def _replay_row(shape: str, tokens: int, chunk: int, slots: int,
                reps: int, *, condition_clock: bool = False,
                retain_samples: bool = False,
                condition_every: int | None = None) -> dict[str, Any]:
    from tessera import runtime as rt
    from tessera.cache import SSMStateHandle

    bsz, d, n = parse_shape(shape)
    if condition_clock:
        from benchmarks.nvidia._clock_conditioning import condition_sm120
    condition_samples: list[float] = []
    waves = replay_wave_offsets(tokens, chunk, slots)
    rng = np.random.default_rng(1200 + bsz + d + n)
    wall_samples: list[float] = []
    device_samples: list[float] = []
    max_abs_error = 0.0
    for rep in range(reps):
        if condition_clock and rep % (condition_every or reps) == 0:
            condition_samples.append(condition_sm120())
        a = -np.abs(rng.standard_normal(d)).astype(np.float32)
        delta = (np.abs(rng.standard_normal((tokens, bsz, d))) * .1).astype(np.float32)
        x = (rng.standard_normal((tokens, bsz, d)) * .1).astype(np.float32)
        bb = (rng.standard_normal((tokens, bsz, n)) * .1).astype(np.float32)
        cc = (rng.standard_normal((tokens, bsz, n)) * .1).astype(np.float32)
        handle = rt.nvidia_ssm_replay_state_handle(
            bsz, d, n, a, capacity=tokens + 1, async_slots=slots)
        reference = SSMStateHandle(
            bsz, d, n, a, capacity=tokens + 1)
        if handle._device is None:
            raise RuntimeError("live CUDA ReplaySSM device state unavailable")
        expected = {}
        for i in range(0, tokens, chunk):
            expected[i] = np.stack([
                reference.step(delta[t], x[t], bb[t], cc[t])
                for t in range(i, i + chunk)])
        t0 = time.perf_counter_ns()
        device_ms = 0.0
        for offsets in waves:
            futures = [handle.submit_block_async(
                delta[i:i + chunk], x[i:i + chunk], bb[i:i + chunk], cc[i:i + chunk])
                for i in offsets]
            # Read device timing before wait() retires each leased slot, then
            # retire the bounded wave so the ring can be reused/backpressured.
            device_ms += sum(f.event.elapsed_ms() for f in futures)
            for i, future in zip(offsets, futures):
                got = future.wait()
                np.testing.assert_allclose(
                    got, expected[i], rtol=2e-4, atol=2e-4)
                max_abs_error = max(
                    max_abs_error, float(np.max(np.abs(got - expected[i]))))
        device_samples.append(device_ms)
        wall_samples.append((time.perf_counter_ns() - t0) / 1e6)
        handle._device.close()
    replay_bytes = replay_state_bytes_per_token(d, n, tokens + 1)
    summary_bytes = summary_state_bytes_per_token(d, n)
    row = {
        "backend": "nvidia", "device": "sm_120", "op": "ssm_replay_decode",
        "shape": shape, "dtype": "f32", "mode": "async_ring",
        "tokens": tokens, "chunk": chunk, "async_slots": slots, "reps": reps,
        "ring_waves": len(waves),
        "max_abs_error": max_abs_error,
        **({"clock_condition_device_ms": _median(condition_samples),
            "clock_condition_batch_ms": condition_samples}
           if condition_samples else {}),
        "latency_ms": _median(wall_samples) / tokens,
        "device_latency_ms": _median(device_samples) / tokens,
        "state_bytes_per_token": replay_bytes,
        "summary_state_bytes_per_token": summary_bytes,
        "state_traffic_ratio": summary_bytes / replay_bytes,
    }
    if retain_samples:
        row["end_to_end_samples_ms_per_token"] = [
            sample / tokens for sample in wall_samples]
        row["device_samples_ms_per_token"] = [
            sample / tokens for sample in device_samples]
    return row


def _paged_kv_row(tokens: int, heads: int, dim: int, page_size: int,
                  reps: int, route: str) -> dict[str, Any]:
    from tessera.compiler.emit.nvidia_cuda import run_paged_attention_resident_f32

    rng = np.random.default_rng(5070 + tokens)
    pages = (tokens + page_size - 1) // page_size
    q = (rng.standard_normal((heads, 1, dim)) * .1).astype(np.float32)
    logical_k = (rng.standard_normal(
        (pages, page_size, heads, dim)) * .1).astype(np.float32)
    logical_v = (rng.standard_normal(logical_k.shape) * .1).astype(np.float32)
    table = np.roll(np.arange(pages, dtype=np.int32), 1)
    k = np.empty_like(logical_k)
    v = np.empty_like(logical_v)
    for logical, physical in enumerate(table):
        k[physical] = logical_k[logical]
        v[physical] = logical_v[logical]
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
        "page_size": page_size, "page_mapping": "permuted",
        "causal_offset": tokens - 1,
        "boundary_relation": ("exact" if tokens % page_size == 0 else "ragged"),
        "reps": reps, "latency_ms": _median(wall),
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
    groups: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for row in rows:
        for timing, field in ((at.TIMING_DEVICE, "device_latency_ms"),
                              (at.TIMING_END_TO_END, "latency_ms")):
            if field not in row:
                continue
            key = (str(row["op"]), str(row["shape"]), str(row["dtype"]), timing)
            groups.setdefault(key, {})[str(row["mode"])] = float(row[field])
    for (op, shape, dtype, timing), candidates in groups.items():
        dims = tuple(int(v) for v in shape.split("x"))
        winner = min(candidates, key=candidates.__getitem__)
        cache.put(("nvidia:sm_120", "nvidia", op,
                   bucket_key(dims, SpecPolicy.BUCKET), dtype, timing),
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
    p.add_argument("--reps", type=int, default=20,
                   help="repeated-median samples per serving route")
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
