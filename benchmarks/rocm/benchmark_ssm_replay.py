"""Exact-gfx1151 ROCM-REPLAY-1 summary-vs-replay serving benchmark."""
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
    return statistics.median(values)


def _data(seed: int, tokens: int, bsz: int, d: int, n: int):
    rng = np.random.default_rng(seed)
    a = -np.abs(rng.standard_normal(d)).astype(np.float32)
    delta = (np.abs(rng.standard_normal((tokens, bsz, d))) * .1).astype(np.float32)
    x = (rng.standard_normal((tokens, bsz, d)) * .1).astype(np.float32)
    b = (rng.standard_normal((tokens, bsz, n)) * .1).astype(np.float32)
    c = (rng.standard_normal((tokens, bsz, n)) * .1).astype(np.float32)
    return a, delta, x, b, c


def _reference_outputs(a: Any, delta: Any, x: Any, b: Any, c: Any,
                       capacity: int, chunk: int) -> np.ndarray:
    from tessera.cache import SSMStateHandle
    tokens, bsz, d = delta.shape; n = b.shape[-1]
    handle = SSMStateHandle(bsz, d, n, a, capacity=capacity)
    out = np.empty((tokens, bsz, d), np.float64)
    for start in range(0, tokens, chunk):
        width = min(chunk, tokens - start)
        if handle.should_flush(width):
            handle.flush()
        for token in range(start, start + width):
            handle.append(delta[token], x[token], b[token], auto_flush=False)
            out[token] = handle.read_output(c[token])
    return out


def _summary_row(shape: str, tokens: int, reps: int) -> dict[str, Any]:
    from tessera.compiler.emit.rocm_hip import RocmReplayDeviceState
    bsz, d, n = parse_shape(shape)
    a, delta, x, b, c = _data(4100 + d + n, tokens, bsz, d, n)
    wall, device = [], []
    observed = np.empty((tokens, bsz, d), np.float32)
    for _ in range(reps):
        state = RocmReplayDeviceState(
            np.zeros((bsz, d, n), np.float32), a, capacity=1)
        start = time.perf_counter_ns(); elapsed = 0.0
        for token in range(tokens):
            observed[token], ms = state.summary_step(
                delta[token], x[token], b[token], c[token])
            elapsed += ms
        wall.append((time.perf_counter_ns() - start) / 1e6)
        device.append(elapsed)
        state.close()
    traffic = summary_state_bytes_per_token(d, n)
    expected = _reference_outputs(a, delta, x, b, c, 1, 1)
    latency = _median(wall) / tokens
    return {"backend": "rocm", "device": "gfx1151",
            "op": "ssm_replay_decode", "shape": shape, "mode": "summary",
            "tokens": tokens, "reps": reps,
            "latency_ms": latency,
            "device_latency_ms": _median(device) / tokens,
            "tokens_per_second": 1000.0 / latency,
            "max_abs_error": float(np.max(np.abs(observed - expected))),
            "state_bytes_per_token": traffic,
            "summary_state_bytes_per_token": traffic,
            "state_traffic_ratio": 1.0}


def _replay_row(shape: str, tokens: int, chunk: int, slots: int,
                reps: int, capacity: int | None = None) -> dict[str, Any]:
    from tessera import runtime as rt
    bsz, d, n = parse_shape(shape)
    if chunk < 1 or chunk > tokens:
        raise ValueError("chunk must be in [1,tokens]")
    capacity = int(capacity or (tokens + 1))
    if capacity < chunk:
        raise ValueError("capacity must fit at least one chunk")
    a, delta, x, b, c = _data(5100 + d + n, tokens, bsz, d, n)
    wall, device, flush_samples = [], [], []
    observed = np.empty((tokens, bsz, d), np.float32)
    for _ in range(reps):
        handle = rt.rocm_ssm_replay_state_handle(
            bsz, d, n, a, capacity=capacity, async_slots=slots)
        if handle._device is None:
            raise RuntimeError("live gfx1151 ReplaySSM device state unavailable")
        start = time.perf_counter_ns()
        offset, elapsed, flushes = 0, 0.0, 0
        while offset < tokens:
            futures: list[tuple[int, Any]] = []
            while offset < tokens and len(futures) < slots:
                width = min(chunk, tokens - offset)
                if handle.should_flush(width):
                    break
                begin = offset
                future = handle.submit_block_async(
                    delta[offset:offset + width], x[offset:offset + width],
                    b[offset:offset + width], c[offset:offset + width])
                futures.append((begin, future))
                offset += width
            if not futures:
                handle.flush(); flushes += 1
                continue
            elapsed += sum(future.event.elapsed_ms() for _, future in futures)
            for begin, future in futures:
                value = future.wait()
                observed[begin:begin + value.shape[0]] = value
        wall.append((time.perf_counter_ns() - start) / 1e6)
        device.append(elapsed); flush_samples.append(flushes)
        handle._device.close()
    replay = replay_state_bytes_per_token(d, n, capacity)
    summary = summary_state_bytes_per_token(d, n)
    expected = _reference_outputs(a, delta, x, b, c, capacity, chunk)
    return {"backend": "rocm", "device": "gfx1151",
            "op": "ssm_replay_decode", "shape": shape,
            "mode": "async_ring", "tokens": tokens, "chunk": chunk,
            "capacity": capacity, "flushes": int(_median(flush_samples)),
            "async_slots": slots, "reps": reps,
            "latency_ms": _median(wall) / tokens,
            "device_latency_ms": _median(device) / tokens,
            "tokens_per_second": 1000.0 / (_median(wall) / tokens),
            "max_abs_error": float(np.max(np.abs(observed - expected))),
            "state_bytes_per_token": replay,
            "summary_state_bytes_per_token": summary,
            "state_traffic_ratio": summary / replay}


def _output_only_row(shape: str, tokens: int, reps: int,
                     capacity: int | None = None) -> dict[str, Any]:
    """True sequential decode: submit and consume exactly one token at a time."""
    from tessera import runtime as rt
    bsz, d, n = parse_shape(shape)
    a, delta, x, b, c = _data(6100 + d + n, tokens, bsz, d, n)
    capacity = int(capacity or (tokens + 1))
    wall, device, flush_samples = [], [], []
    observed = np.empty((tokens, bsz, d), np.float32)
    for _ in range(reps):
        handle = rt.rocm_ssm_replay_state_handle(
            bsz, d, n, a, capacity=capacity, async_slots=2)
        if handle._device is None:
            raise RuntimeError("live gfx1151 ReplaySSM device state unavailable")
        start = time.perf_counter_ns(); elapsed = 0.0; flushes = 0
        for token in range(tokens):
            if handle.should_flush(1):
                handle.flush(); flushes += 1
            future = handle.submit_block_async(
                delta[token:token + 1], x[token:token + 1],
                b[token:token + 1], c[token:token + 1])
            elapsed += future.event.elapsed_ms()
            observed[token] = future.wait()[0]
        wall.append((time.perf_counter_ns() - start) / 1e6)
        device.append(elapsed); flush_samples.append(flushes)
        handle._device.close()
    replay = replay_state_bytes_per_token(d, n, capacity)
    summary = summary_state_bytes_per_token(d, n)
    expected = _reference_outputs(a, delta, x, b, c, capacity, 1)
    return {"backend": "rocm", "device": "gfx1151",
            "op": "ssm_replay_decode", "shape": shape,
            "mode": "output_only", "tokens": tokens, "capacity": capacity,
            "flushes": int(_median(flush_samples)), "reps": reps,
            "latency_ms": _median(wall) / tokens,
            "device_latency_ms": _median(device) / tokens,
            "tokens_per_second": 1000.0 / (_median(wall) / tokens),
            "max_abs_error": float(np.max(np.abs(observed - expected))),
            "state_bytes_per_token": replay,
            "summary_state_bytes_per_token": summary,
            "state_traffic_ratio": summary / replay}


def run_benchmark(shapes: list[str], tokens: int, chunk: int, slots: int,
                  reps: int) -> list[dict[str, Any]]:
    from tessera import runtime as rt
    if rt._rocm_chip() != "gfx1151":
        return []
    rows = []
    for shape in shapes:
        rows.append(_summary_row(shape, tokens, reps))
        rows.append(_output_only_row(shape, tokens, reps))
        rows.append(_replay_row(shape, tokens, chunk, slots, reps))
    return rows


def run_matrix(
    shapes: list[str], cases: list[tuple[int, int]], reps: int,
    schedules: tuple[tuple[int, int], ...] = ((4, 2), (16, 4)),
) -> list[dict[str, Any]]:
    """Compiler matrix over ``(tokens,capacity)`` and async schedules."""
    from tessera import runtime as rt
    if rt._rocm_chip() != "gfx1151":
        return []
    rows: list[dict[str, Any]] = []
    for shape in shapes:
        summary_tokens: set[int] = set()
        for tokens, capacity in cases:
            if tokens not in summary_tokens:
                rows.append(_summary_row(shape, tokens, reps))
                summary_tokens.add(tokens)
            rows.append(_output_only_row(
                shape, tokens, reps, capacity=capacity))
            for chunk, slots in schedules:
                if chunk <= capacity and chunk <= tokens:
                    rows.append(_replay_row(
                        shape, tokens, chunk, slots, reps,
                        capacity=capacity))
    for row in rows:
        row.update({"target": "rocm", "evidence_arch": "gfx1151",
                    "dtype": "f32",
                    "compiler_path": "rocm_replayssm_hip"})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", default=("1x64x64", "1x128x128"))
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--matrix", action="store_true")
    args = parser.parse_args()
    if args.matrix:
        shapes = list(args.shapes)
        if shapes == ["1x64x64", "1x128x128"]:
            shapes = ["1x32x16", "1x64x64", "1x128x64",
                      "1x128x128", "4x64x64"]
        rows = run_matrix(
            shapes, [(16, 16), (64, 16), (64, 64), (256, 64)],
            args.reps)
    else:
        rows = run_benchmark(
            list(args.shapes), args.tokens, args.chunk, args.slots, args.reps)
    payload = {"schema_version": 1, "evidence_arch": "gfx1151", "rows": rows}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
