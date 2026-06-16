"""apple_gpu ReplaySSM decode benchmark — Track-R Phase 5-bench.

Measures Mamba-2 (scalar-A) **decode** across three routes and reports the
ReplaySSM state-traffic reduction:

  - ``summary``          — baseline: load + store the full (D, N) recurrent
    state on every token (the HBM round-trip ReplaySSM removes);
  - ``replay_reference`` — host numpy ``SSMStateHandle`` (output-only +
    periodic flush);
  - ``replay_fused``     — the single-dispatch Metal kernel
    (``tessera_apple_gpu_ssm_replay_decode_f32``) that keeps ``S0`` resident.

Two things are reported per row:

  1. **Analytical state traffic** (``state_bytes_per_token`` /
     ``state_traffic_ratio``) — the architectural claim.  ReplaySSM writes the
     full (D, N) state only once per ``capacity`` tokens (flush) instead of
     every token, and keeps ``S0`` resident, so the dominant 2·D·N HBM
     round-trip collapses to (D·N)/L plus the small replay-input append.  This
     is the guaranteed win and is shape/host-independent.

  2. **Measured per-token latency** (``latency_ms``) — reported for
     transparency.  HONEST CAVEAT: at decode-1 scale the wall-clock is
     dominated by Python-loop + per-call dispatch overhead, NOT HBM bandwidth,
     so the measured latency does NOT yet show the bandwidth win — that needs a
     persistent-command-buffer / batched decode harness.  The architectural
     reduction is the analytical column; the latency column is observed reality.

Output schema matches ``benchmark_gemm.py`` / ``benchmark_fusion.py`` (the
stable fields ``tools/roofline_tools/`` reads), plus ReplaySSM-specific extras.

Usage:
    python benchmarks/apple_gpu/benchmark_ssm_replay.py \\
        --shapes 1x128x128 1x256x128 4x128x64 --tokens 128 --capacity 16 \\
        --reps 20 --output apple_gpu_ssm_replay.json

Best-effort: ``summary`` and ``replay_reference`` run anywhere; ``replay_fused``
runs the kernel (Metal on Darwin, host reference otherwise).
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

import tessera as ts
from tessera import runtime as rt
from tessera.cache import SSMStateHandle


def _parse_shape(spec: str) -> tuple[int, int, int]:
    parts = spec.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"shape must be BxDxN, got {spec!r}")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def _inputs(B: int, D: int, N: int, T: int):
    rng = np.random.RandomState(0)
    x = rng.randn(T, B, D).astype(np.float64)
    a = -np.abs(rng.randn(D)).astype(np.float64)
    b = rng.randn(T, B, N).astype(np.float64)
    c = rng.randn(T, B, N).astype(np.float64)
    dt = np.abs(rng.randn(T, B, D)).astype(np.float64) * 0.5
    return x, a, b, c, dt


# ── Analytical state traffic (the architectural claim) ──────────────────

def _summary_state_bytes_per_token(D: int, N: int, elem: int = 4) -> float:
    """Baseline: load + store the full (D, N) state every token."""
    return 2.0 * elem * D * N


def _replay_state_bytes_per_token(D: int, N: int, L: int, elem: int = 4) -> float:
    """ReplaySSM: full state stored once per L tokens (flush); S0 stays
    resident (not reloaded); plus the small per-token replay-input append
    (delta, x, b = 2D + N)."""
    return (elem * D * N) / max(L, 1) + elem * (2 * D + N)


# ── Decode timers ───────────────────────────────────────────────────────

def _bench_summary(B: int, D: int, N: int, T: int, reps: int) -> tuple[float, float]:
    """Baseline full-state decode: update + read the (D, N) state every token."""
    x, a, b, c, dt = _inputs(B, D, N, T)
    a2 = np.broadcast_to(a[:, None], (D, N))

    def run() -> None:
        S = np.zeros((B, D, N))
        for t in range(T):
            abar = np.exp(dt[t][:, :, None] * a2[None])
            S = abar * S + (dt[t] * x[t])[:, :, None] * b[t][:, None, :]
            _ = np.einsum("bdn,bn->bd", S, c[t])

    samples = _time(run, reps)
    return statistics.median(samples), _stdev(samples)


def _bench_replay(handle_factory, B: int, D: int, N: int, T: int, L: int,
                  reps: int) -> tuple[float, float]:
    x, a, b, c, dt = _inputs(B, D, N, T)

    def run() -> None:
        h = handle_factory(a, L)
        for t in range(T):
            h.step(dt[t], x[t], b[t], c[t])

    # Warm up (compiles the Metal runtime on first fused call).
    run()
    samples = _time(run, reps)
    return statistics.median(samples), _stdev(samples)


def _bench_block(B: int, D: int, N: int, T: int, reps: int) -> tuple[float, float]:
    """All T tokens in ONE block dispatch — the dispatch-overhead fix."""
    x, a, b, c, dt = _inputs(B, D, N, T)
    h = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=8)

    def run() -> None:
        h.decode_block(dt, x, b, c)

    run()
    samples = _time(run, reps)
    return statistics.median(samples), _stdev(samples)


def _time(fn, reps: int) -> list[float]:
    out = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        out.append((time.perf_counter_ns() - t0) / 1e6)
    return out


def _stdev(samples: list[float]) -> float:
    return statistics.stdev(samples) if len(samples) > 1 else 0.0


def _device_name() -> str:
    return "apple_silicon_metal" if sys.platform == "darwin" else "non-darwin-fallback"


def _tessera_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("tessera")
    except Exception:
        return "dev"


def run_benchmark(shapes: list[str], tokens: int, capacity: int,
                  reps: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    device, version = _device_name(), _tessera_version()
    L = capacity

    for shape in shapes:
        B, D, N = _parse_shape(shape)
        summ_bytes = _summary_state_bytes_per_token(D, N)
        repl_bytes = _replay_state_bytes_per_token(D, N, L)
        ratio = summ_bytes / repl_bytes if repl_bytes > 0 else 0.0
        flushes = tokens // max(L, 1)

        def ref(a, cap, B=B, D=D, N=N):
            return SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a,
                                  capacity=cap)

        def fused(a, cap, B=B, D=D, N=N):
            return rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=cap)

        timers = [
            ("summary", lambda: _bench_summary(B, D, N, tokens, reps), summ_bytes),
            ("replay_reference",
             lambda: _bench_replay(ref, B, D, N, tokens, L, reps), repl_bytes),
            ("replay_fused",
             lambda: _bench_replay(fused, B, D, N, tokens, L, reps), repl_bytes),
            # The dispatch-overhead fix: all tokens in ONE block dispatch.
            ("replay_block",
             lambda: _bench_block(B, D, N, tokens, reps), repl_bytes),
        ]
        for mode, timer, state_bytes in timers:
            loop_ms, stdev_ms = timer()
            ms_per_token = loop_ms / max(tokens, 1)
            sec = ms_per_token / 1000.0
            # memory_bw from the analytical per-token state traffic.
            mem_bw = (state_bytes / sec) / 1e9 if sec > 0 else 0.0
            rows.append({
                "backend": "apple_gpu",
                "op": "ssm_replay_decode",
                "shape": shape,
                "dtype": "f32",
                "mode": mode,
                "reps": reps,
                "tokens": tokens,
                "capacity": L,
                "flush_count": flushes if mode != "summary" else tokens,
                "latency_ms": ms_per_token,
                "loop_ms": loop_ms,
                "stdev_ms": stdev_ms,
                "tflops": 0.0,                       # decode is memory-bound
                "memory_bw_gb_s": mem_bw,
                "state_bytes_per_token": state_bytes,
                "state_traffic_ratio": ratio if mode != "summary" else 1.0,
                "device": device,
                "tessera_version": version,
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+",
                        default=["1x128x128", "1x256x128", "4x128x64"],
                        help="BxDxN decode shapes")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = run_benchmark(args.shapes, args.tokens, args.capacity, args.reps)
    payload = {"runs": rows}
    out = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(out)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
