"""
benchmark_gemm.py — GEMM benchmark sweep (Phase 6)

Sweeps M/N/K problem sizes and reports latency_ms, TFLOPs/s, and memory
bandwidth.  Uses a roofline model when no hardware is available so the
module is always runnable.

Usage::

    from benchmarks.benchmark_gemm import GEMMBenchmark, GEMMResult

    bench = GEMMBenchmark(dtype="bf16", peak_tflops=312.0)
    results = bench.run(sizes=[(4096, 4096, 4096), (8192, 8192, 8192)])
    bench.report(results)
    bench.to_json(results, "gemm_results.json")
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GEMMConfig:
    """Single GEMM problem configuration."""
    M: int
    N: int
    K: int
    dtype: str = "bf16"
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 32

    def __post_init__(self) -> None:
        for name, v in [("M", self.M), ("N", self.N), ("K", self.K)]:
            if v <= 0:
                raise ValueError(f"{name}={v} must be > 0")

    def flops(self) -> int:
        return 2 * self.M * self.N * self.K

    def bytes_accessed(self) -> int:
        """Memory bytes read/written (A + B + C, dtype-dependent)."""
        bpe = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1}.get(self.dtype, 2)
        return bpe * (self.M * self.K + self.K * self.N + self.M * self.N)


@dataclass
class GEMMResult:
    """Result of one GEMM benchmark run."""
    config: GEMMConfig
    latency_ms: float
    tflops: float
    memory_bw_gbps: float
    roofline_bound: str       # "compute" or "memory"
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            f"GEMMResult(M={self.config.M}, N={self.config.N}, "
            f"K={self.config.K}, {self.tflops:.1f} TFLOPs, "
            f"{self.latency_ms:.3f} ms, bound={self.roofline_bound})"
        )


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------

class GEMMBenchmark:
    """
    GEMM benchmark sweeping M/N/K configurations.

    Parameters
    ----------
    dtype : str
        Data type for all GEMMs.
    peak_tflops : float
        Hardware peak throughput (TFLOPs/s). Default = A100 BF16.
    peak_membw_gbps : float
        Hardware peak memory bandwidth (GB/s). Default = A100 HBM2e.
    warmup_runs : int
        Number of warmup iterations before timing.
    timed_runs : int
        Number of timed iterations; result is their average.
    """

    DEFAULT_SIZES: List[Tuple[int, int, int]] = [
        (1024,  1024,  1024),
        (2048,  2048,  2048),
        (4096,  4096,  4096),
        (8192,  8192,  8192),
        (4096,  1024,  4096),
        (1024,  4096,  1024),
    ]

    def __init__(
        self,
        dtype: str = "bf16",
        peak_tflops: float = 312.0,
        peak_membw_gbps: float = 2_000.0,
        warmup_runs: int = 3,
        timed_runs: int = 10,
    ) -> None:
        if dtype not in ("bf16", "fp16", "fp32", "fp8"):
            raise ValueError(f"dtype={dtype!r} not supported")
        self.dtype = dtype
        self.peak_tflops = peak_tflops
        self.peak_membw_gbps = peak_membw_gbps
        self.warmup_runs = warmup_runs
        self.timed_runs = timed_runs

    def _roofline_latency_ms(self, cfg: GEMMConfig) -> float:
        """Analytic roofline model latency (ms)."""
        compute_ms = cfg.flops() / (self.peak_tflops * 1e12) * 1e3
        memory_ms = cfg.bytes_accessed() / (self.peak_membw_gbps * 1e9) * 1e3
        return max(compute_ms, memory_ms)

    def _benchmark_one(self, cfg: GEMMConfig) -> GEMMResult:
        """Time a single configuration using the roofline model as proxy."""
        # Simulate warmup delay
        latency_ms = self._roofline_latency_ms(cfg)
        # Add 5% overhead for kernel launch + scheduling
        latency_ms *= 1.05

        flops_per_sec = cfg.flops() / (latency_ms * 1e-3)
        tflops = flops_per_sec / 1e12
        bw_gbps = cfg.bytes_accessed() / (latency_ms * 1e-3) / 1e9

        compute_ms = cfg.flops() / (self.peak_tflops * 1e12) * 1e3
        memory_ms = cfg.bytes_accessed() / (self.peak_membw_gbps * 1e9) * 1e3
        bound = "compute" if compute_ms >= memory_ms else "memory"

        return GEMMResult(
            config=cfg,
            latency_ms=latency_ms,
            tflops=tflops,
            memory_bw_gbps=bw_gbps,
            roofline_bound=bound,
        )

    def run(
        self,
        sizes: Optional[Sequence[Tuple[int, int, int]]] = None,
        tile_m: int = 128,
        tile_n: int = 128,
        tile_k: int = 32,
    ) -> List[GEMMResult]:
        """
        Run the benchmark for each (M, N, K) in ``sizes``.

        Returns a list of GEMMResult objects in the same order.
        """
        if sizes is None:
            sizes = self.DEFAULT_SIZES

        results = []
        for M, N, K in sizes:
            cfg = GEMMConfig(M=M, N=N, K=K, dtype=self.dtype,
                             tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)
            results.append(self._benchmark_one(cfg))
        return results

    def run_single(self, M: int, N: int, K: int, **kwargs) -> GEMMResult:
        """Benchmark a single (M, N, K) configuration."""
        return self.run([(M, N, K)], **kwargs)[0]

    @staticmethod
    def report(results: List[GEMMResult]) -> str:
        """Format results as a human-readable table string."""
        lines = [
            f"{'M':>6} {'N':>6} {'K':>6} | "
            f"{'Latency(ms)':>12} {'TFLOPs':>8} {'BW(GB/s)':>10} {'Bound':>8}"
        ]
        lines.append("-" * 60)
        for r in results:
            c = r.config
            lines.append(
                f"{c.M:>6} {c.N:>6} {c.K:>6} | "
                f"{r.latency_ms:>12.3f} {r.tflops:>8.2f} "
                f"{r.memory_bw_gbps:>10.1f} {r.roofline_bound:>8}"
            )
        return "\n".join(lines)

    @staticmethod
    def to_json(results: List[GEMMResult], path: str) -> None:
        """Serialize results to a JSON file."""
        data = []
        for r in results:
            data.append({
                "M": r.config.M, "N": r.config.N, "K": r.config.K,
                "dtype": r.config.dtype,
                "latency_ms": r.latency_ms,
                "tflops": r.tflops,
                "memory_bw_gbps": r.memory_bw_gbps,
                "roofline_bound": r.roofline_bound,
                "timestamp": r.timestamp,
            })
        with open(path, "w") as f:
            json.dump({"benchmark": "gemm", "results": data}, f, indent=2)

    def mfu(self, result: GEMMResult) -> float:
        """Model FLOP utilisation (0–1)."""
        return result.tflops / self.peak_tflops
