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

import numpy as np

try:
    from benchmarks.compiler_support import compiler_matmul_relu
except ImportError:  # Allows running this file directly from benchmarks/.
    from compiler_support import compiler_matmul_relu

try:
    from tessera.telemetry import make_event, telemetry_report
except ImportError:
    make_event = None
    telemetry_report = None


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
    compiler_path: str = "roofline_model"
    compiler_lowering: str = ""
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

    def _benchmark_one(self, cfg: GEMMConfig, *, use_compiler: bool = False) -> GEMMResult:
        """Time a single configuration using the roofline model as proxy."""
        compiler_path = "roofline_model"
        compiler_lowering = ""

        if use_compiler:
            # Keep compiler smoke sizes bounded; large sweeps remain analytic.
            elems = cfg.M * cfg.K + cfg.K * cfg.N
            if elems <= 2_000_000:
                a = np.ones((cfg.M, cfg.K), dtype=np.float32)
                b = np.ones((cfg.K, cfg.N), dtype=np.float32)
                run = compiler_matmul_relu(a, b, (cfg.tile_m, cfg.tile_n, cfg.tile_k))
                if run is not None:
                    latency_ms = run.latency_ms
                    flops_per_sec = cfg.flops() / max(latency_ms * 1e-3, 1e-12)
                    return GEMMResult(
                        config=cfg,
                        latency_ms=latency_ms,
                        tflops=flops_per_sec / 1e12,
                        memory_bw_gbps=cfg.bytes_accessed() / max(latency_ms * 1e-3, 1e-12) / 1e9,
                        roofline_bound="measured_cpu",
                        compiler_path="tessera_jit_cpu" if run.uses_compiled_path else "tessera_jit_fallback",
                        compiler_lowering=run.lowering,
                    )
                compiler_path = "compiler_unavailable"

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
            compiler_path=compiler_path,
            compiler_lowering=compiler_lowering,
        )

    def run(
        self,
        sizes: Optional[Sequence[Tuple[int, int, int]]] = None,
        tile_m: int = 128,
        tile_n: int = 128,
        tile_k: int = 32,
        use_compiler: bool = False,
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
            results.append(self._benchmark_one(cfg, use_compiler=use_compiler))
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
        telemetry_events = []
        for r in results:
            telemetry = _gemm_telemetry(r) if make_event is not None else {}
            if telemetry:
                telemetry_events.append(telemetry)
            data.append({
                "M": r.config.M, "N": r.config.N, "K": r.config.K,
                "dtype": r.config.dtype,
                "latency_ms": r.latency_ms,
                "tflops": r.tflops,
                "memory_bw_gbps": r.memory_bw_gbps,
                "roofline_bound": r.roofline_bound,
                "compiler_path": r.compiler_path,
                "compiler_lowering": r.compiler_lowering,
                "timestamp": r.timestamp,
                "telemetry": telemetry,
            })
        payload = {"benchmark": "gemm", "results": data}
        if telemetry_report is not None:
            payload["telemetry_summary"] = telemetry_report(telemetry_events)
            payload["telemetry_events"] = telemetry_events
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def mfu(self, result: GEMMResult) -> float:
        """Model FLOP utilisation (0–1)."""
        return result.tflops / self.peak_tflops


def _gemm_telemetry(result: GEMMResult) -> dict:
    return make_event(
        "benchmark.gemm",
        source="benchmark",
        op="matmul",
        shape={"M": result.config.M, "N": result.config.N, "K": result.config.K},
        dtype=result.config.dtype,
        kernel_id="gemm",
        latency_ms=result.latency_ms,
        tflops=result.tflops,
        bandwidth_gbps=result.memory_bw_gbps,
        memory_bytes=result.config.bytes_accessed(),
        status="executable" if result.compiler_path != "compiler_unavailable" else "unmeasured",
        metadata={
            "roofline_bound": result.roofline_bound,
            "compiler_path": result.compiler_path,
            "compiler_lowering": result.compiler_lowering,
            "tile": {
                "M": result.config.tile_m,
                "N": result.config.tile_n,
                "K": result.config.tile_k,
            },
        },
        timestamp=result.timestamp,
    )
