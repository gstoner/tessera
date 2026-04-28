"""
benchmark_collective.py — Collective communication benchmark (Phase 6)

Models all-reduce / reduce-scatter / all-gather bus bandwidth over 2–128 ranks
using the standard bandwidth formulae.

Usage::

    from benchmarks.benchmark_collective import CollectiveBenchmark

    bench = CollectiveBenchmark(peak_bw_gbps=600.0)
    results = bench.run()
    print(bench.report(results))
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence


class CollectiveOp(Enum):
    ALL_REDUCE     = "all_reduce"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_GATHER     = "all_gather"
    ALL_TO_ALL     = "all_to_all"


@dataclass
class CollectiveConfig:
    """Configuration for one collective benchmark run."""
    op: CollectiveOp
    num_ranks: int
    message_bytes: int
    dtype: str = "bf16"

    def __post_init__(self) -> None:
        if self.num_ranks < 2:
            raise ValueError(f"num_ranks={self.num_ranks} must be >= 2")
        if self.message_bytes <= 0:
            raise ValueError(f"message_bytes must be > 0")

    def bus_bytes(self) -> int:
        """
        Effective bytes transferred on the bus per rank, per op.

        - All-reduce (ring):    2*(N-1)/N * msg
        - Reduce-scatter:         (N-1)/N * msg
        - All-gather:             (N-1)/N * msg
        - All-to-all:             (N-1)/N * msg
        """
        N = self.num_ranks
        factor = 2 * (N - 1) / N if self.op == CollectiveOp.ALL_REDUCE \
            else (N - 1) / N
        return int(self.message_bytes * factor)


@dataclass
class CollectiveResult:
    config: CollectiveConfig
    latency_ms: float
    bus_bw_gbps: float
    algbw_gbps: float
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            f"CollectiveResult(op={self.config.op.value}, "
            f"ranks={self.config.num_ranks}, "
            f"msg={self.config.message_bytes//1024}KiB, "
            f"bus_bw={self.bus_bw_gbps:.1f} GB/s)"
        )


class CollectiveBenchmark:
    """
    Collective communication benchmark.

    Parameters
    ----------
    peak_bw_gbps : float
        Per-link peak bandwidth (GB/s). Default = NVLink 4th gen (900 GB/s
        aggregate / 8 GPUs ≈ 112 GB/s per link).
    latency_us : float
        Base latency per collective (µs).
    """

    DEFAULT_RANKS = [2, 4, 8, 16, 32, 64, 128]
    DEFAULT_SIZES = [
        1   * 1024 * 1024,    # 1 MiB
        16  * 1024 * 1024,    # 16 MiB
        128 * 1024 * 1024,    # 128 MiB
        512 * 1024 * 1024,    # 512 MiB
    ]

    def __init__(
        self,
        peak_bw_gbps: float = 600.0,
        latency_us: float = 5.0,
    ) -> None:
        if peak_bw_gbps <= 0:
            raise ValueError("peak_bw_gbps must be > 0")
        self.peak_bw_gbps = peak_bw_gbps
        self.latency_us = latency_us

    def _benchmark_one(self, cfg: CollectiveConfig) -> CollectiveResult:
        bus_bytes = cfg.bus_bytes()
        bus_bw = min(self.peak_bw_gbps, bus_bytes / (self.peak_bw_gbps * 1e9) *
                     self.peak_bw_gbps)
        # Alpha-beta model: latency_us + bus_bytes / peak_bw
        latency_s = self.latency_us * 1e-6 + bus_bytes / (self.peak_bw_gbps * 1e9)
        latency_ms = latency_s * 1e3
        bus_bw_gbps = bus_bytes / latency_s / 1e9
        algbw_gbps = cfg.message_bytes / latency_s / 1e9
        return CollectiveResult(
            config=cfg,
            latency_ms=latency_ms,
            bus_bw_gbps=bus_bw_gbps,
            algbw_gbps=algbw_gbps,
        )

    def run(
        self,
        ops: Optional[Sequence[CollectiveOp]] = None,
        ranks: Optional[Sequence[int]] = None,
        sizes: Optional[Sequence[int]] = None,
    ) -> List[CollectiveResult]:
        if ops is None:
            ops = [CollectiveOp.ALL_REDUCE]
        if ranks is None:
            ranks = self.DEFAULT_RANKS
        if sizes is None:
            sizes = self.DEFAULT_SIZES

        results = []
        for op in ops:
            for n in ranks:
                for s in sizes:
                    cfg = CollectiveConfig(op=op, num_ranks=n, message_bytes=s)
                    results.append(self._benchmark_one(cfg))
        return results

    def run_single(self, op: CollectiveOp, num_ranks: int,
                   message_bytes: int) -> CollectiveResult:
        cfg = CollectiveConfig(op=op, num_ranks=num_ranks,
                               message_bytes=message_bytes)
        return self._benchmark_one(cfg)

    @staticmethod
    def report(results: List[CollectiveResult]) -> str:
        lines = [
            f"{'Op':>14} {'Ranks':>6} {'Msg(MiB)':>9} | "
            f"{'Lat(ms)':>9} {'BusBW(GB/s)':>12} {'AlgBW(GB/s)':>12}"
        ]
        lines.append("-" * 68)
        for r in results:
            c = r.config
            lines.append(
                f"{c.op.value:>14} {c.num_ranks:>6} "
                f"{c.message_bytes/1024/1024:>9.1f} | "
                f"{r.latency_ms:>9.3f} {r.bus_bw_gbps:>12.1f} "
                f"{r.algbw_gbps:>12.1f}"
            )
        return "\n".join(lines)

    @staticmethod
    def to_json(results: List[CollectiveResult], path: str) -> None:
        data = [{
            "op": r.config.op.value,
            "num_ranks": r.config.num_ranks,
            "message_bytes": r.config.message_bytes,
            "dtype": r.config.dtype,
            "latency_ms": r.latency_ms,
            "bus_bw_gbps": r.bus_bw_gbps,
            "algbw_gbps": r.algbw_gbps,
            "timestamp": r.timestamp,
        } for r in results]
        with open(path, "w") as f:
            json.dump({"benchmark": "collective", "results": data}, f, indent=2)

    def bus_utilization(self, result: CollectiveResult) -> float:
        """Bus utilization fraction (0–1)."""
        return min(result.bus_bw_gbps / self.peak_bw_gbps, 1.0)
