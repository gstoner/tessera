"""
run_all.py — Tessera benchmark orchestrator (Phase 6)

Runs all three benchmark suites (GEMM, Flash-Attention, Collective) and emits
a combined JSON report named ``tessera_benchmarks_<timestamp>.json``.

Usage::

    python benchmarks/run_all.py                       # default config
    python benchmarks/run_all.py --peak-tflops 312     # custom hw caps
    python benchmarks/run_all.py --output-dir /tmp     # custom output dir
    python benchmarks/run_all.py --json-only           # skip terminal table

Direct import::

    from benchmarks.run_all import run_all_benchmarks, BenchmarkSuite
    suite = run_all_benchmarks()
    suite.save("/tmp/my_results.json")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Make sure the project root is importable when run as a script.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.benchmark_gemm import GEMMBenchmark, GEMMResult
from benchmarks.benchmark_attention import FlashAttnBenchmark, AttnResult
from benchmarks.benchmark_collective import CollectiveBenchmark, CollectiveResult, CollectiveOp


# ---------------------------------------------------------------------------
# Suite container
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSuite:
    """
    Container for results from all three benchmark families.

    Attributes
    ----------
    gemm_results : list of GEMMResult
    attn_results : list of AttnResult
    collective_results : list of CollectiveResult
    hw_caps : hardware-capability dict used during the run
    start_time : Unix timestamp for the start of the run
    end_time : Unix timestamp for the end of the run
    """
    gemm_results: List[GEMMResult] = field(default_factory=list)
    attn_results: List[AttnResult] = field(default_factory=list)
    collective_results: List[CollectiveResult] = field(default_factory=list)
    hw_caps: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    @property
    def wall_time_s(self) -> float:
        return self.end_time - self.start_time

    def peak_gemm_tflops(self) -> float:
        if not self.gemm_results:
            return 0.0
        return max(r.tflops for r in self.gemm_results)

    def peak_attn_tflops(self) -> float:
        if not self.attn_results:
            return 0.0
        return max(r.tflops for r in self.attn_results)

    def peak_collective_bw(self) -> float:
        if not self.collective_results:
            return 0.0
        return max(r.bus_bw_gbps for r in self.collective_results)

    def peak_attn_mfu(self) -> float:
        if not self.attn_results:
            return 0.0
        return max(r.mfu for r in self.attn_results)

    # ------------------------------------------------------------------
    # Summary / reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "  Tessera Benchmark Suite — Summary",
            "=" * 70,
            f"  Wall time         : {self.wall_time_s:.2f} s",
            f"  GEMM results      : {len(self.gemm_results)}",
            f"  Attn results      : {len(self.attn_results)}",
            f"  Collective results: {len(self.collective_results)}",
            "",
            f"  Peak GEMM TFLOPs  : {self.peak_gemm_tflops():.2f}",
            f"  Peak Attn TFLOPs  : {self.peak_attn_tflops():.2f}",
            f"  Peak Attn MFU     : {self.peak_attn_mfu()*100:.1f}%",
            f"  Peak Coll BW      : {self.peak_collective_bw():.1f} GB/s",
            "=" * 70,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        ts = datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat()
        return {
            "tessera_benchmark_version": "6.0",
            "timestamp_utc": ts,
            "wall_time_s": self.wall_time_s,
            "hw_caps": self.hw_caps,
            "summary": {
                "gemm_count": len(self.gemm_results),
                "attn_count": len(self.attn_results),
                "collective_count": len(self.collective_results),
                "peak_gemm_tflops": self.peak_gemm_tflops(),
                "peak_attn_tflops": self.peak_attn_tflops(),
                "peak_attn_mfu": self.peak_attn_mfu(),
                "peak_collective_bw_gbps": self.peak_collective_bw(),
            },
            "gemm": [
                {
                    "M": r.config.M, "N": r.config.N, "K": r.config.K,
                    "dtype": r.config.dtype,
                    "latency_ms": r.latency_ms,
                    "tflops": r.tflops,
                    "memory_bw_gbps": r.memory_bw_gbps,
                    "roofline_bound": r.roofline_bound,
                    "compiler_path": r.compiler_path,
                    "runtime_status": "executable" if r.compiler_path == "tessera_jit_cpu" else "skipped" if r.compiler_path.startswith("tessera") else "executable",
                    "compiler_lowering": r.compiler_lowering,
                    "timestamp": r.timestamp,
                }
                for r in self.gemm_results
            ],
            "attention": [
                {
                    "batch": r.config.batch, "heads": r.config.heads,
                    "seq_len": r.config.seq_len, "head_dim": r.config.head_dim,
                    "causal": r.config.causal, "dtype": r.config.dtype,
                    "latency_ms": r.latency_ms,
                    "tokens_per_sec": r.tokens_per_sec,
                    "tflops": r.tflops, "mfu": r.mfu,
                    "compiler_path": r.compiler_path,
                    "runtime_status": "skipped" if r.compiler_path == "graph_ir_only" else "executable",
                    "compiler_lowering": r.compiler_lowering,
                    "timestamp": r.timestamp,
                }
                for r in self.attn_results
            ],
            "collective": [
                {
                    "op": r.config.op.value,
                    "num_ranks": r.config.num_ranks,
                    "message_bytes": r.config.message_bytes,
                    "dtype": r.config.dtype,
                    "latency_ms": r.latency_ms,
                    "bus_bw_gbps": r.bus_bw_gbps,
                    "algbw_gbps": r.algbw_gbps,
                    "timestamp": r.timestamp,
                }
                for r in self.collective_results
            ],
        }

    def save(self, path: str) -> None:
        """Write the full suite to *path* as pretty-printed JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def default_output_path(cls, output_dir: str = ".") -> str:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return os.path.join(output_dir, f"tessera_benchmarks_{ts}.json")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_benchmarks(
    *,
    peak_tflops: float = 312.0,
    peak_membw_gbps: float = 2_000.0,
    peak_bw_gbps: float = 600.0,
    latency_us: float = 5.0,
    dtype: str = "bf16",
    causal: bool = True,
    collective_ops: Optional[List[CollectiveOp]] = None,
    collective_ranks: Optional[List[int]] = None,
    collective_sizes: Optional[List[int]] = None,
    gemm_sizes=None,
    attn_configs=None,
    use_compiler: bool = False,
    verbose: bool = True,
) -> BenchmarkSuite:
    """
    Run all benchmark families and return a :class:`BenchmarkSuite`.

    Parameters
    ----------
    peak_tflops : float
        Hardware peak compute (TFLOPs/s) for GEMM + attention benchmarks.
    peak_membw_gbps : float
        Hardware peak memory bandwidth (GB/s) for roofline model.
    peak_bw_gbps : float
        Per-link bandwidth cap for collective benchmark.
    latency_us : float
        Base latency for collective alpha-beta model.
    dtype : str
        Data type for GEMM and collective benchmarks.
    causal : bool
        Use causal masking for flash-attention benchmark.
    collective_ops : list of CollectiveOp, optional
        Collectives to sweep; defaults to all four.
    collective_ranks / collective_sizes : optional
        Override default rank counts or message sizes.
    gemm_sizes : optional
        Override default GEMM (M, N, K) list.
    attn_configs : optional
        Override default attention (B, H, S, D) list.
    verbose : bool
        Print progress and per-suite tables to stdout.
    """
    suite = BenchmarkSuite(
        hw_caps={
            "peak_tflops": peak_tflops,
            "peak_membw_gbps": peak_membw_gbps,
            "peak_bw_gbps": peak_bw_gbps,
            "dtype": dtype,
        },
        start_time=time.time(),
    )

    # ------------------------------------------------------------------
    # 1. GEMM
    # ------------------------------------------------------------------
    if verbose:
        print("\n[1/3] Running GEMM benchmark …")
    gemm_bench = GEMMBenchmark(
        dtype=dtype,
        peak_tflops=peak_tflops,
        peak_membw_gbps=peak_membw_gbps,
    )
    suite.gemm_results = gemm_bench.run(sizes=gemm_sizes, use_compiler=use_compiler)
    if verbose:
        print(gemm_bench.report(suite.gemm_results))

    # ------------------------------------------------------------------
    # 2. Flash-Attention
    # ------------------------------------------------------------------
    if verbose:
        print("\n[2/3] Running Flash-Attention benchmark …")
    attn_bench = FlashAttnBenchmark(
        causal=causal,
        dtype=dtype,
        peak_tflops=peak_tflops,
        peak_membw_gbps=peak_membw_gbps,
    )
    suite.attn_results = attn_bench.run(configs=attn_configs, emit_compiler_ir=use_compiler)
    if verbose:
        print(attn_bench.report(suite.attn_results))

    # ------------------------------------------------------------------
    # 3. Collective
    # ------------------------------------------------------------------
    if verbose:
        print("\n[3/3] Running Collective benchmark …")
    if collective_ops is None:
        collective_ops = list(CollectiveOp)
    coll_bench = CollectiveBenchmark(
        peak_bw_gbps=peak_bw_gbps,
        latency_us=latency_us,
    )
    suite.collective_results = coll_bench.run(
        ops=collective_ops,
        ranks=collective_ranks,
        sizes=collective_sizes,
    )
    if verbose:
        print(coll_bench.report(suite.collective_results))

    suite.end_time = time.time()
    return suite


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tessera benchmark orchestrator — runs GEMM, attention, and collective suites.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--peak-tflops", type=float, default=312.0,
                   help="Hardware peak compute (TFLOPs/s)")
    p.add_argument("--peak-membw", type=float, default=2000.0,
                   help="Hardware peak memory bandwidth (GB/s)")
    p.add_argument("--peak-bw", type=float, default=600.0,
                   help="Per-link collective bandwidth (GB/s)")
    p.add_argument("--latency-us", type=float, default=5.0,
                   help="Collective base latency (µs)")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "fp8"],
                   default="bf16", help="Data type")
    p.add_argument("--no-causal", action="store_true",
                   help="Disable causal masking in attention benchmark")
    p.add_argument("--output-dir", default=".",
                   help="Directory for JSON output")
    p.add_argument("--output", default=None,
                   help="Explicit output file path (overrides --output-dir)")
    p.add_argument("--json-only", action="store_true",
                   help="Suppress per-suite tables; only print summary")
    p.add_argument("--no-save", action="store_true",
                   help="Do not write JSON output file")
    p.add_argument("--use-compiler", action="store_true",
                   help="Exercise current Tessera compiler artifacts where supported")
    p.add_argument("--smoke", action="store_true",
                   help="Use tiny benchmark shapes suitable for local compiler smoke checks")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    verbose = not args.json_only

    gemm_sizes = [(64, 64, 64), (128, 128, 64)] if args.smoke else None
    attn_configs = [(1, 2, 64, 32)] if args.smoke else None
    collective_ranks = [2, 4] if args.smoke else None
    collective_sizes = [1 * 1024 * 1024] if args.smoke else None

    suite = run_all_benchmarks(
        peak_tflops=args.peak_tflops,
        peak_membw_gbps=args.peak_membw,
        peak_bw_gbps=args.peak_bw,
        latency_us=args.latency_us,
        dtype=args.dtype,
        causal=not args.no_causal,
        gemm_sizes=gemm_sizes,
        attn_configs=attn_configs,
        collective_ranks=collective_ranks,
        collective_sizes=collective_sizes,
        use_compiler=args.use_compiler,
        verbose=verbose,
    )

    print(suite.summary())

    if not args.no_save:
        out_path = args.output or BenchmarkSuite.default_output_path(args.output_dir)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        suite.save(out_path)
        print(f"\nResults saved → {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
