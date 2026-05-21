#!/usr/bin/env python3
"""CorrDiff-core benchmark harness — Sub-5 (2026-05-20).

Sweeps (B, H, W, C_hid, heads, window) configurations and reports
forward-pass latency, throughput, and a determinism check.

Emits the canonical Tessera benchmark JSON schema (Architecture
Decision #12) so ``tools/roofline_tools/`` and ``benchmarks/run_all.py``
can ingest the result unchanged::

    {
      "backend": "tessera-reference",
      "op":      "corrdiff_forward",
      "shape":   {"B": 2, "H": 32, "W": 32, "C_hid": 16, "heads": 2,
                  "window": [1, 1]},
      "dtype":   "fp32",
      "latency_ms":     12.3,
      "throughput_msps": 0.85,           # mega-samples / second
      "memory_bw_gb_s": 0.42,            # rough Activation Π /
                                          # (latency * 1e9)
      "device":         "cpu",
      "tessera_version":"pre-alpha"
    }

Status: **reference / artifact_only**.  The numerical contract is
locked (bit-deterministic given the same seed); native backend
lowering for the inner ops (Apple GPU MSL fused matmul→softmax for
window attention, Apple GPU conv2d, eventual NVIDIA WGMMA path) lands
incrementally via the kernel manifest.

Run from the repo root::

    PYTHONPATH=.:python python benchmarks/corrdiff/benchmark_corrdiff.py \\
        --reps 5 --output /tmp/corrdiff_smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmarks.corrdiff.corrdiff_core import CorrDiffConfig, CorrDiffModel  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Result schema
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CorrDiffResult:
    """One sweep row.  Matches Architecture Decision #12 field names."""
    backend: str
    op: str
    shape: dict
    dtype: str
    latency_ms: float
    throughput_msps: float
    memory_bw_gb_s: float
    device: str
    tessera_version: str
    determinism_ok: bool

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Harness
# ─────────────────────────────────────────────────────────────────────────────


class CorrDiffBenchmark:
    """Run a CorrDiff sweep against the reference (CPU) backend.

    Each sweep row times ``reps`` forward passes (after ``warmup``
    untimed passes), computes a memory-bandwidth proxy, and checks
    determinism by re-running the same forward twice and asserting the
    outputs bit-match.
    """

    BACKEND = "tessera-reference"
    OP      = "corrdiff_forward"
    DEVICE  = "cpu"
    VERSION = "pre-alpha"

    def __init__(self, warmup: int = 2, reps: int = 5):
        self.warmup = warmup
        self.reps   = reps

    def _make_input(self, cfg: CorrDiffConfig) -> np.ndarray:
        # Deterministic input from cfg.seed so the benchmark is
        # reproducible across machines.
        rng = np.random.default_rng(cfg.seed + 0xC077D1FF)
        return rng.standard_normal(
            (cfg.B, cfg.H, cfg.W, cfg.C_in)
        ).astype(np.float32) * 0.5

    def run_one(self, cfg: CorrDiffConfig) -> CorrDiffResult:
        model = CorrDiffModel(cfg)
        x = self._make_input(cfg)

        # Warmup.
        for _ in range(self.warmup):
            _ = model(x, step=0)

        # Timed.
        t0 = time.perf_counter()
        last = None
        for r in range(self.reps):
            last = model(x, step=r)
        t1 = time.perf_counter()
        elapsed_s = (t1 - t0) / max(self.reps, 1)
        latency_ms = elapsed_s * 1000.0

        # Determinism check — same input + same step ⇒ same output.
        ref = model(x, step=0)
        det = model(x, step=0)
        determinism_ok = bool(np.array_equal(ref, det))

        # Throughput (mega-samples/sec) and bandwidth proxy.
        samples = cfg.B * cfg.H * cfg.W
        throughput_msps = (samples / 1e6) / max(elapsed_s, 1e-12)
        # Activation footprint (rough): input + hidden + output bytes
        bytes_per_step = (
            (samples * cfg.C_in
             + samples * cfg.C_hid * 2     # h1 + h2 + attn
             + samples * cfg.C_out)
            * 4  # fp32
        )
        memory_bw_gb_s = (bytes_per_step / 1e9) / max(elapsed_s, 1e-12)

        return CorrDiffResult(
            backend=self.BACKEND,
            op=self.OP,
            shape={
                "B":      cfg.B,
                "H":      cfg.H,
                "W":      cfg.W,
                "C_in":   cfg.C_in,
                "C_hid":  cfg.C_hid,
                "C_out":  cfg.C_out,
                "heads":  cfg.heads,
                "window": list(cfg.window),
            },
            dtype="fp32",
            latency_ms=latency_ms,
            throughput_msps=throughput_msps,
            memory_bw_gb_s=memory_bw_gb_s,
            device=self.DEVICE,
            tessera_version=self.VERSION,
            determinism_ok=determinism_ok,
        )

    def run(self, configs: list[CorrDiffConfig]) -> list[CorrDiffResult]:
        return [self.run_one(c) for c in configs]

    def report(self, results: list[CorrDiffResult]) -> None:
        if not results:
            return
        print(f"{'shape':<48} {'latency_ms':>12} {'thr_msps':>10} "
              f"{'bw_gb/s':>10} det")
        for r in results:
            s = r.shape
            shape_str = (
                f"B={s['B']} H={s['H']} W={s['W']} "
                f"C_hid={s['C_hid']} heads={s['heads']} "
                f"window={s['window']}"
            )
            det = "ok" if r.determinism_ok else "FAIL"
            print(f"{shape_str:<48} {r.latency_ms:>12.2f} "
                  f"{r.throughput_msps:>10.3f} "
                  f"{r.memory_bw_gb_s:>10.3f} {det}")

    def to_json(self, results: list[CorrDiffResult], path: str) -> None:
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Default sweep
# ─────────────────────────────────────────────────────────────────────────────


def default_sweep() -> list[CorrDiffConfig]:
    return [
        CorrDiffConfig(B=1, H=16, W=16, C_in=4, C_hid=8,  C_out=4,
                       heads=2, window=(1, 1), seed=0),
        CorrDiffConfig(B=2, H=32, W=32, C_in=4, C_hid=16, C_out=4,
                       heads=2, window=(1, 1), seed=0),
        CorrDiffConfig(B=2, H=32, W=32, C_in=4, C_hid=16, C_out=4,
                       heads=4, window=(2, 2), seed=0),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps",   type=int, default=5)
    ap.add_argument("--output", default=None,
                    help="Write per-row JSON to this path")
    args = ap.parse_args()

    bench = CorrDiffBenchmark(warmup=args.warmup, reps=args.reps)
    results = bench.run(default_sweep())
    bench.report(results)
    if args.output:
        bench.to_json(results, args.output)
        print(f"\nWrote {len(results)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
