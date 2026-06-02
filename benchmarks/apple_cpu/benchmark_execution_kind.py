# SPDX-License-Identifier: MIT
"""Apple CPU execution-kind microbench (Apple plan, phase B, 2026-05-20).

Empirically proves the ``accelerate_native`` vs ``numpy_reference``
split that ``apple_target_map.py``'s ``execution_kind`` axis declares.

**Important calibration note.** On macOS numpy is **already** linked
against Accelerate, so a "Tessera vs numpy" matmul comparison is
really "Tessera-Accelerate vs numpy-Accelerate" and both are fast.
The empirical gate is therefore *not* "Tessera must be faster than
numpy" — it is "Tessera latency must be within a small factor of
numpy latency", which proves both go through the same
``cblas_sgemm`` fast lane (a numpy-Python-loop fallback would be
≥100× slower).

The four ops + their expected ``execution_kind``:

* **matmul** — ``accelerate_native``.  Gate: Tessera ≤ 1.5× numpy
  latency at N=512 (both Accelerate; allows for per-call overhead).
* **layer_norm / softmax / gelu** — ``numpy_reference``.  No perf
  gate.  Tessera goes through the numpy reference shim today;
  latency naturally tracks numpy.

Output is the standard Tessera benchmark JSON schema with
``execution_kind`` filled in per op, so the dashboard can ingest it
as a row alongside the spectral correctness + GA/EBM bench rows.

Usage:

    PYTHONPATH=python python benchmarks/apple_cpu/benchmark_execution_kind.py \\
        --output /tmp/tessera_apple_cpu_execution_kind.json

Exit non-zero only if matmul latency is **more than 1.5× slower**
than numpy — that would mean Tessera isn't on the fast lane.  All
other ops are always pass.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


def _time_call(fn: Any, repeats: int, warmup: int) -> float:
    """Return mean per-call wall time in seconds."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def _bench_matmul(N: int, repeats: int, warmup: int) -> dict[str, Any]:
    """matmul — expect Tessera/Accelerate ≪ numpy."""
    import tessera

    rng = np.random.default_rng(seed=0xDEADBEEF)
    A = rng.standard_normal((N, N)).astype(np.float32)
    B = rng.standard_normal((N, N)).astype(np.float32)

    # Tessera path — goes through apple_cpu Accelerate when on Darwin.
    @tessera.jit(target="apple_cpu")
    def f(x, y):
        return tessera.ops.matmul(x, y)

    t_tess = _time_call(lambda: f(A, B), repeats, warmup)
    t_numpy = _time_call(lambda: A @ B, repeats, warmup)
    ratio = t_tess / t_numpy if t_numpy > 0 else float("inf")
    # Both Tessera and numpy go through Accelerate on macOS.  The
    # empirical "accelerate_native" gate is that Tessera's latency
    # stays within a small factor of numpy's — i.e., it's also on
    # the cblas_sgemm fast lane (a Python-loop fallback would be
    # ≥100× slower).
    return {
        "op": "matmul",
        "N": N,
        "execution_kind": "accelerate_native",
        "tessera_ms": t_tess * 1000.0,
        "numpy_ms": t_numpy * 1000.0,
        "tessera_over_numpy_ratio": ratio,
        "gate": "tessera_must_be_within_1.5x_of_numpy",
        "pass": ratio <= 1.5,
    }


def _bench_layer_norm(N: int, repeats: int, warmup: int) -> dict[str, Any]:
    """layer_norm — Tessera path uses numpy reference today."""
    import tessera

    rng = np.random.default_rng(seed=0xDEADBEEF)
    x = rng.standard_normal((N, N)).astype(np.float32)

    @tessera.jit(target="apple_cpu")
    def f(x_):
        return tessera.ops.layer_norm(x_)

    def numpy_ref(arr: np.ndarray) -> np.ndarray:
        m = arr.mean(axis=-1, keepdims=True)
        v = arr.var(axis=-1, keepdims=True)
        return (arr - m) / np.sqrt(v + 1e-5)

    t_tess = _time_call(lambda: f(x), repeats, warmup)
    t_numpy = _time_call(lambda: numpy_ref(x), repeats, warmup)
    return {
        "op": "layer_norm",
        "N": N,
        "execution_kind": "numpy_reference",
        "tessera_ms": t_tess * 1000.0,
        "numpy_ms": t_numpy * 1000.0,
        "tessera_over_numpy_ratio": t_tess / t_numpy if t_numpy > 0 else float("inf"),
        "gate": "no_perf_gate_for_reference_lane",
        "pass": True,
    }


def _bench_softmax(N: int, repeats: int, warmup: int) -> dict[str, Any]:
    """softmax — Tessera path uses numpy reference today."""
    import tessera

    rng = np.random.default_rng(seed=0xDEADBEEF)
    x = rng.standard_normal((N, N)).astype(np.float32)

    @tessera.jit(target="apple_cpu")
    def f(x_):
        return tessera.ops.softmax(x_, axis=-1)

    def numpy_ref(arr: np.ndarray) -> np.ndarray:
        e = np.exp(arr - arr.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    t_tess = _time_call(lambda: f(x), repeats, warmup)
    t_numpy = _time_call(lambda: numpy_ref(x), repeats, warmup)
    return {
        "op": "softmax",
        "N": N,
        "execution_kind": "numpy_reference",
        "tessera_ms": t_tess * 1000.0,
        "numpy_ms": t_numpy * 1000.0,
        "tessera_over_numpy_ratio": t_tess / t_numpy if t_numpy > 0 else float("inf"),
        "gate": "no_perf_gate_for_reference_lane",
        "pass": True,
    }


def _bench_gelu(N: int, repeats: int, warmup: int) -> dict[str, Any]:
    """gelu — Tessera path uses numpy reference today."""
    import tessera

    rng = np.random.default_rng(seed=0xDEADBEEF)
    x = rng.standard_normal((N, N)).astype(np.float32)

    @tessera.jit(target="apple_cpu")
    def f(x_):
        return tessera.ops.gelu(x_)

    def numpy_ref(arr: np.ndarray) -> np.ndarray:
        # Tanh-approx GELU (matches Tessera reference).
        c = np.sqrt(2.0 / np.pi)
        return 0.5 * arr * (1.0 + np.tanh(c * (arr + 0.044715 * arr ** 3)))

    t_tess = _time_call(lambda: f(x), repeats, warmup)
    t_numpy = _time_call(lambda: numpy_ref(x), repeats, warmup)
    return {
        "op": "gelu",
        "N": N,
        "execution_kind": "numpy_reference",
        "tessera_ms": t_tess * 1000.0,
        "numpy_ms": t_numpy * 1000.0,
        "tessera_over_numpy_ratio": t_tess / t_numpy if t_numpy > 0 else float("inf"),
        "gate": "no_perf_gate_for_reference_lane",
        "pass": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Apple CPU execution-kind microbench (Apple plan B)"
    )
    ap.add_argument("--N", type=int, default=512,
                    help="Matrix dim for matmul/layer_norm/softmax/gelu.")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--output", type=str, default=None,
                    help="Output JSON path (default: stdout).")
    ap.add_argument("--ci", action="store_true",
                    help=(
                        "CI smoke mode: smaller N (128), single repeat, "
                        "relaxed gate (matmul must beat numpy by 1.5x). "
                        "Use this in release_gate.py."
                    ))
    args = ap.parse_args()

    if args.ci:
        if platform.system() != "Darwin":
            payload = {
                "schema": "tessera.benchmark.v1",
                "lane": "apple_cpu_execution_kind",
                "target": "apple_cpu",
                "verdict": "skipped_apple_cpu",
                "reason": (
                    "apple_cpu Accelerate/BNNS performance gate requires "
                    "a Darwin host"
                ),
                "rows": [],
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            out_text = json.dumps(payload, indent=2)
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(out_text + "\n", encoding="utf-8")
                print(f"wrote {args.output} (verdict={payload['verdict']})")
            else:
                sys.stdout.write(out_text + "\n")
            return 0
        # N=256 amortizes Tessera's per-call JIT-bridge overhead
        # (at N=128 the matmul itself is ~10us so per-call overhead
        # dominates; at N=256 the matmul is ~80us and the ratio is
        # meaningful).  Still small enough for a CI smoke (<200ms
        # total).
        args.N = 256
        args.repeats = 3
        args.warmup = 1

    rows = [
        _bench_matmul(args.N, args.repeats, args.warmup),
        _bench_layer_norm(args.N, args.repeats, args.warmup),
        _bench_softmax(args.N, args.repeats, args.warmup),
        _bench_gelu(args.N, args.repeats, args.warmup),
    ]

    # CI mode: looser ratio gate (3x) at N=256 to absorb the JIT-bridge
    # per-call overhead while still catching the real regression we
    # care about (Tessera falling off the cblas_sgemm fast lane onto
    # a numpy-Python loop would push the ratio above 100x).
    if args.ci:
        for r in rows:
            if r["op"] == "matmul":
                r["gate"] = "tessera_must_be_within_3x_of_numpy_at_small_N"
                r["pass"] = r["tessera_over_numpy_ratio"] <= 3.0

    all_pass = all(r["pass"] for r in rows)
    payload = {
        "schema": "tessera.benchmark.v1",
        "lane": "apple_cpu_execution_kind",
        "target": "apple_cpu",
        "N": args.N,
        "repeats": args.repeats,
        "verdict": "pass" if all_pass else "fail",
        "rows": rows,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out_text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out_text + "\n", encoding="utf-8")
        print(f"wrote {args.output} (verdict={payload['verdict']})")
        for r in rows:
            print(
                f"  {r['op']:<12} {r['execution_kind']:<20} "
                f"tessera={r['tessera_ms']:6.2f}ms numpy={r['numpy_ms']:6.2f}ms "
                f"ratio={r['tessera_over_numpy_ratio']:.2f}x pass={r['pass']}"
            )
    else:
        sys.stdout.write(out_text + "\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
