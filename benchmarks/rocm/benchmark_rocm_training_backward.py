"""Operation-total gfx1151 timing for standalone compiled training VJPs.

The current public runtime ABI owns allocation, copies, compilation-cache
lookup, launch, synchronization, and result copies, so these numbers are
deliberately labelled ``operation_total_ms``.  They are not kernel-only
selector evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import tessera as ts  # noqa: E402
from tessera import runtime as rt  # noqa: E402


@ts.jit(
    target="rocm",
    autodiff="reverse",
    wrt=("parameter", "gradient", "moment1", "moment2"),
)
def _adamw(parameter, gradient, moment1, moment2):
    return ts.ops.adamw(
        parameter,
        gradient,
        moment1,
        moment2,
        lr=0.003,
        beta1=0.8,
        beta2=0.95,
        eps=1.0e-6,
        weight_decay=0.02,
        step=3,
    )


@ts.jit(
    target="rocm",
    autodiff="reverse",
    wrt=("parameter", "gradient", "moment"),
)
def _lion(parameter, gradient, moment):
    return ts.ops.lion(
        parameter,
        gradient,
        moment,
        lr=0.004,
        beta1=0.8,
        beta2=0.93,
        weight_decay=0.025,
    )


@ts.jit(target="rocm", autodiff="reverse", wrt=("log_p", "q"))
def _kl(log_p, q):
    return ts.ops.kl_divergence(
        log_p, q, reduction="none", axis=1, epsilon=1.0e-6
    )


@ts.jit(target="rocm", autodiff="reverse", wrt=("p", "q"))
def _js(p, q):
    return ts.ops.js_divergence(
        p, q, reduction="mean", axis=-1, epsilon=1.0e-6
    )


def _measure(call, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        call()
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return samples


def _row(
    name: str,
    shape: tuple[int, ...],
    call,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    samples = _measure(call, warmup, iterations)
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return {
        "name": name,
        "shape": list(shape),
        "dtype": "f32",
        "clock": "time.perf_counter_ns",
        "timing_scope": "operation_total_ms",
        "warmup": warmup,
        "iterations": iterations,
        "median_ms": statistics.median(samples),
        "p95_ms": p95,
        "samples_ms": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be nonnegative and iterations positive")
    if not rt._rocm_wmma_runtime_available():
        raise SystemExit("gfx1151 HIP runtime unavailable")

    rng = np.random.default_rng(20260727)
    shape = (257, 255)
    parameter = rng.normal(size=shape).astype(np.float32)
    gradient = rng.normal(scale=0.2, size=shape).astype(np.float32)
    moment1 = rng.normal(scale=0.1, size=shape).astype(np.float32)
    moment2 = rng.uniform(0.05, 0.25, size=shape).astype(np.float32)
    output_cotangents = tuple(
        rng.normal(scale=0.3, size=shape).astype(np.float32)
        for _ in range(3)
    )
    lion_cotangents = output_cotangents[:2]
    probability = rng.uniform(0.05, 0.95, size=(17, 31, 13)).astype(
        np.float32
    )
    log_p = np.log(probability).astype(np.float32)
    q = rng.uniform(0.05, 0.95, size=probability.shape).astype(np.float32)
    kl_cotangent = rng.normal(size=(17, 13)).astype(np.float32)
    js_p = probability.reshape(527, 13)
    js_q = q.reshape(527, 13)
    scalar_cotangent = np.asarray(1.25, np.float32)

    rows = [
        _row(
            "adamw_vjp",
            shape,
            lambda: _adamw.native_backward(
                parameter,
                gradient,
                moment1,
                moment2,
                out_cotangents=output_cotangents,
            ),
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        _row(
            "lion_vjp",
            shape,
            lambda: _lion.native_backward(
                parameter,
                gradient,
                moment1,
                out_cotangents=lion_cotangents,
            ),
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        _row(
            "kl_vjp_axis_1",
            probability.shape,
            lambda: _kl.native_backward(
                log_p, q, out_cotangents=kl_cotangent
            ),
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        _row(
            "js_vjp_mean",
            js_p.shape,
            lambda: _js.native_backward(
                js_p, js_q, out_cotangents=scalar_cotangent
            ),
            warmup=args.warmup,
            iterations=args.iterations,
        ),
    ]
    record = {
        "schema": "tessera.rocm.training_backward.benchmark.v1",
        "device": "gfx1151",
        "timing_scope": (
            "allocation+copies+cache_lookup+launch+synchronize+result_copies"
        ),
        "selector_eligible": False,
        "rows": rows,
    }
    text = json.dumps(record, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
