"""Public profiling/autotuning facade for Tessera."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from .compiler.autotune_v2 import BayesianAutotuner, GEMMWorkload, TuningConfig, TuningResult


DEFAULT_CACHE_DIR = Path(os.environ.get("TESSERA_CACHE_DIR", Path.home() / ".tessera")) / "autotune"
DEFAULT_CACHE_PATH = DEFAULT_CACHE_DIR / "tuning_cache.db"


@dataclass(frozen=True)
class CostEstimate:
    """Analytical cost-model estimate."""

    latency_ms: float
    compute_ms: float
    memory_ms: float
    arithmetic_intensity: float
    bound: str


class RooflineCostModel:
    """Simple FLOPs/bytes roofline estimator used by Schedule IR planning."""

    def __init__(self, *, peak_tflops: float = 312.0, bandwidth_gbps: float = 2_000.0) -> None:
        if peak_tflops <= 0 or bandwidth_gbps <= 0:
            raise ValueError("peak_tflops and bandwidth_gbps must be > 0")
        self.peak_tflops = peak_tflops
        self.bandwidth_gbps = bandwidth_gbps

    def estimate(self, *, flops: float, bytes_moved: float) -> CostEstimate:
        if flops < 0 or bytes_moved < 0:
            raise ValueError("flops and bytes_moved must be >= 0")
        compute_ms = flops / (self.peak_tflops * 1e12) * 1e3 if flops else 0.0
        memory_ms = bytes_moved / (self.bandwidth_gbps * 1e9) * 1e3 if bytes_moved else 0.0
        ai = flops / bytes_moved if bytes_moved else float("inf")
        return CostEstimate(
            latency_ms=max(compute_ms, memory_ms),
            compute_ms=compute_ms,
            memory_ms=memory_ms,
            arithmetic_intensity=ai,
            bound="compute" if compute_ms >= memory_ms else "memory",
        )


def autotune(
    op: Callable | str,
    *,
    shapes: Sequence[int],
    dtype: str = "bf16",
    arch: str = "generic",
    layout: str = "row_major",
    numeric_policy: Mapping[str, object] | None = None,
    movement: Mapping[str, object] | None = None,
    method: str = "roofline",
    backend: str | None = None,
    max_trials: int = 20,
    cache_path: str | os.PathLike | None = None,
    peak_tflops: float = 312.0,
) -> TuningResult:
    """Tune an operation for a shape and return the best configuration.

    Phase 1 supports GEMM-like shapes ``(M, N, K)`` and reuses the existing
    Bayesian/grid autotuner. ``method="on_device"`` is accepted as the future
    measurement mode; until runtime kernels are wired, it uses the same
    synthetic/roofline evaluator and records the method in the cache wrapper.
    """

    if method not in ("roofline", "on_device", "grid", "bayesian"):
        raise ValueError("method must be roofline, on_device, grid, or bayesian")
    workload = _workload_from(op, shapes, dtype, arch=arch, layout=layout, movement=movement)
    path = _cache_path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tuner = BayesianAutotuner(workload, peak_tflops=peak_tflops)
    tuner.warm_start_from_cache(str(path))
    result = tuner.tune(max_trials=max_trials, method=method)
    if method == "on_device" and backend in {"cpu", "apple_cpu"}:
        result = _measure_gemm_wall_clock(result, shapes, dtype=dtype, backend=backend)
        tuner._results.append(result)
        tuner._best = result
    tuner.save_to_cache(str(path))
    return result


def load(
    op: Callable | str,
    shapes: Sequence[int],
    *,
    dtype: str = "bf16",
    arch: str = "generic",
    layout: str = "row_major",
    movement: Mapping[str, object] | None = None,
    cache_path: str | os.PathLike | None = None,
) -> Optional[TuningResult]:
    """Load the best cached tuning result for an operation/shape."""

    workload = _workload_from(op, shapes, dtype, arch=arch, layout=layout, movement=movement)
    tuner = BayesianAutotuner(workload)
    loaded = tuner.warm_start_from_cache(str(_cache_path(cache_path)))
    if loaded == 0:
        return None
    return tuner.best


def cache_key(
    op: Callable | str,
    shapes: Sequence[int],
    *,
    dtype: str = "bf16",
    arch: str = "generic",
    layout: str = "row_major",
    numeric_policy: Mapping[str, object] | None = None,
    movement: Mapping[str, object] | None = None,
) -> tuple:
    """Stable public cache key tuple including compiler-relevant tuning inputs."""

    policy = dict(numeric_policy or {"storage": dtype, "accum": "f32" if dtype != "int8" else "s32"})
    move = dict(movement or {"prefetch": "auto", "overlap": "compute"})
    return (
        _op_name(op),
        tuple(int(s) for s in shapes),
        dtype,
        arch,
        layout,
        json.dumps(policy, sort_keys=True),
        json.dumps(move, sort_keys=True),
    )


def schedule_artifact(
    result: TuningResult,
    *,
    op: Callable | str,
    shapes: Sequence[int],
    dtype: str = "bf16",
    arch: str = "generic",
    layout: str = "row_major",
    movement: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Create a lightweight public schedule artifact from a tuning result."""

    workload = _workload_from(op, shapes, dtype, arch=arch, layout=layout, movement=movement)
    tuner = BayesianAutotuner(workload)
    tuner._results.append(result)
    tuner._best = result
    return tuner.schedule_artifact(arch=arch)


def _workload_from(
    op: Callable | str,
    shapes: Sequence[int],
    dtype: str,
    *,
    arch: str = "generic",
    layout: str = "row_major",
    movement: Mapping[str, object] | None = None,
) -> GEMMWorkload:
    name = _op_name(op)
    if name not in ("matmul", "gemm", "op.matmul", "tessera.matmul", "tessera.ops.matmul"):
        raise ValueError(f"public autotune currently supports GEMM/matmul, got {name!r}")
    if len(shapes) != 3:
        raise ValueError("GEMM autotune expects shapes=(M, N, K)")
    M, N, K = (int(v) for v in shapes)
    return GEMMWorkload(M=M, N=N, K=K, dtype=dtype, arch=arch, layout=layout, movement=dict(movement or {"prefetch": "auto", "overlap": "compute"}))


def _op_name(op: Callable | str) -> str:
    if isinstance(op, str):
        return op
    return getattr(op, "__name__", repr(op))


def _cache_path(path: str | os.PathLike | None) -> Path:
    return Path(path) if path is not None else DEFAULT_CACHE_PATH


def _measure_gemm_wall_clock(
    result: TuningResult,
    shapes: Sequence[int],
    *,
    dtype: str,
    backend: str,
) -> TuningResult:
    M, N, K = (int(v) for v in shapes)
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K), dtype=np.float32)
    b = rng.standard_normal((K, N), dtype=np.float32)
    warmups = 1
    repeats = 3
    for _ in range(warmups):
        _ = a @ b
    start = time.perf_counter()
    for _ in range(repeats):
        _ = a @ b
    latency_ms = (time.perf_counter() - start) * 1e3 / repeats
    flops = 2.0 * M * N * K
    tflops = flops / max(latency_ms * 1e-3, 1e-12) / 1e12
    return TuningResult(
        result.config,
        latency_ms,
        tflops,
        time.time(),
        result.trial_id,
        "ok",
        f"{backend} wall-clock measurement; dtype request {dtype!r} measured with fp32 reference arrays",
        "on_device",
    )


__all__ = [
    "CostEstimate",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_CACHE_PATH",
    "RooflineCostModel",
    "TuningConfig",
    "TuningResult",
    "autotune",
    "cache_key",
    "load",
    "schedule_artifact",
]
