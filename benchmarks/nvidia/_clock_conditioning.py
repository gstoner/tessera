"""Shared sm_120 clock-conditioning helper for tiny-kernel benchmarks."""
from __future__ import annotations

import numpy as np


def condition_sm120(*, reps: int = 200) -> float:
    """Run a resident 1024^3 TF32 GEMM batch before microsecond-scale timing.

    Tiny kernels do not reliably leave WSL's P8 idle state by themselves. This
    bounded device-only workload raises clocks without entering the measured
    interval. The returned value is diagnostic milliseconds per GEMM.
    """
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession

    rng = np.random.default_rng(20260716)
    a = (rng.standard_normal((1024, 1024)) * .01).astype(np.float32)
    b = (rng.standard_normal((1024, 1024)) * .01).astype(np.float32)
    with NvidiaDeviceSession() as session:
        da, db = session.upload(a), session.upload(b)
        out = session.empty((1024, 1024), np.float32)
        return session.measure(
            lambda: session.gemm(da, db, out, "float32"),
            reps=reps, warmup=5)
