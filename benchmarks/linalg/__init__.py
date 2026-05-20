"""Tessera linalg reference benchmark suite.

Tiny CPU-reference cover for the four linalg primitives that ship
with autodiff: ``cholesky``, ``qr``, ``svd``, ``tri_solve``.

Each row produces a JSON envelope that matches
``benchmarks/benchmark_gemm.py`` so ``benchmarks/run_all.py`` can
ingest it.  Status today: **reference / artifact** — the numerical
result matches numpy to float-precision tolerance; native (Apple GPU /
NVIDIA / ROCm) lowering for these ops is a future M-series milestone.
"""

from __future__ import annotations

__all__ = ()
