"""Compute backends for the Gumiho draft model.

Two interchangeable backends expose the same small op surface so the *exact
same* model/draft code runs on either:

* :class:`NumpyBackend` — float64 reference. The ground truth the demo
  validates against.
* :class:`AppleBackend` — routes the heavy linear algebra through
  ``@tessera.jit(target="apple_gpu")`` functions composed from ``tessera.ops``
  (matmul / linear_general / rmsnorm / silu_mul / relu / softmax). On a Mac
  with Metal these execute on the GPU (``execution_mode="metal_runtime"``);
  off Darwin the jit path degrades to numpy so the demo still runs everywhere.
  ``target="apple_cpu"`` selects the Accelerate path instead.

Only the dense kernels go through ``tessera.ops``; cheap host glue (the RMSNorm
gamma scale, reshapes, concat, the attention score scale) stays in numpy — the
same split the ``test_apple_gpu_batched_mha.py`` block uses.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import tessera as ts
from tessera import runtime as R


# ── module-level jitted ops (so @jit can introspect their source) ────────────
@ts.jit(target="apple_gpu")
def _gpu_matmul(a, b):
    return ts.ops.matmul(a, b)


@ts.jit(target="apple_gpu")
def _gpu_linear(x, w):
    return ts.ops.linear_general(x, w)


@ts.jit(target="apple_gpu")
def _gpu_rmsnorm(x):
    return ts.ops.rmsnorm(x)          # unweighted; gamma applied host-side


@ts.jit(target="apple_gpu")
def _gpu_silu_mul(a, b):
    return ts.ops.silu_mul(a, b)


@ts.jit(target="apple_gpu")
def _gpu_relu(x):
    return ts.ops.relu(x)


@ts.jit(target="apple_gpu")
def _gpu_softmax(x):
    return ts.ops.softmax(x)


# ── CPU-target variants (Accelerate) for the matmul-shaped ops ───────────────
@ts.jit(target="apple_cpu")
def _cpu_matmul(a, b):
    return ts.ops.matmul(a, b)


@ts.jit(target="apple_cpu")
def _cpu_linear(x, w):
    return ts.ops.linear_general(x, w)


class NumpyBackend:
    """float64 reference backend."""

    name = "numpy"

    def __init__(self, eps: float = 1e-5) -> None:
        self.eps = float(eps)

    def linear(self, x: Any, w: Any) -> np.ndarray:
        return np.asarray(x, np.float64) @ np.asarray(w, np.float64)

    def matmul(self, a: Any, b: Any) -> np.ndarray:
        return np.asarray(a, np.float64) @ np.asarray(b, np.float64)

    def rmsnorm(self, x: Any, gamma: Any) -> np.ndarray:
        d = np.asarray(x, np.float64)
        n = d / np.sqrt((d * d).mean(-1, keepdims=True) + self.eps)
        return n * np.asarray(gamma, np.float64)

    def silu_mul(self, a: Any, b: Any) -> np.ndarray:
        a = np.asarray(a, np.float64)
        return (a / (1.0 + np.exp(-a))) * np.asarray(b, np.float64)

    def relu(self, x: Any) -> np.ndarray:
        return np.maximum(np.asarray(x, np.float64), 0.0)

    def softmax(self, x: Any) -> np.ndarray:
        z = np.asarray(x, np.float64)
        z = z - z.max(-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(-1, keepdims=True)


class AppleBackend:
    """Apple compiler-backend path: dense ops via ``tessera.ops`` (GPU or CPU)."""

    def __init__(self, *, target: str = "apple_gpu", eps: float = 1e-5) -> None:
        if target not in ("apple_gpu", "apple_cpu"):
            raise ValueError("target must be apple_gpu or apple_cpu")
        self.target = target
        self.eps = float(eps)
        if target == "apple_gpu":
            self.name = "metal" if R.DeviceTensor.is_metal() else "numpy"
        else:
            self.name = "accelerate" if R.DeviceTensor.is_metal() else "numpy"

    def _matmul(self, a, b):
        fn = _gpu_matmul if self.target == "apple_gpu" else _cpu_matmul
        return np.asarray(fn(np.ascontiguousarray(a, np.float32),
                             np.ascontiguousarray(b, np.float32)))

    def _linear(self, x, w):
        fn = _gpu_linear if self.target == "apple_gpu" else _cpu_linear
        return np.asarray(fn(np.ascontiguousarray(x, np.float32),
                             np.ascontiguousarray(w, np.float32)))

    def linear(self, x: Any, w: Any) -> np.ndarray:
        return self._linear(x, w)

    def matmul(self, a: Any, b: Any) -> np.ndarray:
        return self._matmul(a, b)

    def rmsnorm(self, x: Any, gamma: Any) -> np.ndarray:
        # Heavy normalize on-device (unweighted); gamma scale is host glue.
        if self.target == "apple_gpu":
            n = np.asarray(_gpu_rmsnorm(np.ascontiguousarray(x, np.float32)))
        else:
            d = np.asarray(x, np.float32)
            n = d / np.sqrt((d * d).mean(-1, keepdims=True) + self.eps)
        return n * np.asarray(gamma, np.float32)

    def silu_mul(self, a: Any, b: Any) -> np.ndarray:
        if self.target == "apple_gpu":
            return np.asarray(_gpu_silu_mul(np.ascontiguousarray(a, np.float32),
                                            np.ascontiguousarray(b, np.float32)))
        a = np.asarray(a, np.float32)
        return (a / (1.0 + np.exp(-a))) * np.asarray(b, np.float32)

    def relu(self, x: Any) -> np.ndarray:
        if self.target == "apple_gpu":
            return np.asarray(_gpu_relu(np.ascontiguousarray(x, np.float32)))
        return np.maximum(np.asarray(x, np.float32), 0.0)

    def softmax(self, x: Any) -> np.ndarray:
        if self.target == "apple_gpu":
            return np.asarray(_gpu_softmax(np.ascontiguousarray(x, np.float32)))
        z = np.asarray(x, np.float32)
        z = z - z.max(-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(-1, keepdims=True)


def make_backend(kind: str, eps: float = 1e-5):
    """``kind`` in {"numpy", "apple_gpu", "apple_cpu"}."""
    if kind == "numpy":
        return NumpyBackend(eps=eps)
    return AppleBackend(target=kind, eps=eps)
