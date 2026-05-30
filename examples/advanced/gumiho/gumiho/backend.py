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

    def __init__(self, *, target: str = "apple_gpu", eps: float = 1e-5,
                 compute_dtype: str = "f32") -> None:
        if target not in ("apple_gpu", "apple_cpu"):
            raise ValueError("target must be apple_gpu or apple_cpu")
        self.target = target
        self.eps = float(eps)
        # Half precision is wired for the apple_gpu path only (native f16 MSL +
        # fp32 accumulation; bf16 host-upcast). apple_cpu stays f32.
        self._ct = _resolve_dtype(compute_dtype) if target == "apple_gpu" else np.float32
        self.compute_dtype = compute_dtype if self._ct is not np.float32 else "f32"
        suffix = "" if self._ct is np.float32 else f"-{self.compute_dtype}"
        if target == "apple_gpu":
            self.name = ("metal" if R.DeviceTensor.is_metal() else "numpy") + suffix
        else:
            self.name = "accelerate" if R.DeviceTensor.is_metal() else "numpy"

    def _cast(self, a):
        # A masked attention score (-1e30) overflows half to -inf — which is the
        # intended "masked" value for the downstream softmax — so silence the
        # benign overflow warning on the cast.
        with np.errstate(over="ignore"):
            return np.ascontiguousarray(np.asarray(a), self._ct)

    def _matmul(self, a, b):
        fn = _gpu_matmul if self.target == "apple_gpu" else _cpu_matmul
        return np.asarray(fn(self._cast(a), self._cast(b)))

    def _linear(self, x, w):
        fn = _gpu_linear if self.target == "apple_gpu" else _cpu_linear
        return np.asarray(fn(self._cast(x), self._cast(w)))

    def linear(self, x: Any, w: Any) -> np.ndarray:
        return self._linear(x, w)

    def matmul(self, a: Any, b: Any) -> np.ndarray:
        return self._matmul(a, b)

    def rmsnorm(self, x: Any, gamma: Any) -> np.ndarray:
        # Heavy normalize on-device (unweighted, fp32 internal); gamma is host glue.
        if self.target == "apple_gpu":
            n = np.asarray(_gpu_rmsnorm(self._cast(x)))
        else:
            d = np.asarray(x, np.float32)
            n = d / np.sqrt((d * d).mean(-1, keepdims=True) + self.eps)
        return (n.astype(self._ct) * np.asarray(gamma).astype(self._ct))

    def silu_mul(self, a: Any, b: Any) -> np.ndarray:
        if self.target == "apple_gpu":
            return np.asarray(_gpu_silu_mul(self._cast(a), self._cast(b)))
        a = np.asarray(a, np.float32)
        return (a / (1.0 + np.exp(-a))) * np.asarray(b, np.float32)

    def relu(self, x: Any) -> np.ndarray:
        if self.target == "apple_gpu":
            return np.asarray(_gpu_relu(self._cast(x)))
        return np.maximum(np.asarray(x, np.float32), 0.0)

    def softmax(self, x: Any) -> np.ndarray:
        if self.target == "apple_gpu":
            return np.asarray(_gpu_softmax(self._cast(x)))
        z = np.asarray(x, np.float32)
        z = z - z.max(-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(-1, keepdims=True)


def _resolve_dtype(name: str):
    if name in (None, "f32", "fp32", "float32"):
        return np.float32
    if name in ("f16", "fp16", "float16"):
        return np.float16
    if name in ("bf16", "bfloat16"):
        import ml_dtypes
        return ml_dtypes.bfloat16
    raise ValueError(f"unsupported compute_dtype {name!r}")


def make_backend(kind: str, eps: float = 1e-5, compute_dtype: str = "f32"):
    """``kind`` in {"numpy", "apple_gpu", "apple_cpu"}."""
    if kind == "numpy":
        return NumpyBackend(eps=eps)
    return AppleBackend(target=kind, eps=eps, compute_dtype=compute_dtype)
