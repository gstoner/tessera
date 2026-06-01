"""Phase 2.1c — tessera.ops.* interception under @auto_batch.

Closes the namespace-switching gap: users now write
``tessera.ops.rmsnorm(x, gamma=g, rows=B*S, cols=D, eps=eps)``
INSIDE an ``@auto_batch`` block and the call routes through the
encode-session trace-capture path automatically. Outside the trace
(eager mode), the same call falls through to the existing
``tessera.ops.*`` numpy reference.

Design choice: each intercepted op accepts the apple_gpu_ops-style
kwargs as OPTIONAL parameters. When a trace is active AND the
required kwargs are present, the wrapper routes to
``apple_gpu_ops.<op>`` and returns a ``TraceRef``. When no trace
or insufficient kwargs, the wrapper falls back to the original
``tessera.ops.<op>`` (now extended where natural — e.g.,
``rmsnorm`` now accepts an optional ``gamma`` even in eager mode).

This is a thin layer over the existing surface; backward-compat with
``tessera.ops.<op>(x, eps=...)`` callers is preserved because the
new kwargs default to ``None``.

Coverage: the 8 encode-eligible ops — rmsnorm / layer_norm / softmax
/ bmm / rope / silu / gelu / flash_attn. (matmul/gemm/batched_gemm
all alias to apple_gpu_ops.bmm semantics when inputs allow.)
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional

import numpy as np

from . import apple_gpu_ops as _agpu
from .apple_gpu_ops import _active_trace


def _ndarray_shape(x: Any) -> Optional[tuple[int, ...]]:
    """Best-effort shape extraction for numpy + Tensor-wrapped inputs.
    DeviceTensor + TraceRef return None (no shape carrier today)."""
    if hasattr(x, "_data"):
        x = x._data
    if isinstance(x, np.ndarray):
        return x.shape
    return getattr(x, "shape", None) if hasattr(x, "shape") else None


def _infer_rows_cols(x: Any) -> tuple[Optional[int], Optional[int]]:
    """Infer (rows, cols) for row-op shape: flatten all but the last
    axis into rows, last axis is cols. Returns (None, None) if
    shape isn't recoverable."""
    shape = _ndarray_shape(x)
    if shape is None or len(shape) < 1:
        return None, None
    if len(shape) == 1:
        return 1, int(shape[0])
    return int(np.prod(shape[:-1])), int(shape[-1])


# ---- Per-op intercepting wrappers --------------------------------------

def _wrap_rmsnorm(original_fn: Callable) -> Callable:
    @functools.wraps(original_fn)
    def rmsnorm(x, gamma=None, *, eps: float = 1e-5,
                rows: Optional[int] = None, cols: Optional[int] = None,
                dtype: str = "f32"):
        # Trace-mode interception: route to apple_gpu_ops when active.
        if gamma is not None and _active_trace() is not None:
            if rows is None or cols is None:
                r, c = _infer_rows_cols(x)
                rows = rows if rows is not None else r
                cols = cols if cols is not None else c
            if rows is None or cols is None:
                raise ValueError(
                    "tessera.ops.rmsnorm under @auto_batch needs rows + "
                    "cols (or an x with inferable shape); got x of type "
                    f"{type(x).__name__}")
            return _agpu.rmsnorm(x, gamma, rows=rows, cols=cols, eps=eps,
                                  dtype=dtype)
        # Eager path: extend the numpy reference with optional gamma.
        if hasattr(x, "_data"):
            x = x._data
        out = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
        if gamma is not None:
            out = out * gamma
        return out
    return rmsnorm


def _wrap_layer_norm(original_fn: Callable) -> Callable:
    @functools.wraps(original_fn)
    def layer_norm(x, gamma=None, beta=None, *, eps: float = 1e-5,
                   rows: Optional[int] = None,
                   cols: Optional[int] = None,
                   dtype: str = "f32"):
        if gamma is not None and beta is not None and _active_trace() is not None:
            if rows is None or cols is None:
                r, c = _infer_rows_cols(x)
                rows = rows if rows is not None else r
                cols = cols if cols is not None else c
            if rows is None or cols is None:
                raise ValueError(
                    "tessera.ops.layer_norm under @auto_batch needs "
                    "rows + cols")
            return _agpu.layer_norm(x, gamma, beta,
                                     rows=rows, cols=cols, eps=eps,
                                     dtype=dtype)
        # Eager path — match original_fn behavior + optional gamma/beta.
        if hasattr(x, "_data"):
            x = x._data
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        out = (x - mean) / np.sqrt(var + eps)
        if gamma is not None:
            out = out * gamma
        if beta is not None:
            out = out + beta
        return out
    return layer_norm


def _wrap_softmax(original_fn: Callable) -> Callable:
    @functools.wraps(original_fn)
    def softmax(x, *, axis: int = -1, rows: Optional[int] = None,
                cols: Optional[int] = None, dtype: str = "f32"):
        if _active_trace() is not None and axis in (-1, len(_ndarray_shape(x) or ()) - 1):
            # Only last-axis softmax routes to apple_gpu_ops.
            if rows is None or cols is None:
                r, c = _infer_rows_cols(x)
                rows = rows if rows is not None else r
                cols = cols if cols is not None else c
            if rows is not None and cols is not None:
                return _agpu.softmax(x, rows=rows, cols=cols, dtype=dtype)
        return original_fn(x, axis=axis)
    return softmax


def _wrap_bmm(original_fn: Callable) -> Callable:
    @functools.wraps(original_fn)
    def bmm(A, B, *, batch: Optional[int] = None,
            M: Optional[int] = None, N: Optional[int] = None,
            K: Optional[int] = None, b_broadcast: bool = False,
            dtype: str = "f32"):
        if (_active_trace() is not None
                and batch is not None and M is not None
                and N is not None and K is not None):
            return _agpu.bmm(A, B,
                              batch=batch, M=M, N=N, K=K,
                              b_broadcast=b_broadcast, dtype=dtype)
        # Eager fallback — original_fn handles (A, B[, epilogue]).
        return original_fn(A, B)
    return bmm


def _wrap_rope(original_fn: Optional[Callable]) -> Callable:
    @functools.wraps(original_fn or (lambda *a, **k: None))
    def rope(X, Theta, *, M: Optional[int] = None,
             K: Optional[int] = None, dtype: str = "f32"):
        if _active_trace() is not None:
            if M is None or K is None:
                shape = _ndarray_shape(X)
                if shape is not None and len(shape) >= 1:
                    if M is None:
                        M = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
                    if K is None:
                        K = int(shape[-1])
            if M is None or K is None:
                raise ValueError("rope under @auto_batch needs M + K")
            return _agpu.rope(X, Theta, M=M, K=K, dtype=dtype)
        # Eager fallback — apply the same pair-wise rotation in numpy.
        # The original tessera.ops doesn't ship rope today; provide a
        # reference here.
        if hasattr(X, "_data"):
            X = X._data
        if hasattr(Theta, "_data"):
            Theta = Theta._data
        X_flat = X.reshape(-1)
        T_flat = Theta.reshape(-1)
        n = X_flat.size
        if K is None:
            shape = X.shape
            K = int(shape[-1])
        if M is None:
            M = n // K
        out = np.empty_like(X_flat)
        for m in range(M):
            for pair in range(K // 2):
                ie = m * K + pair * 2
                io = ie + 1
                c, s = np.cos(T_flat[ie]), np.sin(T_flat[ie])
                out[ie] = X_flat[ie] * c - X_flat[io] * s
                out[io] = X_flat[ie] * s + X_flat[io] * c
        return out.reshape(X.shape)
    return rope


def _wrap_silu(original_fn: Optional[Callable]) -> Callable:
    @functools.wraps(original_fn or (lambda *a, **k: None))
    def silu(x, *, n: Optional[int] = None, dtype: str = "f32"):
        if _active_trace() is not None:
            if n is None:
                shape = _ndarray_shape(x)
                if shape is not None:
                    n = int(np.prod(shape))
            if n is None:
                raise ValueError("silu under @auto_batch needs n")
            return _agpu.silu(x, n=n, dtype=dtype)
        # Eager fallback.
        if hasattr(x, "_data"):
            x = x._data
        return x / (1.0 + np.exp(-x))
    return silu


def _wrap_gelu(original_fn: Callable) -> Callable:
    @functools.wraps(original_fn)
    def gelu(x, *, n: Optional[int] = None, dtype: str = "f32"):
        if _active_trace() is not None:
            if n is None:
                shape = _ndarray_shape(x)
                if shape is not None:
                    n = int(np.prod(shape))
            if n is None:
                raise ValueError("gelu under @auto_batch needs n")
            return _agpu.gelu(x, n=n, dtype=dtype)
        return original_fn(x)
    return gelu


def _wrap_flash_attn(original_fn: Optional[Callable]) -> Callable:
    @functools.wraps(original_fn or (lambda *a, **k: None))
    def flash_attn(Q, K, V, *, B: Optional[int] = None,
                    Sq: Optional[int] = None, Sk: Optional[int] = None,
                    D: Optional[int] = None,
                    scale: Optional[float] = None,
                    causal: bool = False, dtype: str = "f32"):
        if _active_trace() is not None:
            if (B is None or Sq is None or Sk is None or D is None):
                # Infer from Q/K shapes (B, S, D).
                q_shape = _ndarray_shape(Q)
                k_shape = _ndarray_shape(K)
                if q_shape is not None and len(q_shape) == 3:
                    B = B if B is not None else int(q_shape[0])
                    Sq = Sq if Sq is not None else int(q_shape[1])
                    D = D if D is not None else int(q_shape[2])
                if k_shape is not None and len(k_shape) == 3:
                    Sk = Sk if Sk is not None else int(k_shape[1])
            if B is None or Sq is None or Sk is None or D is None:
                raise ValueError(
                    "flash_attn under @auto_batch needs B + Sq + Sk + D")
            return _agpu.flash_attn(Q, K, V,
                                     B=B, Sq=Sq, Sk=Sk, D=D,
                                     scale=scale, causal=causal,
                                     dtype=dtype)
        if original_fn is not None:
            return original_fn(Q, K, V)
        raise NotImplementedError(
            "tessera.ops.flash_attn has no eager numpy reference; only "
            "apple_gpu_ops trace path is supported")
    return flash_attn


# ---- Install entry point ----------------------------------------------

_INSTALLED = False


def install_apple_gpu_interception(ops_namespace) -> None:
    """Replace the 8 encode-eligible ops in ``ops_namespace`` with
    intercepting wrappers. Idempotent — calling again is a no-op."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    def _get(name: str) -> Optional[Callable]:
        return getattr(ops_namespace, name, None)

    # rmsnorm + layer_norm + softmax + gelu are guaranteed present
    # (Phase 1 numpy stubs). The others (rope, silu, flash_attn,
    # bmm) may or may not be — the wrappers handle missing original
    # functions gracefully.
    ops_namespace.rmsnorm = _wrap_rmsnorm(_get("rmsnorm"))
    ops_namespace.layer_norm = _wrap_layer_norm(_get("layer_norm"))
    ops_namespace.softmax = _wrap_softmax(_get("softmax"))
    ops_namespace.gelu = _wrap_gelu(_get("gelu"))
    ops_namespace.bmm = _wrap_bmm(_get("bmm") or _get("matmul") or _get("gemm"))
    ops_namespace.rope = _wrap_rope(_get("rope"))
    ops_namespace.silu = _wrap_silu(_get("silu"))
    ops_namespace.flash_attn = _wrap_flash_attn(_get("flash_attn"))


__all__ = [
    "install_apple_gpu_interception",
]
