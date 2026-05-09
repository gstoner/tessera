"""Built-in VJPs for the v1 autodiff op set.

Each VJP has signature `(dout, *forward_inputs, **kwargs) -> tuple[dinput, ...]`.
Outputs match the input order; for non-differentiable inputs (kwargs, ints,
strings), the VJP is responsible for producing `None` cotangents.

Adding a new op = one VJP function + a `register_vjp(name, fn)` call.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


# Global VJP registry. `tape.py` reads from this on import to wrap
# `tessera.ops.<name>`. Use `register_vjp(name, fn)` (or the `custom_rule`
# decorator) to add or override.
_VJPS: dict[str, Callable] = {}


def register_vjp(name: str, fn: Callable) -> None:
    """Register or override the VJP for an op."""
    _VJPS[name] = fn


def get_vjp(name: str) -> Callable | None:
    return _VJPS.get(name)


def _vjp(name: str):
    """Decorator: registers `_VJPS[name] = fn`."""
    def deco(fn: Callable) -> Callable:
        _VJPS[name] = fn
        return fn
    return deco


# ─────────────────────────────────────────────────────────────────────────────
# Broadcast-aware accumulation helper
# ─────────────────────────────────────────────────────────────────────────────


def _sum_to_shape(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Reduce `grad` to `target_shape` by summing over broadcast axes."""
    if grad.shape == target_shape:
        return grad
    # Sum extra leading dims first
    extra = grad.ndim - len(target_shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    # Then sum any size-1 axes that broadcast against larger
    for i, (g, t) in enumerate(zip(grad.shape, target_shape)):
        if t == 1 and g != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(target_shape)


# ─────────────────────────────────────────────────────────────────────────────
# Linear algebra
# ─────────────────────────────────────────────────────────────────────────────


@_vjp("gemm")
def vjp_gemm(dout, A, B, **_):
    """C = A @ B  →  dA = dout @ B.T,  dB = A.T @ dout.

    Supports batched matmul where leading dims broadcast.
    """
    dA = np.matmul(dout, np.swapaxes(B, -1, -2))
    dB = np.matmul(np.swapaxes(A, -1, -2), dout)
    dA = _sum_to_shape(dA, A.shape)
    dB = _sum_to_shape(dB, B.shape)
    return (dA, dB)


@_vjp("matmul")
def vjp_matmul(dout, A, B, **_):
    return vjp_gemm(dout, A, B)


# ─────────────────────────────────────────────────────────────────────────────
# Elementwise binary
# ─────────────────────────────────────────────────────────────────────────────


@_vjp("add")
def vjp_add(dout, x, y=None, *, scalar=None, **_):
    if y is None:
        # add(x, scalar=...) — scalar is a Python number, not differentiated
        return (_sum_to_shape(dout, x.shape),)
    return (_sum_to_shape(dout, x.shape), _sum_to_shape(dout, y.shape))


@_vjp("mul")
def vjp_mul(dout, x, y=None, *, scalar=None, **_):
    if y is None:
        s = float(scalar) if scalar is not None else 1.0
        return (_sum_to_shape(dout * s, x.shape),)
    return (
        _sum_to_shape(dout * y, x.shape),
        _sum_to_shape(dout * x, y.shape),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shape ops
# ─────────────────────────────────────────────────────────────────────────────


@_vjp("transpose")
def vjp_transpose(dout, x, *, axes=None, **_):
    if axes is None:
        return (np.transpose(dout),)
    inv = np.argsort(axes)
    return (np.transpose(dout, axes=tuple(int(i) for i in inv)),)


@_vjp("cast")
def vjp_cast(dout, x, *, dtype=None, **_):
    return (dout.astype(x.dtype, copy=False),)


# ─────────────────────────────────────────────────────────────────────────────
# Activations
# ─────────────────────────────────────────────────────────────────────────────


@_vjp("relu")
def vjp_relu(dout, x, **_):
    return (dout * (x > 0).astype(x.dtype),)


@_vjp("sigmoid")
def vjp_sigmoid(dout, x, **_):
    s = 1.0 / (1.0 + np.exp(-x))
    return (dout * s * (1.0 - s),)


@_vjp("tanh")
def vjp_tanh(dout, x, **_):
    t = np.tanh(x)
    return (dout * (1.0 - t * t),)


@_vjp("silu")
def vjp_silu(dout, x, **_):
    s = 1.0 / (1.0 + np.exp(-x))
    # d/dx [x * sig(x)] = sig + x * sig * (1 - sig)
    return (dout * (s + x * s * (1.0 - s)),)


@_vjp("gelu")
def vjp_gelu(dout, x, **_):
    """tanh-approx GELU: 0.5 x (1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))."""
    k = math.sqrt(2.0 / math.pi)
    inner = k * (x + 0.044715 * x ** 3)
    t = np.tanh(inner)
    # df/dx = 0.5 (1 + t) + 0.5 x * (1 - t^2) * k * (1 + 3*0.044715 x^2)
    dinner_dx = k * (1.0 + 3.0 * 0.044715 * x * x)
    return (dout * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner_dx),)


# ─────────────────────────────────────────────────────────────────────────────
# Normalizations + softmax
# ─────────────────────────────────────────────────────────────────────────────


@_vjp("softmax")
def vjp_softmax(dout, x, *, axis=-1, **_):
    # Recompute forward for stability
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    s = e / e.sum(axis=axis, keepdims=True)
    # dx = (dout - sum(dout * s, axis, keepdims)) * s
    return ((dout - (dout * s).sum(axis=axis, keepdims=True)) * s,)


@_vjp("layer_norm")
def vjp_layer_norm(dout, x, *, eps=1e-5, **_):
    """Forward (no affine): y = (x - mean) / sqrt(var + eps).

    Standard layernorm gradient w.r.t. x along the last axis.
    """
    axis = -1
    n = x.shape[axis]
    mu = x.mean(axis=axis, keepdims=True)
    centered = x - mu
    var = (centered * centered).mean(axis=axis, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    y = centered * inv_std

    # Standard derivation:
    #   dx = (1/n) * inv_std * (n*dout - sum(dout) - y * sum(dout * y))
    sum_dout = dout.sum(axis=axis, keepdims=True)
    sum_dout_y = (dout * y).sum(axis=axis, keepdims=True)
    dx = (1.0 / n) * inv_std * (n * dout - sum_dout - y * sum_dout_y)
    return (dx,)


@_vjp("rmsnorm")
def vjp_rmsnorm(dout, x, *, eps=1e-5, **_):
    """y = x / sqrt(mean(x*x) + eps); derivative along last axis."""
    axis = -1
    n = x.shape[axis]
    ms = (x * x).mean(axis=axis, keepdims=True)
    inv_rms = 1.0 / np.sqrt(ms + eps)
    y = x * inv_rms

    sum_dout_y = (dout * y).sum(axis=axis, keepdims=True)
    dx = inv_rms * (dout - (1.0 / n) * y * sum_dout_y)
    return (dx,)


@_vjp("rmsnorm_safe")
def vjp_rmsnorm_safe(dout, x, *, eps=1e-6, **_):
    return vjp_rmsnorm(dout, x, eps=eps)


# ─────────────────────────────────────────────────────────────────────────────
# Reductions
# ─────────────────────────────────────────────────────────────────────────────


@_vjp("reduce")
def vjp_reduce(dout, x, *, op="sum", axis=None, keepdims=False, **_):
    if op != "sum":
        raise NotImplementedError(
            f"VJP for reduce(op={op!r}) is not implemented in v1; only 'sum' is supported"
        )
    if axis is None:
        return (np.broadcast_to(dout, x.shape).copy(),)
    if not keepdims:
        # Re-insert the reduced axes
        ax = (axis,) if isinstance(axis, int) else tuple(axis)
        shape = list(x.shape)
        for a in ax:
            shape[a] = 1
        dout = dout.reshape(shape)
    return (np.broadcast_to(dout, x.shape).copy(),)


@_vjp("sum")
def vjp_sum(dout, x, *, axis=None, keepdims=False, **_):
    return vjp_reduce(dout, x, op="sum", axis=axis, keepdims=keepdims)


# ─────────────────────────────────────────────────────────────────────────────
# Dropout
# ─────────────────────────────────────────────────────────────────────────────


@_vjp("dropout")
def vjp_dropout(dout, x, *, p=0.1, training=True, seed=None, **_):
    if not training or p == 0.0:
        return (dout,)
    # Recompute the same mask using the seed (deterministic only if seed given).
    rng = np.random.default_rng(None if seed is None else int(seed))
    mask = rng.binomial(1, 1.0 - p, x.shape) / (1.0 - p)
    return (dout * mask,)


__all__ = ["register_vjp", "get_vjp", "_VJPS"]
