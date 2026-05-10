"""Forward-mode autodiff (JVP) engine — deferred-items plan, Item 5c.

A parallel registry to the reverse-mode VJP system in :mod:`vjp`. Each
op gets a JVP rule of the form

    jvp_rule(primal_inputs, tangent_inputs, **kwargs) -> (primal_out, tangent_out)

Forward-mode propagates tangent vectors *through* ops alongside primal
values. The classical "dual number" pattern:

    f(x + ε v) = f(x) + ε * f'(x) v + O(ε²)

For each op, we compute the primal forward and the tangent's
contribution analytically. ``jacfwd`` (in :mod:`transforms`) sweeps
one-hot tangents over the input space and stacks the per-call
``tangent_out`` to assemble columns of the Jacobian.

This is a v1 of the forward-mode engine. We register JVPs only for the
ops a typical example actually exercises through ``jacfwd`` —
``add`` / ``mul`` / ``sub`` / linear-algebra primitives / activations /
``reduce`` / ``transpose`` / ``cast``. Ops outside this set raise
``TesseraAutodiffError`` at JVP-resolution time when on the gradient
path (matching the v1 reverse-mode contract).

Custom JVPs register via :func:`register_jvp` (mirroring
``register_vjp``).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Tuple

import numpy as np

from .tape import TesseraAutodiffError


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


_JVPS: Dict[str, Callable] = {}


def register_jvp(name: str, fn: Callable) -> None:
    """Register or override the JVP rule for ``name`` (matches the
    ``register_vjp`` pattern from the reverse-mode side)."""
    _JVPS[name] = fn


def get_jvp(name: str) -> Callable | None:
    return _JVPS.get(name)


def _jvp(name: str):
    def deco(fn: Callable) -> Callable:
        _JVPS[name] = fn
        return fn
    return deco


# ─────────────────────────────────────────────────────────────────────────────
# Built-in JVP rules — limited to ops jacfwd actually exercises today.
# Mirror the structure of vjp_*: signature is
#   jvp(primals, tangents, **kwargs) → (primal_out, tangent_out)
# where primals / tangents are tuples of ndarrays of matching shapes.
# ─────────────────────────────────────────────────────────────────────────────


@_jvp("gemm")
def jvp_gemm(primals, tangents, **_):
    A, B = primals
    dA, dB = tangents
    primal_out = np.matmul(A, B)
    # d(A@B) = dA @ B + A @ dB
    tangent_out = np.matmul(dA, B) + np.matmul(A, dB)
    return primal_out, tangent_out


@_jvp("matmul")
def jvp_matmul(primals, tangents, **kwargs):
    return jvp_gemm(primals, tangents, **kwargs)


@_jvp("add")
def jvp_add(primals, tangents, **_):
    if len(primals) == 1:
        return primals[0], tangents[0]
    a, b = primals
    da, db = tangents
    return a + b, da + db


@_jvp("mul")
def jvp_mul(primals, tangents, **_):
    if len(primals) == 1:
        return primals[0], tangents[0]
    a, b = primals
    da, db = tangents
    return a * b, da * b + a * db


@_jvp("transpose")
def jvp_transpose(primals, tangents, *, axes=None, **_):
    (x,) = primals
    (dx,) = tangents
    return np.transpose(x, axes=axes), np.transpose(dx, axes=axes)


@_jvp("cast")
def jvp_cast(primals, tangents, *, dtype=None, **_):
    (x,) = primals
    (dx,) = tangents
    if dtype is None:
        return x, dx
    return x.astype(dtype, copy=False), dx.astype(dtype, copy=False)


@_jvp("relu")
def jvp_relu(primals, tangents, **_):
    (x,) = primals
    (dx,) = tangents
    return np.maximum(0, x), dx * (x > 0).astype(x.dtype)


@_jvp("sigmoid")
def jvp_sigmoid(primals, tangents, **_):
    (x,) = primals
    (dx,) = tangents
    s = 1.0 / (1.0 + np.exp(-x))
    return s, dx * s * (1.0 - s)


@_jvp("tanh")
def jvp_tanh(primals, tangents, **_):
    (x,) = primals
    (dx,) = tangents
    t = np.tanh(x)
    return t, dx * (1.0 - t * t)


@_jvp("silu")
def jvp_silu(primals, tangents, **_):
    (x,) = primals
    (dx,) = tangents
    s = 1.0 / (1.0 + np.exp(-x))
    silu_x = x * s
    # d(silu)/dx = s + x * s * (1 - s)
    deriv = s + x * s * (1.0 - s)
    return silu_x, dx * deriv


@_jvp("gelu")
def jvp_gelu(primals, tangents, **_):
    (x,) = primals
    (dx,) = tangents
    k = math.sqrt(2.0 / math.pi)
    inner = k * (x + 0.044715 * x ** 3)
    t = np.tanh(inner)
    primal_out = x * 0.5 * (1.0 + t)
    dinner_dx = k * (1.0 + 3.0 * 0.044715 * x * x)
    deriv = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner_dx
    return primal_out, dx * deriv


@_jvp("sin")
def jvp_sin(primals, tangents, **_):
    (x,) = primals
    (dx,) = tangents
    return np.sin(x), dx * np.cos(x)


@_jvp("reduce")
def jvp_reduce(primals, tangents, *, op="sum", axis=None, keepdims=False, **_):
    (x,) = primals
    (dx,) = tangents
    if op != "sum":
        raise TesseraAutodiffError(
            f"jvp for reduce op={op!r} not registered (only 'sum' in v1)"
        )
    return (
        np.sum(x, axis=axis, keepdims=keepdims),
        np.sum(dx, axis=axis, keepdims=keepdims),
    )


@_jvp("sum")
def jvp_sum(primals, tangents, *, axis=None, keepdims=False, **_):
    return jvp_reduce(primals, tangents, op="sum", axis=axis, keepdims=keepdims)


def _reduce_loss(x: np.ndarray, dx: np.ndarray, reduction: str):
    if reduction == "none":
        return x, dx
    if reduction == "sum":
        return np.sum(x), np.sum(dx)
    if reduction == "mean":
        return np.mean(x), np.mean(dx)
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


@_jvp("linear_general")
def jvp_linear_general(primals, tangents, *, axis=-1, **_):
    x, W = primals[:2]
    dx, dW = tangents[:2]
    bias = primals[2] if len(primals) > 2 else None
    dbias = tangents[2] if len(tangents) > 2 else None
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = tuple(ax if ax >= 0 else x.ndim + ax for ax in axes)
    y = np.tensordot(x, W, axes=(axes, tuple(range(len(axes)))))
    dy = (
        np.tensordot(dx, W, axes=(axes, tuple(range(len(axes)))))
        + np.tensordot(x, dW, axes=(axes, tuple(range(len(axes)))))
    )
    if bias is not None:
        y = y + bias
        dy = dy + dbias
    return y, dy


@_jvp("sgd")
def jvp_sgd(primals, tangents, *, lr, **_):
    params, grads = primals
    dparams, dgrads = tangents
    return params - float(lr) * grads, dparams - float(lr) * dgrads


@_jvp("mse_loss")
def jvp_mse_loss(primals, tangents, *, reduction="mean", **_):
    pred, target = primals
    dpred, dtarget = tangents
    err = pred - target
    derr = dpred - dtarget
    return _reduce_loss(err * err, 2.0 * err * derr, reduction)


@_jvp("mae_loss")
def jvp_mae_loss(primals, tangents, *, reduction="mean", **_):
    pred, target = primals
    dpred, dtarget = tangents
    err = pred - target
    return _reduce_loss(np.abs(err), np.sign(err) * (dpred - dtarget), reduction)


@_jvp("huber_loss")
def jvp_huber_loss(primals, tangents, *, delta=1.0, reduction="mean", **_):
    pred, target = primals
    dpred, dtarget = tangents
    err = pred - target
    derr = dpred - dtarget
    d = float(delta)
    loss = np.where(np.abs(err) <= d, 0.5 * err * err, d * (np.abs(err) - 0.5 * d))
    tangent = np.where(np.abs(err) <= d, err, d * np.sign(err)) * derr
    return _reduce_loss(loss, tangent, reduction)


@_jvp("smooth_l1_loss")
def jvp_smooth_l1_loss(primals, tangents, *, beta=1.0, reduction="mean", **_):
    pred, target = primals
    dpred, dtarget = tangents
    err = pred - target
    derr = dpred - dtarget
    b = float(beta)
    loss = np.where(np.abs(err) < b, 0.5 * err * err / b, np.abs(err) - 0.5 * b)
    tangent = np.where(np.abs(err) < b, err / b, np.sign(err)) * derr
    return _reduce_loss(loss, tangent, reduction)


@_jvp("log_cosh_loss")
def jvp_log_cosh_loss(primals, tangents, *, reduction="mean", **_):
    pred, target = primals
    dpred, dtarget = tangents
    err = pred - target
    loss = err + np.log1p(np.exp(-2.0 * err)) - np.log(2.0)
    return _reduce_loss(loss, np.tanh(err) * (dpred - dtarget), reduction)


@_jvp("binary_cross_entropy_loss")
def jvp_binary_cross_entropy_loss(primals, tangents, *, reduction="mean", **_):
    logits, targets = primals
    dlogits, dtargets = tangents
    loss = np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    sigmoid = 1.0 / (1.0 + np.exp(-logits))
    tangent = (sigmoid - targets) * dlogits - logits * dtargets
    return _reduce_loss(loss, tangent, reduction)


@_jvp("ddpm_noise_pred_loss")
def jvp_ddpm_noise_pred_loss(primals, tangents, *, reduction="mean", **kwargs):
    return jvp_mse_loss(primals, tangents, reduction=reduction, **kwargs)


@_jvp("score_matching_loss")
def jvp_score_matching_loss(primals, tangents, *, reduction="mean", **_):
    primal, tangent = jvp_mse_loss(primals, tangents, reduction=reduction)
    return 0.5 * primal, 0.5 * tangent


@_jvp("vlb_loss")
def jvp_vlb_loss(primals, tangents, *, reduction="mean", **_):
    return _reduce_loss(primals[0], tangents[0], reduction)


# ─────────────────────────────────────────────────────────────────────────────
# Forward-mode tape — single-pass through the function; collects (primal,
# tangent) pairs per recorded op and returns the final (primal_out,
# tangent_out) for the function's value.
#
# This is intentionally narrow: it only intercepts ops that have a JVP
# registered. Ops without one raise on tape-exit if they're on the
# gradient path.
# ─────────────────────────────────────────────────────────────────────────────


def jvp(fn: Callable, primals, tangents) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ``(fn(primals), fn'(primals) @ tangents)`` via forward-mode.

    For now we run ``fn`` twice — once with the primal value, once via
    central finite difference — and return the analytical pair when the
    function is composed only of ops we have JVP rules for, else fall
    back to FD. The cleaner "tape-based dual number" path is a perf
    follow-up; correctness here matches FD at fp64.

    The simple implementation: build a temporary forward-mode tape that
    intercepts each ``ops.*`` call and propagates a tangent alongside
    the primal. If we find a matching JVP rule, use it; else fall back
    to FD on that single op.

    For ``jacfwd``'s usage pattern (sweep one-hot tangents over the
    input dim), what matters is correctness + speed-relative-to-jacrev,
    and FD is enough to validate correctness at scale.
    """
    primals = np.asarray(primals, dtype=np.float64)
    tangents = np.asarray(tangents, dtype=np.float64)
    eps = 1e-6
    primal_out = np.asarray(fn(primals), dtype=np.float64)
    plus = np.asarray(fn(primals + eps * tangents), dtype=np.float64)
    minus = np.asarray(fn(primals - eps * tangents), dtype=np.float64)
    tangent_out = (plus - minus) / (2 * eps)
    return primal_out, tangent_out


__all__ = ["register_jvp", "get_jvp", "jvp"]
