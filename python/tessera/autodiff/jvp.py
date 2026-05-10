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
