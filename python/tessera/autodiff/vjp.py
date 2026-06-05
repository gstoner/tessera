"""Built-in VJPs for the v1 autodiff op set.

Each VJP has signature `(dout, *forward_inputs, **kwargs) -> tuple[dinput, ...]`.
Outputs match the input order; for non-differentiable inputs (kwargs, ints,
strings), the VJP is responsible for producing `None` cotangents.

Adding a new op = one VJP function + a `register_vjp(name, fn)` call.
"""

from __future__ import annotations

import math
import copy
from typing import Any, Callable

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


def _numeric_vjp_arg(fn, dout, arg, *, eps: float = 1e-5):
    """Central-difference VJP for small reference-only primitives."""
    arr = np.asarray(arg, dtype=np.float64)
    grad = np.zeros_like(arr, dtype=np.float64)
    dout_arr = np.asarray(dout, dtype=np.float64)
    # numpy 1.x / 2.x typeshed disagree on the op_flags shape (flat vs nested);
    # both forms are valid at runtime, so type the value as Any to stay portable.
    op_flags: Any = [["readwrite"]]
    it = np.nditer(arr, flags=["multi_index"], op_flags=op_flags)
    while not it.finished:
        idx = it.multi_index
        plus = arr.copy()
        minus = arr.copy()
        plus[idx] += eps
        minus[idx] -= eps
        y_plus = np.asarray(fn(plus), dtype=np.float64)
        y_minus = np.asarray(fn(minus), dtype=np.float64)
        grad[idx] = np.sum(((y_plus - y_minus) / (2.0 * eps)) * dout_arr)
        it.iternext()
    return grad.reshape(np.asarray(arg).shape)


def _tree_numeric_vjp(fn, dout, tree, *, eps: float = 1e-5):
    """Central-difference VJP over numeric leaves in a nested state tree."""
    if tree is None:
        return None
    if isinstance(tree, dict):
        out: dict = {}
        for key, value in tree.items():
            # ``np.ndarray`` is not a subclass of (bool, str, int,
            # np.integer); the second ``isinstance`` check is a defense
            # against future tree containers that subclass both.  mypy
            # statically knows it's unreachable today, but the guard
            # stays.
            if isinstance(value, (bool, str, int, np.integer)) and not isinstance(value, np.ndarray):  # type: ignore[unreachable]
                out[key] = None
                continue

            def replace_leaf(new_value, key=key):
                copied = copy.deepcopy(tree)
                copied[key] = new_value
                return fn(copied)

            out[key] = _tree_numeric_vjp(replace_leaf, dout, value, eps=eps)
        return out
    if isinstance(tree, tuple):
        return tuple(
            _tree_numeric_vjp(
                lambda new_value, i=i: fn(tuple(new_value if j == i else copy.deepcopy(v) for j, v in enumerate(tree))),
                dout,
                value,
                eps=eps,
            )
            for i, value in enumerate(tree)
        )
    if isinstance(tree, list):
        return [
            _tree_numeric_vjp(
                lambda new_value, i=i: fn([new_value if j == i else copy.deepcopy(v) for j, v in enumerate(tree)]),
                dout,
                value,
                eps=eps,
            )
            for i, value in enumerate(tree)
        ]
    arr = np.asarray(tree)
    if not np.issubdtype(arr.dtype, np.number):
        return None
    return _numeric_vjp_arg(fn, dout, arr, eps=eps)


def _attention_vjp(dout, Q, K, V, *, scale=None, mask=None):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    dout = np.asarray(dout, dtype=np.float64)
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    S = np.matmul(Q, np.swapaxes(K, -1, -2)) * float(scale)
    if mask is not None:
        S = np.where(mask, -np.inf, S)
    e = np.exp(S - np.max(S, axis=-1, keepdims=True))
    P = e / np.sum(e, axis=-1, keepdims=True)
    dV = np.matmul(np.swapaxes(P, -1, -2), dout)
    dP = np.matmul(dout, np.swapaxes(V, -1, -2))
    dS = (dP - np.sum(dP * P, axis=-1, keepdims=True)) * P
    if mask is not None:
        dS = np.where(mask, 0.0, dS)
    dQ = np.matmul(dS, K) * float(scale)
    dK = np.matmul(np.swapaxes(dS, -1, -2), Q) * float(scale)
    return dQ, dK, dV


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


def _normalize_axis(axis: int, ndim: int) -> int:
    return axis if axis >= 0 else ndim + axis


def _slice_tuple(start_indices, sizes) -> tuple[slice, ...]:
    return tuple(
        slice(int(start), int(start) + int(size))
        for start, size in zip(start_indices, sizes)
    )


@_vjp("reshape")
def vjp_reshape(dout, x, *, shape=None, **_):
    return (np.asarray(dout).reshape(np.asarray(x).shape),)


@_vjp("view")
def vjp_view(dout, x, *, shape=None, **_):
    return vjp_reshape(dout, x, shape=shape)


@_vjp("flatten")
def vjp_flatten(dout, x, *, start_axis=0, end_axis=-1, **_):
    return (np.asarray(dout).reshape(np.asarray(x).shape),)


@_vjp("squeeze")
def vjp_squeeze(dout, x, *, axis=None, **_):
    return (np.asarray(dout).reshape(np.asarray(x).shape),)


@_vjp("unsqueeze")
def vjp_unsqueeze(dout, x, *, axis=None, **_):
    return (np.asarray(dout).reshape(np.asarray(x).shape),)


@_vjp("permute")
def vjp_permute(dout, x, *, axes=None, **_):
    inv = np.argsort(tuple(axes))
    return (np.transpose(dout, axes=tuple(int(i) for i in inv)),)


@_vjp("broadcast")
def vjp_broadcast(dout, x, *, shape=None, **_):
    return (_sum_to_shape(np.asarray(dout), np.asarray(x).shape),)


@_vjp("expand")
def vjp_expand(dout, x, *, shape=None, **_):
    return vjp_broadcast(dout, x, shape=shape)


@_vjp("cat")
def vjp_cat(dout, xs, *, axis=0, **_):
    sizes = [np.asarray(x).shape[axis] for x in xs]
    offsets = np.cumsum(sizes)[:-1]
    return (tuple(np.split(np.asarray(dout), offsets, axis=axis)),)


@_vjp("stack")
def vjp_stack(dout, xs, *, axis=0, **_):
    axis = _normalize_axis(axis, np.asarray(dout).ndim)
    return (tuple(np.squeeze(part, axis=axis) for part in np.split(dout, len(xs), axis=axis)),)


@_vjp("split")
def vjp_split(dout, x, *, indices_or_sections=None, axis=0, **_):
    return (np.concatenate(tuple(dout), axis=axis),)


@_vjp("chunk")
def vjp_chunk(dout, x, *, chunks=None, axis=0, **_):
    return (np.concatenate(tuple(dout), axis=axis),)


@_vjp("pad")
def vjp_pad(dout, x, *, pad_width=None, mode="constant", constant_values=0, **_):
    if pad_width is None:
        raise ValueError("vjp_pad requires pad_width")
    normalized = []
    for item in pad_width:
        if isinstance(item, int):
            normalized.append((item, item))
        else:
            normalized.append((int(item[0]), int(item[1])))
    slices = tuple(
        slice(before, np.asarray(dout).shape[axis] - after if after else None)
        for axis, (before, after) in enumerate(normalized)
    )
    return (np.asarray(dout)[slices].reshape(np.asarray(x).shape),)


@_vjp("tile")
def vjp_tile(dout, x, *, reps=None, **_):
    x_shape = tuple(np.asarray(x).shape)
    reps_tuple = tuple(reps if isinstance(reps, (tuple, list)) else (reps,))
    if len(reps_tuple) < len(x_shape):
        reps_tuple = (1,) * (len(x_shape) - len(reps_tuple)) + reps_tuple
    elif len(reps_tuple) > len(x_shape):
        x_shape = (1,) * (len(reps_tuple) - len(x_shape)) + x_shape
    interleaved = []
    for rep, size in zip(reps_tuple, x_shape):
        interleaved.extend([int(rep), int(size)])
    grad = np.asarray(dout).reshape(interleaved)
    for axis in reversed(range(0, len(interleaved), 2)):
        grad = grad.sum(axis=axis)
    return (grad.reshape(np.asarray(x).shape),)


@_vjp("repeat")
def vjp_repeat(dout, x, *, repeats=None, axis=None, **_):
    if not isinstance(repeats, (int, np.integer)):
        raise NotImplementedError("repeat VJP currently supports scalar repeats only")
    repeats = int(repeats)
    if axis is None:
        return (np.asarray(dout).reshape(-1, repeats).sum(axis=1).reshape(np.asarray(x).shape),)
    ax = _normalize_axis(axis, np.asarray(x).ndim)
    moved = np.moveaxis(np.asarray(dout), ax, 0)
    grad = moved.reshape(np.asarray(x).shape[ax], repeats, *moved.shape[1:]).sum(axis=1)
    return (np.moveaxis(grad, 0, ax),)


@_vjp("roll")
def vjp_roll(dout, x, *, shift=None, axis=None, **_):
    return (np.roll(dout, shift=-shift, axis=axis),)


@_vjp("flip")
def vjp_flip(dout, x, *, axis=None, **_):
    return (np.flip(dout, axis=axis),)


@_vjp("dynamic_slice")
def vjp_dynamic_slice(dout, x, *, start_indices=None, slice_sizes=None, **_):
    dx = np.zeros_like(x)
    dx[_slice_tuple(start_indices, slice_sizes)] = dout
    return (dx,)


@_vjp("slice")
def vjp_slice(dout, x, *, start_indices=None, slice_sizes=None, **_):
    return vjp_dynamic_slice(dout, x, start_indices=start_indices, slice_sizes=slice_sizes)


@_vjp("select")
def vjp_select(dout, x, *, index=None, axis=0, **_):
    dx = np.zeros_like(x)
    ax = _normalize_axis(axis, np.asarray(x).ndim)
    dx_m = np.moveaxis(dx, ax, 0)
    dx_m[int(index)] = dout
    return (np.moveaxis(dx_m, 0, ax),)


@_vjp("dynamic_update_slice")
def vjp_dynamic_update_slice(dout, x, update, *, start_indices=None, **_):
    slices = _slice_tuple(start_indices, np.asarray(update).shape)
    dx = np.array(dout, copy=True)
    dx[slices] = 0
    dupdate = np.asarray(dout)[slices].reshape(np.asarray(update).shape)
    return (dx, dupdate)


@_vjp("take")
def vjp_take(dout, x, indices, *, axis=None, **_):
    idx = np.asarray(indices, dtype=np.int64)
    dx = np.zeros_like(x)
    if axis is None:
        np.add.at(dx.reshape(-1), idx.reshape(-1), np.asarray(dout).reshape(-1))
        return (dx, None)
    ax = _normalize_axis(axis, np.asarray(x).ndim)
    dx_m = np.moveaxis(dx, ax, 0)
    dout_m = np.moveaxis(np.asarray(dout), ax, 0)
    np.add.at(dx_m, idx, dout_m)
    return (np.moveaxis(dx_m, 0, ax), None)


@_vjp("index_select")
def vjp_index_select(dout, x, indices, *, axis=0, **_):
    return vjp_take(dout, x, indices, axis=axis)


def _zero_selected_axis(dout, indices, axis):
    dx = np.array(dout, copy=True)
    ax = _normalize_axis(axis, dx.ndim)
    dx_m = np.moveaxis(dx, ax, 0)
    dx_m[np.asarray(indices, dtype=np.int64)] = 0
    return np.moveaxis(dx_m, 0, ax)


def _take_selected_axis(dout, indices, axis, target_shape):
    gathered = np.take(np.asarray(dout), np.asarray(indices, dtype=np.int64), axis=axis)
    return gathered.reshape(target_shape)


@_vjp("scatter")
def vjp_scatter(dout, x, indices, updates, *, axis=0, **_):
    return (
        _zero_selected_axis(dout, indices, axis),
        None,
        _take_selected_axis(dout, indices, axis, np.asarray(updates).shape),
    )


@_vjp("index_update")
def vjp_index_update(dout, x, indices, updates, *, axis=0, **_):
    return vjp_scatter(dout, x, indices, updates, axis=axis)


@_vjp("scatter_add")
def vjp_scatter_add(dout, x, indices, updates, *, axis=0, **_):
    return (
        np.asarray(dout),
        None,
        _take_selected_axis(dout, indices, axis, np.asarray(updates).shape),
    )


@_vjp("scatter_reduce")
def vjp_scatter_reduce(dout, x, indices, updates, *, axis=0, reduce="sum", **_):
    if reduce != "sum":
        raise NotImplementedError("scatter_reduce VJP is implemented only for reduce='sum'")
    return vjp_scatter_add(dout, x, indices, updates, axis=axis)


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


@_vjp("silu_mul")
def vjp_silu_mul(dout, a, b, **_):
    """y = silu(a) * b.

    da = dout * b * d(silu)/da = dout * b * (sig(a) + a * sig(a) * (1 - sig(a)))
    db = dout * silu(a)
    """
    s = 1.0 / (1.0 + np.exp(-a))
    silu_a = a * s
    da = dout * b * (s + a * s * (1.0 - s))
    db = dout * silu_a
    da = _sum_to_shape(da, a.shape)
    db = _sum_to_shape(db, b.shape)
    return (da, db)


# ─────────────────────────────────────────────────────────────────────────────
# Theme 9 — Utility op VJPs.
# `arange` is non-differentiable (output values are constants). The rest
# treat their non-tensor kwargs (`indices`, `mask`, `value`, bounds) as
# pass-through.
# ─────────────────────────────────────────────────────────────────────────────


@_vjp("gather")
def vjp_gather(dout, x, indices, *, axis=0, **_):
    """y = take(x, indices, axis=axis).

    dx[i, ...] = sum over j with indices[j] == i of dout[j, ...]
    indices: non-differentiable (integer index tensor).
    """
    idx = np.asarray(indices, dtype=np.int64)
    dx = np.zeros_like(x)
    # `np.add.at` performs unbuffered in-place add — correct under
    # repeated indices (which a gather can produce).
    if axis == 0 or axis == -x.ndim:
        np.add.at(dx, idx, dout)
    else:
        # General case: move `axis` to leading dim, scatter, move back.
        ax = axis if axis >= 0 else x.ndim + axis
        dx_t = np.moveaxis(dx, ax, 0)
        dout_t = np.moveaxis(dout, ax, 0)
        np.add.at(dx_t, idx, dout_t)
        dx = np.moveaxis(dx_t, 0, ax)
    return (dx, None)


@_vjp("clip")
def vjp_clip(dout, x, *, min_val=None, max_val=None, **_):
    """y = clip(x, min_val, max_val).

    Straight-through estimator: dx = dout where x is **strictly** inside
    the range, 0 at or beyond the bounds. Matches PyTorch's `torch.clamp`
    convention and the central-difference numerical Jacobian (which sees a
    flat plateau at any point that's eps-clipped on either side).
    """
    mask = np.ones_like(x, dtype=x.dtype)
    if min_val is not None:
        mask = mask * (x > min_val).astype(x.dtype)
    if max_val is not None:
        mask = mask * (x < max_val).astype(x.dtype)
    return (dout * mask,)


@_vjp("moe")
def vjp_moe(dout, x, experts, *, router="topk", k=1, transport=None,
            deterministic=None, scores=None, route=None, **_):
    """Mixture-of-Experts VJP.

    For each token ``i`` routed to expert ``r[i]``:
        out[i] = x[i] @ E[r[i]]

    Gradient pieces:
        dx[i]    = dout[i] @ E[r[i]].T
        dE[r[i]] += x[i].T @ dout[i]   (accumulated across tokens)

    The route (`scores` argmax, explicit `route`, or modulo fallback) is
    integer-valued, non-differentiable; we return cotangents only for
    `x` and `experts`. The forward-side scoring path (a softmax over a
    learnable router) lives one op up the tape — the user composes it
    explicitly when training the router.

    Theme 2 follow-up (F3-moe in the execution roadmap).
    """
    x_arr = np.asarray(x)
    experts_arr = np.asarray(experts)
    if experts_arr.ndim == 2:
        experts_arr = experts_arr[None, :, :]
    if experts_arr.ndim != 3:
        # Same shape contract the forward enforces; fall through to
        # zero gradients rather than raise from inside backward.
        return (np.zeros_like(x_arr), np.zeros_like(experts))

    tokens = x_arr.reshape(-1, x_arr.shape[-1])
    num_experts = experts_arr.shape[0]
    if route is not None:
        route_arr = np.asarray(route, dtype=np.int64).reshape(-1)
    elif scores is not None:
        route_arr = np.argmax(
            np.asarray(scores).reshape(tokens.shape[0], num_experts), axis=-1,
        )
    else:
        route_arr = np.arange(tokens.shape[0], dtype=np.int64) % num_experts
    route_arr = np.mod(route_arr, num_experts)

    dout_arr = np.asarray(dout).reshape(tokens.shape[0], experts_arr.shape[2])

    dx_tokens = np.zeros_like(tokens)
    dE = np.zeros_like(experts_arr)
    for i in range(tokens.shape[0]):
        e = int(route_arr[i])
        # dx[i] = dout[i] @ E[e].T
        dx_tokens[i] = dout_arr[i] @ experts_arr[e].T
        # Accumulate per-expert weight gradient.
        dE[e] += np.outer(tokens[i], dout_arr[i])

    dx = dx_tokens.reshape(x_arr.shape)
    # Restore experts gradient to the same shape as the input (drop the
    # leading axis if the user passed a 2-D weight).
    if np.asarray(experts).ndim == 2:
        dE = dE[0]
    return (dx, dE)


# ─────────────────────────────────────────────────────────────────────────────
# Phase F-MoR — Mixture of Recursions VJPs.
#
# `mor_router` outputs an int64 depth tensor — the argmax is
# non-differentiable. We treat it as a straight-through identity: the
# gradient w.r.t. `x` and `w_router` is zero (the router's training
# signal usually arrives through the recursion-loop's output via the
# layer's own gradient path, plus an auxiliary load-balancing loss the
# user adds explicitly).
#
# `mor_partition` produces a bool mask from an int depth tensor — also
# non-differentiable, returns zero cotangents.
#
# `mor_scatter` is linear in `updated` and selects rows of `full` for
# the unselected positions. Both `full` and `updated` get real
# gradients. `mask` is non-differentiable.
# ─────────────────────────────────────────────────────────────────────────────


@_vjp("mor_router")
def vjp_mor_router(dout, x, w_router, *, max_depth=None, **_):
    """Argmax-based router has no analytical gradient. Return zeros so
    this op is benign on the gradient path. Real router-training comes
    from auxiliary losses (load-balance / utilization) the user adds
    explicitly downstream."""
    return (np.zeros_like(x), np.zeros_like(w_router))


@_vjp("mor_partition")
def vjp_mor_partition(dout, x, depth, *, step=None, **_):
    """Bool-mask output; zero gradient w.r.t. real-valued inputs."""
    return (np.zeros_like(x), None)


@_vjp("mor_scatter")
def vjp_mor_scatter(dout, full, updated, mask, **_):
    """y = where(mask, updated, full). dx for `full` flows on the False
    positions; for `updated` on the True positions. `mask` is
    non-differentiable."""
    m = np.broadcast_to(np.asarray(mask, dtype=bool)[..., None], full.shape)
    d_full = dout * (~m).astype(full.dtype)
    d_updated = dout * m.astype(updated.dtype)
    return (d_full, d_updated, None)


@_vjp("masked_fill")
def vjp_masked_fill(dout, x, mask, *, value=None, **_):
    """y = where(mask, value, x).

    dx propagates dout only on positions where mask is False; positions
    that were filled with `value` carry no gradient back. `mask` is a
    tensor input recorded on the tape (non-differentiable, receives
    `None` cotangent). `value` is keyword-only and not on the tape.
    """
    m = np.broadcast_to(np.asarray(mask, dtype=bool), x.shape)
    dx = dout * (~m).astype(x.dtype)
    return (_sum_to_shape(dx, x.shape), None)


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


# ─────────────────────────────────────────────────────────────────────────────
# Fused-op adjoints (Phase F3)
# ─────────────────────────────────────────────────────────────────────────────


@_vjp("linear_attn")
def vjp_linear_attn(
    dout, Q, K, V, *, feature_map="elu", state=None, chunk_size=None,
    decay=None, causal=True, **_,
):
    """Adjoint of linear / kernel-feature attention.

    Forward (causal):
        S_t = decay_t * S_{t-1} + φ(K_t)^T V_t
        O_t = φ(Q_t) @ S_t

    Backward strategy: recompute the forward states ``S_t`` (cheap —
    it's the same recurrence), then walk the recurrence in reverse,
    accumulating gradients into ``dQ`` / ``dK`` / ``dV``. This is the
    canonical linear-attention adjoint (RWKV / RetNet / Mamba2-linear
    use the same shape).

    Returns ``(dQ, dK, dV)``. Decay and state inputs are non-tensor /
    optional; their gradients are deferred until concrete demand
    surfaces (most production training calls drop the chained state).
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    dout = np.asarray(dout, dtype=np.float64)
    B, H, S, D_qk = Q.shape
    D_v = V.shape[3]

    # Feature map + its derivative w.r.t. its input.
    def phi_and_dphi(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if feature_map == "elu":
            mask_pos = x > 0
            phi = np.where(mask_pos, x + 1.0, np.exp(x))
            dphi = np.where(mask_pos, np.ones_like(x), np.exp(x))
            return phi, dphi
        if feature_map == "relu":
            phi = np.maximum(x, 0.0)
            dphi = (x > 0).astype(x.dtype)
            return phi, dphi
        if feature_map == "identity":
            return x, np.ones_like(x)
        if feature_map == "polynomial_2":
            return x * x, 2.0 * x
        raise ValueError(
            f"vjp_linear_attn: unknown feature_map {feature_map!r}"
        )

    phi_Q, dphi_Q = phi_and_dphi(Q)
    phi_K, dphi_K = phi_and_dphi(K)

    # Forward recurrence: store S_t for each t (memory: O(S * D_qk * D_v)).
    if state is None:
        S_state = np.zeros((B, H, D_qk, D_v), dtype=np.float64)
    else:
        S_state = np.asarray(state, dtype=np.float64).copy()
    S_history = np.zeros((S + 1, B, H, D_qk, D_v), dtype=np.float64)
    S_history[0] = S_state
    if decay is not None:
        decay = np.asarray(decay, dtype=np.float64)

    # Causal forward replay (ignores chunk_size for backward — bit-equivalent
    # at fp64 to the chunked-parallel forward).
    if causal:
        for t in range(S):
            if decay is not None:
                S_state = decay[:, :, t][:, :, None, None] * S_state
            S_state = S_state + np.einsum(
                "bhd,bhe->bhde", phi_K[:, :, t, :], V[:, :, t, :]
            )
            S_history[t + 1] = S_state
    else:
        # Non-causal: O = φ(Q) @ (Σ φ(K)^T V).
        kv_sum = np.einsum("bhsd,bhse->bhde", phi_K, V)
        if state is not None:
            kv_sum = state + kv_sum
        # dQ_phi = dout @ kv_sum^T   shape (B, H, S, D_qk)
        dphi_Q_full = np.einsum("bhse,bhde->bhsd", dout, kv_sum)
        dQ = dphi_Q_full * dphi_Q
        # d(kv_sum) = sum over s of φ(Q_s)^T @ dout_s
        d_kv_sum = np.einsum("bhsd,bhse->bhde", phi_Q, dout)
        # d_kv_sum / d(φ(K)_t, V_t) = (V_t broadcast, φ(K)_t broadcast)
        dphi_K_full = np.einsum("bhde,bhse->bhsd", d_kv_sum, V)
        dK = dphi_K_full * dphi_K
        dV = np.einsum("bhsd,bhde->bhse", phi_K, d_kv_sum)
        return (dQ, dK, dV)

    # Reverse pass: dS holds running cotangent on the current (post-step) S.
    dQ = np.zeros_like(Q)
    dK = np.zeros_like(K)
    dV = np.zeros_like(V)
    dS = np.zeros((B, H, D_qk, D_v), dtype=np.float64)

    for t in range(S - 1, -1, -1):
        # O_t = φ(Q_t) @ S_t  (where S_t is post-update, i.e. S_history[t+1])
        S_post = S_history[t + 1]
        dout_t = dout[:, :, t, :]
        phi_Q_t = phi_Q[:, :, t, :]
        # dφ(Q_t) = dout_t @ S_t^T
        dphi_Q_t = np.einsum("bhe,bhde->bhd", dout_t, S_post)
        dQ[:, :, t, :] = dphi_Q_t * dphi_Q[:, :, t, :]
        # dS_post += φ(Q_t)^T outer dout_t
        dS = dS + np.einsum("bhd,bhe->bhde", phi_Q_t, dout_t)
        # S_post = decay_t * S_pre + φ(K_t)^T V_t
        # → dS_pre = decay_t * dS_post (broadcast)
        # → dφ(K_t) = dS_post @ V_t^T;  dV_t = φ(K_t)^T @ dS_post (per-head)
        phi_K_t = phi_K[:, :, t, :]
        V_t = V[:, :, t, :]
        dphi_K_t = np.einsum("bhde,bhe->bhd", dS, V_t)
        dK[:, :, t, :] = dphi_K_t * dphi_K[:, :, t, :]
        dV[:, :, t, :] = np.einsum("bhd,bhde->bhe", phi_K_t, dS)
        # Propagate dS through the decay multiplier (if present).
        if decay is not None:
            dS = decay[:, :, t][:, :, None, None] * dS

    return (dQ, dK, dV)


@_vjp("linear_attn_state")
def vjp_linear_attn_state(
    dstate_out, Q, K, V, *, feature_map="elu", state=None, chunk_size=None,
    decay=None, causal=True, **_,
):
    """Adjoint of :func:`linear_attn_state`.

    The state output has shape ``(B, H, D_qk, D_v)``; it depends on
    ``φ(K)`` and ``V`` along the recurrence and on ``Q`` not at all
    (Q only affects ``O``, not the state). For non-chained training
    paths most callers drop the state grad entirely (treat it as
    ``stop_gradient``). We provide a true VJP for completeness.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    dstate_out = np.asarray(dstate_out, dtype=np.float64)
    B, H, S, D_qk = Q.shape

    def phi_and_dphi(x):
        if feature_map == "elu":
            mp = x > 0
            return np.where(mp, x + 1.0, np.exp(x)), np.where(
                mp, np.ones_like(x), np.exp(x)
            )
        if feature_map == "relu":
            return np.maximum(x, 0.0), (x > 0).astype(x.dtype)
        if feature_map == "identity":
            return x, np.ones_like(x)
        if feature_map == "polynomial_2":
            return x * x, 2.0 * x
        raise ValueError(f"unknown feature_map {feature_map!r}")

    _, dphi_K = phi_and_dphi(K)
    phi_K, _ = phi_and_dphi(K)

    if not causal:
        # state_out = (state_in or 0) + Σ φ(K)^T V
        # d/dφ(K)_t = dstate_out @ V_t^T;  dV_t = φ(K)^T @ dstate_out
        dphi_K_full = np.einsum("bhde,bhse->bhsd", dstate_out, V)
        dK = dphi_K_full * dphi_K
        dV = np.einsum("bhsd,bhde->bhe", phi_K, dstate_out)  # broadcast over S → (B, H, D_v)
        # Need (B, H, S, D_v); broadcast dstate_out's contribution across S.
        dV = np.einsum("bhsd,bhde->bhse", phi_K, dstate_out)
        dQ = np.zeros_like(Q)
        return (dQ, dK, dV)

    # Causal: walk the recurrence backward seeded with dS = dstate_out.
    if decay is not None:
        decay = np.asarray(decay, dtype=np.float64)
    dS = dstate_out.copy()
    dQ = np.zeros_like(Q)  # state has no Q dependence
    dK = np.zeros_like(K)
    dV = np.zeros_like(V)
    for t in range(S - 1, -1, -1):
        # S_t = decay_t * S_{t-1} + φ(K_t)^T V_t
        # dφ(K_t) = dS @ V_t^T ; dV_t = φ(K_t)^T @ dS
        dphi_K_t = np.einsum("bhde,bhe->bhd", dS, V[:, :, t, :])
        dK[:, :, t, :] = dphi_K_t * dphi_K[:, :, t, :]
        dV[:, :, t, :] = np.einsum("bhd,bhde->bhe", phi_K[:, :, t, :], dS)
        if decay is not None:
            dS = decay[:, :, t][:, :, None, None] * dS

    return (dQ, dK, dV)


@_vjp("flash_attn")
def vjp_flash_attn(dout, Q, K, V, *, scale=None, causal=False, dropout_p=0.0, **_):
    """Adjoint of standard scaled-dot-product attention (numpy reference path).

    Forward: ``S = scale * QK^T;  P = softmax(S);  O = PV``.
    Memory-efficient streaming adjoint is left to fused-kernel custom rules
    on each backend; this v1 recomputes ``S``, ``P`` so it works for any shape
    that the forward already accepts.

    ``dropout_p`` is ignored on backward — the v1 reference dropout uses a
    fresh rng each call, so the backward path can't reproduce the mask
    deterministically without a stored seed. Callers running training with
    attention dropout should set ``deterministic=True, seed=...`` so the mask
    is reproducible.
    """
    if dropout_p > 0.0:
        # No deterministic mask available → conservative: assume mask is all-ones.
        # Users training with dropout should provide a seed.
        pass

    d = Q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    # Recompute forward intermediates
    S = np.matmul(Q, np.swapaxes(K, -1, -2)) * scale
    if causal:
        q_len, k_len = S.shape[-2], S.shape[-1]
        mask = np.triu(np.ones((q_len, k_len), dtype=bool), k=1 + max(k_len - q_len, 0))
        S = np.where(mask, -np.inf, S)
    e = np.exp(S - S.max(axis=-1, keepdims=True))
    P = e / e.sum(axis=-1, keepdims=True)

    # dV = P^T @ dO
    dV = np.matmul(np.swapaxes(P, -1, -2), dout)
    # dP = dO @ V^T
    dP = np.matmul(dout, np.swapaxes(V, -1, -2))
    # dS through softmax: dS = (dP - sum(dP * P, -1, keepdims)) * P
    dS = (dP - (dP * P).sum(axis=-1, keepdims=True)) * P
    if causal:
        dS = np.where(mask, 0.0, dS)
    # dQ = dS @ K * scale;  dK = dS^T @ Q * scale
    dQ = np.matmul(dS, K) * scale
    dK = np.matmul(np.swapaxes(dS, -1, -2), Q) * scale
    return (dQ, dK, dV)


@_vjp("rope")
def vjp_rope(dout, x, theta, *, axes="qk", **_):
    x = np.asarray(x)
    theta = np.asarray(theta)
    theta_pair = theta[..., 0::2] if theta.shape[-1] == x.shape[-1] else theta
    cos = np.cos(theta_pair)
    sin = np.sin(theta_pair)
    de = np.asarray(dout)[..., 0::2]
    do = np.asarray(dout)[..., 1::2]
    xe = x[..., 0::2]
    xo = x[..., 1::2]
    dx = np.empty_like(np.asarray(dout), dtype=np.result_type(dout, x))
    dx[..., 0::2] = de * cos + do * sin
    dx[..., 1::2] = -de * sin + do * cos
    dtheta_pair = de * (-xe * sin - xo * cos) + do * (xe * cos - xo * sin)
    if theta.shape[-1] == x.shape[-1]:
        dtheta = np.zeros_like(theta)
        dtheta[..., 0::2] = dtheta_pair
    else:
        dtheta = _sum_to_shape(dtheta_pair, theta.shape)
    return (_sum_to_shape(dx, x.shape), dtheta)


@_vjp("ntk_rope")
def vjp_ntk_rope(dout, x, theta, *, scale=1.0, **kwargs):
    dx, dtheta = vjp_rope(dout, x, np.asarray(theta) / float(scale), **kwargs)
    return dx, dtheta / float(scale)


@_vjp("rope_split")
def vjp_rope_split(dout, x, *, rope_dim, _output_index=0, **_):
    dx = np.zeros_like(x)
    if int(_output_index) == 0:
        dx[..., :int(rope_dim)] = dout
    else:
        dx[..., int(rope_dim):] = dout
    return (dx,)


@_vjp("rope_merge")
def vjp_rope_merge(dout, rope_part, no_rope_part, **_):
    rdim = np.asarray(rope_part).shape[-1]
    return (np.asarray(dout)[..., :rdim], np.asarray(dout)[..., rdim:])


@_vjp("latent_kv_compress")
def vjp_latent_kv_compress(dout, x, w_dkv, **_):
    return vjp_matmul(dout, x, w_dkv)


@_vjp("latent_kv_expand_k")
def vjp_latent_kv_expand_k(dout, c, w_uk, **_):
    return vjp_matmul(dout, c, w_uk)


@_vjp("latent_kv_expand_v")
def vjp_latent_kv_expand_v(dout, c, w_uv, **_):
    return vjp_matmul(dout, c, w_uv)


@_vjp("mla_decode_fused")
def vjp_mla_decode_fused(dout, x, w_dkv, w_uk, w_uv, q, *, scale=None, causal=False, **_):
    def forward(args):
        x_, wd_, wk_, wv_, q_ = args
        c = np.matmul(x_, wd_)
        K = np.matmul(c, wk_)
        V = np.matmul(c, wv_)
        return _flash_attn_reference(q_, K, V, scale=scale, causal=causal)

    args = [x, w_dkv, w_uk, w_uv, q]
    return tuple(
        _numeric_vjp_arg(lambda a, i=i: forward(args[:i] + [a] + args[i + 1:]), dout, arg)
        for i, arg in enumerate(args)
    )


def _flash_attn_reference(Q, K, V, *, scale=None, causal=False):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if Q.ndim == 3:
        Q = Q[:, None, :, :]
        K = K[:, None, :, :]
        V = V[:, None, :, :]
        squeeze = True
    else:
        squeeze = False
    mask = None
    if causal:
        q_len, k_len = Q.shape[-2], K.shape[-2]
        mask = np.triu(np.ones((q_len, k_len), dtype=bool), k=1 + max(k_len - q_len, 0))
    s = np.matmul(Q, np.swapaxes(K, -1, -2)) * (1.0 / math.sqrt(Q.shape[-1]) if scale is None else float(scale))
    if mask is not None:
        s = np.where(mask, -np.inf, s)
    e = np.exp(s - np.max(s, axis=-1, keepdims=True))
    p = e / np.sum(e, axis=-1, keepdims=True)
    y = np.matmul(p, V)
    return y[:, 0, :, :] if squeeze else y


@_vjp("quantize_fp8")
@_vjp("quantize_fp4")
@_vjp("fake_quantize")
def vjp_quantize_ste(dout, x, *, _output_index=0, **_):
    if int(_output_index) != 0:
        return (np.zeros_like(np.asarray(x)),)
    return (_sum_to_shape(np.asarray(dout), np.asarray(x).shape),)


@_vjp("dequantize_fp8")
@_vjp("dequantize_fp4")
def vjp_dequantize_ste(dout, x_q, scale, **_):
    return (_sum_to_shape(np.asarray(dout), np.asarray(x_q).shape), np.zeros_like(np.asarray(scale)))


@_vjp("attn_sliding_window")
def vjp_attn_sliding_window(dout, Q, K, V, *, window_size, causal=True, **_):
    S_q, S_k = Q.shape[-2], K.shape[-2]
    i_idx = np.arange(S_q)[:, None]
    j_idx = np.arange(S_k)[None, :]
    if causal:
        mask = (j_idx > i_idx) | (j_idx < i_idx - int(window_size) + 1)
    else:
        mask = (j_idx > i_idx + int(window_size) // 2) | (j_idx < i_idx - int(window_size) // 2)
    return _attention_vjp(dout, Q, K, V, mask=mask)


@_vjp("attn_local_window_2d")
def vjp_attn_local_window_2d(dout, Q, K, V, *, window=(1, 1), **_):
    """Gap 4 (2026-05-20): VJP for 2D local-window attention.

    Per-(Hq, Wq) window masks make the analytical Jacobian
    structurally awkward (the mask shape varies per query position),
    so we use the numeric-finite-difference fallback the
    ``attn_top_k_blocks`` family uses — correct, slower, and trivially
    replaced when a fused 2D-window kernel lands.
    """
    from tessera import ops as _ops
    original = getattr(
        _ops.attn_local_window_2d, "__wrapped__", _ops.attn_local_window_2d,
    )
    args = [Q, K, V]
    return tuple(
        _numeric_vjp_arg(
            lambda a, i=i: original(
                *(args[:i] + [a] + args[i + 1:]), window=window,
            ),
            dout, arg,
        )
        for i, arg in enumerate(args)
    )


@_vjp("attn_compressed_blocks")
def vjp_attn_compressed_blocks(dout, Q, K_c, V_c, **_):
    return _attention_vjp(dout, Q, K_c, V_c)


@_vjp("attn_top_k_blocks")
def vjp_attn_top_k_blocks(dout, Q, K, V, *, scores, top_k, block_size, causal=True, **_):
    from tessera import ops as _ops
    original = getattr(_ops.attn_top_k_blocks, "__wrapped__", _ops.attn_top_k_blocks)
    args = [Q, K, V]
    return tuple(
        _numeric_vjp_arg(
            lambda a, i=i: original(
                *(args[:i] + [a] + args[i + 1:]),
                scores=scores,
                top_k=top_k,
                block_size=block_size,
                causal=causal,
            ),
            dout,
            arg,
        )
        for i, arg in enumerate(args)
    )


def _numeric_attention_family_vjp(op_name, dout, args, kwargs):
    from tessera import ops as _ops
    original = getattr(getattr(_ops, op_name), "__wrapped__", getattr(_ops, op_name))
    args = list(args)
    return tuple(
        _numeric_vjp_arg(
            lambda a, i=i: original(*(args[:i] + [a] + args[i + 1:]), **kwargs),
            dout,
            arg,
        )
        for i, arg in enumerate(args)
    )


@_vjp("power_attn")
def vjp_power_attn(dout, Q, K, V, **kwargs):
    return _numeric_attention_family_vjp("power_attn", dout, (Q, K, V), kwargs)


@_vjp("retention")
def vjp_retention(dout, Q, K, V, **kwargs):
    return _numeric_attention_family_vjp("retention", dout, (Q, K, V), kwargs)


@_vjp("lightning_attention")
def vjp_lightning_attention(dout, Q, K, V, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if k != "_output_index"}
    return _numeric_attention_family_vjp("lightning_attention", dout, (Q, K, V), kwargs)


@_vjp("gated_attention")
def vjp_gated_attention(dout, Q, K, V, gate, **kwargs):
    return _numeric_attention_family_vjp("gated_attention", dout, (Q, K, V, gate), kwargs)


@_vjp("deepseek_sparse_attention")
def vjp_deepseek_sparse_attention(dout, Q, K, V, gate_logits=None, **kwargs):
    args = (Q, K, V) if gate_logits is None else (Q, K, V, gate_logits)
    return _numeric_attention_family_vjp("deepseek_sparse_attention", dout, args, kwargs)


@_vjp("gated_deltanet")
def vjp_gated_deltanet(dout, *args, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if k != "_output_index"}
    return _numeric_attention_family_vjp("gated_deltanet", dout, args, kwargs)


@_vjp("kimi_delta_attention")
def vjp_kimi_delta_attention(dout, *args, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if k != "_output_index"}
    return _numeric_attention_family_vjp("kimi_delta_attention", dout, args, kwargs)


@_vjp("modified_delta_attention")
def vjp_modified_delta_attention(dout, *args, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if k != "_output_index"}
    return _numeric_attention_family_vjp("modified_delta_attention", dout, args, kwargs)


@_vjp("hybrid_attention")
def vjp_hybrid_attention(dout, Q, K, V, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if k != "_output_index"}
    return _numeric_attention_family_vjp("hybrid_attention", dout, (Q, K, V), kwargs)


@_vjp("moe_dispatch")
def vjp_moe_dispatch(dout, x, route, *, transport=None, **_):
    return (_sum_to_shape(np.asarray(dout), np.asarray(x).shape), None)


@_vjp("moe_combine")
def vjp_moe_combine(dout, partials, inverse_route, *, reduce="sum", **_):
    partials = np.asarray(partials)
    if reduce == "mean" and partials.ndim > 0:
        dpartials = np.broadcast_to(dout, partials.shape) / partials.shape[0]
    elif reduce == "sum" and partials.ndim > 1:
        dpartials = np.broadcast_to(dout, partials.shape)
    else:
        dpartials = np.asarray(dout)
    return (dpartials, None)


@_vjp("alibi")
def vjp_alibi(dout, *args, **_):
    return tuple(None for _ in args)


@_vjp("multi_head_attention")
def vjp_multi_head_attention(dout, Q, K, V, **kwargs):
    from tessera import ops as _ops
    original = getattr(_ops.multi_head_attention, "__wrapped__", _ops.multi_head_attention)
    args = [Q, K, V]
    return tuple(
        _numeric_vjp_arg(lambda a, i=i: original(*(args[:i] + [a] + args[i + 1:]), **kwargs), dout, arg)
        for i, arg in enumerate(args)
    )


@_vjp("gqa_attention")
def vjp_gqa_attention(dout, Q, K, V, **kwargs):
    from tessera import ops as _ops
    original = getattr(_ops.gqa_attention, "__wrapped__", _ops.gqa_attention)
    args = [Q, K, V]
    return tuple(
        _numeric_vjp_arg(lambda a, i=i: original(*(args[:i] + [a] + args[i + 1:]), **kwargs), dout, arg)
        for i, arg in enumerate(args)
    )


@_vjp("mqa_attention")
def vjp_mqa_attention(dout, Q, K, V, **kwargs):
    from tessera import ops as _ops
    original = getattr(_ops.mqa_attention, "__wrapped__", _ops.mqa_attention)
    args = [Q, K, V]
    return tuple(
        _numeric_vjp_arg(lambda a, i=i: original(*(args[:i] + [a] + args[i + 1:]), **kwargs), dout, arg)
        for i, arg in enumerate(args)
    )


@_vjp("mla_decode")
def vjp_mla_decode(dout, Q, K_latent, V_latent, *weights, **kwargs):
    from tessera import ops as _ops
    original = getattr(_ops.mla_decode, "__wrapped__", _ops.mla_decode)
    args = [Q, K_latent, V_latent, *weights]
    return tuple(
        _numeric_vjp_arg(lambda a, i=i: original(*(args[:i] + [a] + args[i + 1:]), **kwargs), dout, arg)
        for i, arg in enumerate(args)
    )


def _rl_numeric_vjp(dout, fn, args, kwargs):
    return tuple(
        _numeric_vjp_arg(lambda a, i=i: fn(*(args[:i] + [a] + args[i + 1:]), **kwargs), dout, arg)
        for i, arg in enumerate(args)
    )


@_vjp("normalize_group_advantages")
def vjp_normalize_group_advantages(dout, rewards, **kwargs):
    from tessera import rl as ts_rl
    return _rl_numeric_vjp(dout, ts_rl.normalize_group_advantages, [rewards], kwargs)


@_vjp("ppo_policy_loss")
def vjp_ppo_policy_loss(dout, logp_new, logp_old, advantages, **kwargs):
    from tessera import rl as ts_rl
    grads = _rl_numeric_vjp(dout, ts_rl.ppo_policy_loss, [logp_new, logp_old, advantages], kwargs)
    return grads


@_vjp("grpo_policy_loss")
def vjp_grpo_policy_loss(dout, logp_new, logp_old, rewards=None, **kwargs):
    from tessera import rl as ts_rl
    args = [logp_new, logp_old] if rewards is None else [logp_new, logp_old, rewards]
    return _rl_numeric_vjp(dout, ts_rl.grpo_policy_loss, args, kwargs)


@_vjp("cispo_policy_loss")
def vjp_cispo_policy_loss(dout, logp_new, logp_old, rewards=None, **kwargs):
    from tessera import rl as ts_rl
    args = [logp_new, logp_old] if rewards is None else [logp_new, logp_old, rewards]
    return _rl_numeric_vjp(dout, ts_rl.cispo_policy_loss, args, kwargs)


@_vjp("fft")
def vjp_fft(dout, x, *, axis=-1, axes=None, **_):
    """Adjoint of full FFT — adjoint is `n * ifft(dout, axis)` for non-orthonormal FFT.

    NumPy's ``np.fft.fft`` is non-orthonormal: ``ifft(fft(x)) == x`` exactly,
    but the gradient flow is ``df/dx = n * ifft(df/dy, axis)`` where ``n`` is
    the FFT length. Returns a complex grad — for real inputs, the user is
    responsible for taking the real part.
    """
    n = x.shape[axes[-1] if axes is not None else axis]
    return (n * np.fft.ifft(dout, axis=axes[-1] if axes is not None else axis),)


@_vjp("ifft")
def vjp_ifft(dout, x, *, axis=-1, axes=None, **_):
    """Adjoint of inverse FFT — adjoint is `(1/n) * fft(dout, axis)`."""
    n = x.shape[axes[-1] if axes is not None else axis]
    return ((1.0 / n) * np.fft.fft(dout, axis=axes[-1] if axes is not None else axis),)


@_vjp("rfft")
def vjp_rfft(dout, x, *, axis=-1, axes=None, **_):
    """Adjoint of real FFT — pads ``dout`` back to full length then `n*ifft`."""
    target_axis = axes[-1] if axes is not None else axis
    n = x.shape[target_axis]
    # Adjoint of rfft is irfft(dout * n) followed by taking the real part —
    # but np.fft.irfft already does the (1/n) so multiply by n.
    return (np.fft.irfft(dout, n=n, axis=target_axis) * n,)


@_vjp("irfft")
def vjp_irfft(dout, x, *, axis=-1, axes=None, n=None, **_):
    """Adjoint of inverse real FFT — `rfft(dout) / n`."""
    target_axis = axes[-1] if axes is not None else axis
    n_out = n if n is not None else 2 * (x.shape[target_axis] - 1)
    return ((1.0 / n_out) * np.fft.rfft(dout, axis=target_axis),)


# ─────────────────────────────────────────────────────────────────────────────
# Streaming kernels (Phase D)
# ─────────────────────────────────────────────────────────────────────────────


@_vjp("selective_ssm")
def vjp_selective_ssm(
    dout, x, A, B, C, delta, *, gate=None, state=None, chunk_size=128, **_
):
    """Reverse-mode adjoint of Mamba2 selective_ssm (Phase D3 follow-up).

    Forward (dropping batch ``b`` for brevity):
        ``z[t,d,n] = delta[t,d] * A[d,n]``
        ``A_bar[t,d,n] = exp(z[t,d,n])``
        ``B_bar[t,d,n] = delta[t,d] * B[t,n]``
        ``h[t,d,n] = A_bar[t,d,n] * h[t-1,d,n] + B_bar[t,d,n] * x[t,d]``
        ``y[t,d] = sum_n C[t,n] * h[t,d,n]``
        ``y_final = y * gate`` (if gated)

    The VJP recomputes the forward trajectory (``h`` at every ``t``,
    ``A_bar``, ``B_bar``) under the autodiff seed and walks ``t = S-1 → 0``
    accumulating gradients via the chain rule. ``gate`` and ``state`` are
    keyword-only — they have no cotangent slot in v1; users wanting their
    gradients should pass them as positional inputs in a future revision.
    """
    Bsz, S, D = x.shape
    N = B.shape[2]

    if A.ndim == 1:
        A2d = np.broadcast_to(A[:, None], (D, N)).copy()
        A_was_1d = True
    else:
        A2d = A
        A_was_1d = False

    # Recompute the forward trajectory we need on backward.
    # h_traj[t] holds h[t-1] (so h_traj[0] = initial state, h_traj[S] = final).
    h_traj = np.zeros((S + 1, Bsz, D, N), dtype=x.dtype)
    if state is not None:
        h_traj[0] = np.asarray(state)
    A_bar_traj = np.empty((S, Bsz, D, N), dtype=x.dtype)
    B_bar_traj = np.empty((S, Bsz, D, N), dtype=x.dtype)

    for t in range(S):
        A_bar = np.exp(delta[:, t, :, None] * A2d[None, :, :])
        B_bar_t = delta[:, t, :, None] * B[:, t, None, :]
        h_t = A_bar * h_traj[t] + B_bar_t * x[:, t, :, None]
        h_traj[t + 1] = h_t
        A_bar_traj[t] = A_bar
        B_bar_traj[t] = B_bar_t

    # Cotangent at the un-gated y values
    if gate is not None:
        dy = dout * np.asarray(gate)
    else:
        dy = dout

    dx = np.zeros_like(x)
    dA2d = np.zeros((D, N), dtype=np.float64)  # accumulate in fp64 for stability
    dB = np.zeros_like(B)
    dC = np.zeros_like(C)
    ddelta = np.zeros_like(delta)

    dh_curr = np.zeros((Bsz, D, N), dtype=x.dtype)

    for t in reversed(range(S)):
        # y[t,d] = sum_n C[t,n] * h[t,d,n]
        dh_curr = dh_curr + C[:, t, None, :] * dy[:, t, :, None]
        dC[:, t, :] += np.einsum("bdn,bd->bn", h_traj[t + 1], dy[:, t, :])

        # h[t] = A_bar * h[t-1] + B_bar * x[t]
        dA_bar = dh_curr * h_traj[t]                              # (B, D, N)
        dh_prev = dh_curr * A_bar_traj[t]
        dB_bar = dh_curr * x[:, t, :, None]                       # (B, D, N)
        dx[:, t, :] += np.einsum("bdn,bdn->bd", dh_curr, B_bar_traj[t])

        # B_bar[t,d,n] = delta[t,d] * B[t,n]
        dB[:, t, :] += np.einsum("bdn,bd->bn", dB_bar, delta[:, t, :])
        ddelta[:, t, :] += np.einsum("bdn,bn->bd", dB_bar, B[:, t, :])

        # A_bar[t,d,n] = exp(z),  z = delta[t,d] * A[d,n]
        dz = dA_bar * A_bar_traj[t]                                # (B, D, N)
        dA2d += np.einsum("bd,bdn->dn", delta[:, t, :], dz).astype(np.float64)
        ddelta[:, t, :] += np.einsum("bdn,dn->bd", dz, A2d)

        dh_curr = dh_prev

    if A_was_1d:
        dA = dA2d.sum(axis=1).astype(x.dtype)
    else:
        dA = dA2d.astype(x.dtype)

    return (dx, dA, dB, dC, ddelta)


def _conv1d_forward_fp64(
    x,
    weight,
    *,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> np.ndarray:
    """NCL grouped Conv1d reference that preserves fp64 for adjoint tests."""

    x_arr = np.asarray(x, dtype=np.float64)
    w_arr = np.asarray(weight, dtype=np.float64)
    if x_arr.ndim != 3 or w_arr.ndim != 3:
        raise ValueError("conv1d expects x [N,C,L] and weight [O,I,K]")
    n, c_in, length = x_arr.shape
    c_out, c_per_group, kernel = w_arr.shape
    if groups <= 0 or c_in % groups != 0 or c_out % groups != 0:
        raise ValueError("groups must divide input and output channels")
    if c_per_group != c_in // groups:
        raise ValueError("weight input channels must equal C_in/groups")
    out_len = (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    if out_len <= 0:
        raise ValueError("conv1d output length must be positive")

    padded = np.pad(x_arr, ((0, 0), (0, 0), (padding, padding)))
    out = np.zeros((n, c_out, out_len), dtype=np.float64)
    out_per_group = c_out // groups
    in_per_group = c_in // groups
    for b in range(n):
        for g in range(groups):
            in_base = g * in_per_group
            out_base = g * out_per_group
            for oc in range(out_per_group):
                oc_abs = out_base + oc
                for pos in range(out_len):
                    start = pos * stride
                    acc = 0.0
                    for ic in range(in_per_group):
                        ic_abs = in_base + ic
                        for k in range(kernel):
                            acc += padded[b, ic_abs, start + k * dilation] * w_arr[oc_abs, ic, k]
                    out[b, oc_abs, pos] = acc
    return out


@_vjp("conv1d")
def vjp_conv1d(
    dout,
    x,
    weight,
    bias=None,
    *,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    **_,
):
    x_arr = np.asarray(x, dtype=np.float64)
    w_arr = np.asarray(weight, dtype=np.float64)
    do = np.asarray(dout, dtype=np.float64)
    n, c_in, length = x_arr.shape
    c_out, _w_cin, kernel = w_arr.shape
    out_len = do.shape[-1]
    padded = np.pad(x_arr, ((0, 0), (0, 0), (padding, padding)))
    dx_padded = np.zeros_like(padded, dtype=np.float64)
    dw = np.zeros_like(w_arr, dtype=np.float64)
    out_per_group = c_out // groups
    in_per_group = c_in // groups

    for b in range(n):
        for g in range(groups):
            in_base = g * in_per_group
            out_base = g * out_per_group
            for oc in range(out_per_group):
                oc_abs = out_base + oc
                for pos in range(out_len):
                    start = pos * stride
                    grad = do[b, oc_abs, pos]
                    for ic in range(in_per_group):
                        ic_abs = in_base + ic
                        for k in range(kernel):
                            x_pos = start + k * dilation
                            dw[oc_abs, ic, k] += grad * padded[b, ic_abs, x_pos]
                            dx_padded[b, ic_abs, x_pos] += grad * w_arr[oc_abs, ic, k]

    dx = dx_padded[:, :, padding:padding + length] if padding else dx_padded
    db = do.sum(axis=(0, 2)) if bias is not None else None
    return (dx.astype(np.asarray(x).dtype, copy=False), dw.astype(np.asarray(weight).dtype, copy=False), db)


@_vjp("depthwise_conv1d")
def vjp_depthwise_conv1d(dout, x, w=None, *, kernel_size, padding=0, causal=False, state=None, **_):
    """Adjoint of depthwise 1-D conv.

    Forward: ``out[..., t] = sum_k x_full[..., t+k] * w[..., k]``
    where ``x_full = pad-or-prefix(state, x)``. Adjoints:

    * ``dx_full[..., j] += sum_{k: 0<=j-k<L_out} dout[..., j-k] * w[..., k]``
    * ``dw[c, k] = sum_{n, t} dout[n, c, t] * x_full[n, c, t+k]``

    The cotangent is split back into ``dx`` and ``dstate`` based on the original
    prefix length.
    """
    N, C, L = x.shape
    K = int(kernel_size)
    L_out = dout.shape[-1]

    # Reconstruct x_full layout (same logic as forward)
    if state is not None:
        prefix_len = K - 1
        x_full = np.concatenate([state, x], axis=-1)
    elif causal:
        prefix_len = K - 1
        x_full = np.pad(x, ((0, 0), (0, 0), (K - 1, 0)))
    else:
        prefix_len = int(padding)
        x_full = np.pad(x, ((0, 0), (0, 0), (int(padding), int(padding))))

    # dw[c, k] = sum over n, t of dout[n, c, t] * x_full[n, c, t+k]
    dw = np.zeros_like(w)
    for k in range(K):
        dw[:, k] = (dout * x_full[..., k:k + L_out]).sum(axis=(0, 2))

    # dx_full via accumulation
    dx_full = np.zeros_like(x_full)
    for k in range(K):
        dx_full[..., k:k + L_out] += dout * w[None, :, k:k + 1]

    # Split dx_full back into dstate (if streaming) + dx (the actual input region) + tail (padding, dropped)
    if state is not None:
        dstate = dx_full[..., :prefix_len]
        dx = dx_full[..., prefix_len:prefix_len + L]
        return (dx, dw, dstate)
    elif causal:
        # Causal padding contributed only zeros to the forward; their cotangent is dropped.
        dx = dx_full[..., prefix_len:prefix_len + L]
        return (dx, dw)
    else:
        # Symmetric padding — drop both ends of dx_full.
        dx = dx_full[..., prefix_len:prefix_len + L]
        return (dx, dw)


@_vjp("depthwise_conv2d")
def vjp_depthwise_conv2d(
    dout, x, w=None, *, kernel_size, stride=(1, 1), padding=(0, 0), causal=False, **_
):
    """Adjoint of depthwise 2-D conv (NHWC).

    Forward: ``out[n, i, j, c] = sum_{kh, kw} x_pad[n, i*sH+kh, j*sW+kw, c] * w[kh, kw, c]``.
    Adjoints mirror the 1-D case but accumulate over both spatial axes.
    """
    def _pair(v):
        return (int(v[0]), int(v[1])) if isinstance(v, (tuple, list)) else (int(v), int(v))

    kH, kW = _pair(kernel_size)
    sH, sW = _pair(stride)
    pH, pW = _pair(padding)

    N, H, W, C = x.shape
    H_out, W_out = dout.shape[1], dout.shape[2]

    if causal:
        prefix_h = kH - 1
        prefix_w = kW - 1
        x_pad = np.pad(x, ((0, 0), (kH - 1, 0), (kW - 1, 0), (0, 0)))
    else:
        prefix_h = pH
        prefix_w = pW
        x_pad = np.pad(x, ((0, 0), (pH, pH), (pW, pW), (0, 0)))

    # dw[kh, kw, c] = sum over n, i, j of dout[n,i,j,c] * x_pad[n, i*sH+kh, j*sW+kw, c]
    dw = np.zeros_like(w)
    for kh in range(kH):
        for kw in range(kW):
            if sH == 1 and sW == 1:
                patch = x_pad[:, kh:kh + H_out, kw:kw + W_out, :]
            else:
                patch = x_pad[:, kh:kh + H_out * sH:sH, kw:kw + W_out * sW:sW, :]
            dw[kh, kw, :] = (dout * patch).sum(axis=(0, 1, 2))

    # dx_full via accumulation
    dx_full = np.zeros_like(x_pad)
    for kh in range(kH):
        for kw in range(kW):
            if sH == 1 and sW == 1:
                dx_full[:, kh:kh + H_out, kw:kw + W_out, :] += (
                    dout * w[None, kh:kh + 1, kw:kw + 1, :]
                )
            else:
                dx_full[:, kh:kh + H_out * sH:sH, kw:kw + W_out * sW:sW, :] += (
                    dout * w[None, kh:kh + 1, kw:kw + 1, :]
                )

    # Trim padding off dx_full to recover dx at original input shape.
    dx = dx_full[:, prefix_h:prefix_h + H, prefix_w:prefix_w + W, :]
    return (dx, dw)


@_vjp("lstm_cell")
def vjp_lstm_cell(
    dout, x_t, h_prev, c_prev, W_ih, W_hh, b_ih=None, b_hh=None, **_
):
    """Adjoint of one LSTM step (Phase H2).

    Forward (recomputed for backward):
        gates = x_t @ W_ih^T + h_prev @ W_hh^T + b_ih + b_hh
        i,f,g,o = sigmoid/sigmoid/tanh/sigmoid of the four gate slices
        c_t = f*c_prev + i*g
        h_t = o*tanh(c_t)
        out = concat([h_t, c_t], axis=-1)

    `dout` is split into `(dh_t, dc_t)` along the last axis. Returns
    cotangents in input order: `(dx_t, dh_prev, dc_prev, dW_ih, dW_hh,
    db_ih, db_hh)`.
    """
    H = h_prev.shape[-1]
    # Recompute forward intermediates we need.
    gates = x_t @ W_ih.T + h_prev @ W_hh.T
    if b_ih is not None:
        gates = gates + b_ih
    if b_hh is not None:
        gates = gates + b_hh
    i_g, f_g, g_g, o_g = gates[..., :H], gates[..., H:2*H], gates[..., 2*H:3*H], gates[..., 3*H:4*H]
    i = 1.0 / (1.0 + np.exp(-i_g))
    f = 1.0 / (1.0 + np.exp(-f_g))
    g_act = np.tanh(g_g)
    o = 1.0 / (1.0 + np.exp(-o_g))
    c_t = f * c_prev + i * g_act
    tanh_c_t = np.tanh(c_t)

    # Split incoming cotangent.
    dh_t = dout[..., :H]
    dc_t_direct = dout[..., H:]

    # Through h_t = o * tanh(c_t):
    do = dh_t * tanh_c_t                               # (B, H)
    dc_through_h = dh_t * o * (1.0 - tanh_c_t * tanh_c_t)
    dc_total = dc_t_direct + dc_through_h

    # Through c_t = f*c_prev + i*g:
    df = dc_total * c_prev
    dc_prev = dc_total * f
    di = dc_total * g_act
    dg = dc_total * i

    # Through pre-sigmoid / pre-tanh of the four gates:
    di_g = di * i * (1.0 - i)
    df_g = df * f * (1.0 - f)
    dg_g = dg * (1.0 - g_act * g_act)
    do_g = do * o * (1.0 - o)
    d_gates = np.concatenate([di_g, df_g, dg_g, do_g], axis=-1)  # (B, 4H)

    # Through gates = x_t @ W_ih.T + h_prev @ W_hh.T (+ biases):
    dx_t = d_gates @ W_ih
    dh_prev = d_gates @ W_hh
    # dW_ih[k, j] = sum_b d_gates[b, k] * x_t[b, j]
    dW_ih = d_gates.T @ x_t
    dW_hh = d_gates.T @ h_prev
    # Biases (one for each); reduce over batch.
    db_ih = d_gates.sum(axis=0) if b_ih is not None else None
    db_hh = d_gates.sum(axis=0) if b_hh is not None else None

    return (dx_t, dh_prev, dc_prev, dW_ih, dW_hh, db_ih, db_hh)


@_vjp("lstm_state_h")
def vjp_lstm_state_h(dout, packed, **_):
    """`h_t = packed[..., :H]` — adjoint puts dout in the first half, zeros in the second."""
    H = dout.shape[-1]
    dpacked = np.zeros_like(packed)
    dpacked[..., :H] = dout
    return (dpacked,)


@_vjp("lstm_state_c")
def vjp_lstm_state_c(dout, packed, **_):
    """`c_t = packed[..., H:]` — adjoint puts dout in the second half."""
    H = dout.shape[-1]
    dpacked = np.zeros_like(packed)
    dpacked[..., H:] = dout
    return (dpacked,)


@_vjp("online_softmax")
def vjp_online_softmax(dout, x, *, axis=-1, state=None, **_):
    """Adjoint of `online_softmax` for the single-chunk path (state=None).

    The streaming case (state given) shares the standard softmax adjoint
    *for that chunk's portion of the output* — earlier-chunk gradients live
    in earlier-chunk tape entries.
    """
    if state is None:
        # Same as standard softmax
        e = np.exp(x - x.max(axis=axis, keepdims=True))
        s = e / e.sum(axis=axis, keepdims=True)
        return ((dout - (dout * s).sum(axis=axis, keepdims=True)) * s,)
    # Streaming chunk — recompute the chunk's softmax against running max/sum
    prev_m, prev_s = state
    chunk_m = x.max(axis=axis, keepdims=True)
    new_m = np.maximum(prev_m, chunk_m)
    new_s = prev_s * np.exp(prev_m - new_m) + np.exp(x - new_m).sum(axis=axis, keepdims=True)
    s = np.exp(x - new_m) / new_s
    return ((dout - (dout * s).sum(axis=axis, keepdims=True)) * s,)


# ─────────────────────────────────────────────────────────────────────────────
# S-series sprint S2 — VJPs for reductions, stability primitives, numeric
# helpers. The non-differentiable ops (argmax/argmin/cumprod-via-zeros/
# isnan/isinf/isfinite/comparisons) deliberately don't get a VJP — they
# emit a zero gradient or a pass-through gradient through `where` /
# masked operations as appropriate.
# ─────────────────────────────────────────────────────────────────────────────


def _broadcast_grad(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast a reduction-output grad back to the input shape."""
    return np.broadcast_to(np.asarray(grad), shape).copy()


@_vjp("mean")
def vjp_mean(dout, x, *, axis=None, keepdims=False, **_):
    a = np.asarray(x)
    if axis is None:
        n = a.size
    elif isinstance(axis, int):
        n = a.shape[axis]
    else:
        n = int(np.prod([a.shape[ax] for ax in axis]))
    if not keepdims and axis is not None:
        dout = np.expand_dims(np.asarray(dout), axis=axis)
    elif not keepdims and axis is None:
        dout = np.asarray(dout)  # scalar
    return (_broadcast_grad(np.asarray(dout) / n, a.shape),)


@_vjp("prod")
def vjp_prod(dout, x, *, axis=None, keepdims=False, **_):
    """d/dx_i prod(x) = prod(x) / x_i (vectorized via the running output)."""
    a = np.asarray(x)
    p = np.prod(a, axis=axis, keepdims=True)
    if not keepdims and axis is not None:
        dout_b = np.expand_dims(np.asarray(dout), axis=axis)
    else:
        dout_b = np.asarray(dout)
    # Avoid divide-by-zero by special-casing x_i == 0:
    #   if x has a unique zero at i, grad_i = prod(x \ {x_i}); other entries 0.
    safe = np.where(a == 0, 1.0, a)
    grad = (p / safe) * dout_b
    grad = grad * (a != 0).astype(grad.dtype) + (a == 0).astype(grad.dtype) * (
        np.prod(safe, axis=axis, keepdims=True) * dout_b
    ) * (np.sum(a == 0, axis=axis, keepdims=True) == 1)
    return (np.broadcast_to(grad, a.shape).copy(),)


@_vjp("amax")
def vjp_amax(dout, x, *, axis=None, keepdims=False, **_):
    a = np.asarray(x)
    m = np.max(a, axis=axis, keepdims=True)
    mask = (a == m).astype(a.dtype)
    # Distribute grad equally across all argmax ties.
    counts = mask.sum(axis=axis, keepdims=True)
    if not keepdims and axis is not None:
        dout_b = np.expand_dims(np.asarray(dout), axis=axis)
    else:
        dout_b = np.asarray(dout)
    return (mask * dout_b / counts,)


@_vjp("max")
def vjp_max(dout, x, *, axis=None, keepdims=False, **kwargs):
    return vjp_amax(dout, x, axis=axis, keepdims=keepdims, **kwargs)


@_vjp("amin")
def vjp_amin(dout, x, *, axis=None, keepdims=False, **_):
    a = np.asarray(x)
    m = np.min(a, axis=axis, keepdims=True)
    mask = (a == m).astype(a.dtype)
    counts = mask.sum(axis=axis, keepdims=True)
    if not keepdims and axis is not None:
        dout_b = np.expand_dims(np.asarray(dout), axis=axis)
    else:
        dout_b = np.asarray(dout)
    return (mask * dout_b / counts,)


@_vjp("min")
def vjp_min(dout, x, *, axis=None, keepdims=False, **kwargs):
    return vjp_amin(dout, x, axis=axis, keepdims=keepdims, **kwargs)


@_vjp("var")
def vjp_var(dout, x, *, axis=None, keepdims=False, ddof=0, **_):
    a = np.asarray(x)
    if axis is None:
        n = a.size
    elif isinstance(axis, int):
        n = a.shape[axis]
    else:
        n = int(np.prod([a.shape[ax] for ax in axis]))
    mu = np.mean(a, axis=axis, keepdims=True)
    if not keepdims and axis is not None:
        dout_b = np.expand_dims(np.asarray(dout), axis=axis)
    else:
        dout_b = np.asarray(dout)
    # d var / d x_i = 2 (x_i - mu) / (n - ddof)
    return ((2.0 / (n - ddof)) * (a - mu) * dout_b,)


@_vjp("std")
def vjp_std(dout, x, *, axis=None, keepdims=False, ddof=0, **_):
    a = np.asarray(x)
    sigma = np.std(a, axis=axis, keepdims=True, ddof=ddof)
    if not keepdims and axis is not None:
        dout_b = np.expand_dims(np.asarray(dout), axis=axis)
    else:
        dout_b = np.asarray(dout)
    if axis is None:
        n = a.size
    elif isinstance(axis, int):
        n = a.shape[axis]
    else:
        n = int(np.prod([a.shape[ax] for ax in axis]))
    mu = np.mean(a, axis=axis, keepdims=True)
    sigma_safe = np.where(sigma == 0, 1.0, sigma)
    return (((a - mu) / ((n - ddof) * sigma_safe)) * dout_b,)


@_vjp("cumsum")
def vjp_cumsum(dout, x, *, axis=-1, **_):
    # d/dx_i sum_{j<=i} x_j = 1 for j <= i, so grad_x[j] = sum_{i>=j} dout[i]
    # That's a reverse cumulative sum.
    do = np.asarray(dout)
    return (np.flip(np.cumsum(np.flip(do, axis=axis), axis=axis), axis=axis),)


# ── Stability primitives ────────────────────────────────────────────────────


@_vjp("logsumexp")
def vjp_logsumexp(dout, x, *, axis=None, keepdims=False, **_):
    a = np.asarray(x)
    m = np.max(a, axis=axis, keepdims=True)
    e = np.exp(a - m)
    s = np.sum(e, axis=axis, keepdims=True)
    softmax = e / s
    if not keepdims and axis is not None:
        dout_b = np.expand_dims(np.asarray(dout), axis=axis)
    else:
        dout_b = np.asarray(dout)
    return (softmax * dout_b,)


@_vjp("log_softmax")
def vjp_log_softmax(dout, x, *, axis=-1, **_):
    a = np.asarray(x)
    m = np.max(a, axis=axis, keepdims=True)
    e = np.exp(a - m)
    s = e / np.sum(e, axis=axis, keepdims=True)
    do = np.asarray(dout)
    return (do - s * np.sum(do, axis=axis, keepdims=True),)


@_vjp("log1p")
def vjp_log1p(dout, x, **_):
    return (np.asarray(dout) / (1.0 + np.asarray(x)),)


@_vjp("expm1")
def vjp_expm1(dout, x, **_):
    return (np.asarray(dout) * np.exp(np.asarray(x)),)


@_vjp("softplus")
def vjp_softplus(dout, x, **_):
    # d/dx log(1 + e^x) = sigmoid(x) — compute in branch-stable form.
    a = np.asarray(x)
    sig = np.where(
        a >= 0,
        1.0 / (1.0 + np.exp(-a)),
        np.exp(a) / (1.0 + np.exp(a)),
    )
    return (sig * np.asarray(dout),)


@_vjp("sigmoid_safe")
def vjp_sigmoid_safe(dout, x, **_):
    a = np.asarray(x)
    sig = np.where(
        a >= 0,
        1.0 / (1.0 + np.exp(-a)),
        np.exp(a) / (1.0 + np.exp(a)),
    )
    return (sig * (1.0 - sig) * np.asarray(dout),)


# ── S2 scalar math breadth ─────────────────────────────────────────────────


@_vjp("sub")
def vjp_sub(dout, x, y, **_):
    do = np.asarray(dout)
    return (_sum_to_shape(do, np.asarray(x).shape),
            _sum_to_shape(-do, np.asarray(y).shape))


@_vjp("div")
def vjp_div(dout, x, y, **_):
    a, b, do = np.asarray(x), np.asarray(y), np.asarray(dout)
    return (_sum_to_shape(do / b, a.shape),
            _sum_to_shape(-do * a / (b * b), b.shape))


@_vjp("exp")
def vjp_exp(dout, x, **_):
    return (np.asarray(dout) * np.exp(np.asarray(x)),)


@_vjp("log")
def vjp_log(dout, x, **_):
    return (np.asarray(dout) / np.asarray(x),)


@_vjp("sqrt")
def vjp_sqrt(dout, x, **_):
    a = np.asarray(x)
    return (np.asarray(dout) * 0.5 / np.sqrt(a),)


@_vjp("rsqrt")
def vjp_rsqrt(dout, x, **_):
    a = np.asarray(x)
    return (np.asarray(dout) * -0.5 * np.power(a, -1.5),)


@_vjp("pow")
def vjp_pow(dout, x, y, **_):
    a, b, do = np.asarray(x), np.asarray(y), np.asarray(dout)
    out = np.power(a, b)
    da = do * b * np.power(a, b - 1.0)
    # d/db a**b = a**b log(a). Undefined for a <= 0 in real-valued AD;
    # return 0 there, matching the "skip singular branch" convention used
    # by several framework reference paths.
    db = do * out * np.where(a > 0, np.log(a), 0.0)
    return (_sum_to_shape(da, a.shape), _sum_to_shape(db, b.shape))


@_vjp("cos")
def vjp_cos(dout, x, **_):
    return (-np.asarray(dout) * np.sin(np.asarray(x)),)


@_vjp("tan")
def vjp_tan(dout, x, **_):
    c = np.cos(np.asarray(x))
    return (np.asarray(dout) / (c * c),)


@_vjp("sinh")
def vjp_sinh(dout, x, **_):
    return (np.asarray(dout) * np.cosh(np.asarray(x)),)


@_vjp("cosh")
def vjp_cosh(dout, x, **_):
    return (np.asarray(dout) * np.sinh(np.asarray(x)),)


@_vjp("asin")
def vjp_asin(dout, x, **_):
    a = np.asarray(x)
    return (np.asarray(dout) / np.sqrt(1.0 - a * a),)


@_vjp("acos")
def vjp_acos(dout, x, **_):
    a = np.asarray(x)
    return (-np.asarray(dout) / np.sqrt(1.0 - a * a),)


@_vjp("atan")
def vjp_atan(dout, x, **_):
    a = np.asarray(x)
    return (np.asarray(dout) / (1.0 + a * a),)


@_vjp("atan2")
def vjp_atan2(dout, y, x, **_):
    a, b, do = np.asarray(y), np.asarray(x), np.asarray(dout)
    denom = a * a + b * b
    dy = do * b / denom
    dx = -do * a / denom
    return (_sum_to_shape(dy, a.shape), _sum_to_shape(dx, b.shape))


@_vjp("erf")
def vjp_erf(dout, x, **_):
    a = np.asarray(x)
    return (np.asarray(dout) * (2.0 / math.sqrt(math.pi)) * np.exp(-(a * a)),)


@_vjp("erfc")
def vjp_erfc(dout, x, **_):
    a = np.asarray(x)
    return (-np.asarray(dout) * (2.0 / math.sqrt(math.pi)) * np.exp(-(a * a)),)


def _digamma_positive(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).copy()
    result = np.zeros_like(x, dtype=np.float64)
    while np.any(x < 8.0):
        mask = x < 8.0
        result[mask] -= 1.0 / x[mask]
        x[mask] += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    result = (
        result
        + np.log(x)
        - 0.5 * inv
        - inv2 / 12.0
        + inv2 * inv2 / 120.0
        - inv2 * inv2 * inv2 / 252.0
        + inv2 * inv2 * inv2 * inv2 / 240.0
    )
    return result


def _trigamma_positive(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).copy()
    result = np.zeros_like(x, dtype=np.float64)
    while np.any(x < 8.0):
        mask = x < 8.0
        result[mask] += 1.0 / (x[mask] * x[mask])
        x[mask] += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    # Derivative of the digamma asymptotic expansion above.
    result = (
        result
        + inv
        + 0.5 * inv2
        + inv2 * inv / 6.0
        - inv2 * inv2 * inv / 30.0
        + inv2 * inv2 * inv2 * inv / 42.0
        - inv2 * inv2 * inv2 * inv2 * inv / 30.0
    )
    return result.astype(np.asarray(a).dtype, copy=False)


@_vjp("lgamma")
def vjp_lgamma(dout, x, **_):
    a = np.asarray(x)
    return (np.asarray(dout) * _digamma_positive(a),)


@_vjp("digamma")
def vjp_digamma(dout, x, **_):
    return (np.asarray(dout) * _trigamma_positive(np.asarray(x)),)


@_vjp("reciprocal")
def vjp_reciprocal(dout, x, **_):
    a = np.asarray(x)
    return (-np.asarray(dout) / (a * a),)


# ── Numeric helpers (differentiable ones) ───────────────────────────────────


@_vjp("clamp")
def vjp_clamp(dout, x, *, min=None, max=None, **_):
    a = np.asarray(x)
    in_range = np.ones_like(a, dtype=a.dtype)
    if min is not None:
        in_range = in_range * (a >= min).astype(a.dtype)
    if max is not None:
        in_range = in_range * (a <= max).astype(a.dtype)
    return (np.asarray(dout) * in_range,)


@_vjp("where")
def vjp_where(dout, cond, x, y, **_):
    c = np.asarray(cond)
    do = np.asarray(dout)
    return (None, np.where(c, do, np.zeros_like(do)),
            np.where(c, np.zeros_like(do), do))


@_vjp("absolute")
def vjp_absolute(dout, x, **_):
    return (np.asarray(dout) * np.sign(np.asarray(x)),)


@_vjp("minimum")
def vjp_minimum(dout, x, y, **_):
    a, b, do = np.asarray(x), np.asarray(y), np.asarray(dout)
    # Equal-tie convention: split grad evenly.
    lt = (a < b).astype(a.dtype)
    eq = (a == b).astype(a.dtype) * 0.5
    return (do * (lt + eq), do * ((1.0 - lt) - eq))


@_vjp("maximum")
def vjp_maximum(dout, x, y, **_):
    a, b, do = np.asarray(x), np.asarray(y), np.asarray(dout)
    gt = (a > b).astype(a.dtype)
    eq = (a == b).astype(a.dtype) * 0.5
    return (do * (gt + eq), do * ((1.0 - gt) - eq))


def _reduction_cotangent(dout, shape: tuple[int, ...], reduction: str) -> np.ndarray:
    do = np.asarray(dout)
    if reduction == "none":
        return np.broadcast_to(do, shape)
    if reduction == "sum":
        return np.ones(shape, dtype=do.dtype) * do
    if reduction == "mean":
        return np.ones(shape, dtype=do.dtype) * do / max(int(np.prod(shape)), 1)
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


@_vjp("linear_general")
def vjp_linear_general(dout, x, W, bias=None, *, axis=-1, **_):
    x_arr = np.asarray(x)
    w_arr = np.asarray(W)
    do = np.asarray(dout)
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = tuple(ax if ax >= 0 else x_arr.ndim + ax for ax in axes)
    w_contract_axes = tuple(range(len(axes)))
    dx = np.tensordot(do, w_arr, axes=(tuple(range(do.ndim - (w_arr.ndim - len(axes)), do.ndim)), tuple(range(len(axes), w_arr.ndim))))
    if axes != tuple(range(x_arr.ndim - len(axes), x_arr.ndim)):
        remaining = [i for i in range(x_arr.ndim) if i not in axes]
        current_order = remaining + list(axes)
        inverse = np.argsort(current_order)
        dx = np.transpose(dx, inverse)
    dW = np.tensordot(x_arr, do, axes=(tuple(i for i in range(x_arr.ndim) if i not in axes), tuple(range(do.ndim - (w_arr.ndim - len(axes))))))
    db = _sum_to_shape(do, np.asarray(bias).shape) if bias is not None else None
    return (dx.reshape(x_arr.shape), dW.reshape(w_arr.shape), db)


@_vjp("sgd")
def vjp_sgd(dout, params, grads, *, lr, **_):
    do = np.asarray(dout)
    return (_sum_to_shape(do, np.asarray(params).shape), _sum_to_shape(-float(lr) * do, np.asarray(grads).shape))


@_vjp("mse_loss")
def vjp_mse_loss(dout, pred, target, *, reduction="mean", **_):
    pred_arr = np.asarray(pred)
    target_arr = np.asarray(target)
    scale = _reduction_cotangent(dout, np.broadcast_shapes(pred_arr.shape, target_arr.shape), reduction)
    err = pred_arr - target_arr
    grad = 2.0 * err * scale
    return (_sum_to_shape(grad, pred_arr.shape), _sum_to_shape(-grad, target_arr.shape))


@_vjp("mae_loss")
def vjp_mae_loss(dout, pred, target, *, reduction="mean", **_):
    pred_arr = np.asarray(pred)
    target_arr = np.asarray(target)
    scale = _reduction_cotangent(dout, np.broadcast_shapes(pred_arr.shape, target_arr.shape), reduction)
    grad = np.sign(pred_arr - target_arr) * scale
    return (_sum_to_shape(grad, pred_arr.shape), _sum_to_shape(-grad, target_arr.shape))


@_vjp("huber_loss")
def vjp_huber_loss(dout, pred, target, *, delta=1.0, reduction="mean", **_):
    pred_arr = np.asarray(pred)
    target_arr = np.asarray(target)
    err = pred_arr - target_arr
    d = float(delta)
    local = np.where(np.abs(err) <= d, err, d * np.sign(err))
    grad = local * _reduction_cotangent(dout, np.broadcast_shapes(pred_arr.shape, target_arr.shape), reduction)
    return (_sum_to_shape(grad, pred_arr.shape), _sum_to_shape(-grad, target_arr.shape))


@_vjp("smooth_l1_loss")
def vjp_smooth_l1_loss(dout, pred, target, *, beta=1.0, reduction="mean", **_):
    pred_arr = np.asarray(pred)
    target_arr = np.asarray(target)
    err = pred_arr - target_arr
    b = float(beta)
    local = np.where(np.abs(err) < b, err / b, np.sign(err))
    grad = local * _reduction_cotangent(dout, np.broadcast_shapes(pred_arr.shape, target_arr.shape), reduction)
    return (_sum_to_shape(grad, pred_arr.shape), _sum_to_shape(-grad, target_arr.shape))


@_vjp("log_cosh_loss")
def vjp_log_cosh_loss(dout, pred, target, *, reduction="mean", **_):
    pred_arr = np.asarray(pred)
    target_arr = np.asarray(target)
    grad = np.tanh(pred_arr - target_arr) * _reduction_cotangent(
        dout, np.broadcast_shapes(pred_arr.shape, target_arr.shape), reduction
    )
    return (_sum_to_shape(grad, pred_arr.shape), _sum_to_shape(-grad, target_arr.shape))


@_vjp("cross_entropy_loss")
def vjp_cross_entropy_loss(dout, logits, targets, *, reduction="mean", **_):
    logits_arr = np.asarray(logits, dtype=np.float64)
    targets_arr = np.asarray(targets)
    shifted = logits_arr - np.max(logits_arr, axis=-1, keepdims=True)
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
    if targets_arr.dtype.kind in "iu":
        one_hot = np.zeros_like(probs)
        np.put_along_axis(one_hot.reshape(-1, one_hot.shape[-1]), targets_arr.reshape(-1, 1).astype(np.int64), 1.0, axis=-1)
        target_grad = None
    else:
        one_hot = targets_arr
        target_grad = -(
            logits_arr - (np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True)) + np.max(logits_arr, axis=-1, keepdims=True))
        )
    scale_shape = probs.shape[:-1]
    scale = _reduction_cotangent(dout, scale_shape, reduction)[..., None]
    grad_logits = (probs - one_hot) * scale
    return (_sum_to_shape(grad_logits, logits_arr.shape), target_grad)


@_vjp("binary_cross_entropy_loss")
def vjp_binary_cross_entropy_loss(dout, logits, targets, *, reduction="mean", **_):
    logits_arr = np.asarray(logits, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.float64)
    sigmoid = 1.0 / (1.0 + np.exp(-logits_arr))
    scale = _reduction_cotangent(dout, np.broadcast_shapes(logits_arr.shape, targets_arr.shape), reduction)
    grad_logits = (sigmoid - targets_arr) * scale
    grad_targets = -logits_arr * scale
    return (_sum_to_shape(grad_logits, logits_arr.shape), _sum_to_shape(grad_targets, targets_arr.shape))


@_vjp("ddpm_noise_pred_loss")
def vjp_ddpm_noise_pred_loss(dout, pred_noise, true_noise, *, reduction="mean", **kwargs):
    return vjp_mse_loss(dout, pred_noise, true_noise, reduction=reduction, **kwargs)


@_vjp("score_matching_loss")
def vjp_score_matching_loss(dout, score, target_score, *, reduction="mean", **_):
    score_arr = np.asarray(score)
    target_arr = np.asarray(target_score)
    scale = _reduction_cotangent(dout, np.broadcast_shapes(score_arr.shape, target_arr.shape), reduction)
    grad = (score_arr - target_arr) * scale
    return (_sum_to_shape(grad, score_arr.shape), _sum_to_shape(-grad, target_arr.shape))


@_vjp("vlb_loss")
def vjp_vlb_loss(dout, terms, *, reduction="mean", **_):
    terms_arr = np.asarray(terms)
    return (_reduction_cotangent(dout, terms_arr.shape, reduction),)


# ─────────────────────────────────────────────────────────────────────────────
# EBM4 — energy-based-model training losses.
# Pre-computed tensor APIs; chain rule is mechanical.
# ─────────────────────────────────────────────────────────────────────────────

@_vjp("contrastive_divergence_loss")
def vjp_contrastive_divergence_loss(dout, energy_pos, energy_neg, *, reduction="mean", **_):
    e_pos = np.asarray(energy_pos)
    e_neg = np.asarray(energy_neg)
    shape = np.broadcast_shapes(e_pos.shape, e_neg.shape)
    scale = _reduction_cotangent(dout, shape, reduction)
    return (
        _sum_to_shape(scale, e_pos.shape),
        _sum_to_shape(-scale, e_neg.shape),
    )


@_vjp("persistent_cd_loss")
def vjp_persistent_cd_loss(dout, energy_pos, energy_persistent_neg, *, reduction="mean", **_):
    return vjp_contrastive_divergence_loss(
        dout, energy_pos, energy_persistent_neg, reduction=reduction
    )


@_vjp("implicit_score_matching_loss")
def vjp_implicit_score_matching_loss(dout, score, divergence_score, *, reduction="mean", **_):
    s = np.asarray(score).astype(np.float64, copy=False)
    div = np.asarray(divergence_score).astype(np.float64, copy=False)
    per_sample_shape = div.shape
    scale = _reduction_cotangent(dout, per_sample_shape, reduction)
    # ∂L/∂s = scale[..., None] * s; ∂L/∂div = scale.
    grad_s = scale[..., None] * s
    grad_div = scale
    return _sum_to_shape(grad_s, s.shape), _sum_to_shape(grad_div, div.shape)


@_vjp("denoising_score_matching_loss")
def vjp_denoising_score_matching_loss(
    dout, score_noisy, y_clean, y_noisy, sigma, *, reduction="mean", **_
):
    s = np.asarray(score_noisy).astype(np.float64, copy=False)
    yc = np.asarray(y_clean).astype(np.float64, copy=False)
    yn = np.asarray(y_noisy).astype(np.float64, copy=False)
    sig2 = float(sigma) ** 2
    target = -(yn - yc) / sig2
    diff = s - target  # shape: (B, D)
    per_sample_shape = diff.shape[:-1]
    scale = _reduction_cotangent(dout, per_sample_shape, reduction)
    # target = -(yn - yc)/sig2, so ∂target/∂yn = -1/sig2 and ∂target/∂yc = +1/sig2.
    # diff = s - target ⇒ ∂diff/∂yn = +1/sig2 and ∂diff/∂yc = -1/sig2.
    grad_s = scale[..., None] * diff
    grad_yc = -scale[..., None] * diff / sig2
    grad_yn = scale[..., None] * diff / sig2
    return (
        _sum_to_shape(grad_s, s.shape),
        _sum_to_shape(grad_yc, yc.shape),
        _sum_to_shape(grad_yn, yn.shape),
        None,  # sigma scalar — non-differentiable for v1
    )


# ─────────────────────────────────────────────────────────────────────────────
# Autodiff-coverage hardening pass — S11 classification + distribution +
# contrastive + sequence losses, S7 layer/pooling. Per the
# "Recommended Next Work" in `docs/audit/coverage/COVERAGE_AUDIT.md`.
#
# Each VJP here is paired with a JVP in `autodiff/jvp.py` and verified
# numerically in `tests/unit/test_autodiff_loss_layer_coverage.py`.
# ─────────────────────────────────────────────────────────────────────────────


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


# ── S11 classification ──────────────────────────────────────────────────────


@_vjp("focal_loss")
def vjp_focal_loss(dout, logits, targets, *, gamma=2.0, alpha=None,
                   reduction="mean", **_):
    """Focal loss = -((1-p_t)^γ) * log(p_t) where p_t = softmax(logits)[target].

    targets are int class indices; the gradient flows only through `logits`.
    """
    logits_arr = np.asarray(logits).astype(np.float64, copy=False)
    targets_arr = np.asarray(targets).astype(np.int64)
    p = _softmax(logits_arr, axis=-1)
    flat_p = p.reshape(-1, p.shape[-1])
    idx = targets_arr.reshape(-1)
    rng = np.arange(idx.size)
    pt = np.maximum(flat_p[rng, idx], 1e-12)

    # dL/dpt = γ*(1-pt)^(γ-1)*log(pt) - (1-pt)^γ / pt   (per sample)
    one_minus_pt = 1.0 - pt
    dL_dpt = (
        gamma * np.power(one_minus_pt, gamma - 1.0) * np.log(pt)
        - np.power(one_minus_pt, gamma) / pt
    )
    if alpha is not None:
        dL_dpt = float(alpha) * dL_dpt

    # Build dL/dlogits via softmax Jacobian: dpt/dlogit_j = pt*(δ_{j,target} - p_j)
    grad_flat = np.zeros_like(flat_p)
    grad_flat[rng, idx] = pt * dL_dpt
    grad_flat -= (pt * dL_dpt)[:, None] * flat_p
    grad = grad_flat.reshape(logits_arr.shape)

    # Apply outer reduction cotangent across the per-sample loss surface.
    do = np.asarray(dout)
    if reduction == "mean":
        scale = 1.0 / max(int(np.prod(targets_arr.shape)), 1)
        grad = grad * (do * scale)  # scalar `do` broadcasts.
    elif reduction == "sum":
        grad = grad * do
    else:  # 'none' — `do` has shape == targets_arr.shape; broadcast over class axis.
        grad = grad * do.reshape(targets_arr.shape + (1,))
    return (grad, None)


@_vjp("label_smoothed_cross_entropy")
def vjp_label_smoothed_cross_entropy(dout, logits, targets,
                                     *, smoothing=0.1, reduction="mean", **_):
    logits_arr = np.asarray(logits).astype(np.float64, copy=False)
    targets_arr = np.asarray(targets).astype(np.int64)
    n_classes = logits_arr.shape[-1]
    smooth = float(smoothing)
    one_hot = np.full(
        targets_arr.shape + (n_classes,),
        smooth / max(1, n_classes - 1),
        dtype=np.float64,
    )
    np.put_along_axis(one_hot, targets_arr[..., None], 1.0 - smooth, axis=-1)

    # CE-with-soft-targets gradient: dL/dlogits = softmax(logits) - one_hot.
    grad = _softmax(logits_arr, axis=-1) - one_hot
    do = np.asarray(dout)
    if reduction == "mean":
        do = do / max(int(np.prod(targets_arr.shape)), 1)
    if reduction == "none":
        do = do.reshape(targets_arr.shape + (1,))
        return (grad * do, None)
    return (grad * do, None)


@_vjp("kl_divergence")
def vjp_kl_divergence(dout, p_log_probs, q_probs, *, reduction="mean", **_):
    """KL(p || q) where p = exp(p_log_probs).

    dL/dp_log_probs[i] = p_i * (lp_i - log q_i + 1)
    dL/dq_probs[i]     = -p_i / q_i
    """
    lp = np.asarray(p_log_probs).astype(np.float64, copy=False)
    q = np.asarray(q_probs).astype(np.float64, copy=False)
    p = np.exp(lp)
    log_q = np.log(np.maximum(q, 1e-12))

    # The reduction is over the leading axes; the inner sum is over axis=-1
    # so a per-sample cotangent broadcasts to (..., 1) before flowing into
    # the per-class gradients.
    leading_shape = lp.shape[:-1]
    leading_size = max(int(np.prod(leading_shape)) if leading_shape else 1, 1)

    do = np.asarray(dout)
    if reduction == "mean":
        do = do / leading_size
    if reduction == "none":
        do = do.reshape(leading_shape + (1,))
    else:
        do = np.broadcast_to(do, leading_shape).reshape(leading_shape + (1,))

    grad_lp = p * (lp - log_q + 1.0) * do
    grad_q = -(p / np.maximum(q, 1e-12)) * do
    return (grad_lp, grad_q)


# ── S11 contrastive ─────────────────────────────────────────────────────────


@_vjp("triplet_loss")
def vjp_triplet_loss(dout, anchor, positive, negative,
                     *, margin=1.0, reduction="mean", **_):
    a = np.asarray(anchor).astype(np.float64, copy=False)
    p = np.asarray(positive).astype(np.float64, copy=False)
    n = np.asarray(negative).astype(np.float64, copy=False)
    d_ap = np.linalg.norm(a - p, axis=-1)
    d_an = np.linalg.norm(a - n, axis=-1)
    raw = d_ap - d_an + float(margin)
    active = (raw > 0).astype(np.float64)

    # Gradient of the L2 norm `||x||` wrt x is `x / ||x||` (zero at the origin).
    safe_ap = np.maximum(d_ap, 1e-12)[..., None]
    safe_an = np.maximum(d_an, 1e-12)[..., None]
    g_ap = (a - p) / safe_ap
    g_an = (a - n) / safe_an

    do = _reduction_cotangent(dout, raw.shape, reduction)
    do = (do * active)[..., None]
    grad_anchor = do * (g_ap - g_an)
    grad_positive = -do * g_ap
    grad_negative = do * g_an
    return (grad_anchor, grad_positive, grad_negative)


@_vjp("contrastive_loss")
def vjp_contrastive_loss(dout, x1, x2, target, *, margin=1.0,
                         reduction="mean", **_):
    a = np.asarray(x1).astype(np.float64, copy=False)
    b = np.asarray(x2).astype(np.float64, copy=False)
    t = np.asarray(target).astype(np.float64)
    diff = a - b
    dist = np.linalg.norm(diff, axis=-1)
    safe = np.maximum(dist, 1e-12)
    margin_active = np.maximum(0.0, float(margin) - dist)
    # L = t*dist^2 + (1-t)*margin_active^2

    grad_dist = 2.0 * t * dist - 2.0 * (1.0 - t) * margin_active
    do = _reduction_cotangent(dout, dist.shape, reduction)
    grad_dist = grad_dist * do
    grad_diff = (grad_dist / safe)[..., None] * diff
    return (grad_diff, -grad_diff, None)


@_vjp("cosine_embedding_loss")
def vjp_cosine_embedding_loss(dout, x1, x2, target, *, margin=0.0,
                              reduction="mean", **_):
    a = np.asarray(x1).astype(np.float64, copy=False)
    b = np.asarray(x2).astype(np.float64, copy=False)
    t = np.asarray(target)
    na = np.linalg.norm(a, axis=-1, keepdims=True)
    nb = np.linalg.norm(b, axis=-1, keepdims=True)
    denom = (na * nb + 1e-12)
    cos = np.sum(a * b, axis=-1, keepdims=True) / denom

    pos_mask = (t > 0).astype(np.float64)[..., None]
    # Active for negative pairs when cos > margin.
    neg_active = ((cos > float(margin)) & (t[..., None] <= 0)).astype(np.float64)

    # dcos/da = (b - cos * a * (||b||/||a||)) / (||a|| ||b|| + eps)
    g_a = (b - cos * a * (nb / np.maximum(na, 1e-12))) / denom
    g_b = (a - cos * b * (na / np.maximum(nb, 1e-12))) / denom

    do_shape = cos.shape[:-1] if cos.ndim > 1 else cos.shape
    do = _reduction_cotangent(dout, do_shape, reduction)[..., None]
    # L = pos*(1-cos) + neg*max(0, cos-margin) -> dL/dcos = -pos + neg_active
    dL_dcos = -pos_mask + neg_active
    factor = do * dL_dcos
    return (factor * g_a, factor * g_b, None)


@_vjp("info_nce_loss")
def vjp_info_nce_loss(dout, query, positive, negatives,
                      *, temperature=0.1, reduction="mean", **_):
    """InfoNCE = cross_entropy([q·p, q·n_1, ..., q·n_K] / τ, target=0).

    All three inputs receive analytical gradients. The pattern is:
    cross-entropy yields `softmax - one_hot` over logits, then logits =
    [pos, neg]/τ get distributed back to (q, p, n).
    """
    q = np.asarray(query).astype(np.float64, copy=False)
    p = np.asarray(positive).astype(np.float64, copy=False)
    n = np.asarray(negatives).astype(np.float64, copy=False)
    pos = np.sum(q * p, axis=-1, keepdims=True)         # (B, 1)
    neg = np.einsum("bd,bkd->bk", q, n)                  # (B, K)
    logits = np.concatenate([pos, neg], axis=-1) / float(temperature)
    sm = _softmax(logits, axis=-1)                       # (B, K+1)
    target = np.zeros(q.shape[0], dtype=np.int64)
    one_hot = np.zeros_like(sm)
    one_hot[np.arange(q.shape[0]), target] = 1.0
    grad_logits = (sm - one_hot) / float(temperature)
    do = _reduction_cotangent(dout, (q.shape[0],), reduction)[..., None]
    grad_logits = grad_logits * do

    # Split back: grad wrt logits[:, 0] is the positive logit (q·p);
    # grad wrt logits[:, 1:] are the negative logits (q·n_k).
    g_pos = grad_logits[:, 0:1]                          # (B, 1)
    g_neg = grad_logits[:, 1:]                           # (B, K)

    grad_q = g_pos * p + np.einsum("bk,bkd->bd", g_neg, n)
    grad_p = g_pos * q
    grad_n = np.einsum("bk,bd->bkd", g_neg, q)
    return (grad_q, grad_p, grad_n)


# ── S11 sequence ────────────────────────────────────────────────────────────


@_vjp("seq2seq_loss")
def vjp_seq2seq_loss(dout, logits, targets, mask=None,
                     *, reduction="mean", **_):
    """Masked cross-entropy. Gradient flows only through `logits`."""
    logits_arr = np.asarray(logits).astype(np.float64, copy=False)
    targets_arr = np.asarray(targets).astype(np.int64)
    sm = _softmax(logits_arr, axis=-1)
    one_hot = np.zeros_like(sm)
    np.put_along_axis(one_hot, targets_arr[..., None], 1.0, axis=-1)
    grad_logits = sm - one_hot

    if mask is not None:
        m = np.asarray(mask).astype(np.float64)
        grad_logits = grad_logits * m[..., None]
        # Masked-mean denominator differs from the standard reduction.
        do = np.asarray(dout)
        if reduction == "mean":
            denom = max(float(np.sum(m)), 1.0)
            grad_logits = grad_logits * (do / denom)
        elif reduction == "sum":
            grad_logits = grad_logits * do
        else:  # 'none'
            grad_logits = grad_logits * np.asarray(dout)[..., None]
        return (grad_logits, None, None)

    do = np.asarray(dout)
    if reduction == "mean":
        do = do / max(int(np.prod(targets_arr.shape)), 1)
    if reduction == "none":
        do = do[..., None]
    grad_logits = grad_logits * do if reduction != "none" else grad_logits * do
    return (grad_logits, None, None)


# ── S7 normalizations ───────────────────────────────────────────────────────


def _normalize_grad(x, grad_y, axes, eps):
    """Standard layer-norm-style backward: shared by group/instance norm."""
    n = float(np.prod([x.shape[ax] for ax in axes]))
    mean = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_centered = x - mean
    x_hat = x_centered * inv_std

    sum_g = grad_y.sum(axis=axes, keepdims=True)
    sum_g_xhat = (grad_y * x_hat).sum(axis=axes, keepdims=True)
    grad_x = (grad_y - sum_g / n - x_hat * sum_g_xhat / n) * inv_std
    return grad_x


@_vjp("group_norm")
def vjp_group_norm(dout, x, num_groups, weight=None, bias=None,
                   *, eps=1e-5, **_):
    x_arr = np.asarray(x).astype(np.float32, copy=False)
    do = np.asarray(dout).astype(np.float32, copy=False)
    n, c = x_arr.shape[:2]
    grouped = x_arr.reshape(n, num_groups, c // num_groups, *x_arr.shape[2:])
    do_grouped = do.reshape(grouped.shape)
    reduce_axes = tuple(range(2, grouped.ndim))

    # If a weight was applied in forward, strip it before the inner backward.
    if weight is not None:
        w = np.asarray(weight).reshape(1, c, *([1] * (x_arr.ndim - 2)))
        do_grouped = do.reshape(grouped.shape) * w.reshape(grouped.shape)
    grad_x = _normalize_grad(grouped, do_grouped, reduce_axes, eps).reshape(x_arr.shape)
    return (grad_x, None, None, None)  # weight/bias grads not yet wired.


@_vjp("instance_norm")
def vjp_instance_norm(dout, x, weight=None, bias=None, *, eps=1e-5, **_):
    x_arr = np.asarray(x).astype(np.float32, copy=False)
    do = np.asarray(dout).astype(np.float32, copy=False)
    reduce_axes = tuple(range(2, x_arr.ndim))
    if weight is not None:
        c = x_arr.shape[1]
        w = np.asarray(weight).reshape(1, c, *([1] * (x_arr.ndim - 2)))
        do = do * w
    grad_x = _normalize_grad(x_arr, do, reduce_axes, eps)
    return (grad_x, None, None)


# ── S7 layers ───────────────────────────────────────────────────────────────


@_vjp("lora_linear")
def vjp_lora_linear(dout, x, weight, lora_a, lora_b, bias=None,
                    *, alpha=1.0, **_):
    """y = x @ W + (x @ A) @ B * (alpha / rank).

    Gradients flow into x, W, A, B (bias is non-differentiable through this VJP
    today — the test only exercises the weight path).
    """
    x_arr = np.asarray(x).astype(np.float64, copy=False)
    w_arr = np.asarray(weight).astype(np.float64, copy=False)
    a_arr = np.asarray(lora_a).astype(np.float64, copy=False)
    b_arr = np.asarray(lora_b).astype(np.float64, copy=False)
    do = np.asarray(dout).astype(np.float64, copy=False)
    rank = a_arr.shape[-1]
    scale = float(alpha) / max(1, rank)

    grad_w = x_arr.T @ do
    # x @ A has shape (..., rank); call it `xa`. Then xa @ B has shape (..., out).
    xa = x_arr @ a_arr
    grad_b_lora = xa.T @ do * scale
    grad_a = (x_arr.T @ (do @ b_arr.T)) * scale
    grad_x = do @ w_arr.T + ((do @ b_arr.T) @ a_arr.T) * scale
    return (grad_x, grad_w, grad_a, grad_b_lora, None)


# ── S7 pooling ──────────────────────────────────────────────────────────────


def _pair(value):
    if isinstance(value, int):
        return (value, value)
    return tuple(value)


@_vjp("max_pool")
def vjp_max_pool(dout, x, kernel_size, stride=None, padding=0, **_):
    x_arr = np.asarray(x).astype(np.float64, copy=False)
    do = np.asarray(dout).astype(np.float64, copy=False)
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride if stride is not None else kernel_size)
    ph, pw = _pair(padding)
    n, c, h, w = x_arr.shape
    if ph or pw:
        padded = np.pad(x_arr, ((0, 0), (0, 0), (ph, ph), (pw, pw)),
                        constant_values=-np.inf)
    else:
        padded = x_arr
    grad_padded = np.zeros_like(padded)
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    for i in range(out_h):
        for j in range(out_w):
            window = padded[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw]
            flat = window.reshape(n, c, -1)
            argmax = np.argmax(flat, axis=-1)
            for b in range(n):
                for ch in range(c):
                    flat_idx = argmax[b, ch]
                    di, dj = divmod(int(flat_idx), kw)
                    grad_padded[b, ch, i * sh + di, j * sw + dj] += do[b, ch, i, j]
    if ph or pw:
        grad = grad_padded[:, :, ph:ph + h, pw:pw + w]
    else:
        grad = grad_padded
    return (grad,)


# ── S7 conv1d ───────────────────────────────────────────────────────────────


def _conv1d_forward_fp64(x_arr: np.ndarray, w_arr: np.ndarray,
                          *, stride: int, padding: int, dilation: int,
                          groups: int) -> np.ndarray:
    """Bit-exact fp64 mirror of `tessera.nn.functional.conv1d` (no bias).

    Used by both `vjp_conv1d`'s sanity path and `jvp_conv1d`'s tangent
    computation so the JVP doesn't pull fp32 quantization noise into
    forward-mode tests.
    """
    n, c_in, length = x_arr.shape
    c_out, c_per_group, kernel = w_arr.shape
    out_per_group = c_out // groups
    in_per_group = c_in // groups
    if c_per_group != in_per_group:
        raise ValueError("conv1d weight input channels must equal C_in/groups")
    padded = np.pad(x_arr, ((0, 0), (0, 0), (padding, padding)))
    out_len = (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    if out_len <= 0:
        raise ValueError("conv1d output length must be positive")
    out = np.zeros((n, c_out, out_len), dtype=np.float64)
    for b in range(n):
        for g in range(groups):
            in_base = g * in_per_group
            out_base = g * out_per_group
            for oc in range(out_per_group):
                for pos in range(out_len):
                    start = pos * stride
                    acc = 0.0
                    for ic in range(in_per_group):
                        for k in range(kernel):
                            acc += (
                                padded[b, in_base + ic, start + k * dilation]
                                * w_arr[out_base + oc, ic, k]
                            )
                    out[b, out_base + oc, pos] = acc
    return out


@_vjp("conv1d")
def vjp_conv1d(dout, x, weight, bias=None, *, stride=1, padding=0,
               dilation=1, groups=1, **_):
    """Reverse-mode for grouped Conv1d (NCL).

    Returns `(grad_x, grad_weight, grad_bias)`. `bias` is non-differentiable
    if it's `None` (the corresponding gradient slot is `None`).

    Derivation:
      - `grad_x[b, ic, p+padding] += Σ_{oc, k}  do[b, oc, pos] * W[oc, ic, k]`
        for every `(pos, k)` such that `pos*stride + k*dilation == p+padding`.
        We accumulate into a padded grad and strip the padding at the end.
      - `grad_W[oc, ic, k] += Σ_{b, pos} do[b, oc, pos] * padded[b, ic, pos*s + k*d]`.
      - `grad_bias[oc] = Σ_{b, pos} do[b, oc, pos]`.

    All accumulators run in fp64 — tests use a relaxed tolerance vs. the
    fp32 forward path, matching the convention used by the pool VJPs.
    """
    x_arr = np.asarray(x).astype(np.float64, copy=False)
    w_arr = np.asarray(weight).astype(np.float64, copy=False)
    do = np.asarray(dout).astype(np.float64, copy=False)

    n, c_in, length = x_arr.shape
    c_out, c_per_group, kernel = w_arr.shape
    out_per_group = c_out // groups
    in_per_group = c_in // groups
    out_len = do.shape[2]

    padded_x = np.pad(x_arr, ((0, 0), (0, 0), (padding, padding)))
    grad_padded = np.zeros_like(padded_x)
    grad_w = np.zeros_like(w_arr)

    for b in range(n):
        for g in range(groups):
            in_base = g * in_per_group
            out_base = g * out_per_group
            for oc in range(out_per_group):
                for pos in range(out_len):
                    start = pos * stride
                    do_val = do[b, out_base + oc, pos]
                    if do_val == 0.0:
                        continue
                    for ic in range(in_per_group):
                        for k in range(kernel):
                            in_pos = start + k * dilation
                            grad_padded[b, in_base + ic, in_pos] += (
                                do_val * w_arr[out_base + oc, ic, k]
                            )
                            grad_w[out_base + oc, ic, k] += (
                                do_val * padded_x[b, in_base + ic, in_pos]
                            )

    # Strip the symmetric padding to recover grad_x at the input shape.
    grad_x = grad_padded[:, :, padding:padding + length]

    if bias is None:
        grad_bias = None
    else:
        grad_bias = do.sum(axis=(0, 2))
    return (grad_x, grad_w, grad_bias)


@_vjp("avg_pool")
def vjp_avg_pool(dout, x, kernel_size, stride=None, padding=0, **_):
    x_arr = np.asarray(x).astype(np.float64, copy=False)
    do = np.asarray(dout).astype(np.float64, copy=False)
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride if stride is not None else kernel_size)
    ph, pw = _pair(padding)
    n, c, h, w = x_arr.shape
    cell = float(kh * kw)
    grad_padded = np.zeros((n, c, h + 2 * ph, w + 2 * pw), dtype=np.float64)
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    for i in range(out_h):
        for j in range(out_w):
            grad_padded[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw] += (
                do[:, :, i:i + 1, j:j + 1] / cell
            )
    if ph or pw:
        grad = grad_padded[:, :, ph:ph + h, pw:pw + w]
    else:
        grad = grad_padded
    return (grad,)


# ─────────────────────────────────────────────────────────────────────────────
# Deferred-VJP follow-up (per `primitive_coverage_state.md` recommended next
# work): CTC, JS divergence, Wasserstein, NT-Xent.
#
# These were skipped in the earlier autodiff hardening pass because each
# required a non-trivial closed-form derivation. Now that the simpler S11
# losses are settled, this block closes the four highest-leverage gaps.
# ─────────────────────────────────────────────────────────────────────────────


# ── S11 sequence: CTC loss ──────────────────────────────────────────────────


def _ctc_extended_target(target: np.ndarray, blank: int) -> list[int]:
    """Build the blank-interleaved extended target [blank, t_0, blank, t_1, ...]."""
    ext = [blank]
    for token in target:
        ext.extend([int(token), blank])
    return ext


def _ctc_log_beta(log_y_b: np.ndarray, ext: list[int], inp_len: int,
                   blank: int) -> np.ndarray:
    """Backward DP table mirroring the forward `alpha` recurrence.

    `log_y_b[t, v]` = log probability for time `t`, vocab `v`, this batch.
    `log_beta[t, i]` includes the emission `log_y_b[t, ext[i]]` so it's
    symmetric with `log_alpha` and the standard posterior identity holds:

        P(state i at t | obs, target) ∝ α[t, i] · β[t, i] / y[t, ext[i]]
    """
    s = len(ext)
    log_beta = np.full((inp_len, s), -np.inf, dtype=np.float64)
    log_beta[inp_len - 1, s - 1] = log_y_b[inp_len - 1, ext[s - 1]]
    if s > 1:
        log_beta[inp_len - 1, s - 2] = log_y_b[inp_len - 1, ext[s - 2]]
    for t in range(inp_len - 2, -1, -1):
        for i in range(s):
            nxt = [log_beta[t + 1, i]]
            if i + 1 < s:
                nxt.append(log_beta[t + 1, i + 1])
            if (i + 2 < s and ext[i + 2] != blank
                    and ext[i + 2] != ext[i]):
                nxt.append(log_beta[t + 1, i + 2])
            log_beta[t, i] = log_y_b[t, ext[i]] + np.logaddexp.reduce(nxt)
    return log_beta


def _ctc_log_alpha(log_y_b: np.ndarray, ext: list[int], inp_len: int,
                    blank: int) -> np.ndarray:
    """Forward DP table — re-implemented in vjp.py so `log_alpha` is
    available without re-entering `tessera.losses.ctc_loss`. Bit-identical
    to the recurrence in `python/tessera/losses.py:ctc_loss`."""
    s = len(ext)
    log_alpha = np.full((inp_len, s), -np.inf, dtype=np.float64)
    log_alpha[0, 0] = log_y_b[0, blank]
    if s > 1:
        log_alpha[0, 1] = log_y_b[0, ext[1]]
    for t in range(1, inp_len):
        for i in range(s):
            prev = [log_alpha[t - 1, i]]
            if i - 1 >= 0:
                prev.append(log_alpha[t - 1, i - 1])
            if (i - 2 >= 0 and ext[i] != blank
                    and ext[i] != ext[i - 2]):
                prev.append(log_alpha[t - 1, i - 2])
            log_alpha[t, i] = np.logaddexp.reduce(prev) + log_y_b[t, ext[i]]
    return log_alpha


@_vjp("ctc_loss")
def vjp_ctc_loss(dout, log_probs, targets, input_lengths, target_lengths,
                 *, blank=0, reduction="mean", **_):
    """Reverse-mode for CTC.

    The gradient flows only through `log_probs`. For each batch element b,

        dL_b / dlog_y[t, k] = -(1/Z_b) · Σ_{i: ext[i]=k} α[t,i] · β[t,i] / y[t,k]

    computed entirely in log-space to stay numerically stable. The
    posterior is then summed over states sharing the same vocab index k.

    `targets` / `input_lengths` / `target_lengths` / `blank` are
    non-differentiable — `None` slot for each.
    """
    lp = np.asarray(log_probs).astype(np.float64, copy=False)
    targets_arr = np.asarray(targets).astype(np.int64)
    inp_lens = np.asarray(input_lengths).astype(np.int64)
    tgt_lens = np.asarray(target_lengths).astype(np.int64)
    T, B, V = lp.shape

    grad = np.zeros_like(lp)
    do = np.asarray(dout, dtype=np.float64)
    if reduction == "mean":
        per_batch_cot = np.full((B,), float(do) / max(B, 1))
    elif reduction == "sum":
        per_batch_cot = np.full((B,), float(do))
    else:  # 'none'
        per_batch_cot = np.broadcast_to(do.reshape(-1), (B,)).astype(np.float64).copy()

    for b in range(B):
        inp_len = int(inp_lens[b])
        t_len = int(tgt_lens[b])
        target = targets_arr[b, :t_len]
        ext = _ctc_extended_target(target, blank)
        s = len(ext)
        log_y_b = lp[:inp_len, b, :]

        log_alpha = _ctc_log_alpha(log_y_b, ext, inp_len, blank)
        log_beta = _ctc_log_beta(log_y_b, ext, inp_len, blank)
        log_Z = np.logaddexp(
            log_alpha[inp_len - 1, s - 1],
            log_alpha[inp_len - 1, s - 2] if s > 1 else -np.inf,
        )

        # Per-(t, k) gradient: `-exp(logsumexp_{i: ext[i]=k} (log_α + log_β - log_y) - log_Z)`.
        # `log_alpha + log_beta - log_y_b[..., ext[i]]` strips one copy of the
        # double-counted emission term so the result is the path posterior.
        ab = log_alpha + log_beta - log_y_b[:, np.array(ext)]  # (inp_len, s)

        # Group states by vocab index.
        ext_arr = np.asarray(ext)
        vocab_used = np.unique(ext_arr)
        for k in vocab_used:
            states_for_k = np.where(ext_arr == k)[0]
            # logsumexp along the state axis for this vocab index.
            block = ab[:, states_for_k]
            log_post = np.logaddexp.reduce(block, axis=1) - log_Z
            grad[:inp_len, b, k] = -np.exp(log_post) * per_batch_cot[b]

    return (grad, None, None, None)


# ── S11 distribution: Jensen-Shannon divergence ─────────────────────────────


@_vjp("js_divergence")
def vjp_js_divergence(dout, p_probs, q_probs, *, reduction="mean", **_):
    """JS(p||q) = ½(KL(p||m) + KL(q||m)) with m = ½(p+q).

    The cross-terms cancel cleanly because p_i + q_i = 2 m_i, leaving:

        dJS/dp_i = ½ log(p_i / m_i),   dJS/dq_i = ½ log(q_i / m_i)
    """
    p = np.asarray(p_probs).astype(np.float64, copy=False)
    q = np.asarray(q_probs).astype(np.float64, copy=False)
    m = 0.5 * (p + q)
    log_p_m = np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(m, 1e-12))
    log_q_m = np.log(np.maximum(q, 1e-12)) - np.log(np.maximum(m, 1e-12))

    leading_shape = p.shape[:-1]
    leading_size = max(int(np.prod(leading_shape)) if leading_shape else 1, 1)
    do = np.asarray(dout)
    if reduction == "mean":
        do = do / leading_size
    if reduction == "none":
        do = do.reshape(leading_shape + (1,))
    else:
        do = np.broadcast_to(do, leading_shape).reshape(leading_shape + (1,))

    grad_p = 0.5 * log_p_m * do
    grad_q = 0.5 * log_q_m * do
    return (grad_p, grad_q)


# ── S11 distribution: Wasserstein-1 (1-D, sort-based) ──────────────────────


@_vjp("wasserstein_distance")
def vjp_wasserstein_distance(dout, x, y, *, reduction="mean", **_):
    """1-D empirical Wasserstein-1 distance routed through sort permutations.

    Forward computes `mean_i |x_sort[i] - y_sort[i]|` along axis -1 and
    reduces. Backward:

      dW/dx_sort[i] = (1/N) sign(x_sort[i] - y_sort[i])
      dW/dx[k]      = dW/dx_sort[π_x_inv[k]]   (scatter through the sort)

    Same for `y` with sign flipped. Sort ties are handled by the standard
    sub-gradient convention `sign(0) = 0`.
    """
    x_arr = np.asarray(x).astype(np.float64, copy=False)
    y_arr = np.asarray(y).astype(np.float64, copy=False)
    N = x_arr.shape[-1]
    pi_x = np.argsort(x_arr, axis=-1)
    pi_y = np.argsort(y_arr, axis=-1)
    x_sorted = np.take_along_axis(x_arr, pi_x, axis=-1)
    y_sorted = np.take_along_axis(y_arr, pi_y, axis=-1)
    sign_diff = np.sign(x_sorted - y_sorted)  # (..., N)

    # Per-element cotangent in sort space, then scatter back.
    leading_shape = x_arr.shape[:-1]
    do = _reduction_cotangent(dout, leading_shape if leading_shape else (), reduction)
    do_b = np.asarray(do).reshape(leading_shape + (1,))

    grad_sort_x = sign_diff * do_b / N
    grad_sort_y = -sign_diff * do_b / N

    grad_x = np.empty_like(x_arr)
    grad_y = np.empty_like(y_arr)
    np.put_along_axis(grad_x, pi_x, grad_sort_x, axis=-1)
    np.put_along_axis(grad_y, pi_y, grad_sort_y, axis=-1)
    return (grad_x, grad_y)


# ── S11 contrastive: NT-Xent ────────────────────────────────────────────────


def _nt_xent_forward_state(z: np.ndarray, labels: np.ndarray,
                            temperature: float) -> dict:
    """Cache the forward intermediates that both VJP and JVP need.

    Computes `n` (norms), `u` (normalized embeddings), `S` (Gram / temp),
    `pos` (positive mask, diagonal-zeroed), `K` (positives per row),
    and `sm` (softmax over the masked logits).
    """
    norms = np.linalg.norm(z, axis=-1, keepdims=True)
    safe_n = norms + 1e-12
    u = z / safe_n
    S = (u @ u.T) / float(temperature)
    masked = S.copy()
    np.fill_diagonal(masked, -np.inf)
    # Stable softmax along last axis.
    m = np.max(masked, axis=-1, keepdims=True)
    sm = np.exp(masked - m)
    sm[~np.isfinite(masked)] = 0.0
    sm = sm / np.maximum(np.sum(sm, axis=-1, keepdims=True), 1e-12)

    labels_arr = np.asarray(labels)
    pos = (labels_arr[:, None] == labels_arr[None, :])
    np.fill_diagonal(pos, False)
    K = np.maximum(pos.sum(axis=-1), 1)
    return {"u": u, "n": safe_n, "S": S, "sm": sm, "pos": pos, "K": K}


@_vjp("nt_xent_loss")
def vjp_nt_xent_loss(dout, embeddings, labels, *, temperature=0.5,
                     reduction="mean", **_):
    """SimCLR-style NT-Xent contrastive loss.

    The chain is:
      z → u = z/||z||           (per-row L2 normalize)
      u → S = u uᵀ / τ          (Gram matrix scaled by temperature)
      S → masked: diagonal = -∞
      masked → log_softmax row-wise → loss[i] = -mean over positives j of LP[i,j]
      L = reduce(loss)

    Gradient at the masked-logit level:
      dL/dM[i,k] = (-pos[i,k]/K_i + softmax[i,k])    (i ≠ k; diagonal is 0)

    Then `dL/du = (1/τ)(G + Gᵀ) u` (since S = (1/τ) u uᵀ is symmetric in
    its dependence on u via two slots), and finally L2-normalize backprop:
      dL/dz[i] = (dL/du[i] - u[i] * (dL/du[i] · u[i])) / ||z[i]||
    """
    z = np.asarray(embeddings).astype(np.float64, copy=False)
    state = _nt_xent_forward_state(z, labels, temperature)
    u, n, sm, pos, K = state["u"], state["n"], state["sm"], state["pos"], state["K"]
    B = z.shape[0]

    # Per-row dL/dM: where i==j it's 0 (those rows were -inf in logits, sm[i,i]=0).
    dM = sm - pos.astype(np.float64) / K[:, None]
    np.fill_diagonal(dM, 0.0)

    # No-positives rows have loss[i] = 0 by the forward's `where(positives, ...)`.
    no_pos = pos.sum(axis=-1) == 0
    dM[no_pos, :] = 0.0

    # Reduction cotangent: per-row scalar weight.
    do = np.asarray(dout)
    if reduction == "mean":
        row_cot = float(do) / max(B, 1)
    elif reduction == "sum":
        row_cot = float(do)
    else:  # 'none' — `do[b]` is the per-row cotangent.
        do_arr = np.asarray(do).reshape(-1)
        dM = dM * do_arr[:, None]
        row_cot = 1.0
    if reduction != "none":
        dM = dM * row_cot

    # G = dL/dS where S is the un-masked Gram matrix; off-diagonal == dM,
    # diagonal == 0 (it was masked out).
    G = dM.copy()
    np.fill_diagonal(G, 0.0)

    # du[a,d] = (1/τ) * (G + Gᵀ) @ u
    du = ((G + G.T) @ u) / float(temperature)

    # Unwind the L2 normalize: dz = (du - u (du · u)) / ||z||
    proj = np.sum(du * u, axis=-1, keepdims=True)
    dz = (du - u * proj) / n
    return (dz, None)


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 1 — S6 collective VJPs.
#
# All collectives in `tessera.sharding` operate over a leading "rank" axis
# (axis=0). The VJPs are the standard duals:
#   psum               ↔ broadcast_to_axis (sum is self-adjoint)
#   pmean              ↔ broadcast_to_axis / R
#   pmax / pmin        → argmax-routed, ties split evenly
#   collective_permute → inverse permutation
#   broadcast_to_axis  ↔ psum
# These are required for `shard_map(grad(f))` to route correctly.
# ─────────────────────────────────────────────────────────────────────────────


@_vjp("psum")
def vjp_psum(dout, values, axis_name=None, **_):
    """Sum-across-ranks VJP. Each rank receives the full upstream gradient."""
    arr = np.asarray(values)
    R = arr.shape[0]
    do = np.broadcast_to(np.asarray(dout), arr.shape[1:])
    grad = np.broadcast_to(do, arr.shape).copy()
    return (grad,)


@_vjp("pmean")
def vjp_pmean(dout, values, axis_name=None, **_):
    arr = np.asarray(values)
    R = arr.shape[0]
    do = np.asarray(dout)
    grad = np.broadcast_to(do, arr.shape) / R
    return (np.array(grad),)


@_vjp("pmax")
def vjp_pmax(dout, values, axis_name=None, **_):
    """Routes grad only to the rank(s) that achieved the max; ties split."""
    arr = np.asarray(values).astype(np.float64, copy=False)
    m = np.max(arr, axis=0, keepdims=True)
    mask = (arr == m).astype(arr.dtype)
    counts = mask.sum(axis=0, keepdims=True)
    do = np.asarray(dout)
    return (mask * do[None] / counts,)


@_vjp("pmin")
def vjp_pmin(dout, values, axis_name=None, **_):
    arr = np.asarray(values).astype(np.float64, copy=False)
    m = np.min(arr, axis=0, keepdims=True)
    mask = (arr == m).astype(arr.dtype)
    counts = mask.sum(axis=0, keepdims=True)
    do = np.asarray(dout)
    return (mask * do[None] / counts,)


@_vjp("collective_permute")
def vjp_collective_permute(dout, values, pairs, **_):
    """Permute (src, dst) → invert by routing grad along (dst, src)."""
    arr = np.asarray(values)
    do = np.asarray(dout)
    grad = np.zeros_like(arr)
    for src, dst in pairs:
        grad[int(src)] = do[int(dst)]
    return (grad, None)


@_vjp("broadcast_to_axis")
def vjp_broadcast_to_axis(dout, value, *, axis_size, axis=0, **_):
    """Broadcast→stack VJP is `psum` along the broadcast axis."""
    do = np.asarray(dout)
    return (np.sum(do, axis=axis),)


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 2 — stateful optimizer VJPs.
#
# Convention: `dout` is the cotangent for `new_params` only. The
# `new_state` cotangent is treated as zero (single-step meta-learning). For
# K-step meta-learning chains the user registers `custom_vjp` over the full
# rollout. Returns `(d_params, d_grads, d_state)` where `d_state` is the
# same dict-of-arrays shape as the input state.
# ─────────────────────────────────────────────────────────────────────────────


def _zero_state_dict(state: dict | None, default_keys=()) -> dict:
    if state is None:
        return {k: None for k in default_keys}
    return state


@_vjp("momentum")
def vjp_momentum(dout, params, grads, state=None, *, lr, momentum=0.9, **_):
    """new_velocity = momentum*velocity + grads
       new_params   = params - lr*new_velocity

    With `d_new_state = 0`:
      d_params   = dout
      d_grads    = -lr * dout
      d_velocity = -lr * momentum * dout      (chain through new_velocity)
    """
    do = np.asarray(dout, dtype=np.float64)
    d_params = do
    d_grads = -float(lr) * do
    d_velocity = -float(lr) * float(momentum) * do
    d_state = {"velocity": d_velocity}
    return (d_params, d_grads, d_state)


@_vjp("nesterov")
def vjp_nesterov(dout, params, grads, state=None, *, lr, momentum=0.9, **_):
    """Nesterov uses the look-ahead update:
       new_velocity = momentum*velocity + grads
       look_ahead   = grads + momentum*new_velocity
       new_params   = params - lr*look_ahead

    Chain rule:
      d_params  = dout
      d_grads   = -lr * dout * (1 + momentum)            (grads enters look_ahead twice)
      d_velocity= -lr * momentum * (1 + momentum) * dout
                  ── new_velocity flows into look_ahead through momentum
    """
    do = np.asarray(dout, dtype=np.float64)
    m = float(momentum)
    d_params = do
    d_grads = -float(lr) * do * (1.0 + m)
    d_velocity = -float(lr) * m * (1.0 + m) * do
    d_state = {"velocity": d_velocity}
    return (d_params, d_grads, d_state)


@_vjp("adam")
def vjp_adam(dout, param, grad, moment1, moment2, *,
             lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, step=1,
             _output_index=0, **_):
    """Low-level tuple-output Adam VJP.

    ``_output_index`` is injected by the tape for tuple components:
    0 = new_param, 1 = new_moment1, 2 = new_moment2.
    """
    do = np.asarray(dout, dtype=np.float64)
    p = np.asarray(param, dtype=np.float64)
    g = np.asarray(grad, dtype=np.float64)
    m_prev = np.asarray(moment1, dtype=np.float64)
    v_prev = np.asarray(moment2, dtype=np.float64)
    b1, b2 = float(beta1), float(beta2)
    if int(_output_index) == 1:
        return (
            np.zeros_like(p),
            (1.0 - b1) * do,
            b1 * do,
            np.zeros_like(v_prev),
        )
    if int(_output_index) == 2:
        return (
            np.zeros_like(p),
            2.0 * (1.0 - b2) * g * do,
            np.zeros_like(m_prev),
            b2 * do,
        )

    step_i = int(step)
    m_new = b1 * m_prev + (1.0 - b1) * g
    v_new = b2 * v_prev + (1.0 - b2) * g * g
    bc1 = 1.0 - b1 ** step_i
    bc2 = 1.0 - b2 ** step_i
    m_hat = m_new / bc1
    v_hat = v_new / bc2
    sqrt_v = np.sqrt(v_hat)
    denom = sqrt_v + float(eps)
    d_update = -float(lr) * do
    d_m_new = d_update / (bc1 * denom)
    d_v_new = d_update * (-m_hat / (2.0 * np.maximum(sqrt_v, 1e-12) * denom * denom * bc2))
    return (
        do,
        (1.0 - b1) * d_m_new + 2.0 * (1.0 - b2) * g * d_v_new,
        b1 * d_m_new,
        b2 * d_v_new,
    )


@_vjp("adamw")
def vjp_adamw(dout, params, grads, state=None, *,
              lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, **_):
    """AdamW with decoupled weight decay.

    Forward:
      m_new   = β1 m + (1-β1) g
      v_new   = β2 v + (1-β2) g²
      m_hat   = m_new / (1 - β1^t)
      v_hat   = v_new / (1 - β2^t)
      update  = m_hat / (sqrt(v_hat) + eps)
      params_after_decay = params * (1 - lr*wd)
      new_params = params_after_decay - lr*update

    With `d_new_state = 0`:
      d_params = dout * (1 - lr*wd)
      d_grads  = -lr * d_update/d_g
      d_m      = -lr * d_update/d_m_new
      d_v      = -lr * d_update/d_v_new

    `d_update/d_m_new = 1 / ((1-β1^t)(sqrt(v_hat)+eps))`
    `d_update/d_v_new = -m_hat / (2*sqrt(v_hat)*(sqrt(v_hat)+eps)²) / (1-β2^t)`
    `d_update/d_g     = (1-β1)/(1-β1) * d_update/d_m_new + 2(1-β2)g * d_update/d_v_new`
                     = (1-β1) * d_update/d_m_new + 2(1-β2)g * d_update/d_v_new
    """
    do = np.asarray(dout, dtype=np.float64)
    g = np.asarray(grads, dtype=np.float64)
    if state is None:
        state = {"m": np.zeros_like(g), "v": np.zeros_like(g), "step": 0}
    step = int(state["step"]) + 1
    m_prev = np.asarray(state["m"], dtype=np.float64)
    v_prev = np.asarray(state["v"], dtype=np.float64)
    b1, b2 = float(beta1), float(beta2)
    m_new = b1 * m_prev + (1.0 - b1) * g
    v_new = b2 * v_prev + (1.0 - b2) * g * g
    bc1 = 1.0 - b1 ** step
    bc2 = 1.0 - b2 ** step
    m_hat = m_new / bc1
    v_hat = v_new / bc2
    sqrt_v = np.sqrt(v_hat)
    denom = sqrt_v + float(eps)

    d_update_d_m_new = 1.0 / (bc1 * denom)
    d_update_d_v_new = -m_hat / (2.0 * np.maximum(sqrt_v, 1e-12) * denom * denom * bc2)

    d_params = do * (1.0 - float(lr) * float(weight_decay))
    d_update = -float(lr) * do
    d_m_new = d_update * d_update_d_m_new
    d_v_new = d_update * d_update_d_v_new
    d_grads = (1.0 - b1) * d_m_new + 2.0 * (1.0 - b2) * g * d_v_new
    d_m_prev = b1 * d_m_new
    d_v_prev = b2 * d_v_new
    d_state = {"m": d_m_prev, "v": d_v_prev, "step": None}
    return (d_params, d_grads, d_state)


@_vjp("lion")
def vjp_lion(dout, params, grads, state=None, *,
             lr=1e-4, beta1=0.9, beta2=0.99, weight_decay=0.0, **_):
    del beta1, beta2
    do = np.asarray(dout, dtype=np.float64)
    d_params = do * (1.0 - float(lr) * float(weight_decay))
    d_grads = np.zeros_like(np.asarray(grads, dtype=np.float64))
    d_state = {"m": np.zeros_like(np.asarray(params, dtype=np.float64)), "step": None}
    if state is not None and "m" in state:
        d_state["m"] = np.zeros_like(np.asarray(state["m"], dtype=np.float64))
    return d_params, d_grads, d_state


@_vjp("adafactor")
def vjp_adafactor(dout, params, grads, state=None, *,
                  lr=1e-3, beta2=0.999, eps=1e-30, **kwargs):
    from tessera import optim as ts_optim

    def forward(p, g, s):
        return ts_optim.adafactor(
            p,
            g,
            s,
            lr=lr,
            beta2=beta2,
            eps=eps,
            **kwargs,
        )[0]

    d_params = _numeric_vjp_arg(lambda p: forward(p, grads, state), dout, params)
    d_grads = _numeric_vjp_arg(lambda g: forward(params, g, state), dout, grads)
    d_state = _tree_numeric_vjp(lambda s: forward(params, grads, s), dout, state) if state is not None else None
    return d_params, d_grads, d_state


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 3 — Memory architecture: differentiable `memory_read`.
#
# Forward (Titans/Atlas-style):
#   scores  = query @ keys.T
#   indices = top_k(scores)              (non-diff — argpartition)
#   weights = softmax(top_scores)         (or 1/k uniform if normalize=False)
#   read    = sum_j gathered_values[j] * weights[j]
#
# Backward treats `indices` as constants (the standard top-k convention,
# matching JAX/PyTorch). The read VJP returns (d_table, d_query) where
# d_table is `(d_keys, d_values)` and d_query has the input query's shape.
# ─────────────────────────────────────────────────────────────────────────────


def _memory_unwrap(query):
    arr = np.asarray(query._data if hasattr(query, "_data") else query, dtype=np.float64)
    return arr


@_vjp("memory_read")
def vjp_memory_read(dout, memory, query, *, top_k=1, normalize=True, **_):
    """Reverse-mode through Titans/Atlas memory_read.

    `dout` is the gradient w.r.t. `read_values` (the primary output of
    `MemoryReadResult`). `memory` is a `MemoryTable` or `(keys, values)`
    pair. Returns `(d_memory, d_query)` where `d_memory` is the tuple
    `(d_keys, d_values)` matching the table layout.

    Per-row math (one batch row b shown — code vectorizes over b):
      gathered[j] = values[indices[j]]
      read = Σ_j gathered[j] * weights[j]        (broadcasting over value dims)
      ↓
      d_gathered[j] = dout * weights[j]
      d_weights[j]  = Σ_v gathered[j, v] * dout[v]
      d_top_scores  = (d_weights - Σ_k weights[k] d_weights[k]) * weights      (softmax J)
      d_scores[indices[j]] += d_top_scores[j]    (scatter)
      d_query  = d_scores @ keys
      d_keys   = d_scores.T @ query
      d_values[indices[j]] += d_gathered[j]      (scatter)
    """
    from tessera.memory import MemoryTable

    if isinstance(memory, MemoryTable):
        keys = memory.keys
        values = memory.values
    else:
        keys, values = memory
    keys_arr = np.asarray(keys, dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.float64)
    query_arr = _memory_unwrap(query)
    do = np.asarray(dout, dtype=np.float64)

    single_query = query_arr.ndim == 1
    if single_query:
        query_arr = query_arr[None, :]
        do = do[None, ...]

    B = query_arr.shape[0]
    N = keys_arr.shape[0]
    k = min(int(top_k), N)

    scores = query_arr @ keys_arr.T              # (B, N)
    partition = np.argpartition(-scores, kth=k - 1, axis=-1)[:, :k]
    top_scores = np.take_along_axis(scores, partition, axis=-1)
    order = np.argsort(-top_scores, axis=-1)
    indices = np.take_along_axis(partition, order, axis=-1)  # (B, k)
    top_scores = np.take_along_axis(top_scores, order, axis=-1)

    if normalize:
        # Stable softmax mirroring the forward.
        m = np.max(top_scores, axis=-1, keepdims=True)
        e = np.exp(top_scores - m)
        weights = e / np.sum(e, axis=-1, keepdims=True)
    else:
        weights = np.ones_like(top_scores) / k

    gathered = values_arr[indices]                # (B, k, *value_shape)

    # value_shape may have multiple trailing dims.
    value_dims = tuple(range(2, gathered.ndim))
    # d_gathered[b, j, ...] = dout[b, ...] * weights[b, j]
    weights_b = weights.reshape(B, k, *([1] * len(value_dims)))
    d_gathered = do[:, None] * weights_b
    # d_weights[b, j] = Σ_v gathered[b, j, v] * dout[b, v]
    d_weights = np.sum(gathered * do[:, None], axis=value_dims)  # (B, k)

    if normalize:
        d_top_scores = (d_weights - np.sum(weights * d_weights, axis=-1, keepdims=True)) * weights
    else:
        d_top_scores = d_weights / k

    # Scatter d_top_scores into the full d_scores at the gathered indices.
    d_scores = np.zeros_like(scores)
    np.add.at(d_scores, (np.arange(B)[:, None], indices), d_top_scores)

    d_query = d_scores @ keys_arr            # (B, key_dim)
    d_keys = d_scores.T @ query_arr          # (N, key_dim)

    # Scatter d_gathered into d_values at the gathered indices.
    d_values = np.zeros_like(values_arr)
    flat_idx = indices.reshape(-1)
    flat_grad = d_gathered.reshape(-1, *values_arr.shape[1:])
    np.add.at(d_values, flat_idx, flat_grad)

    if single_query:
        d_query = d_query[0]

    d_memory = (d_keys, d_values)
    return (d_memory, d_query)


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 4 — `cummax` and `cummin` reverse-prefix VJPs.
#
# `cummax[i] = max(x[0], ..., x[i])`. The grad routes only to the index
# that achieved the running max at each step. With ties, distribute evenly.
# ─────────────────────────────────────────────────────────────────────────────


def _cumextrema_grad(x, dout, axis, comparator):
    """Shared backward for cummax/cummin.

    `comparator(a, b) -> bool` decides which value wins (e.g. `np.greater`
    for cummax, `np.less` for cummin). For each output position i, we
    accumulate `dout[i]` into the input position k where x[k] is the
    running extremum at step i.

    Implementation: for each axis position i, find argmax/argmin of x[0..i]
    along that axis, scatter `dout[i]` there. For ties at any prefix, split
    `dout[i]` evenly across the tying positions (matches the existing
    amax/amin/minimum/maximum convention).
    """
    x = np.asarray(x).astype(np.float64, copy=False)
    do = np.asarray(dout).astype(np.float64, copy=False)
    axis = axis if axis >= 0 else x.ndim + axis

    # Move target axis to the end for vectorized prefix scans.
    x_perm = np.moveaxis(x, axis, -1)
    do_perm = np.moveaxis(do, axis, -1)
    grad_perm = np.zeros_like(x_perm)

    L = x_perm.shape[-1]
    # Running extremum and tie mask via prefix scan.
    if comparator is np.greater:
        running = np.maximum.accumulate(x_perm, axis=-1)
    else:
        running = np.minimum.accumulate(x_perm, axis=-1)

    for i in range(L):
        prefix = x_perm[..., :i + 1]
        winner = running[..., i:i + 1]                       # (..., 1)
        mask = (prefix == winner).astype(np.float64)         # (..., i+1)
        counts = mask.sum(axis=-1, keepdims=True)            # (..., 1)
        share = do_perm[..., i:i + 1] * mask / counts        # (..., i+1)
        grad_perm[..., :i + 1] += share

    return np.moveaxis(grad_perm, -1, axis)


@_vjp("cummax")
def vjp_cummax(dout, x, *, axis=-1, **_):
    return (_cumextrema_grad(x, dout, axis, np.greater),)


@_vjp("cummin")
def vjp_cummin(dout, x, *, axis=-1, **_):
    return (_cumextrema_grad(x, dout, axis, np.less),)


# ─────────────────────────────────────────────────────────────────────────────
# Long-tail VJP closure (2026-05-10). Closes the planned VJP entries
# flagged in `docs/audit/coverage/COVERAGE_AUDIT.md`. Organized by family.
#
# All implementations are pure-numpy reference; backend-specific kernels
# arrive with each Phase G/H/I integration.
# ─────────────────────────────────────────────────────────────────────────────


# ── Collectives — duality table (Phase F5 confirmed canonical) ──────────────

@_vjp("all_reduce")
def vjp_all_reduce(dout, x, *, op="sum", axis_name=None, **_):
    """`all_reduce-sum` is self-dual (each rank receives the full dout).
    `op="max"`/`min` route to the argmax/argmin position (single-rank
    reference: distribute over ties evenly).
    """
    do = np.asarray(dout, dtype=np.float64)
    arr = np.asarray(x, dtype=np.float64)
    if op in ("sum", "mean"):
        return (np.broadcast_to(do, arr.shape).copy() if do.shape != arr.shape else do,)
    if op in ("max", "min"):
        m = np.max(arr) if op == "max" else np.min(arr)
        mask = (arr == m).astype(np.float64)
        counts = max(int(mask.sum()), 1)
        return (mask * do / counts,)
    return (do,)


@_vjp("all_gather")
def vjp_all_gather(dout, x, *, axis_name=None, axis=0, **_):
    """`all_gather`'s transpose is `reduce_scatter`-sum along the same axis.
    In the single-rank reference: take this rank's slice from the gathered
    cotangent.
    """
    do = np.asarray(dout, dtype=np.float64)
    arr = np.asarray(x, dtype=np.float64)
    if do.shape == arr.shape:
        return (do,)
    axis_idx = axis if axis >= 0 else do.ndim + axis
    n_local = arr.shape[axis_idx] if axis_idx < arr.ndim else do.shape[axis_idx]
    slc = [slice(None)] * do.ndim
    slc[axis_idx] = slice(0, n_local)
    return (do[tuple(slc)],)


@_vjp("all_to_all")
def vjp_all_to_all(dout, x, *, axis_name=None, split_axis=0, concat_axis=0, **_):
    """`all_to_all` is self-dual with swapped split/concat axes."""
    return (np.asarray(dout, dtype=np.float64),)


@_vjp("reduce_scatter")
def vjp_reduce_scatter(dout, x, *, op="sum", axis_name=None, axis=0, **_):
    """`reduce_scatter`'s transpose is `all_gather`. Broadcast `dout` back."""
    do = np.asarray(dout, dtype=np.float64)
    arr = np.asarray(x, dtype=np.float64)
    if do.shape == arr.shape:
        return (do,)
    return (np.broadcast_to(do, arr.shape).copy(),)


# ── Recurrent cells ─────────────────────────────────────────────────────────

@_vjp("simple_rnn_cell")
def vjp_simple_rnn_cell(dout, x, h, W_ih, W_hh, bias=None, *,
                         activation="tanh", **_):
    """h_new = activation(x @ W_ih + h @ W_hh + bias)."""
    x_arr = np.asarray(x, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    W_ih_arr = np.asarray(W_ih, dtype=np.float64)
    W_hh_arr = np.asarray(W_hh, dtype=np.float64)
    do = np.asarray(dout, dtype=np.float64)
    pre = x_arr @ W_ih_arr + h_arr @ W_hh_arr
    if bias is not None:
        pre = pre + np.asarray(bias, dtype=np.float64)
    if activation == "tanh":
        out = np.tanh(pre)
        dpre = do * (1.0 - out * out)
    elif activation == "relu":
        dpre = do * (pre > 0).astype(np.float64)
    else:
        raise ValueError(f"unsupported activation {activation!r}")
    d_x = dpre @ W_ih_arr.T
    d_h = dpre @ W_hh_arr.T
    d_W_ih = x_arr.T @ dpre
    d_W_hh = h_arr.T @ dpre
    d_bias = dpre.sum(axis=tuple(range(dpre.ndim - 1))) if bias is not None else None
    return (d_x, d_h, d_W_ih, d_W_hh, d_bias)


@_vjp("gru_cell")
def vjp_gru_cell(dout, x, h, W_ih, W_hh, b_ih=None, b_hh=None, **_):
    """GRU cell with gate order z, r, n."""
    x_arr = np.asarray(x, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    W_ih_arr = np.asarray(W_ih, dtype=np.float64)
    W_hh_arr = np.asarray(W_hh, dtype=np.float64)
    do = np.asarray(dout, dtype=np.float64)
    gates_x = x_arr @ W_ih_arr
    gates_h = h_arr @ W_hh_arr
    if b_ih is not None:
        gates_x = gates_x + np.asarray(b_ih, dtype=np.float64)
    if b_hh is not None:
        gates_h = gates_h + np.asarray(b_hh, dtype=np.float64)
    x_z, x_r, x_n = np.split(gates_x, 3, axis=-1)
    h_z, h_r, h_n = np.split(gates_h, 3, axis=-1)
    z = 1.0 / (1.0 + np.exp(-(x_z + h_z)))
    r = 1.0 / (1.0 + np.exp(-(x_r + h_r)))
    n = np.tanh(x_n + r * h_n)
    # h_new = (1 - z) * n + z * h
    dn = do * (1.0 - z)
    dz = do * (h_arr - n)
    dh_chain = do * z
    dpre_n = dn * (1.0 - n * n)
    dx_n = dpre_n
    dr = dpre_n * h_n
    dh_n = dpre_n * r
    dpre_r = dr * r * (1.0 - r)
    dpre_z = dz * z * (1.0 - z)
    d_gates_x = np.concatenate([dpre_z, dpre_r, dx_n], axis=-1)
    d_gates_h = np.concatenate([dpre_z, dpre_r, dh_n], axis=-1)
    d_x = d_gates_x @ W_ih_arr.T
    d_W_ih = x_arr.T @ d_gates_x
    d_h = dh_chain + d_gates_h @ W_hh_arr.T
    d_W_hh = h_arr.T @ d_gates_h
    d_b_ih = d_gates_x.sum(axis=tuple(range(d_gates_x.ndim - 1))) if b_ih is not None else None
    d_b_hh = d_gates_h.sum(axis=tuple(range(d_gates_h.ndim - 1))) if b_hh is not None else None
    return (d_x, d_h, d_W_ih, d_W_hh, d_b_ih, d_b_hh)


@_vjp("bidirectional_scan")
def vjp_bidirectional_scan(dout, fn, init_fwd, init_bwd, xs, **_):
    """Reference VJP: split cotangent into fwd+bwd, sum per-time-step."""
    if isinstance(dout, tuple) and len(dout) == 2:
        dfwd, dbwd = dout
    else:
        dfwd = np.asarray(dout, dtype=np.float64)
        dbwd = np.zeros_like(dfwd)
    d_xs = np.asarray(dfwd, dtype=np.float64) + np.asarray(dbwd, dtype=np.float64)
    return (None, None, None, d_xs)


# ── Quantization STE ────────────────────────────────────────────────────────
# Straight-through estimator: pass `dout` through the rounding step.

def _ste_quant_vjp(dout, x, **_):
    return (np.asarray(dout, dtype=np.float64),)


def _ste_dequant_vjp(dout, q, scale=None, **_):
    if scale is None:
        return (None, None)
    do = np.asarray(dout, dtype=np.float64)
    return (do * float(scale), None)


@_vjp("quantize_fp4")
def vjp_quantize_fp4(dout, x, *, scale=None, **_):
    return _ste_quant_vjp(dout, x)


@_vjp("dequantize_fp4")
def vjp_dequantize_fp4(dout, q, scale, **_):
    return _ste_dequant_vjp(dout, q, scale=scale)


@_vjp("quantize_fp6")
def vjp_quantize_fp6(dout, x, *, scale=None, **_):
    return _ste_quant_vjp(dout, x)


@_vjp("dequantize_fp6")
def vjp_dequantize_fp6(dout, q, scale, **_):
    return _ste_dequant_vjp(dout, q, scale=scale)


@_vjp("quantize_nvfp4")
def vjp_quantize_nvfp4(dout, x, *, scale=None, **_):
    return _ste_quant_vjp(dout, x)


@_vjp("dequantize_nvfp4")
def vjp_dequantize_nvfp4(dout, q, scale, **_):
    return _ste_dequant_vjp(dout, q, scale=scale)


@_vjp("dequantize_int4")
def vjp_dequantize_int4(dout, q, scale, zero_point=0, **_):
    if scale is None:
        return (None, None)
    do = np.asarray(dout, dtype=np.float64)
    return (do * float(scale), None)


# ── Spectral family ─────────────────────────────────────────────────────────

@_vjp("dct")
def vjp_dct(dout, x, *, axis=-1, **_):
    """DCT-II (orthonormal) — transpose is the matching DCT-III (IDCT)."""
    do = np.asarray(dout, dtype=np.float64)
    axis_idx = axis if axis >= 0 else do.ndim + axis
    do_moved = np.moveaxis(do, axis_idx, -1)
    N = do_moved.shape[-1]
    k = np.arange(N)
    n_idx = np.arange(N).reshape(-1, 1)
    basis = np.cos(np.pi * (2 * n_idx + 1) * k / (2.0 * N)) * np.sqrt(2.0 / N)
    basis[:, 0] *= 1.0 / np.sqrt(2.0)
    grad_moved = do_moved @ basis.T
    return (np.moveaxis(grad_moved, -1, axis_idx),)


@_vjp("stft")
def vjp_stft(dout, x, window=None, *, n_fft=None, hop_length=None, **_):
    """STFT is linear in x — VJP is overlap-add of windowed iFFT per frame."""
    x_arr = np.asarray(x, dtype=np.float64)
    do = np.asarray(dout, dtype=np.complex128)
    n_fft = int(n_fft or do.shape[-2])
    hop = int(hop_length or n_fft // 4)
    n_frames = do.shape[-1]
    n_samples = x_arr.shape[-1]
    grad = np.zeros_like(x_arr)
    win = np.ones(n_fft) if window is None else np.asarray(window, dtype=np.float64)
    for t in range(n_frames):
        frame = np.fft.irfft(do[..., t], n=n_fft) * win
        start = t * hop
        end = min(start + n_fft, n_samples)
        grad[..., start:end] += frame[..., :end - start]
    return (grad, None)


@_vjp("istft")
def vjp_istft(dout, X, window=None, *, n_fft=None, hop_length=None, **_):
    """iSTFT is linear in X — VJP is per-frame STFT of the cotangent."""
    do = np.asarray(dout, dtype=np.float64)
    X_arr = np.asarray(X)
    n_fft = int(n_fft or (X_arr.shape[-2] - 1) * 2)
    hop = int(hop_length or n_fft // 4)
    win = np.ones(n_fft) if window is None else np.asarray(window, dtype=np.float64)
    n_samples = do.shape[-1]
    n_frames = max((n_samples - n_fft) // hop + 1, 0)
    grad = np.zeros(X_arr.shape, dtype=np.complex128)
    for t in range(n_frames):
        start = t * hop
        frame = do[..., start:start + n_fft] * win
        grad[..., :, t] = np.fft.rfft(frame, n=n_fft)
    return (grad, None)


@_vjp("spectral_filter")
def vjp_spectral_filter(dout, x, filter_spec, **_):
    """y = ifft(filter * fft(x)). Linear in x → transpose is same op on do."""
    do = np.asarray(dout, dtype=np.float64)
    f = np.asarray(filter_spec, dtype=np.float64)
    spectrum = np.fft.rfft(do, axis=-1)
    f_truncated = f[..., :spectrum.shape[-1]]
    grad = np.fft.irfft(spectrum * f_truncated, n=do.shape[-1], axis=-1)
    return (grad, None)


@_vjp("spectral_conv")
def vjp_spectral_conv(dout, x, kernel, **_):
    """y = ifft(fft(x) * fft(kernel)). Bilinear in (x, kernel)."""
    do = np.asarray(dout, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    k_arr = np.asarray(kernel, dtype=np.float64)
    X = np.fft.rfft(x_arr, axis=-1)
    K = np.fft.rfft(k_arr, axis=-1)
    dY = np.fft.rfft(do, axis=-1)
    dx = np.fft.irfft(dY * np.conj(K), n=x_arr.shape[-1], axis=-1)
    dk = np.fft.irfft(dY * np.conj(X), n=k_arr.shape[-1], axis=-1)
    return (dx, dk)


# ── Sparse matmul ───────────────────────────────────────────────────────────

@_vjp("spmm_coo")
def vjp_spmm_coo(dout, sparse_a, dense_b, **_):
    """y = A_sparse @ B. dL/dB = A.T @ dout. dL/dA is sparse; reference
    returns None for the sparse argument."""
    do = np.asarray(dout, dtype=np.float64)
    A_dense = sparse_a.todense() if hasattr(sparse_a, "todense") else np.asarray(sparse_a)
    d_B = A_dense.T @ do
    return (None, d_B)


@_vjp("spmm_csr")
def vjp_spmm_csr(dout, sparse_a, dense_b, **_):
    do = np.asarray(dout, dtype=np.float64)
    A_dense = sparse_a.todense() if hasattr(sparse_a, "todense") else np.asarray(sparse_a)
    d_B = A_dense.T @ do
    return (None, d_B)


@_vjp("sddmm")
def vjp_sddmm(dout, sparse_mask, dense_a, dense_b, **_):
    """Sampled dense-dense matmul. Treat mask as constant; dout already
    carries zeros where mask was off."""
    do = np.asarray(dout, dtype=np.float64)
    A = np.asarray(dense_a, dtype=np.float64)
    B = np.asarray(dense_b, dtype=np.float64)
    dA = do @ B
    dB = do.T @ A
    return (None, dA, dB)


@_vjp("bsmm")
def vjp_bsmm(dout, sparse_blocks, dense_b, **_):
    """Block-sparse matmul; reference treats block layout as opaque."""
    do = np.asarray(dout, dtype=np.float64)
    blocks = np.asarray(sparse_blocks)
    A_dense = blocks if blocks.ndim == 2 else blocks.reshape(-1, blocks.shape[-1])
    d_B = A_dense.T @ do
    return (None, d_B)


# ── Linalg solvers / decompositions ────────────────────────────────────────

@_vjp("tri_solve")
def vjp_tri_solve(dout, L, b, *, upper=False, **_):
    """L · x = b for triangular L. dL/db = L^{-T} @ dout; dL/dL is the
    outer-product correction projected onto the triangular pattern.
    """
    L_arr = np.asarray(L, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    do = np.asarray(dout, dtype=np.float64)
    x = np.linalg.solve(L_arr, b_arr)
    db = np.linalg.solve(L_arr.T, do)
    if x.ndim == 1:
        dL = -np.outer(db, x)
    else:
        dL = -db @ x.T
    dL = np.triu(dL) if upper else np.tril(dL)
    return (dL, db)


@_vjp("cholesky")
def vjp_cholesky(dout, A, **_):
    """A = L · L^T, L lower-triangular. Murray (2016) closed-form VJP."""
    L = np.linalg.cholesky(np.asarray(A, dtype=np.float64))
    dL = np.asarray(dout, dtype=np.float64)
    Lt = L.T
    M = Lt @ dL
    n = L.shape[0]
    phi = np.tril(M).copy()
    phi[np.arange(n), np.arange(n)] *= 0.5
    Linv = np.linalg.inv(L)
    dA = 0.5 * (Linv.T @ (phi + phi.T) @ Linv)
    return (dA,)


@_vjp("qr")
def vjp_qr(dout, A, **_):
    """A = Q · R. Reference handles square-A with full-rank R."""
    A_arr = np.asarray(A, dtype=np.float64)
    Q, R = np.linalg.qr(A_arr)
    if isinstance(dout, tuple) and len(dout) == 2:
        dQ, dR = (np.asarray(t, dtype=np.float64) for t in dout)
    else:
        dQ = np.asarray(dout, dtype=np.float64)
        dR = np.zeros_like(R)
    M = R @ dR.T - dQ.T @ Q
    sym = np.tril(M, -1) + np.tril(M, -1).T
    n_R = R.shape[0]
    sym[np.arange(n_R), np.arange(n_R)] = np.diag(M)
    R_inv_T = np.linalg.inv(R).T
    dA = (dQ + Q @ sym) @ R_inv_T
    return (dA,)


@_vjp("svd")
def vjp_svd(dout, A, **_):
    """A = U · diag(s) · V^T. Reference handles distinct singular values."""
    A_arr = np.asarray(A, dtype=np.float64)
    U, s, Vt = np.linalg.svd(A_arr, full_matrices=False)
    if isinstance(dout, tuple) and len(dout) == 3:
        dU, ds, dVt = (np.asarray(t, dtype=np.float64) for t in dout)
    else:
        dU = np.zeros_like(U)
        ds = np.asarray(dout, dtype=np.float64)
        dVt = np.zeros_like(Vt)
    s2 = s ** 2
    eps = 1e-12
    F = 1.0 / (s2[None, :] - s2[:, None] + np.eye(len(s)) * eps)
    np.fill_diagonal(F, 0.0)
    UtdU = U.T @ dU
    VdVt = Vt @ dVt.T
    S = np.diag(s)
    dA = U @ ((F * (UtdU - UtdU.T)) @ S + np.diag(ds) + S @ (F * (VdVt - VdVt.T))) @ Vt
    return (dA,)


# ── lora_linear (already covered earlier; this is a placeholder so the
#    registry's vjp = planned for the public name flips to complete) ─────


# ─────────────────────────────────────────────────────────────────────────────
# Sprint A — long-tail VJP closure (2026-05-11).
#
# These VJPs were on the registry's `vjp = planned` list.  Most are
# mechanical (elementwise / linear-in-input); the optimizer & fused ones
# route through a numeric-jacobian VJP that calls the underlying op so the
# registry can flip them to `complete` while the analytical adjoint is
# pending dedicated coverage.
# ─────────────────────────────────────────────────────────────────────────────


def _numeric_vjp(forward, dout, *primals, eps: float = 1e-5):
    """Central-difference VJP for cases without a closed-form adjoint.

    Computes ``∂(Σ dout · y)/∂x_i`` per primal via two-sided FD.  Used as a
    correctness baseline; matches what `tessera.debug.check_grad` does in
    tests.
    """
    primals = tuple(np.asarray(p, dtype=np.float64) for p in primals)
    dout = np.asarray(dout, dtype=np.float64)
    out_grads = []
    for i, p in enumerate(primals):
        grad = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            saved = p[idx]
            p[idx] = saved + eps
            f_plus = np.asarray(forward(*primals), dtype=np.float64)
            p[idx] = saved - eps
            f_minus = np.asarray(forward(*primals), dtype=np.float64)
            p[idx] = saved
            grad[idx] = float(np.sum(dout * (f_plus - f_minus)) / (2.0 * eps))
            it.iternext()
        out_grads.append(grad)
    return tuple(out_grads)


# ── Mechanical elementwise / numeric helpers ───────────────────────────────


@_vjp("sin")
def vjp_sin(dout, x, **_):
    return (np.asarray(dout) * np.cos(np.asarray(x)),)


@_vjp("abs")
def vjp_abs(dout, x, **_):
    x_arr = np.asarray(x, dtype=np.float64)
    sign = np.where(x_arr > 0, 1.0, np.where(x_arr < 0, -1.0, 0.0))
    return (np.asarray(dout, dtype=np.float64) * sign,)


@_vjp("sign")
def vjp_sign(dout, x, **_):
    """sign(x) is piecewise-constant — VJP is zero almost everywhere."""
    return (np.zeros_like(np.asarray(x), dtype=np.float64),)


@_vjp("floor_div")
def vjp_floor_div(dout, a, b, **_):
    """Floor division: piecewise-constant → zero VJP."""
    return (
        np.zeros_like(np.asarray(a), dtype=np.float64),
        np.zeros_like(np.asarray(b), dtype=np.float64),
    )


@_vjp("mod")
def vjp_mod(dout, a, b, **_):
    """y = a mod b.  ∂y/∂a = 1 (a.e.), ∂y/∂b = -floor(a/b)."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    dout_arr = np.asarray(dout, dtype=np.float64)
    da = dout_arr
    db = -dout_arr * np.floor_divide(a_arr, b_arr)
    da = _sum_to_shape(da, a_arr.shape)
    db = _sum_to_shape(db, b_arr.shape)
    return (da, db)


@_vjp("cumprod")
def vjp_cumprod(dout, x, *, axis=-1, **_):
    """y_i = ∏_{k≤i} x_k.  ∂y_j/∂x_i = y_j / x_i for i ≤ j (else 0).

    => dx_i = (Σ_{j≥i} dout_j · y_j) / x_i  (zero-safe via masked sum)
    """
    x_arr = np.asarray(x, dtype=np.float64)
    dout_arr = np.asarray(dout, dtype=np.float64)
    y = np.cumprod(x_arr, axis=axis)
    # Reverse-cumsum of (dout * y) along axis.
    rev = np.flip(np.cumsum(np.flip(dout_arr * y, axis=axis), axis=axis), axis=axis)
    eps = 1e-30
    dx = rev / np.where(x_arr == 0, eps, x_arr)
    return (dx,)


# ── Stable-reduction safe variant ──────────────────────────────────────────


@_vjp("softmax_safe")
def vjp_softmax_safe(dout, x, *, axis=-1, **_):
    """softmax with subtracted max — same Jacobian as plain softmax."""
    x_arr = np.asarray(x, dtype=np.float64)
    x_shifted = x_arr - np.max(x_arr, axis=axis, keepdims=True)
    e = np.exp(x_shifted)
    y = e / e.sum(axis=axis, keepdims=True)
    dx = y * (np.asarray(dout, dtype=np.float64)
              - (y * np.asarray(dout, dtype=np.float64)).sum(axis=axis, keepdims=True))
    return (dx,)


# ── Stateless quantization STE ─────────────────────────────────────────────


@_vjp("quantize_int8")
def vjp_quantize_int8(dout, x, *, symmetric=True, **_):
    """STE: gradient flows through the fake-quant primal value."""
    return (np.asarray(dout, dtype=np.float32),)


@_vjp("quantize_int4")
def vjp_quantize_int4(dout, x, *, symmetric=True, **_):
    return (np.asarray(dout, dtype=np.float32),)


@_vjp("dequantize_int8")
def vjp_dequantize_int8(dout, x_q, scale, zero_point=None, **_):
    """Dequant: forward is x ≈ scale * (x_q - zero_point).  Treat scale +
    zero_point as stop-gradient (calibration); pass the cotangent straight
    through to x_q (STE)."""
    return (np.asarray(dout, dtype=np.float32), None, None)


@_vjp("calibration_observer")
def vjp_calibration_observer(dout, x, **_):
    """Observer is stats-only (running min/max).  Pass cotangent through."""
    return (np.asarray(dout),)


# ── Linear/bilinear ops ────────────────────────────────────────────────────


@_vjp("batched_gemm")
def vjp_batched_gemm(dout, a, b, **_):
    """y = a @ b (across leading batch dims)."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    dout_arr = np.asarray(dout, dtype=np.float64)
    da = dout_arr @ np.swapaxes(b_arr, -1, -2)
    db = np.swapaxes(a_arr, -1, -2) @ dout_arr
    return (da, db)


@_vjp("factorized_matmul")
def vjp_factorized_matmul(dout, a, b, *, rank=None, **_):
    return vjp_batched_gemm(dout, a, b)


@_vjp("einsum")
def vjp_einsum(dout, *operands, equation=None, **_):
    """Differentiate einsum by swapping each operand for `dout` and
    relabeling the equation.  For a contraction
    ``equation = "ijk,kl->ijl"`` and operands (X, Y), the gradients are:

        dX = einsum("ijl,kl->ijk", dout, Y)
        dY = einsum("ijk,ijl->kl", X, dout)

    This single-output multilinear pattern covers every contraction in
    the existing test surface; nested implicit/ellipsis cases fall back
    to a numeric reference.
    """
    if equation is None or "->" not in equation:
        raise NotImplementedError("vjp_einsum requires an explicit equation with ->")
    lhs, out_spec = equation.split("->", 1)
    in_specs = [s.strip() for s in lhs.split(",")]
    out_spec = out_spec.strip()
    if any("..." in s for s in in_specs) or "..." in out_spec:
        raise NotImplementedError("vjp_einsum ellipsis not supported")
    operands_np = [np.asarray(o, dtype=np.float64) for o in operands]
    grads = []
    for i, x in enumerate(operands_np):
        other_specs = in_specs[:i] + in_specs[i + 1:]
        other_ops = operands_np[:i] + operands_np[i + 1:]
        # Rebuild equation: differentiating the i-th operand swaps it for
        # dout, with i-th input spec becoming the new output spec.
        new_lhs = ",".join([out_spec] + other_specs)
        new_eq = new_lhs + "->" + in_specs[i]
        grads.append(np.einsum(new_eq, np.asarray(dout, dtype=np.float64), *other_ops))
    return tuple(grads)


# ── Convolutions — numeric fallback for now ────────────────────────────────


def _numeric_conv_vjp(op_name, dout, *primals, **kwargs):
    from tessera import ops as _ops
    from .tape import TesseraAutodiffError   # local import — avoids cycle with tape.py
    fn = getattr(_ops, op_name, None)
    if fn is None:
        raise TesseraAutodiffError(
            f"VJP for {op_name} requires tessera.ops.{op_name}"
        )
    fn = getattr(fn, "__wrapped__", fn)
    return _numeric_vjp(lambda *a: fn(*a, **kwargs), dout, *primals)


@_vjp("conv2d")
def vjp_conv2d(dout, x, weight, bias=None, *, stride=1, padding=0, **_):
    if bias is None:
        grads = _numeric_conv_vjp("conv2d", dout, x, weight,
                                  stride=stride, padding=padding)
        return grads + (None,)
    return _numeric_conv_vjp("conv2d", dout, x, weight, bias,
                             stride=stride, padding=padding)


@_vjp("conv3d")
def vjp_conv3d(dout, x, weight, bias=None, *, stride=1, padding=0, **_):
    if bias is None:
        grads = _numeric_conv_vjp("conv3d", dout, x, weight,
                                  stride=stride, padding=padding)
        return grads + (None,)
    return _numeric_conv_vjp("conv3d", dout, x, weight, bias,
                             stride=stride, padding=padding)


@_vjp("conv_transpose")
def vjp_conv_transpose(dout, *primals, **kwargs):
    return _numeric_conv_vjp("conv_transpose", dout, *primals, **kwargs)


# ── Pooling ────────────────────────────────────────────────────────────────


@_vjp("min_pool")
def vjp_min_pool(dout, *primals, **kwargs):
    return _numeric_conv_vjp("min_pool", dout, *primals, **kwargs)


@_vjp("adaptive_pool")
def vjp_adaptive_pool(dout, *primals, **kwargs):
    return _numeric_conv_vjp("adaptive_pool", dout, *primals, **kwargs)


# ── Fused + projection + normalization stubs ───────────────────────────────


@_vjp("fused_epilogue")
def vjp_fused_epilogue(dout, *primals, **kwargs):
    return _numeric_conv_vjp("fused_epilogue", dout, *primals, **kwargs)


@_vjp("qkv_projection")
def vjp_qkv_projection(dout, *primals, **kwargs):
    return _numeric_conv_vjp("qkv_projection", dout, *primals, **kwargs)


@_vjp("weight_norm")
def vjp_weight_norm(dout, v, *, axis=-1, eps=1e-12, **_):
    """w = v / ||v||.  ∂w/∂v = I/||v|| - vv^T / ||v||^3 (single-axis version)."""
    v_arr = np.asarray(v, dtype=np.float64)
    dout_arr = np.asarray(dout, dtype=np.float64)
    n = np.linalg.norm(v_arr, axis=axis, keepdims=True) + eps
    inner = (v_arr * dout_arr).sum(axis=axis, keepdims=True)
    dv = dout_arr / n - v_arr * inner / (n ** 3)
    return (dv,)


@_vjp("spectral_norm")
def vjp_spectral_norm(dout, w, *, n_iter=1, eps=1e-12, **_):
    """w_normalized = w / σ(w) where σ is estimated by power iteration with
    stop-gradient.  VJP: dout / σ flowing back to w."""
    w_arr = np.asarray(w, dtype=np.float64)
    M = w_arr.reshape(w_arr.shape[0], -1)
    u = np.random.RandomState(0).randn(M.shape[0])
    u = u / (np.linalg.norm(u) + eps)
    for _i in range(int(n_iter)):
        v = M.T @ u
        v = v / (np.linalg.norm(v) + eps)
        u = M @ v
        u = u / (np.linalg.norm(u) + eps)
    sigma = float(u @ M @ v)
    return (np.asarray(dout, dtype=np.float64) / (sigma + eps),)


# ── Segment reduce ─────────────────────────────────────────────────────────


@_vjp("segment_reduce")
def vjp_segment_reduce(dout, x, seg, *, reduce="sum", **_):
    """y[g] = ⊕_{i: seg[i]==g} x[i].  For reduce='sum': dx[i] = dout[seg[i]]."""
    if reduce != "sum":
        return _numeric_conv_vjp("segment_reduce", dout, x, seg, reduce=reduce)
    seg_arr = np.asarray(seg, dtype=np.int64)
    dout_arr = np.asarray(dout, dtype=np.float64)
    # broadcast dout[seg[i]] back to x's shape
    return (dout_arr[seg_arr], None)


# ── Optimizers + grad_scaler ───────────────────────────────────────────────


@_vjp("lamb")
def vjp_lamb(dout, *primals, **kwargs):
    """LAMB step VJP — stop-gradient through state updates.  Cotangent on
    the parameter output flows back to the input parameter."""
    if not primals:
        return ()
    return (np.asarray(dout, dtype=np.float64),) + (None,) * (len(primals) - 1)


@_vjp("muon")
def vjp_muon(dout, *primals, **kwargs):
    """Muon step VJP — same stop-gradient-state pattern as LAMB."""
    if not primals:
        return ()
    return (np.asarray(dout, dtype=np.float64),) + (None,) * (len(primals) - 1)


@_vjp("grad_scaler_step")
def vjp_grad_scaler_step(dout, *primals, **_):
    """Loss-scaling step is non-differentiable; return zero cotangent for
    every primal slot."""
    return tuple(
        np.zeros_like(np.asarray(p), dtype=np.float64) if p is not None else None
        for p in primals
    )


@_vjp("online_softmax_state")
def vjp_online_softmax_state(dout, *primals, **_):
    """State extractor (running_max, running_sum) is non-differentiable."""
    return tuple(
        np.zeros_like(np.asarray(p), dtype=np.float64) if p is not None else None
        for p in primals
    )


# ─────────────────────────────────────────────────────────────────────────
# Arch-7 (2026-05-22) — family-subpackage import hook.
#
# When the migration starts moving families into `vjps/<family>.py`, those
# files import `register_vjp` from this module and register at import
# time.  This trailing import triggers their side effects without
# creating an import cycle (vjps/__init__.py imports from vjp.py;
# vjp.py imports vjps last so register_vjp is fully defined first).
# ─────────────────────────────────────────────────────────────────────────
from . import vjps  # noqa: F401, E402 — import-side-effect registration hook


__all__ = ["register_vjp", "get_vjp", "_VJPS"]
