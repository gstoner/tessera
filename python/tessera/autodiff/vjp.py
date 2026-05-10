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
# Autodiff-coverage hardening pass — S11 classification + distribution +
# contrastive + sequence losses, S7 layer/pooling. Per the
# "Recommended Next Work" in `docs/audit/primitive_coverage_state.md`.
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


__all__ = ["register_vjp", "get_vjp", "_VJPS"]
