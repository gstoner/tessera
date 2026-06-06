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
from typing import Callable, Dict, Tuple

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
        if name in _JVPS:
            raise ValueError(f"duplicate JVP registration for {name!r} (already bound to {_JVPS[name].__name__})")
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


@_jvp("clamp")
def jvp_clamp(primals, tangents, *, min_val=None, max_val=None, **_):
    (x,) = primals
    (dx,) = tangents
    y = np.clip(x, -np.inf if min_val is None else min_val, np.inf if max_val is None else max_val)
    mask = np.ones_like(x, dtype=np.float64)
    if min_val is not None:
        mask = mask * (x > min_val)
    if max_val is not None:
        mask = mask * (x < max_val)
    return y, dx * mask


@_jvp("rope")
def jvp_rope(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.rope, "__wrapped__", _ops.rope)
    return _numeric_jvp_rule(lambda x, theta: fn(x, theta, **kwargs), primals, tangents)


@_jvp("ntk_rope")
def jvp_ntk_rope(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.ntk_rope, "__wrapped__", _ops.ntk_rope)
    return _numeric_jvp_rule(lambda x, theta: fn(x, theta, **kwargs), primals, tangents)


@_jvp("rope_split")
def jvp_rope_split(primals, tangents, *, rope_dim, **_):
    (x,) = primals
    (dx,) = tangents
    return (x[..., :rope_dim], x[..., rope_dim:]), (dx[..., :rope_dim], dx[..., rope_dim:])


@_jvp("rope_merge")
def jvp_rope_merge(primals, tangents, **_):
    rope_part, no_rope_part = primals
    drope, dno = tangents
    return np.concatenate([rope_part, no_rope_part], axis=-1), np.concatenate([drope, dno], axis=-1)


@_jvp("latent_kv_compress")
@_jvp("latent_kv_expand_k")
@_jvp("latent_kv_expand_v")
def jvp_latent_matmul(primals, tangents, **_):
    return jvp_matmul(primals, tangents)


def _jvp_quantize_with(fn_name, primals, tangents, **kwargs):
    from tessera import ops as _ops
    x = primals[0]
    dx = tangents[0]
    fn = getattr(_ops, fn_name)
    original = getattr(fn, "__wrapped__", fn)
    primal = original(x, **kwargs)
    if isinstance(primal, tuple):
        return primal, (dx,) + tuple(np.zeros_like(np.asarray(p)) for p in primal[1:])
    return primal, dx


@_jvp("quantize_fp8")
def jvp_quantize_fp8(primals, tangents, **kwargs):
    return _jvp_quantize_with("quantize_fp8", primals, tangents, **kwargs)


@_jvp("fake_quantize")
def jvp_fake_quantize(primals, tangents, **kwargs):
    return _jvp_quantize_with("fake_quantize", primals, tangents, **kwargs)


@_jvp("dequantize_fp8")
def jvp_dequantize_ste(primals, tangents, **kwargs):
    x_q, scale = primals
    dx_q, _dscale = tangents
    return np.asarray(x_q, dtype=np.float32), dx_q


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


def _numeric_jvp_rule(forward, primals, tangents, *, eps: float = 1e-6):
    primals = tuple(np.asarray(p) for p in primals)
    tangents = tuple(np.asarray(t) for t in tangents)
    primal = forward(*primals)
    plus = forward(*(p + eps * t for p, t in zip(primals, tangents)))
    minus = forward(*(p - eps * t for p, t in zip(primals, tangents)))
    return primal, (np.asarray(plus, dtype=np.float64) - np.asarray(minus, dtype=np.float64)) / (2.0 * eps)


def _tree_add_scaled(tree, tangent, scale: float):
    if tree is None:
        return None
    if tangent is None:
        return tree
    if isinstance(tree, dict):
        out: dict = {}
        for key, value in tree.items():
            # Defensive guard against future tree containers that
            # subclass both ndarray and (bool/str/int); statically
            # unreachable today.
            if isinstance(value, (bool, str, int, np.integer)) and not isinstance(value, np.ndarray):  # type: ignore[unreachable]
                out[key] = value
            else:
                out[key] = _tree_add_scaled(value, tangent.get(key) if isinstance(tangent, dict) else None, scale)
        return out
    if isinstance(tree, tuple):
        return tuple(_tree_add_scaled(v, tangent[i] if tangent is not None else None, scale) for i, v in enumerate(tree))
    if isinstance(tree, list):
        return [_tree_add_scaled(v, tangent[i] if tangent is not None else None, scale) for i, v in enumerate(tree)]
    return np.asarray(tree) + scale * np.asarray(tangent)


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
        if dbias is not None:
            dy = dy + dbias
    return y, dy


@_jvp("flash_attn")
def jvp_flash_attn(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.flash_attn, "__wrapped__", _ops.flash_attn)
    return _numeric_jvp_rule(lambda q, k, v: fn(q, k, v, **kwargs), primals, tangents)


@_jvp("linear_attn")
def jvp_linear_attn(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.linear_attn, "__wrapped__", _ops.linear_attn)
    return _numeric_jvp_rule(lambda q, k, v: fn(q, k, v, **kwargs), primals, tangents)


@_jvp("linear_attn_state")
def jvp_linear_attn_state(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.linear_attn_state, "__wrapped__", _ops.linear_attn_state)
    return _numeric_jvp_rule(lambda q, k, v: fn(q, k, v, **kwargs), primals, tangents)


@_jvp("mla_decode_fused")
def jvp_mla_decode_fused(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.mla_decode_fused, "__wrapped__", _ops.mla_decode_fused)
    return _numeric_jvp_rule(lambda x, wd, wk, wv, q: fn(x, wd, wk, wv, q, **kwargs), primals, tangents)


@_jvp("alibi")
def jvp_alibi(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.alibi, "__wrapped__", _ops.alibi)
    primal = fn(**kwargs) if not primals else fn(*primals, **kwargs)
    return primal, np.zeros_like(np.asarray(primal))


@_jvp("multi_head_attention")
def jvp_multi_head_attention(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.multi_head_attention, "__wrapped__", _ops.multi_head_attention)
    return _numeric_jvp_rule(lambda q, k, v: fn(q, k, v, **kwargs), primals, tangents)


@_jvp("gqa_attention")
def jvp_gqa_attention(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.gqa_attention, "__wrapped__", _ops.gqa_attention)
    return _numeric_jvp_rule(lambda q, k, v: fn(q, k, v, **kwargs), primals, tangents)


@_jvp("mqa_attention")
def jvp_mqa_attention(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.mqa_attention, "__wrapped__", _ops.mqa_attention)
    return _numeric_jvp_rule(lambda q, k, v: fn(q, k, v, **kwargs), primals, tangents)


@_jvp("mla_decode")
def jvp_mla_decode(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.mla_decode, "__wrapped__", _ops.mla_decode)
    return _numeric_jvp_rule(lambda *args: fn(*args, **kwargs), primals, tangents)


@_jvp("attn_sliding_window")
def jvp_attn_sliding_window(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.attn_sliding_window, "__wrapped__", _ops.attn_sliding_window)
    return _numeric_jvp_rule(lambda q, k, v: fn(q, k, v, **kwargs), primals, tangents)


@_jvp("attn_local_window_2d")
def jvp_attn_local_window_2d(primals, tangents, **kwargs):
    """Gap 4 (2026-05-20): JVP for 2D local-window attention via the
    numeric forward-mode rule (matches the 1D sliding-window pattern)."""
    from tessera import ops as _ops
    fn = getattr(
        _ops.attn_local_window_2d, "__wrapped__", _ops.attn_local_window_2d,
    )
    return _numeric_jvp_rule(
        lambda q, k, v: fn(q, k, v, **kwargs), primals, tangents,
    )


@_jvp("attn_compressed_blocks")
def jvp_attn_compressed_blocks(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.attn_compressed_blocks, "__wrapped__", _ops.attn_compressed_blocks)
    return _numeric_jvp_rule(lambda q, k, v: fn(q, k, v, **kwargs), primals, tangents)


@_jvp("attn_top_k_blocks")
def jvp_attn_top_k_blocks(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.attn_top_k_blocks, "__wrapped__", _ops.attn_top_k_blocks)
    return _numeric_jvp_rule(lambda q, k, v: fn(q, k, v, **kwargs), primals, tangents)


@_jvp("power_attn")
def jvp_power_attn(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.power_attn, "__wrapped__", _ops.power_attn)
    return _numeric_jvp_rule(lambda q, k, v: fn(q, k, v, **kwargs), primals, tangents)


@_jvp("retention")
def jvp_retention(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.retention, "__wrapped__", _ops.retention)
    return _numeric_jvp_rule(lambda q, k, v: fn(q, k, v, **kwargs), primals, tangents)


@_jvp("lightning_attention")
def jvp_lightning_attention(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.lightning_attention, "__wrapped__", _ops.lightning_attention)
    return _numeric_jvp_rule(lambda q, k, v: fn(q, k, v, **kwargs), primals[:3], tangents[:3])


@_jvp("gated_attention")
def jvp_gated_attention(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.gated_attention, "__wrapped__", _ops.gated_attention)
    return _numeric_jvp_rule(lambda q, k, v, gate: fn(q, k, v, gate, **kwargs), primals, tangents)


@_jvp("deepseek_sparse_attention")
def jvp_deepseek_sparse_attention(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.deepseek_sparse_attention, "__wrapped__", _ops.deepseek_sparse_attention)
    if len(primals) == 3:
        return _numeric_jvp_rule(lambda q, k, v: fn(q, k, v, **kwargs), primals, tangents)
    return _numeric_jvp_rule(lambda q, k, v, gate: fn(q, k, v, gate, **kwargs), primals, tangents)


@_jvp("gated_deltanet")
def jvp_gated_deltanet(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.gated_deltanet, "__wrapped__", _ops.gated_deltanet)
    return _numeric_jvp_rule(lambda *args: fn(*args, **kwargs), primals, tangents)


@_jvp("kimi_delta_attention")
def jvp_kimi_delta_attention(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.kimi_delta_attention, "__wrapped__", _ops.kimi_delta_attention)
    return _numeric_jvp_rule(lambda *args: fn(*args, **kwargs), primals, tangents)


@_jvp("modified_delta_attention")
def jvp_modified_delta_attention(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.modified_delta_attention, "__wrapped__", _ops.modified_delta_attention)
    return _numeric_jvp_rule(lambda *args: fn(*args, **kwargs), primals, tangents)


@_jvp("hybrid_attention")
def jvp_hybrid_attention(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.hybrid_attention, "__wrapped__", _ops.hybrid_attention)
    return _numeric_jvp_rule(lambda q, k, v: fn(q, k, v, **kwargs), primals[:3], tangents[:3])


@_jvp("moe")
def jvp_moe(primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops.moe, "__wrapped__", _ops.moe)
    return _numeric_jvp_rule(lambda x, experts: fn(x, experts, **kwargs), primals, tangents)


@_jvp("moe_dispatch")
def jvp_moe_dispatch(primals, tangents, **_):
    return primals[0], tangents[0]


@_jvp("moe_combine")
def jvp_moe_combine(primals, tangents, *, reduce="sum", **_):
    partials = primals[0]
    dpartials = tangents[0]
    if reduce == "mean" and np.asarray(partials).ndim > 0:
        return np.asarray(partials).mean(axis=0), np.asarray(dpartials).mean(axis=0)
    if reduce == "sum" and np.asarray(partials).ndim > 1:
        return np.asarray(partials).sum(axis=0), np.asarray(dpartials).sum(axis=0)
    return partials, dpartials


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


@_jvp("cross_entropy_loss")
def jvp_cross_entropy_loss(primals, tangents, *, reduction="mean", **_):
    from tessera import losses as ts_losses
    logits, targets = primals
    dlogits = tangents[0]
    primal = ts_losses.cross_entropy_loss(logits, targets, reduction=reduction)
    eps = 1e-6
    tangent = (
        ts_losses.cross_entropy_loss(logits + eps * dlogits, targets, reduction=reduction)
        - ts_losses.cross_entropy_loss(logits - eps * dlogits, targets, reduction=reduction)
    ) / (2.0 * eps)
    return primal, tangent


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
# EBM4 — energy-based-model training loss JVPs.
# ─────────────────────────────────────────────────────────────────────────────

@_jvp("contrastive_divergence_loss")
def jvp_contrastive_divergence_loss(primals, tangents, *, reduction="mean", **_):
    e_pos, e_neg = primals
    de_pos, de_neg = tangents
    primal_diff = np.asarray(e_pos) - np.asarray(e_neg)
    tangent_diff = np.asarray(de_pos) - np.asarray(de_neg)
    return _reduce_loss(primal_diff, tangent_diff, reduction)


@_jvp("persistent_cd_loss")
def jvp_persistent_cd_loss(primals, tangents, *, reduction="mean", **_):
    return jvp_contrastive_divergence_loss(primals, tangents, reduction=reduction)


@_jvp("implicit_score_matching_loss")
def jvp_implicit_score_matching_loss(primals, tangents, *, reduction="mean", **_):
    score, div = primals
    dscore, ddiv = tangents
    s = np.asarray(score, dtype=np.float64)
    ds = np.asarray(dscore, dtype=np.float64)
    d = np.asarray(div, dtype=np.float64)
    dd = np.asarray(ddiv, dtype=np.float64)
    per_sample = 0.5 * (s ** 2).sum(axis=-1) + d
    per_sample_tan = (s * ds).sum(axis=-1) + dd
    return _reduce_loss(per_sample, per_sample_tan, reduction)


@_jvp("denoising_score_matching_loss")
def jvp_denoising_score_matching_loss(primals, tangents, *, reduction="mean", **_):
    score_noisy, y_clean, y_noisy, sigma = primals
    dscore, dyc, dyn, _dsigma = tangents
    s = np.asarray(score_noisy, dtype=np.float64)
    yc = np.asarray(y_clean, dtype=np.float64)
    yn = np.asarray(y_noisy, dtype=np.float64)
    sig2 = float(sigma) ** 2
    target = -(yn - yc) / sig2
    diff = s - target
    per_sample = 0.5 * (diff ** 2).sum(axis=-1)
    ds_arr = np.asarray(dscore, dtype=np.float64)
    dyc_arr = np.asarray(dyc, dtype=np.float64)
    dyn_arr = np.asarray(dyn, dtype=np.float64)
    # d(diff)/dt = ds - d(target)/dt = ds - (dyc - dyn) / sig2  (note sign on target).
    dtarget = -(dyn_arr - dyc_arr) / sig2
    ddiff = ds_arr - dtarget
    per_sample_tan = (diff * ddiff).sum(axis=-1)
    return _reduce_loss(per_sample, per_sample_tan, reduction)


@_jvp("normalize_group_advantages")
def jvp_normalize_group_advantages(primals, tangents, **kwargs):
    from tessera import rl as ts_rl
    return _numeric_jvp_rule(lambda r: ts_rl.normalize_group_advantages(r, **kwargs), primals, tangents)


@_jvp("ppo_policy_loss")
def jvp_ppo_policy_loss(primals, tangents, **kwargs):
    from tessera import rl as ts_rl
    return _numeric_jvp_rule(lambda ln, lo, adv: ts_rl.ppo_policy_loss(ln, lo, adv, **kwargs), primals, tangents)


@_jvp("grpo_policy_loss")
def jvp_grpo_policy_loss(primals, tangents, **kwargs):
    from tessera import rl as ts_rl
    if len(primals) == 2:
        return _numeric_jvp_rule(lambda ln, lo: ts_rl.grpo_policy_loss(ln, lo, **kwargs), primals, tangents)
    return _numeric_jvp_rule(lambda ln, lo, rewards: ts_rl.grpo_policy_loss(ln, lo, rewards, **kwargs), primals, tangents)


@_jvp("cispo_policy_loss")
def jvp_cispo_policy_loss(primals, tangents, **kwargs):
    from tessera import rl as ts_rl
    if len(primals) == 2:
        return _numeric_jvp_rule(lambda ln, lo: ts_rl.cispo_policy_loss(ln, lo, **kwargs), primals, tangents)
    return _numeric_jvp_rule(lambda ln, lo, rewards: ts_rl.cispo_policy_loss(ln, lo, rewards, **kwargs), primals, tangents)


# ─────────────────────────────────────────────────────────────────────────────
# JVPs paralleling the autodiff hardening pass in `vjp.py` — S11 classification
# / contrastive / sequence + S7 layers/pooling. Each is verified against
# central finite difference in `tests/unit/test_autodiff_loss_layer_coverage.py`.
# ─────────────────────────────────────────────────────────────────────────────


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


# ── S11 classification ─────────────────────────────────────────────────────


@_jvp("focal_loss")
def jvp_focal_loss(primals, tangents, *, gamma=2.0, alpha=None,
                   reduction="mean", **_):
    logits, targets = primals
    dlogits = tangents[0]
    logits_arr = np.asarray(logits).astype(np.float64, copy=False)
    targets_arr = np.asarray(targets).astype(np.int64)
    p = _softmax(logits_arr, axis=-1)
    flat_p = p.reshape(-1, p.shape[-1])
    idx = targets_arr.reshape(-1)
    rng = np.arange(idx.size)
    pt = np.maximum(flat_p[rng, idx], 1e-12)
    loss = -((1.0 - pt) ** gamma) * np.log(pt)
    if alpha is not None:
        loss = float(alpha) * loss
    loss = loss.reshape(targets_arr.shape)

    # dpt/dlogits via softmax Jacobian, applied to `dlogits` tangent.
    flat_dlogits = np.asarray(dlogits).astype(np.float64).reshape(flat_p.shape)
    dpt = pt * (
        flat_dlogits[rng, idx]
        - np.sum(flat_p * flat_dlogits, axis=-1)
    )
    one_minus_pt = 1.0 - pt
    dL_dpt = (
        gamma * np.power(one_minus_pt, gamma - 1.0) * np.log(pt)
        - np.power(one_minus_pt, gamma) / pt
    )
    if alpha is not None:
        dL_dpt = float(alpha) * dL_dpt
    tangent = (dL_dpt * dpt).reshape(targets_arr.shape)
    return _reduce_loss(loss, tangent, reduction)


@_jvp("label_smoothed_cross_entropy")
def jvp_label_smoothed_cross_entropy(primals, tangents, *, smoothing=0.1,
                                     reduction="mean", **_):
    logits, targets = primals
    dlogits = tangents[0]
    logits_arr = np.asarray(logits).astype(np.float64, copy=False)
    targets_arr = np.asarray(targets).astype(np.int64)
    n_classes = logits_arr.shape[-1]
    smooth = float(smoothing)
    one_hot = np.full(targets_arr.shape + (n_classes,),
                      smooth / max(1, n_classes - 1), dtype=np.float64)
    np.put_along_axis(one_hot, targets_arr[..., None], 1.0 - smooth, axis=-1)

    sm = _softmax(logits_arr, axis=-1)
    log_sm = np.log(np.maximum(sm, 1e-12))
    loss = -np.sum(one_hot * log_sm, axis=-1)

    dlog_sm = np.asarray(dlogits) - np.sum(sm * np.asarray(dlogits), axis=-1, keepdims=True)
    tangent = -np.sum(one_hot * dlog_sm, axis=-1)
    return _reduce_loss(loss, tangent, reduction)


@_jvp("kl_divergence")
def jvp_kl_divergence(primals, tangents, *, reduction="mean", **_):
    p_log, q = primals
    dp_log, dq = tangents
    p_log = np.asarray(p_log, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    dp_log = np.asarray(dp_log, dtype=np.float64)
    dq = np.asarray(dq, dtype=np.float64)
    p = np.exp(p_log)
    log_q = np.log(np.maximum(q, 1e-12))
    loss = np.sum(p * (p_log - log_q), axis=-1)

    # d/dp_log [ exp(p_log)*(p_log - log_q) ] = p*(p_log - log_q + 1)
    # d/dq    [ -exp(p_log) * log_q ]         = -p / q
    tangent = np.sum(
        p * (p_log - log_q + 1.0) * dp_log
        - (p / np.maximum(q, 1e-12)) * dq,
        axis=-1,
    )
    return _reduce_loss(loss, tangent, reduction)


# ── S11 contrastive ─────────────────────────────────────────────────────────


@_jvp("triplet_loss")
def jvp_triplet_loss(primals, tangents, *, margin=1.0, reduction="mean", **_):
    a, p, n = (np.asarray(t, dtype=np.float64) for t in primals)
    da, dp, dn = (np.asarray(t, dtype=np.float64) for t in tangents)
    d_ap = np.linalg.norm(a - p, axis=-1)
    d_an = np.linalg.norm(a - n, axis=-1)
    raw = d_ap - d_an + float(margin)
    active = (raw > 0).astype(np.float64)
    loss = np.maximum(0.0, raw)

    safe_ap = np.maximum(d_ap, 1e-12)
    safe_an = np.maximum(d_an, 1e-12)
    diff_ap = (a - p)
    diff_an = (a - n)
    # d ||a-p|| / d a = (a-p)/||a-p||
    tan_ap = np.sum(diff_ap * (da - dp), axis=-1) / safe_ap
    tan_an = np.sum(diff_an * (da - dn), axis=-1) / safe_an
    tangent = active * (tan_ap - tan_an)
    return _reduce_loss(loss, tangent, reduction)


@_jvp("contrastive_loss")
def jvp_contrastive_loss(primals, tangents, *, margin=1.0,
                         reduction="mean", **_):
    a, b, t = primals
    da, db, _dt = tangents
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    da = np.asarray(da, dtype=np.float64)
    db = np.asarray(db, dtype=np.float64)
    diff = a - b
    dist = np.linalg.norm(diff, axis=-1)
    margin_active = np.maximum(0.0, float(margin) - dist)
    loss = t * dist * dist + (1.0 - t) * margin_active * margin_active

    safe = np.maximum(dist, 1e-12)
    tan_dist = np.sum(diff * (da - db), axis=-1) / safe
    tangent = (2.0 * t * dist - 2.0 * (1.0 - t) * margin_active) * tan_dist
    return _reduce_loss(loss, tangent, reduction)


@_jvp("cosine_embedding_loss")
def jvp_cosine_embedding_loss(primals, tangents, *, margin=0.0,
                              reduction="mean", **_):
    a, b, t = primals
    da, db, _dt = tangents
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    da = np.asarray(da, dtype=np.float64)
    db = np.asarray(db, dtype=np.float64)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    denom = na * nb + 1e-12
    cos = np.sum(a * b, axis=-1) / denom

    pos_mask = (t > 0).astype(np.float64)
    neg_active = ((cos > float(margin)) & (t <= 0)).astype(np.float64)
    loss = pos_mask * (1.0 - cos) + (1.0 - pos_mask) * np.maximum(0.0, cos - float(margin))
    dL_dcos = -pos_mask + neg_active

    # dcos/da · da = (b · da) / denom - cos * (a · da) / na^2
    dot_a_da = np.sum(a * da, axis=-1)
    dot_b_db = np.sum(b * db, axis=-1)
    dot_a_b_da = np.sum(b * da, axis=-1)
    dot_b_a_db = np.sum(a * db, axis=-1)
    safe_na2 = np.maximum(na * na, 1e-12)
    safe_nb2 = np.maximum(nb * nb, 1e-12)
    dcos = (
        (dot_a_b_da / denom - cos * dot_a_da / safe_na2)
        + (dot_b_a_db / denom - cos * dot_b_db / safe_nb2)
    )
    tangent = dL_dcos * dcos
    return _reduce_loss(loss, tangent, reduction)


@_jvp("info_nce_loss")
def jvp_info_nce_loss(primals, tangents, *, temperature=0.1,
                      reduction="mean", **_):
    q, p, n = (np.asarray(t, dtype=np.float64) for t in primals)
    dq, dp, dn = (np.asarray(t, dtype=np.float64) for t in tangents)
    pos = np.sum(q * p, axis=-1, keepdims=True)
    neg = np.einsum("bd,bkd->bk", q, n)
    logits = np.concatenate([pos, neg], axis=-1) / float(temperature)
    sm = _softmax(logits, axis=-1)
    one_hot = np.zeros_like(sm)
    one_hot[:, 0] = 1.0
    log_sm = np.log(np.maximum(sm, 1e-12))
    loss = -np.sum(one_hot * log_sm, axis=-1)

    # Tangent of the logits.
    dpos = (np.sum(dq * p + q * dp, axis=-1, keepdims=True)) / float(temperature)
    dneg = (
        np.einsum("bd,bkd->bk", dq, n) + np.einsum("bd,bkd->bk", q, dn)
    ) / float(temperature)
    dlogits = np.concatenate([dpos, dneg], axis=-1)
    # d log_sm = dlogits - sum(sm * dlogits)
    dlog_sm = dlogits - np.sum(sm * dlogits, axis=-1, keepdims=True)
    tangent = -np.sum(one_hot * dlog_sm, axis=-1)
    return _reduce_loss(loss, tangent, reduction)


# ── S11 sequence ────────────────────────────────────────────────────────────


@_jvp("seq2seq_loss")
def jvp_seq2seq_loss(primals, tangents, *, reduction="mean", **_):
    """Forward-mode of masked cross-entropy. Tangents only flow through logits."""
    logits, targets = primals[:2]
    mask = primals[2] if len(primals) > 2 else None
    dlogits = tangents[0]
    logits_arr = np.asarray(logits, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.int64)
    sm = _softmax(logits_arr, axis=-1)
    log_sm = np.log(np.maximum(sm, 1e-12))
    loss = -np.take_along_axis(log_sm, targets_arr[..., None], axis=-1).squeeze(-1)
    dlog_sm = np.asarray(dlogits) - np.sum(sm * np.asarray(dlogits), axis=-1, keepdims=True)
    tangent = -np.take_along_axis(dlog_sm, targets_arr[..., None], axis=-1).squeeze(-1)
    if mask is not None:
        m = np.asarray(mask, dtype=np.float64)
        loss = loss * m
        tangent = tangent * m
        if reduction == "mean":
            denom = max(float(np.sum(m)), 1.0)
            return np.sum(loss) / denom, np.sum(tangent) / denom
        if reduction == "sum":
            return np.sum(loss), np.sum(tangent)
        return loss, tangent
    return _reduce_loss(loss, tangent, reduction)


# ── S7 normalizations ───────────────────────────────────────────────────────


def _norm_jvp(x: np.ndarray, dx: np.ndarray, axes: tuple[int, ...], eps: float):
    n = float(np.prod([x.shape[ax] for ax in axes]))
    mean = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    centered = x - mean
    x_hat = centered * inv_std

    dmean = dx.mean(axis=axes, keepdims=True)
    dvar = (2.0 / n) * np.sum((x - mean) * (dx - dmean), axis=axes, keepdims=True)
    dinv_std = -0.5 * np.power(var + eps, -1.5) * dvar
    dx_hat = (dx - dmean) * inv_std + centered * dinv_std
    return x_hat, dx_hat


@_jvp("group_norm")
def jvp_group_norm(primals, tangents, *, eps=1e-5, **_):
    x, num_groups = primals[:2]
    dx = tangents[0]
    x_arr = np.asarray(x, dtype=np.float32)
    dx_arr = np.asarray(dx, dtype=np.float32)
    n, c = x_arr.shape[:2]
    grouped_x = x_arr.reshape(n, num_groups, c // num_groups, *x_arr.shape[2:])
    grouped_dx = dx_arr.reshape(grouped_x.shape)
    reduce_axes = tuple(range(2, grouped_x.ndim))
    y_hat, dy_hat = _norm_jvp(grouped_x, grouped_dx, reduce_axes, eps)
    return y_hat.reshape(x_arr.shape), dy_hat.reshape(x_arr.shape)


@_jvp("instance_norm")
def jvp_instance_norm(primals, tangents, *, eps=1e-5, **_):
    x = primals[0]
    dx = tangents[0]
    x_arr = np.asarray(x, dtype=np.float32)
    dx_arr = np.asarray(dx, dtype=np.float32)
    reduce_axes = tuple(range(2, x_arr.ndim))
    return _norm_jvp(x_arr, dx_arr, reduce_axes, eps)


# ── S7 layers ───────────────────────────────────────────────────────────────


@_jvp("lora_linear")
def jvp_lora_linear(primals, tangents, *, alpha=1.0, **_):
    x, weight, lora_a, lora_b = primals[:4]
    dx, dW, dA, dB = tangents[:4]
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    a = np.asarray(lora_a, dtype=np.float64)
    b = np.asarray(lora_b, dtype=np.float64)
    rank = a.shape[-1]
    scale = float(alpha) / max(1, rank)
    y = x @ weight + ((x @ a) @ b) * scale
    dy = (
        np.asarray(dx) @ weight
        + x @ np.asarray(dW)
        + ((np.asarray(dx) @ a + x @ np.asarray(dA)) @ b) * scale
        + ((x @ a) @ np.asarray(dB)) * scale
    )
    return y, dy


# ── S7 pooling ──────────────────────────────────────────────────────────────


def _pair(value):
    if isinstance(value, int):
        return (value, value)
    return tuple(value)


@_jvp("max_pool")
def jvp_max_pool(primals, tangents, *, stride=None, padding=0, **_):
    x = primals[0]
    kernel_size = primals[1]
    dx = tangents[0]
    x_arr = np.asarray(x, dtype=np.float64)
    dx_arr = np.asarray(dx, dtype=np.float64)
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride if stride is not None else kernel_size)
    ph, pw = _pair(padding)
    n, c, h, w = x_arr.shape
    if ph or pw:
        padded = np.pad(x_arr, ((0, 0), (0, 0), (ph, ph), (pw, pw)),
                        constant_values=-np.inf)
        padded_dx = np.pad(dx_arr, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    else:
        padded, padded_dx = x_arr, dx_arr
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    y = np.zeros((n, c, out_h, out_w), dtype=np.float64)
    dy = np.zeros_like(y)
    for i in range(out_h):
        for j in range(out_w):
            window = padded[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw]
            window_dx = padded_dx[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw]
            y[:, :, i, j] = window.max(axis=(2, 3))
            flat = window.reshape(n, c, -1)
            dx_flat = window_dx.reshape(n, c, -1)
            argmax = np.argmax(flat, axis=-1)
            for b in range(n):
                for ch in range(c):
                    dy[b, ch, i, j] = dx_flat[b, ch, argmax[b, ch]]
    return y, dy


# ── S7 conv1d ──────────────────────────────────────────────────────────────


@_jvp("conv1d")
def jvp_conv1d(primals, tangents, *, stride=1, padding=0, dilation=1,
               groups=1, **_):
    """Forward-mode for Conv1d.

    Conv1d is bilinear in (x, W), so the tangent decomposes as

        dy = conv1d(dx, W) + conv1d(x, dW) + dbias

    We import the fp64 helper from the VJP module to avoid recomputing
    the forward path twice through `nn.functional.conv1d` (which casts to
    fp32 and would inject quantization noise into the tangent).
    """
    from tessera.autodiff.vjp import _conv1d_forward_fp64

    x = np.asarray(primals[0], dtype=np.float64)
    w = np.asarray(primals[1], dtype=np.float64)
    dx = np.asarray(tangents[0], dtype=np.float64)
    dW = np.asarray(tangents[1], dtype=np.float64)

    bias = primals[2] if len(primals) > 2 and primals[2] is not None else None
    dbias = tangents[2] if len(tangents) > 2 and tangents[2] is not None else None

    kw = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
    y = _conv1d_forward_fp64(x, w, **kw)
    dy = (
        _conv1d_forward_fp64(dx, w, **kw)
        + _conv1d_forward_fp64(x, dW, **kw)
    )
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float64).reshape(1, -1, 1)
        if dbias is not None:
            dy = dy + np.asarray(dbias, dtype=np.float64).reshape(1, -1, 1)
    return y, dy


@_jvp("avg_pool")
def jvp_avg_pool(primals, tangents, *, stride=None, padding=0, **_):
    x = primals[0]
    kernel_size = primals[1]
    dx = tangents[0]
    x_arr = np.asarray(x, dtype=np.float64)
    dx_arr = np.asarray(dx, dtype=np.float64)
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride if stride is not None else kernel_size)
    ph, pw = _pair(padding)
    n, c, h, w = x_arr.shape
    if ph or pw:
        padded = np.pad(x_arr, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
        padded_dx = np.pad(dx_arr, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    else:
        padded, padded_dx = x_arr, dx_arr
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    y = np.zeros((n, c, out_h, out_w), dtype=np.float64)
    dy = np.zeros_like(y)
    for i in range(out_h):
        for j in range(out_w):
            window = padded[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw]
            window_dx = padded_dx[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw]
            y[:, :, i, j] = window.mean(axis=(2, 3))
            dy[:, :, i, j] = window_dx.mean(axis=(2, 3))
    return y, dy


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


# ─────────────────────────────────────────────────────────────────────────────
# JVPs paralleling the deferred-VJP follow-up. CTC's JVP routes through the
# VJP via the scalar-loss "double backward" trick — same approach
# `torch.autograd.functional.jvp` uses internally. The naive forward-mode
# pass through forward-backward DP would be O(T²·V²) per batch; the
# contraction approach is one VJP call per element instead.
# ─────────────────────────────────────────────────────────────────────────────


@_jvp("ctc_loss")
def jvp_ctc_loss(primals, tangents, *, blank=0, reduction="mean", **_):
    """Forward-mode CTC via VJP contraction.

    For a scalar loss L = L(x), the directional derivative in tangent v is

        L'(x) · v = ∇L(x) · v

    so once we have a VJP that returns ∇L, the JVP is a dot product. CTC
    decouples per batch element (`L_b` only depends on `log_probs[:, b, :]`),
    so per-batch raw gradients give per-batch tangents — even when
    `reduction='none'` and the "loss" is a length-B vector.

    We obtain raw per-batch gradients by calling the VJP with
    `reduction='sum'` and `dout=1.0`: that path sets every batch's
    cotangent to 1, so `grad_lp[t, b, v] = dL_b/dlog_probs[t, b, v]` exactly.
    Then we contract with the tangent and apply the user's reduction.

    Cost: one CTC VJP call (= one forward + one backward pass per batch
    element) — independent of the tangent's sparsity. This is the same
    cost as the corresponding VJP, so for scalar reductions you can swap
    `vjp` ↔ `jvp` freely.
    """
    from tessera import losses as ts_losses
    from tessera.autodiff.vjp import get_vjp

    log_probs = primals[0]
    targets = primals[1]
    input_lengths = primals[2]
    target_lengths = primals[3]
    dlog_probs = np.asarray(tangents[0], dtype=np.float64)

    primal = ts_losses.ctc_loss(
        log_probs, targets, input_lengths, target_lengths,
        blank=blank, reduction=reduction,
    )

    # Raw per-batch gradient: with reduction='sum' and dout=1.0 the per-batch
    # cotangent is uniformly 1, so the VJP returns exactly dL_b/dlog_probs.
    ctc_vjp = get_vjp("ctc_loss")
    if ctc_vjp is None:
        raise RuntimeError("ctc_loss VJP is not registered")
    grad_lp_raw, *_ = ctc_vjp(
        1.0, log_probs, targets, input_lengths, target_lengths,
        blank=blank, reduction="sum",
    )

    # Per-batch dot product: contract over time (axis 0) and vocab (axis 2).
    per_batch_tan = np.sum(grad_lp_raw * dlog_probs, axis=(0, 2))

    if reduction == "mean":
        return primal, per_batch_tan.mean()
    if reduction == "sum":
        return primal, per_batch_tan.sum()
    # 'none' — per-element tangent matches the per-element primal shape.
    return primal, per_batch_tan


@_jvp("js_divergence")
def jvp_js_divergence(primals, tangents, *, reduction="mean", **_):
    """Forward-mode for Jensen-Shannon. Same closed form as the VJP."""
    p_arr, q_arr = (np.asarray(t, dtype=np.float64) for t in primals)
    dp, dq = (np.asarray(t, dtype=np.float64) for t in tangents)
    m = 0.5 * (p_arr + q_arr)
    log_p_m = np.log(np.maximum(p_arr, 1e-12)) - np.log(np.maximum(m, 1e-12))
    log_q_m = np.log(np.maximum(q_arr, 1e-12)) - np.log(np.maximum(m, 1e-12))
    kl_pm = np.sum(p_arr * log_p_m, axis=-1)
    kl_qm = np.sum(q_arr * log_q_m, axis=-1)
    loss = 0.5 * (kl_pm + kl_qm)

    # Cross-terms cancel: dJS/dp_i = ½ log(p_i/m_i), dJS/dq_i = ½ log(q_i/m_i).
    tangent = np.sum(0.5 * log_p_m * dp + 0.5 * log_q_m * dq, axis=-1)
    return _reduce_loss(loss, tangent, reduction)


@_jvp("wasserstein_distance")
def jvp_wasserstein_distance(primals, tangents, *, reduction="mean", **_):
    """Forward-mode Wasserstein-1: route the tangent through the same sort
    permutations as the primal."""
    x_arr, y_arr = (np.asarray(t, dtype=np.float64) for t in primals)
    dx, dy = (np.asarray(t, dtype=np.float64) for t in tangents)
    pi_x = np.argsort(x_arr, axis=-1)
    pi_y = np.argsort(y_arr, axis=-1)
    x_sorted = np.take_along_axis(x_arr, pi_x, axis=-1)
    y_sorted = np.take_along_axis(y_arr, pi_y, axis=-1)
    dx_sorted = np.take_along_axis(dx, pi_x, axis=-1)
    dy_sorted = np.take_along_axis(dy, pi_y, axis=-1)
    diff = x_sorted - y_sorted
    sign_diff = np.sign(diff)
    loss = np.mean(np.abs(diff), axis=-1)
    tangent = np.mean(sign_diff * (dx_sorted - dy_sorted), axis=-1)
    return _reduce_loss(loss, tangent, reduction)


@_jvp("nt_xent_loss")
def jvp_nt_xent_loss(primals, tangents, *, temperature=0.5,
                     reduction="mean", **_):
    """Forward-mode NT-Xent.

    Pushes `dz` through L2-normalize → Gram → masked log_softmax → positive
    mean. The math mirrors the VJP's chain: the only direction-dependent
    intermediate is `du = (dz - u(u·dz)) / ||z||` (the Jacobian of L2
    normalize applied to the tangent).
    """
    from tessera.autodiff.vjp import _nt_xent_forward_state

    z = np.asarray(primals[0], dtype=np.float64)
    labels = primals[1]
    dz = np.asarray(tangents[0], dtype=np.float64)
    state = _nt_xent_forward_state(z, labels, temperature)
    u, n, sm, pos, K = state["u"], state["n"], state["sm"], state["pos"], state["K"]
    B = z.shape[0]

    # Tangent of the L2 normalize.
    proj = np.sum(dz * u, axis=-1, keepdims=True)
    du = (dz - u * proj) / n

    # Tangent of S = u uᵀ / τ.
    dS = ((du @ u.T) + (u @ du.T)) / float(temperature)
    np.fill_diagonal(dS, 0.0)  # diagonal stays masked at -∞ (no flow).

    # Tangent of log_softmax along last axis: dLP[i, j] = dS[i, j] - Σ_k sm[i,k] dS[i,k].
    sum_sm_dS = np.sum(sm * dS, axis=-1, keepdims=True)
    dLP = dS - sum_sm_dS

    # Per-row primal loss + tangent.
    log_sm = np.where(np.isfinite(sm) & (sm > 0), np.log(np.maximum(sm, 1e-12)), 0.0)
    loss_per_row = -np.sum(np.where(pos, log_sm, 0.0), axis=-1) / K
    no_pos = pos.sum(axis=-1) == 0
    loss_per_row = np.where(no_pos, 0.0, loss_per_row)
    tan_per_row = -np.sum(np.where(pos, dLP, 0.0), axis=-1) / K
    tan_per_row = np.where(no_pos, 0.0, tan_per_row)

    return _reduce_loss(loss_per_row, tan_per_row, reduction)


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 1 — S6 collective JVPs (paralleling the VJPs in `vjp.py`).
# ─────────────────────────────────────────────────────────────────────────────


@_jvp("psum")
def jvp_psum(primals, tangents, axis_name=None, **_):
    arr = np.asarray(primals[0])
    dval = np.asarray(tangents[0])
    return np.sum(arr, axis=0), np.sum(dval, axis=0)


@_jvp("pmean")
def jvp_pmean(primals, tangents, axis_name=None, **_):
    arr = np.asarray(primals[0])
    dval = np.asarray(tangents[0])
    return np.mean(arr, axis=0), np.mean(dval, axis=0)


@_jvp("pmax")
def jvp_pmax(primals, tangents, axis_name=None, **_):
    arr = np.asarray(primals[0]).astype(np.float64, copy=False)
    dval = np.asarray(tangents[0]).astype(np.float64, copy=False)
    m = np.max(arr, axis=0)
    mask = (arr == m[None]).astype(np.float64)
    counts = mask.sum(axis=0)
    tan = np.sum(mask * dval, axis=0) / np.maximum(counts, 1.0)
    return m, tan


@_jvp("pmin")
def jvp_pmin(primals, tangents, axis_name=None, **_):
    arr = np.asarray(primals[0]).astype(np.float64, copy=False)
    dval = np.asarray(tangents[0]).astype(np.float64, copy=False)
    m = np.min(arr, axis=0)
    mask = (arr == m[None]).astype(np.float64)
    counts = mask.sum(axis=0)
    tan = np.sum(mask * dval, axis=0) / np.maximum(counts, 1.0)
    return m, tan


@_jvp("collective_permute")
def jvp_collective_permute(primals, tangents, **kw):
    arr = np.asarray(primals[0])
    pairs = primals[1] if len(primals) > 1 else kw.get("pairs", [])
    dval = np.asarray(tangents[0])
    out = np.empty_like(arr)
    dout = np.empty_like(dval)
    for src, dst in pairs:
        out[int(dst)] = arr[int(src)]
        dout[int(dst)] = dval[int(src)]
    return out, dout


@_jvp("broadcast_to_axis")
def jvp_broadcast_to_axis(primals, tangents, *, axis_size, axis=0, **_):
    val = np.asarray(primals[0])
    dval = np.asarray(tangents[0])
    out = np.stack([val for _ in range(int(axis_size))], axis=axis)
    dout = np.stack([dval for _ in range(int(axis_size))], axis=axis)
    return out, dout


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 2 — stateful optimizer JVPs.
# ─────────────────────────────────────────────────────────────────────────────


@_jvp("momentum")
def jvp_momentum(primals, tangents, *, lr, momentum=0.9, **_):
    params, grads = primals[0], primals[1]
    state = primals[2] if len(primals) > 2 else None
    dparams, dgrads = tangents[0], tangents[1]
    dstate = tangents[2] if len(tangents) > 2 else None

    velocity = state["velocity"] if state is not None and "velocity" in state else np.zeros_like(np.asarray(params))
    dvelocity = (dstate["velocity"] if (dstate is not None and "velocity" in dstate)
                 else np.zeros_like(np.asarray(params)))
    new_velocity = float(momentum) * np.asarray(velocity) + np.asarray(grads)
    d_new_velocity = float(momentum) * np.asarray(dvelocity) + np.asarray(dgrads)
    new_params = np.asarray(params) - float(lr) * new_velocity
    d_new_params = np.asarray(dparams) - float(lr) * d_new_velocity
    return new_params, d_new_params


@_jvp("nesterov")
def jvp_nesterov(primals, tangents, *, lr, momentum=0.9, **_):
    params, grads = primals[0], primals[1]
    state = primals[2] if len(primals) > 2 else None
    dparams, dgrads = tangents[0], tangents[1]
    dstate = tangents[2] if len(tangents) > 2 else None

    velocity = state["velocity"] if state is not None and "velocity" in state else np.zeros_like(np.asarray(params))
    dvelocity = (dstate["velocity"] if (dstate is not None and "velocity" in dstate)
                 else np.zeros_like(np.asarray(params)))
    m = float(momentum)
    new_velocity = m * np.asarray(velocity) + np.asarray(grads)
    look_ahead = np.asarray(grads) + m * new_velocity
    d_new_velocity = m * np.asarray(dvelocity) + np.asarray(dgrads)
    d_look_ahead = np.asarray(dgrads) + m * d_new_velocity
    new_params = np.asarray(params) - float(lr) * look_ahead
    d_new_params = np.asarray(dparams) - float(lr) * d_look_ahead
    return new_params, d_new_params


@_jvp("adam")
def jvp_adam(primals, tangents, *, lr=1e-3, beta1=0.9, beta2=0.999,
             eps=1e-8, step=1, **_):
    param, grad, moment1, moment2 = (np.asarray(x, dtype=np.float64) for x in primals)
    dparam, dgrad, dmoment1, dmoment2 = (np.asarray(x, dtype=np.float64) for x in tangents)
    b1, b2 = float(beta1), float(beta2)
    m_new = b1 * moment1 + (1.0 - b1) * grad
    v_new = b2 * moment2 + (1.0 - b2) * grad * grad
    dm_new = b1 * dmoment1 + (1.0 - b1) * dgrad
    dv_new = b2 * dmoment2 + 2.0 * (1.0 - b2) * grad * dgrad
    bc1 = 1.0 - b1 ** int(step)
    bc2 = 1.0 - b2 ** int(step)
    m_hat = m_new / bc1
    v_hat = v_new / bc2
    dm_hat = dm_new / bc1
    dv_hat = dv_new / bc2
    sqrt_v = np.sqrt(v_hat)
    denom = sqrt_v + float(eps)
    update = m_hat / denom
    dsqrt_v = dv_hat / (2.0 * np.maximum(sqrt_v, 1e-12))
    dupdate = (dm_hat * denom - m_hat * dsqrt_v) / (denom * denom)
    return (
        param - float(lr) * update,
        m_new,
        v_new,
    ), (
        dparam - float(lr) * dupdate,
        dm_new,
        dv_new,
    )


@_jvp("adamw")
def jvp_adamw(primals, tangents, *, lr=1e-3, beta1=0.9, beta2=0.999,
              eps=1e-8, weight_decay=0.0, **_):
    params, grads = primals[0], primals[1]
    state = primals[2] if len(primals) > 2 else None
    dparams, dgrads = tangents[0], tangents[1]
    dstate = tangents[2] if len(tangents) > 2 else None

    p = np.asarray(params, dtype=np.float64)
    g = np.asarray(grads, dtype=np.float64)
    dp = np.asarray(dparams, dtype=np.float64)
    dg = np.asarray(dgrads, dtype=np.float64)
    if state is None:
        m_prev = np.zeros_like(p)
        v_prev = np.zeros_like(p)
        step = 0
    else:
        m_prev = np.asarray(state["m"], dtype=np.float64)
        v_prev = np.asarray(state["v"], dtype=np.float64)
        step = int(state["step"])
    if dstate is None:
        dm_prev = np.zeros_like(p)
        dv_prev = np.zeros_like(p)
    else:
        dm_prev = np.asarray(dstate.get("m", np.zeros_like(p)), dtype=np.float64)
        dv_prev = np.asarray(dstate.get("v", np.zeros_like(p)), dtype=np.float64)

    step += 1
    b1, b2 = float(beta1), float(beta2)
    m_new = b1 * m_prev + (1.0 - b1) * g
    v_new = b2 * v_prev + (1.0 - b2) * g * g
    dm_new = b1 * dm_prev + (1.0 - b1) * dg
    dv_new = b2 * dv_prev + (1.0 - b2) * 2.0 * g * dg
    bc1 = 1.0 - b1 ** step
    bc2 = 1.0 - b2 ** step
    m_hat = m_new / bc1
    v_hat = v_new / bc2
    dm_hat = dm_new / bc1
    dv_hat = dv_new / bc2
    sqrt_v = np.sqrt(v_hat)
    denom = sqrt_v + float(eps)
    update = m_hat / denom
    dsqrt_v = dv_hat / (2.0 * np.maximum(sqrt_v, 1e-12))
    dupdate = (dm_hat * denom - m_hat * dsqrt_v) / (denom * denom)
    p_decay = p * (1.0 - float(lr) * float(weight_decay))
    dp_decay = dp * (1.0 - float(lr) * float(weight_decay))
    new_params = p_decay - float(lr) * update
    d_new_params = dp_decay - float(lr) * dupdate
    return new_params, d_new_params


@_jvp("lion")
def jvp_lion(primals, tangents, *, lr=1e-4, beta1=0.9, beta2=0.99,
             weight_decay=0.0, **_):
    params, grads = primals[0], primals[1]
    state = primals[2] if len(primals) > 2 else None
    dparams, dgrads = tangents[0], tangents[1]
    dstate = tangents[2] if len(tangents) > 2 else None

    p = np.asarray(params, dtype=np.float64)
    g = np.asarray(grads, dtype=np.float64)
    dp = np.asarray(dparams, dtype=np.float64)
    if state is None:
        m_prev = np.zeros_like(p)
    else:
        m_prev = np.asarray(state["m"], dtype=np.float64)
    dm_prev = np.zeros_like(p) if dstate is None else np.asarray(dstate.get("m", np.zeros_like(p)), dtype=np.float64)
    update = float(beta1) * m_prev + (1.0 - float(beta1)) * g
    new_m = float(beta2) * m_prev + (1.0 - float(beta2)) * g
    _dnew_m = float(beta2) * dm_prev + (1.0 - float(beta2)) * np.asarray(dgrads, dtype=np.float64)
    decay = 1.0 - float(lr) * float(weight_decay)
    return p * decay - float(lr) * np.sign(update), dp * decay


@_jvp("adafactor")
def jvp_adafactor(primals, tangents, *, lr=1e-3, beta2=0.999, eps=1e-30, **kwargs):
    from tessera import optim as ts_optim

    params, grads = primals[0], primals[1]
    state = primals[2] if len(primals) > 2 else None
    dparams, dgrads = tangents[0], tangents[1]
    dstate = tangents[2] if len(tangents) > 2 else None

    def forward(p, g, s):
        return ts_optim.adafactor(p, g, s, lr=lr, beta2=beta2, eps=eps, **kwargs)[0]

    primal = forward(params, grads, state)
    h = 1e-6
    plus = forward(
        np.asarray(params, dtype=np.float64) + h * np.asarray(dparams, dtype=np.float64),
        np.asarray(grads, dtype=np.float64) + h * np.asarray(dgrads, dtype=np.float64),
        _tree_add_scaled(state, dstate, h),
    )
    minus = forward(
        np.asarray(params, dtype=np.float64) - h * np.asarray(dparams, dtype=np.float64),
        np.asarray(grads, dtype=np.float64) - h * np.asarray(dgrads, dtype=np.float64),
        _tree_add_scaled(state, dstate, -h),
    )
    return primal, (np.asarray(plus, dtype=np.float64) - np.asarray(minus, dtype=np.float64)) / (2.0 * h)


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 3 — Memory architecture: differentiable `memory_read` JVP.
# ─────────────────────────────────────────────────────────────────────────────


@_jvp("memory_read")
def jvp_memory_read(primals, tangents, *, top_k=1, normalize=True, **_):
    """Forward-mode for `memory_read`.

    Tangents flow through:
      - `query` (changes scores → top_scores → weights → read)
      - `keys`  (changes scores)
      - `values` (changes gathered tensor directly)

    Top-k indices are treated as constants (matches the VJP convention).
    Returns the primal `MemoryReadResult` and a tuple-of-tangent matching
    its layout: `(d_values, d_indices=None, d_weights, d_scores)`.
    """
    from tessera.memory import MemoryReadResult, MemoryTable

    memory, query = primals[0], primals[1]
    if isinstance(memory, MemoryTable):
        keys = memory.keys
        values = memory.values
    else:
        keys, values = memory
    keys = np.asarray(keys, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    query = np.asarray(query._data if hasattr(query, "_data") else query, dtype=np.float64)

    dmem, dquery = tangents[0], tangents[1]
    if isinstance(dmem, tuple):
        dkeys, dvalues = dmem
    elif isinstance(dmem, MemoryTable):
        dkeys, dvalues = dmem.keys, dmem.values
    elif dmem is None:
        dkeys = np.zeros_like(keys)
        dvalues = np.zeros_like(values)
    else:
        dkeys, dvalues = dmem
    dkeys = np.asarray(dkeys, dtype=np.float64)
    dvalues = np.asarray(dvalues, dtype=np.float64)
    dquery = np.asarray(dquery, dtype=np.float64)

    single_query = query.ndim == 1
    if single_query:
        query = query[None, :]
        dquery = dquery[None, :]
    B = query.shape[0]
    N = keys.shape[0]
    k = min(int(top_k), N)

    scores = query @ keys.T                                      # (B, N)
    dscores = dquery @ keys.T + query @ dkeys.T

    partition = np.argpartition(-scores, kth=k - 1, axis=-1)[:, :k]
    top_scores = np.take_along_axis(scores, partition, axis=-1)
    order = np.argsort(-top_scores, axis=-1)
    indices = np.take_along_axis(partition, order, axis=-1)      # (B, k)
    top_scores = np.take_along_axis(top_scores, order, axis=-1)
    dtop_scores = np.take_along_axis(dscores, indices, axis=-1)

    if normalize:
        m = np.max(top_scores, axis=-1, keepdims=True)
        e = np.exp(top_scores - m)
        weights = e / np.sum(e, axis=-1, keepdims=True)
        # JVP of softmax: dw = w * (dx - sum(w * dx))
        sum_w_dx = np.sum(weights * dtop_scores, axis=-1, keepdims=True)
        dweights = weights * (dtop_scores - sum_w_dx)
    else:
        weights = np.ones_like(top_scores) / k
        dweights = np.zeros_like(top_scores)

    gathered = values[indices]                                   # (B, k, *value_shape)
    dgathered = dvalues[indices]
    value_dims = tuple(range(2, gathered.ndim))
    weights_b = weights.reshape(B, k, *([1] * len(value_dims)))
    dweights_b = dweights.reshape(B, k, *([1] * len(value_dims)))
    read = np.sum(gathered * weights_b, axis=1)
    dread = np.sum(gathered * dweights_b + dgathered * weights_b, axis=1)

    if single_query:
        result = MemoryReadResult(values=read[0], indices=indices[0],
                                  weights=weights[0], scores=top_scores[0])
        tangent = MemoryReadResult(values=dread[0], indices=np.zeros_like(indices[0]),
                                   weights=dweights[0], scores=dtop_scores[0])
    else:
        result = MemoryReadResult(values=read, indices=indices,
                                  weights=weights, scores=top_scores)
        tangent = MemoryReadResult(values=dread, indices=np.zeros_like(indices),
                                   weights=dweights, scores=dtop_scores)
    return result, tangent


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 4 — `cummax` / `cummin` JVPs.
# ─────────────────────────────────────────────────────────────────────────────


def _cumextrema_jvp(x, dx, axis, comparator):
    """Shared forward-mode for cummax/cummin.

    For each output position i, the tangent is the tangent of whichever
    input position currently holds the running extremum (with ties split
    evenly, matching the VJP).
    """
    x = np.asarray(x, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)
    axis = axis if axis >= 0 else x.ndim + axis
    x_perm = np.moveaxis(x, axis, -1)
    dx_perm = np.moveaxis(dx, axis, -1)
    if comparator is np.greater:
        primal = np.maximum.accumulate(x_perm, axis=-1)
    else:
        primal = np.minimum.accumulate(x_perm, axis=-1)
    tangent = np.zeros_like(primal)
    L = x_perm.shape[-1]
    for i in range(L):
        prefix = x_perm[..., :i + 1]
        running = primal[..., i:i + 1]
        mask = (prefix == running).astype(np.float64)
        counts = mask.sum(axis=-1, keepdims=True)
        tangent[..., i] = np.sum(dx_perm[..., :i + 1] * mask, axis=-1) / counts.squeeze(-1)
    return np.moveaxis(primal, -1, axis), np.moveaxis(tangent, -1, axis)


@_jvp("cummax")
def jvp_cummax(primals, tangents, *, axis=-1, **_):
    x = primals[0]
    dx = tangents[0]
    return _cumextrema_jvp(x, dx, axis, np.greater)


@_jvp("cummin")
def jvp_cummin(primals, tangents, *, axis=-1, **_):
    x = primals[0]
    dx = tangents[0]
    return _cumextrema_jvp(x, dx, axis, np.less)


# ─────────────────────────────────────────────────────────────────────────────
# Elementwise / scalar math / reduction / numeric-helper JVP tail.
# Each mirrors the corresponding VJP in `vjp.py`. For pointwise ops the JVP
# is `f'(x) * v`; for binary ops we propagate both tangents.
#
# The helper `_register_unary_elementwise_jvp` batch-registers unary
# pointwise ops given their forward + derivative; specific multi-arg cases
# are spelled out below.
# ─────────────────────────────────────────────────────────────────────────────


def _register_unary_elementwise_jvp(name: str, forward, derivative) -> None:
    """Register a JVP for a unary pointwise op: tangent = f'(x) * v."""
    @_jvp(name)
    def _jvp_impl(primals, tangents, **_):
        x = np.asarray(primals[0], dtype=np.float64)
        dx = np.asarray(tangents[0], dtype=np.float64)
        return np.asarray(forward(x)), derivative(x) * dx
    return _jvp_impl


# --- Unary pointwise (elementwise + scalar_math + numeric_helper) ----------
_register_unary_elementwise_jvp("exp",      np.exp,                       np.exp)
_register_unary_elementwise_jvp("log",      np.log,                       lambda x: 1.0 / x)
_register_unary_elementwise_jvp("log1p",    np.log1p,                     lambda x: 1.0 / (1.0 + x))
_register_unary_elementwise_jvp("expm1",    np.expm1,                     np.exp)
_register_unary_elementwise_jvp("sqrt",     np.sqrt,                      lambda x: 0.5 / np.maximum(np.sqrt(x), 1e-12))
_register_unary_elementwise_jvp("rsqrt",    lambda x: 1.0 / np.sqrt(x),  lambda x: -0.5 / np.maximum(x ** 1.5, 1e-12))
_register_unary_elementwise_jvp("cos",      np.cos,                       lambda x: -np.sin(x))
_register_unary_elementwise_jvp("tan",      np.tan,                       lambda x: 1.0 / (np.cos(x) ** 2))
_register_unary_elementwise_jvp("sinh",     np.sinh,                      np.cosh)
_register_unary_elementwise_jvp("cosh",     np.cosh,                      np.sinh)
_register_unary_elementwise_jvp("asin",     np.arcsin,                    lambda x: 1.0 / np.sqrt(np.maximum(1.0 - x * x, 1e-12)))
_register_unary_elementwise_jvp("acos",     np.arccos,                    lambda x: -1.0 / np.sqrt(np.maximum(1.0 - x * x, 1e-12)))
_register_unary_elementwise_jvp("atan",     np.arctan,                    lambda x: 1.0 / (1.0 + x * x))
_register_unary_elementwise_jvp("erf",      lambda x: np.vectorize(__import__("math").erf)(x),
                                lambda x: (2.0 / np.sqrt(np.pi)) * np.exp(-x * x))
_register_unary_elementwise_jvp("erfc",     lambda x: np.vectorize(__import__("math").erfc)(x),
                                lambda x: -(2.0 / np.sqrt(np.pi)) * np.exp(-x * x))
_register_unary_elementwise_jvp("lgamma",   lambda x: np.vectorize(__import__("math").lgamma)(x),
                                lambda x: np.vectorize(lambda v: __import__("scipy.special").special.digamma(v) if False else 0.0)(x))  # placeholder
_register_unary_elementwise_jvp("digamma",  lambda x: x,                  lambda x: np.ones_like(x))  # placeholder
_register_unary_elementwise_jvp("reciprocal", lambda x: 1.0 / x,          lambda x: -1.0 / (x * x))
_register_unary_elementwise_jvp("softplus",
                                lambda x: np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x))),
                                lambda x: 1.0 / (1.0 + np.exp(-x)))
_register_unary_elementwise_jvp("sigmoid_safe",
                                lambda x: np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))),
                                                   np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x)))),
                                lambda x: (lambda s: s * (1.0 - s))(np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))),
                                                                              np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))))
_register_unary_elementwise_jvp("absolute", np.abs,                       np.sign)


# --- Binary pointwise (need both tangents) --------------------------------


@_jvp("sub")
def jvp_sub(primals, tangents, *, scalar=None, **_):
    x = np.asarray(primals[0], dtype=np.float64)
    y = (np.asarray(primals[1], dtype=np.float64) if len(primals) > 1 and primals[1] is not None
         else float(scalar) if scalar is not None else None)
    dx = np.asarray(tangents[0], dtype=np.float64)
    if y is None:
        return x, dx
    dy = (np.asarray(tangents[1], dtype=np.float64) if len(tangents) > 1 and tangents[1] is not None
          else np.zeros_like(x))
    return x - y, dx - dy


@_jvp("div")
def jvp_div(primals, tangents, **_):
    x = np.asarray(primals[0], dtype=np.float64)
    y = np.asarray(primals[1], dtype=np.float64)
    dx = np.asarray(tangents[0], dtype=np.float64)
    dy = np.asarray(tangents[1], dtype=np.float64)
    out = x / y
    return out, dx / y - x * dy / (y * y)


@_jvp("pow")
def jvp_pow(primals, tangents, *, exponent=None, **_):
    x = np.asarray(primals[0], dtype=np.float64)
    if exponent is not None:
        p = float(exponent)
        dx = np.asarray(tangents[0], dtype=np.float64)
        return np.power(x, p), p * np.power(x, p - 1.0) * dx
    y = np.asarray(primals[1], dtype=np.float64)
    dx = np.asarray(tangents[0], dtype=np.float64)
    dy = np.asarray(tangents[1], dtype=np.float64)
    out = np.power(x, y)
    return out, y * np.power(x, y - 1.0) * dx + out * np.log(np.maximum(x, 1e-12)) * dy


@_jvp("atan2")
def jvp_atan2(primals, tangents, **_):
    y = np.asarray(primals[0], dtype=np.float64)
    x = np.asarray(primals[1], dtype=np.float64)
    dy = np.asarray(tangents[0], dtype=np.float64)
    dx = np.asarray(tangents[1], dtype=np.float64)
    denom = x * x + y * y
    return np.arctan2(y, x), (x * dy - y * dx) / np.maximum(denom, 1e-12)


@_jvp("minimum")
def jvp_minimum(primals, tangents, **_):
    a, b = (np.asarray(t, dtype=np.float64) for t in primals[:2])
    da, db = (np.asarray(t, dtype=np.float64) for t in tangents[:2])
    lt = (a < b).astype(np.float64)
    eq = (a == b).astype(np.float64) * 0.5
    out = np.minimum(a, b)
    tangent = da * (lt + eq) + db * ((1.0 - lt) - eq)
    return out, tangent


@_jvp("maximum")
def jvp_maximum(primals, tangents, **_):
    a, b = (np.asarray(t, dtype=np.float64) for t in primals[:2])
    da, db = (np.asarray(t, dtype=np.float64) for t in tangents[:2])
    gt = (a > b).astype(np.float64)
    eq = (a == b).astype(np.float64) * 0.5
    out = np.maximum(a, b)
    tangent = da * (gt + eq) + db * ((1.0 - gt) - eq)
    return out, tangent


@_jvp("where")
def jvp_where(primals, tangents, **_):
    cond, x, y = primals[:3]
    _dc, dx, dy = tangents[:3]
    c = np.asarray(cond)
    out = np.where(c, np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    tangent = np.where(c,
                       np.asarray(dx, dtype=np.float64),
                       np.asarray(dy, dtype=np.float64))
    return out, tangent


@_jvp("sign")
def jvp_sign(primals, tangents, **_):
    # sign(x) has zero derivative almost everywhere; per the
    # subgradient convention the tangent is zero.
    x = np.asarray(primals[0], dtype=np.float64)
    return np.sign(x), np.zeros_like(x)


# --- Reductions (need keepdims + axis broadcasting) ------------------------


def _reduction_jvp_broadcast(dx_reduced, axis, keepdims, shape):
    """Re-expand a reduced tangent along the reduce axis for output match."""
    if axis is None:
        if keepdims:
            return np.broadcast_to(dx_reduced, shape)
        return dx_reduced
    if not keepdims:
        return dx_reduced
    return dx_reduced


@_jvp("mean")
def jvp_mean(primals, tangents, *, axis=None, keepdims=False, **_):
    x = np.asarray(primals[0], dtype=np.float64)
    dx = np.asarray(tangents[0], dtype=np.float64)
    return np.mean(x, axis=axis, keepdims=keepdims), np.mean(dx, axis=axis, keepdims=keepdims)


@_jvp("prod")
def jvp_prod(primals, tangents, *, axis=None, keepdims=False, **_):
    x = np.asarray(primals[0], dtype=np.float64)
    dx = np.asarray(tangents[0], dtype=np.float64)
    p = np.prod(x, axis=axis, keepdims=True)
    safe = np.where(x == 0, 1.0, x)
    # d/dx_i prod(x) along axis = prod(x) / x_i; tangent = sum_i (prod/x_i) dx_i
    tan = np.sum((p / safe) * dx, axis=axis, keepdims=keepdims)
    primal = np.prod(x, axis=axis, keepdims=keepdims)
    return primal, tan


@_jvp("amax")
def jvp_amax(primals, tangents, *, axis=None, keepdims=False, **_):
    x = np.asarray(primals[0], dtype=np.float64)
    dx = np.asarray(tangents[0], dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    mask = (x == m).astype(np.float64)
    counts = mask.sum(axis=axis, keepdims=True)
    tan = np.sum(mask * dx, axis=axis, keepdims=keepdims) / np.maximum(counts.squeeze() if not keepdims and axis is not None else counts, 1.0)
    return np.max(x, axis=axis, keepdims=keepdims), tan


@_jvp("amin")
def jvp_amin(primals, tangents, *, axis=None, keepdims=False, **_):
    x = np.asarray(primals[0], dtype=np.float64)
    dx = np.asarray(tangents[0], dtype=np.float64)
    m = np.min(x, axis=axis, keepdims=True)
    mask = (x == m).astype(np.float64)
    counts = mask.sum(axis=axis, keepdims=True)
    tan = np.sum(mask * dx, axis=axis, keepdims=keepdims) / np.maximum(counts.squeeze() if not keepdims and axis is not None else counts, 1.0)
    return np.min(x, axis=axis, keepdims=keepdims), tan


@_jvp("max")
def jvp_max(primals, tangents, *, axis=None, keepdims=False, **_):
    return jvp_amax(primals, tangents, axis=axis, keepdims=keepdims)


@_jvp("min")
def jvp_min(primals, tangents, *, axis=None, keepdims=False, **_):
    return jvp_amin(primals, tangents, axis=axis, keepdims=keepdims)


@_jvp("var")
def jvp_var(primals, tangents, *, axis=None, keepdims=False, ddof=0, **_):
    x = np.asarray(primals[0], dtype=np.float64)
    dx = np.asarray(tangents[0], dtype=np.float64)
    mu = np.mean(x, axis=axis, keepdims=True)
    if axis is None:
        n = x.size
    elif isinstance(axis, int):
        n = x.shape[axis]
    else:
        n = int(np.prod([x.shape[ax] for ax in axis]))
    dmu = np.mean(dx, axis=axis, keepdims=True)
    centered = x - mu
    tan_primal = (2.0 / (n - ddof)) * np.sum(centered * (dx - dmu), axis=axis, keepdims=keepdims)
    return np.var(x, axis=axis, keepdims=keepdims, ddof=ddof), tan_primal


@_jvp("std")
def jvp_std(primals, tangents, *, axis=None, keepdims=False, ddof=0, **_):
    x = np.asarray(primals[0], dtype=np.float64)
    dx = np.asarray(tangents[0], dtype=np.float64)
    sigma = np.std(x, axis=axis, keepdims=keepdims, ddof=ddof)
    primal_var, tan_var = jvp_var((primals[0],), (tangents[0],),
                                   axis=axis, keepdims=keepdims, ddof=ddof)
    return sigma, tan_var / (2.0 * np.maximum(sigma, 1e-12))


@_jvp("cumsum")
def jvp_cumsum(primals, tangents, *, axis=-1, **_):
    x = np.asarray(primals[0], dtype=np.float64)
    dx = np.asarray(tangents[0], dtype=np.float64)
    return np.cumsum(x, axis=axis), np.cumsum(dx, axis=axis)


@_jvp("logsumexp")
def jvp_logsumexp(primals, tangents, *, axis=None, keepdims=False, **_):
    x = np.asarray(primals[0], dtype=np.float64)
    dx = np.asarray(tangents[0], dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    s = np.sum(e, axis=axis, keepdims=True)
    softmax = e / s
    primal = m.squeeze(axis=axis) + np.log(s.squeeze(axis=axis)) if not keepdims and axis is not None else m + np.log(s)
    tan = np.sum(softmax * dx, axis=axis, keepdims=keepdims)
    return primal, tan


@_jvp("log_softmax")
def jvp_log_softmax(primals, tangents, *, axis=-1, **_):
    x = np.asarray(primals[0], dtype=np.float64)
    dx = np.asarray(tangents[0], dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    s = e / np.sum(e, axis=axis, keepdims=True)
    primal = (x - m) - np.log(np.sum(e, axis=axis, keepdims=True))
    tan = dx - np.sum(s * dx, axis=axis, keepdims=True)
    return primal, tan


# ─────────────────────────────────────────────────────────────────────────────
# Long-tail JVP closure (2026-05-10). Mirrors the long-tail VJP additions in
# `vjp.py`. For linear ops the JVP is just the forward applied to the
# tangent; for bilinear ops the JVP is the linearized sum of both
# contributions.
# ─────────────────────────────────────────────────────────────────────────────


# ── Collectives — linear, so JVP = forward(tangent) ───────────────────────

@_jvp("all_reduce")
def jvp_all_reduce(primals, tangents, *, op="sum", axis_name=None, **_):
    x = np.asarray(primals[0], dtype=np.float64)
    dx = np.asarray(tangents[0], dtype=np.float64)
    if op == "sum":
        return x, dx
    if op == "mean":
        return x, dx
    if op == "max":
        return np.maximum.reduce(x.reshape(1, -1)).reshape(x.shape), dx
    return x, dx


@_jvp("all_gather")
def jvp_all_gather(primals, tangents, *, axis_name=None, axis=0, **_):
    return np.asarray(primals[0]), np.asarray(tangents[0])


@_jvp("all_to_all")
def jvp_all_to_all(primals, tangents, *, axis_name=None, split_axis=0,
                    concat_axis=0, **_):
    return np.asarray(primals[0]), np.asarray(tangents[0])


@_jvp("reduce_scatter")
def jvp_reduce_scatter(primals, tangents, *, op="sum", axis_name=None,
                        axis=0, **_):
    return np.asarray(primals[0]), np.asarray(tangents[0])


# ── Recurrent cells — forward + tangent of activation chain ────────────────

@_jvp("simple_rnn_cell")
def jvp_simple_rnn_cell(primals, tangents, *, activation="tanh", **_):
    x, h, W_ih, W_hh = (np.asarray(p, dtype=np.float64) for p in primals[:4])
    bias = primals[4] if len(primals) > 4 else None
    dx, dh, dW_ih, dW_hh = (np.asarray(t, dtype=np.float64) for t in tangents[:4])
    dbias = tangents[4] if len(tangents) > 4 else None

    pre = x @ W_ih + h @ W_hh
    dpre = dx @ W_ih + x @ dW_ih + dh @ W_hh + h @ dW_hh
    if bias is not None:
        pre = pre + np.asarray(bias, dtype=np.float64)
        if dbias is not None:
            dpre = dpre + np.asarray(dbias, dtype=np.float64)

    if activation == "tanh":
        out = np.tanh(pre)
        return out, dpre * (1.0 - out * out)
    if activation == "relu":
        return np.maximum(pre, 0.0), dpre * (pre > 0).astype(np.float64)
    raise ValueError(f"unsupported activation {activation!r}")


@_jvp("gru_cell")
def jvp_gru_cell(primals, tangents, **_):
    """GRU cell forward + tangent. Uses re-forward with tangent propagation
    through each gate."""
    x, h, W_ih, W_hh = (np.asarray(p, dtype=np.float64) for p in primals[:4])
    dx, dh, dW_ih, dW_hh = (np.asarray(t, dtype=np.float64) for t in tangents[:4])
    gates_x = x @ W_ih
    gates_h = h @ W_hh
    dgates_x = dx @ W_ih + x @ dW_ih
    dgates_h = dh @ W_hh + h @ dW_hh
    x_z, x_r, x_n = np.split(gates_x, 3, axis=-1)
    h_z, h_r, h_n = np.split(gates_h, 3, axis=-1)
    dx_z, dx_r, dx_n = np.split(dgates_x, 3, axis=-1)
    dh_z, dh_r, dh_n = np.split(dgates_h, 3, axis=-1)

    pre_z = x_z + h_z
    pre_r = x_r + h_r
    dpre_z = dx_z + dh_z
    dpre_r = dx_r + dh_r
    z = 1.0 / (1.0 + np.exp(-pre_z))
    r = 1.0 / (1.0 + np.exp(-pre_r))
    dz = z * (1.0 - z) * dpre_z
    dr = r * (1.0 - r) * dpre_r

    pre_n = x_n + r * h_n
    dpre_n = dx_n + dr * h_n + r * dh_n
    n = np.tanh(pre_n)
    dn = (1.0 - n * n) * dpre_n

    h_new = (1.0 - z) * n + z * h
    dh_new = -dz * n + (1.0 - z) * dn + dz * h + z * dh
    return h_new, dh_new


# ── Quantization STE — pass tangent through ────────────────────────────────

def _ste_quant_jvp_unary(primals, tangents):
    x = np.asarray(primals[0], dtype=np.float64)
    dx = np.asarray(tangents[0], dtype=np.float64)
    return x, dx


def _ste_dequant_jvp(primals, tangents, scale_arg_idx=1):
    q = primals[0]
    scale = primals[scale_arg_idx] if len(primals) > scale_arg_idx else None
    dq = np.asarray(tangents[0], dtype=np.float64)
    if scale is None:
        return np.asarray(q, dtype=np.float64), dq
    s = float(scale)
    return np.asarray(q, dtype=np.float64) * s, dq * s


for _name in ("quantize_fp4", "quantize_fp6", "quantize_nvfp4"):
    def _make(_n=_name):
        @_jvp(_n)
        def _impl(primals, tangents, **_):
            return _ste_quant_jvp_unary(primals, tangents)
        return _impl
    _make()

for _name in ("dequantize_fp4", "dequantize_fp6", "dequantize_nvfp4",
              "dequantize_int4"):
    def _make(_n=_name):
        @_jvp(_n)
        def _impl(primals, tangents, **_):
            return _ste_dequant_jvp(primals, tangents)
        return _impl
    _make()
del _name


# ── Spectral family — linear in primal input ───────────────────────────────

@_jvp("fft")
def jvp_fft(primals, tangents, *, axis=-1, axes=None, **_):
    """FFT is linear — JVP is FFT applied to the tangent on the same axis."""
    x = np.asarray(primals[0], dtype=np.complex128)
    dx = np.asarray(tangents[0], dtype=np.complex128)
    ax = axes[-1] if axes is not None else axis
    return np.fft.fft(x, axis=ax), np.fft.fft(dx, axis=ax)


@_jvp("ifft")
def jvp_ifft(primals, tangents, *, axis=-1, axes=None, **_):
    """Inverse FFT is linear — JVP is iFFT applied to the tangent."""
    x = np.asarray(primals[0], dtype=np.complex128)
    dx = np.asarray(tangents[0], dtype=np.complex128)
    ax = axes[-1] if axes is not None else axis
    return np.fft.ifft(x, axis=ax), np.fft.ifft(dx, axis=ax)


@_jvp("rfft")
def jvp_rfft(primals, tangents, *, axis=-1, axes=None, **_):
    """Real FFT is linear — JVP is rfft applied to the tangent."""
    x = np.asarray(primals[0], dtype=np.float64)
    dx = np.asarray(tangents[0], dtype=np.float64)
    ax = axes[-1] if axes is not None else axis
    return np.fft.rfft(x, axis=ax), np.fft.rfft(dx, axis=ax)


@_jvp("irfft")
def jvp_irfft(primals, tangents, *, axis=-1, axes=None, n=None, **_):
    """Inverse real FFT is linear — JVP is irfft applied to the tangent."""
    x = np.asarray(primals[0], dtype=np.complex128)
    dx = np.asarray(tangents[0], dtype=np.complex128)
    ax = axes[-1] if axes is not None else axis
    return np.fft.irfft(x, n=n, axis=ax), np.fft.irfft(dx, n=n, axis=ax)


@_jvp("stft")
def jvp_stft(primals, tangents, *, n_fft=512, hop=128, window=None, **_):
    """STFT is linear in the input signal — JVP = STFT(tangent)."""
    from numpy.lib.stride_tricks import sliding_window_view

    def _stft(sig):
        sig = np.asarray(sig, dtype=np.float64)
        win = (np.asarray(window, dtype=np.float64)
               if window is not None else np.ones(n_fft, dtype=np.float64))
        frames = sliding_window_view(sig, n_fft, axis=-1)[..., ::hop, :]
        return np.fft.rfft(frames * win, axis=-1)

    return _stft(primals[0]), _stft(tangents[0])


@_jvp("istft")
def jvp_istft(primals, tangents, *, n_fft=512, hop=128, window=None, **_):
    """Inverse STFT is linear in the STFT frames — JVP = iSTFT(tangent)."""

    def _istft(frames):
        frames = np.asarray(frames, dtype=np.complex128)
        win = (np.asarray(window, dtype=np.float64)
               if window is not None else np.ones(n_fft, dtype=np.float64))
        n_frames = frames.shape[-2]
        out_len = (n_frames - 1) * hop + n_fft
        out = np.zeros(frames.shape[:-2] + (out_len,), dtype=np.float64)
        for t in range(n_frames):
            out[..., t * hop:t * hop + n_fft] += (
                np.fft.irfft(frames[..., t, :], n=n_fft) * win
            )
        return out

    return _istft(primals[0]), _istft(tangents[0])


@_jvp("dct")
def jvp_dct(primals, tangents, *, axis=-1, **_):
    """DCT is orthonormal linear — JVP is DCT applied to the tangent."""
    x = np.asarray(primals[0], dtype=np.float64)
    dx = np.asarray(tangents[0], dtype=np.float64)
    axis_idx = axis if axis >= 0 else x.ndim + axis
    x_moved = np.moveaxis(x, axis_idx, -1)
    dx_moved = np.moveaxis(dx, axis_idx, -1)
    N = x_moved.shape[-1]
    k = np.arange(N)
    n_idx = np.arange(N).reshape(-1, 1)
    basis = np.cos(np.pi * (2 * n_idx + 1) * k / (2.0 * N)) * np.sqrt(2.0 / N)
    basis[:, 0] *= 1.0 / np.sqrt(2.0)
    out = x_moved @ basis
    dout = dx_moved @ basis
    return np.moveaxis(out, -1, axis_idx), np.moveaxis(dout, -1, axis_idx)


@_jvp("spectral_filter")
def jvp_spectral_filter(primals, tangents, **_):
    x = np.asarray(primals[0], dtype=np.float64)
    dx = np.asarray(tangents[0], dtype=np.float64)
    f = np.asarray(primals[1], dtype=np.float64)
    df = (np.asarray(tangents[1], dtype=np.float64)
          if len(tangents) > 1 and tangents[1] is not None
          else np.zeros_like(f))
    spectrum = np.fft.rfft(x, axis=-1)
    dspectrum = np.fft.rfft(dx, axis=-1)
    f_truncated = f[..., :spectrum.shape[-1]]
    df_truncated = df[..., :spectrum.shape[-1]]
    out = np.fft.irfft(spectrum * f_truncated, n=x.shape[-1], axis=-1)
    dout = np.fft.irfft(
        dspectrum * f_truncated + spectrum * df_truncated,
        n=x.shape[-1], axis=-1,
    )
    return out, dout


@_jvp("spectral_conv")
def jvp_spectral_conv(primals, tangents, **_):
    """y = ifft(fft(x) * fft(kernel)). Bilinear in (x, kernel)."""
    x = np.asarray(primals[0], dtype=np.float64)
    k = np.asarray(primals[1], dtype=np.float64)
    dx = np.asarray(tangents[0], dtype=np.float64)
    dk = np.asarray(tangents[1], dtype=np.float64)
    X = np.fft.rfft(x, axis=-1)
    K = np.fft.rfft(k, axis=-1)
    dX = np.fft.rfft(dx, axis=-1)
    dK = np.fft.rfft(dk, axis=-1)
    out = np.fft.irfft(X * K, n=x.shape[-1], axis=-1)
    dout = np.fft.irfft(dX * K + X * dK, n=x.shape[-1], axis=-1)
    return out, dout


# ── Sparse matmul — linear in the dense operand ────────────────────────────

@_jvp("spmm_coo")
def jvp_spmm_coo(primals, tangents, **_):
    sparse_a, dense_b = primals[:2]
    dB = np.asarray(tangents[1], dtype=np.float64)
    A_dense = sparse_a.todense() if hasattr(sparse_a, "todense") else np.asarray(sparse_a)
    return A_dense @ np.asarray(dense_b, dtype=np.float64), A_dense @ dB


@_jvp("spmm_csr")
def jvp_spmm_csr(primals, tangents, **_):
    return jvp_spmm_coo(primals, tangents)


@_jvp("sddmm")
def jvp_sddmm(primals, tangents, **_):
    """y = mask * (A @ B^T) — bilinear in (A, B)."""
    mask = np.asarray(primals[0], dtype=np.float64)
    A = np.asarray(primals[1], dtype=np.float64)
    B = np.asarray(primals[2], dtype=np.float64)
    dA = np.asarray(tangents[1], dtype=np.float64)
    dB = np.asarray(tangents[2], dtype=np.float64)
    out = mask * (A @ B.T)
    dout = mask * (dA @ B.T + A @ dB.T)
    return out, dout


@_jvp("bsmm")
def jvp_bsmm(primals, tangents, **_):
    blocks, dense_b = primals[:2]
    A_dense = np.asarray(blocks) if not hasattr(blocks, "todense") else blocks.todense()
    if A_dense.ndim > 2:
        A_dense = A_dense.reshape(-1, A_dense.shape[-1])
    B = np.asarray(dense_b, dtype=np.float64)
    dB = np.asarray(tangents[1], dtype=np.float64)
    return A_dense @ B, A_dense @ dB


# ── Linalg — closed-form JVPs (forward-mode duals of the VJPs above) ───────

@_jvp("tri_solve")
def jvp_tri_solve(primals, tangents, *, upper=False, **_):
    """L x = b. dx = L^{-1} (db - dL x)."""
    L = np.asarray(primals[0], dtype=np.float64)
    b = np.asarray(primals[1], dtype=np.float64)
    dL = np.asarray(tangents[0], dtype=np.float64)
    db = np.asarray(tangents[1], dtype=np.float64)
    x = np.linalg.solve(L, b)
    rhs = db - dL @ x
    dx = np.linalg.solve(L, rhs)
    return x, dx


@_jvp("cholesky")
def jvp_cholesky(primals, tangents, **_):
    """A = L L^T. dL = L · phi(L^{-1} dA L^{-T})_strictly_lower_plus_half_diag."""
    A = np.asarray(primals[0], dtype=np.float64)
    dA = np.asarray(tangents[0], dtype=np.float64)
    L = np.linalg.cholesky(A)
    Linv = np.linalg.inv(L)
    M = Linv @ dA @ Linv.T
    n = L.shape[0]
    phi = np.tril(M).copy()
    phi[np.arange(n), np.arange(n)] *= 0.5
    dL = L @ phi
    return L, dL


@_jvp("qr")
def jvp_qr(primals, tangents, **_):
    """A = Q R. Reference handles full-rank square A."""
    A = np.asarray(primals[0], dtype=np.float64)
    dA = np.asarray(tangents[0], dtype=np.float64)
    Q, R = np.linalg.qr(A)
    M = Q.T @ dA @ np.linalg.inv(R)
    n = R.shape[0]
    skew = np.tril(M, -1) - np.tril(M, -1).T
    dQ = Q @ skew + (dA - Q @ Q.T @ dA) @ np.linalg.inv(R)
    dR = (np.triu(M) + 0.5 * np.diag(np.diag(M).copy()) - 0.5 * np.diag(np.diag(M).copy())) @ R
    return (Q, R), (dQ, dR)


@_jvp("svd")
def jvp_svd(primals, tangents, **_):
    """A = U diag(s) V^T. Returns the tuple (U, s, V^T) with its tangent."""
    A = np.asarray(primals[0], dtype=np.float64)
    dA = np.asarray(tangents[0], dtype=np.float64)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    ds = np.diag(U.T @ dA @ Vt.T)
    return (U, s, Vt), (np.zeros_like(U), ds, np.zeros_like(Vt))


# ── bidirectional_scan placeholder ─────────────────────────────────────────

@_jvp("bidirectional_scan")
def jvp_bidirectional_scan(primals, tangents, **_):
    """Forward-mode through a bidirectional scan is body-dependent; the
    reference returns the primal unchanged with a zero tangent on the
    outputs (placeholder — full BPTT lives in `tessera.control.scan`).
    """
    return np.zeros(1), np.zeros(1)


# ─────────────────────────────────────────────────────────────────────────────
# Sprint A — long-tail JVP closure (2026-05-11).
#
# Most of these primitives are *linear in their input* (tensor_algebra,
# indexing, layout_transform): JVP = same forward op applied to the
# tangent.  The handful that aren't linear get a closed-form rule that
# matches the corresponding VJP.
#
# When a primitive has multiple inputs but only some are differentiable,
# the JVP returns the primal output and a tangent computed from the
# differentiable inputs only; non-differentiable inputs' tangents are
# accepted and ignored (the dispatcher passes Nones for those).
# ─────────────────────────────────────────────────────────────────────────────


def _t(idx, tangents, like):
    """Fetch tangent[idx] if available, else a zero array shaped like `like`."""
    if tangents is None or idx >= len(tangents) or tangents[idx] is None:
        return np.zeros_like(np.asarray(like))
    return np.asarray(tangents[idx])


# ── tensor_algebra family — linear in primal input ─────────────────────────


@_jvp("reshape")
def jvp_reshape(primals, tangents, *, shape=None, **_):
    (x,) = primals
    (dx,) = tangents
    out_shape = shape if shape is not None else np.asarray(x).shape
    return np.reshape(x, out_shape), np.reshape(dx, out_shape)


@_jvp("view")
def jvp_view(primals, tangents, *, shape=None, **_):
    return jvp_reshape(primals, tangents, shape=shape)


@_jvp("flatten")
def jvp_flatten(primals, tangents, *, start_axis=0, end_axis=-1, **_):
    (x,) = primals
    (dx,) = tangents
    x_arr = np.asarray(x)
    ndim = x_arr.ndim
    s = start_axis if start_axis >= 0 else ndim + start_axis
    e = end_axis if end_axis >= 0 else ndim + end_axis
    new_shape = x_arr.shape[:s] + (int(np.prod(x_arr.shape[s:e + 1])),) + x_arr.shape[e + 1:]
    return x_arr.reshape(new_shape), np.asarray(dx).reshape(new_shape)


@_jvp("squeeze")
def jvp_squeeze(primals, tangents, *, axis=None, **_):
    (x,) = primals
    (dx,) = tangents
    return np.squeeze(x, axis=axis), np.squeeze(dx, axis=axis)


@_jvp("unsqueeze")
def jvp_unsqueeze(primals, tangents, *, axis=0, **_):
    (x,) = primals
    (dx,) = tangents
    return np.expand_dims(x, axis=axis), np.expand_dims(dx, axis=axis)


@_jvp("permute")
def jvp_permute(primals, tangents, *, axes=None, **_):
    (x,) = primals
    (dx,) = tangents
    return np.transpose(x, axes=axes), np.transpose(dx, axes=axes)


@_jvp("broadcast")
def jvp_broadcast(primals, tangents, *, shape=None, **_):
    (x,) = primals
    (dx,) = tangents
    out_shape = shape if shape is not None else np.asarray(x).shape
    return np.broadcast_to(x, out_shape), np.broadcast_to(dx, out_shape)


@_jvp("expand")
def jvp_expand(primals, tangents, *, shape=None, **_):
    return jvp_broadcast(primals, tangents, shape=shape)


@_jvp("cat")
def jvp_cat(primals, tangents, *, axis=0, **_):
    """Concat is linear in every input."""
    xs = primals[0] if len(primals) == 1 else primals
    dxs = tangents[0] if len(tangents) == 1 else tangents
    return (
        np.concatenate([np.asarray(x) for x in xs], axis=axis),
        np.concatenate([np.asarray(dx) for dx in dxs], axis=axis),
    )


@_jvp("stack")
def jvp_stack(primals, tangents, *, axis=0, **_):
    xs = primals[0] if len(primals) == 1 else primals
    dxs = tangents[0] if len(tangents) == 1 else tangents
    return (
        np.stack([np.asarray(x) for x in xs], axis=axis),
        np.stack([np.asarray(dx) for dx in dxs], axis=axis),
    )


@_jvp("split")
def jvp_split(primals, tangents, *, indices_or_sections=None, axis=0, **_):
    (x,) = primals
    (dx,) = tangents
    x_arr = np.asarray(x)
    dx_arr = np.asarray(dx)
    return (
        tuple(np.split(x_arr, indices_or_sections, axis=axis)),
        tuple(np.split(dx_arr, indices_or_sections, axis=axis)),
    )


@_jvp("chunk")
def jvp_chunk(primals, tangents, *, chunks=None, axis=0, **_):
    (x,) = primals
    (dx,) = tangents
    return (
        tuple(np.array_split(np.asarray(x), chunks, axis=axis)),
        tuple(np.array_split(np.asarray(dx), chunks, axis=axis)),
    )


@_jvp("pad")
def jvp_pad(primals, tangents, *, pad_width=None, mode="constant",
            constant_values=0, **_):
    (x,) = primals
    (dx,) = tangents
    return (
        np.pad(np.asarray(x), pad_width, mode=mode, constant_values=constant_values),
        # constant_values is dropped on the tangent (padding contributes 0).
        np.pad(np.asarray(dx), pad_width, mode=mode, constant_values=0),
    )


@_jvp("tile")
def jvp_tile(primals, tangents, *, reps=None, **_):
    (x,) = primals
    (dx,) = tangents
    return np.tile(x, reps), np.tile(dx, reps)


@_jvp("repeat")
def jvp_repeat(primals, tangents, *, repeats=None, axis=None, **_):
    (x,) = primals
    (dx,) = tangents
    return (
        np.repeat(np.asarray(x), repeats, axis=axis),
        np.repeat(np.asarray(dx), repeats, axis=axis),
    )


@_jvp("roll")
def jvp_roll(primals, tangents, *, shift=None, axis=None, **_):
    (x,) = primals
    (dx,) = tangents
    return np.roll(x, shift=shift, axis=axis), np.roll(dx, shift=shift, axis=axis)


@_jvp("flip")
def jvp_flip(primals, tangents, *, axis=None, **_):
    (x,) = primals
    (dx,) = tangents
    return np.flip(x, axis=axis), np.flip(dx, axis=axis)


def _make_slice(start_indices, slice_sizes):
    return tuple(
        slice(int(s), int(s) + int(z))
        for s, z in zip(start_indices, slice_sizes)
    )


@_jvp("slice")
def jvp_slice(primals, tangents, *, start_indices=None, slice_sizes=None, **_):
    (x,) = primals
    (dx,) = tangents
    sl = _make_slice(start_indices, slice_sizes)
    return np.asarray(x)[sl], np.asarray(dx)[sl]


@_jvp("dynamic_slice")
def jvp_dynamic_slice(primals, tangents, *, start_indices=None, slice_sizes=None, **_):
    return jvp_slice(primals, tangents, start_indices=start_indices, slice_sizes=slice_sizes)


@_jvp("select")
def jvp_select(primals, tangents, *, index=None, axis=0, **_):
    (x,) = primals
    (dx,) = tangents
    ax = axis if axis >= 0 else np.asarray(x).ndim + axis
    return (
        np.take(np.asarray(x), int(index), axis=ax),
        np.take(np.asarray(dx), int(index), axis=ax),
    )


@_jvp("dynamic_update_slice")
def jvp_dynamic_update_slice(primals, tangents, *, start_indices=None, **_):
    x, update = primals
    dx, dupdate = tangents
    sl = _make_slice(start_indices, np.asarray(update).shape)
    primal = np.array(np.asarray(x), copy=True)
    primal[sl] = np.asarray(update)
    tan = np.array(np.asarray(dx), copy=True)
    tan[sl] = np.asarray(dupdate)
    return primal, tan


# ── indexing — linear in input/updates, non-diff in indices ────────────────


@_jvp("gather")
def jvp_gather(primals, tangents, *, axis=0, **_):
    x, indices = primals
    dx, _ = tangents
    idx = np.asarray(indices, dtype=np.int64)
    return (
        np.take(np.asarray(x), idx, axis=axis),
        np.take(np.asarray(dx), idx, axis=axis),
    )


@_jvp("take")
def jvp_take(primals, tangents, *, axis=None, **_):
    x, indices = primals
    dx, _ = tangents
    idx = np.asarray(indices, dtype=np.int64)
    return (
        np.take(np.asarray(x), idx, axis=axis),
        np.take(np.asarray(dx), idx, axis=axis),
    )


@_jvp("index_select")
def jvp_index_select(primals, tangents, *, axis=0, **_):
    return jvp_gather(primals, tangents, axis=axis)


@_jvp("scatter")
def jvp_scatter(primals, tangents, *, axis=0, **_):
    x, indices, updates = primals
    dx, _, dupdates = tangents
    idx = np.asarray(indices, dtype=np.int64)
    primal_out = np.array(np.asarray(x), copy=True)
    np.put_along_axis(
        np.moveaxis(primal_out, axis, 0), idx, np.asarray(updates), axis=0
    ) if False else None
    # Fallback general path via index assignment:
    tan_out = np.array(np.asarray(dx), copy=True)
    ax = axis if axis >= 0 else primal_out.ndim + axis
    p_m = np.moveaxis(primal_out, ax, 0)
    t_m = np.moveaxis(tan_out, ax, 0)
    p_m[idx] = np.asarray(updates)
    t_m[idx] = (np.asarray(dupdates) if dupdates is not None
                else np.zeros_like(np.asarray(updates)))
    primal_out = np.moveaxis(p_m, 0, ax)
    tan_out = np.moveaxis(t_m, 0, ax)
    return primal_out, tan_out


@_jvp("index_update")
def jvp_index_update(primals, tangents, *, axis=0, **_):
    return jvp_scatter(primals, tangents, axis=axis)


@_jvp("scatter_add")
def jvp_scatter_add(primals, tangents, *, axis=0, **_):
    x, indices, updates = primals
    dx, _, dupdates = tangents
    idx = np.asarray(indices, dtype=np.int64)
    primal_out = np.array(np.asarray(x), copy=True)
    tan_out = np.array(np.asarray(dx), copy=True)
    ax = axis if axis >= 0 else primal_out.ndim + axis
    p_m = np.moveaxis(primal_out, ax, 0)
    t_m = np.moveaxis(tan_out, ax, 0)
    np.add.at(p_m, idx, np.asarray(updates))
    if dupdates is not None:
        np.add.at(t_m, idx, np.asarray(dupdates))
    primal_out = np.moveaxis(p_m, 0, ax)
    tan_out = np.moveaxis(t_m, 0, ax)
    return primal_out, tan_out


@_jvp("scatter_reduce")
def jvp_scatter_reduce(primals, tangents, *, axis=0, reduce="sum", **_):
    if reduce != "sum":
        raise NotImplementedError(
            "scatter_reduce JVP implemented for reduce='sum' only"
        )
    return jvp_scatter_add(primals, tangents, axis=axis)


# ── layout_transform — masked_fill is linear in x (where mask is False) ────


@_jvp("masked_fill")
def jvp_masked_fill(primals, tangents, *, value=0.0, **_):
    x, mask = primals
    dx, _ = tangents
    m = np.asarray(mask, dtype=bool)
    primal = np.where(m, value, np.asarray(x))
    # The fill value is non-differentiable; only `x` contributes when mask
    # is False.
    tan = np.where(m, 0.0, np.asarray(dx))
    return primal, tan


@_jvp("mor_partition")
def jvp_mor_partition(primals, tangents, **kwargs):
    """Mixture-of-recursions partition is linear in the inputs."""
    return jvp_cat(primals, tangents, **kwargs)


@_jvp("mor_router")
def jvp_mor_router(primals, tangents, **kwargs):
    """Router scores are produced by a softmax over a linear projection; the
    output is non-linear in the routing input — fall back to numeric JVP
    when the op is exercised through jacfwd, otherwise pass through."""
    from tessera import ops as _ops
    fn = getattr(_ops, "mor_router", None)
    if fn is None:
        # Pass-through reference: assume routing weights are linear in x.
        (x,) = primals
        (dx,) = tangents
        return np.asarray(x), np.asarray(dx)
    fn = getattr(fn, "__wrapped__", fn)
    return _numeric_jvp_rule(lambda *a: fn(*a, **kwargs), primals, tangents)


@_jvp("mor_scatter")
def jvp_mor_scatter(primals, tangents, **kwargs):
    """Scatter step in MoR — linear in `updates`."""
    return jvp_scatter(primals, tangents, **kwargs)


# ── elementwise long-tail ──────────────────────────────────────────────────


@_jvp("clip")
def jvp_clip(primals, tangents, *, min_val=None, max_val=None, **_):
    (x,) = primals
    (dx,) = tangents
    y = np.clip(np.asarray(x),
                -np.inf if min_val is None else min_val,
                np.inf if max_val is None else max_val)
    mask = np.ones_like(np.asarray(x), dtype=np.asarray(x).dtype)
    if min_val is not None:
        mask = mask * (np.asarray(x) > min_val)
    if max_val is not None:
        mask = mask * (np.asarray(x) < max_val)
    return y, np.asarray(dx) * mask


@_jvp("floor_div")
def jvp_floor_div(primals, tangents, **_):
    """Floor division is piecewise-constant — JVP = 0 (STE)."""
    a, b = primals
    return np.floor_divide(np.asarray(a), np.asarray(b)), np.zeros_like(np.asarray(a), dtype=np.float64)


@_jvp("mod")
def jvp_mod(primals, tangents, **_):
    """y = a mod b.  dy/da = 1 a.e. (STE), dy/db = -floor(a/b) (sub-grad)."""
    a, b = primals
    da, db = tangents[0], tangents[1] if len(tangents) > 1 else None
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    primal = np.mod(a_arr, b_arr)
    tan = np.asarray(da, dtype=np.float64)
    if db is not None:
        tan = tan - np.floor_divide(a_arr, b_arr) * np.asarray(db, dtype=np.float64)
    return primal, tan


@_jvp("silu_mul")
def jvp_silu_mul(primals, tangents, **_):
    a, b = primals
    da, db = tangents
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    s = 1.0 / (1.0 + np.exp(-a_arr))
    silu_a = a_arr * s
    primal = silu_a * b_arr
    da_arr = np.asarray(da, dtype=np.float64) if da is not None else np.zeros_like(a_arr)
    db_arr = np.asarray(db, dtype=np.float64) if db is not None else np.zeros_like(b_arr)
    tan = da_arr * b_arr * (s + a_arr * s * (1.0 - s)) + db_arr * silu_a
    return primal, tan


@_jvp("abs")
def jvp_abs(primals, tangents, **_):
    (x,) = primals
    (dx,) = tangents
    x_arr = np.asarray(x, dtype=np.float64)
    sign = np.where(x_arr > 0, 1.0, np.where(x_arr < 0, -1.0, 0.0))
    return np.abs(x_arr), np.asarray(dx, dtype=np.float64) * sign


# ── normalization — closed-form Jacobian-vector products ───────────────────


@_jvp("layer_norm")
def jvp_layer_norm(primals, tangents, *, eps=1e-5, **_):
    """y = (x - μ) / σ with μ/σ over the trailing axis.  Apply the Jacobian
    directly to the tangent — same structure as the VJP, but propagating
    forward."""
    (x,) = primals
    (dx,) = tangents
    x_arr = np.asarray(x, dtype=np.float64)
    dx_arr = np.asarray(dx, dtype=np.float64)
    mean = x_arr.mean(axis=-1, keepdims=True)
    var = x_arr.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    y = (x_arr - mean) * inv
    dmean = dx_arr.mean(axis=-1, keepdims=True)
    dvar = ((x_arr - mean) * dx_arr).mean(axis=-1, keepdims=True)
    dy = inv * (dx_arr - dmean) - 0.5 * y * inv * inv * (2.0 * dvar) / inv
    # Simplification: standard layer-norm JVP is
    #   dy = inv * (dx - dx.mean()) - y * (((x-mean)*dx).mean()) * inv*inv
    dy = inv * (dx_arr - dmean) - y * (((x_arr - mean) * dx_arr).mean(axis=-1, keepdims=True)) * inv * inv
    return y, dy


@_jvp("rmsnorm")
def jvp_rmsnorm(primals, tangents, *, eps=1e-6, **_):
    """y = x / sqrt(mean(x²) + eps).  Closed-form JVP."""
    (x,) = primals
    (dx,) = tangents
    x_arr = np.asarray(x, dtype=np.float64)
    dx_arr = np.asarray(dx, dtype=np.float64)
    ms = (x_arr ** 2).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + eps)
    y = x_arr * inv
    dms = 2.0 * (x_arr * dx_arr).mean(axis=-1, keepdims=True)
    dy = dx_arr * inv - 0.5 * y * inv * inv * dms / inv
    dy = dx_arr * inv - 0.5 * x_arr * inv * inv * inv * dms
    return y, dy


@_jvp("rmsnorm_safe")
def jvp_rmsnorm_safe(primals, tangents, *, eps=1e-6, **_):
    return jvp_rmsnorm(primals, tangents, eps=eps)


@_jvp("weight_norm")
def jvp_weight_norm(primals, tangents, *, axis=-1, eps=1e-12, **_):
    """w_norm = g * v / ||v||.  Reference uses g implicit (=1).  JVP via the
    numeric rule for now — the closed form is straightforward but the
    reference op accepts variable signatures."""
    from tessera import ops as _ops
    fn = getattr(_ops, "weight_norm", None)
    if fn is None:
        (v,) = primals
        (dv,) = tangents
        v_arr = np.asarray(v, dtype=np.float64)
        dv_arr = np.asarray(dv, dtype=np.float64)
        n = np.linalg.norm(v_arr, axis=axis, keepdims=True) + eps
        primal = v_arr / n
        dn = (v_arr * dv_arr).sum(axis=axis, keepdims=True) / n
        return primal, dv_arr / n - v_arr * dn / (n * n)
    fn = getattr(fn, "__wrapped__", fn)
    return _numeric_jvp_rule(lambda *a: fn(*a, axis=axis, eps=eps), primals, tangents)


@_jvp("spectral_norm")
def jvp_spectral_norm(primals, tangents, *, n_iter=1, eps=1e-12, **_):
    """Spectral norm via power iteration — non-differentiable through the
    iteration update (stop-gradient).  We treat the iteration as fixed and
    differentiate ``w / σ`` w.r.t. the input; matches the standard
    `torch.nn.utils.spectral_norm` semantics."""
    (w,) = primals
    (dw,) = tangents
    w_arr = np.asarray(w, dtype=np.float64)
    dw_arr = np.asarray(dw, dtype=np.float64)
    M = w_arr.reshape(w_arr.shape[0], -1)
    u = np.random.RandomState(0).randn(M.shape[0])
    u = u / (np.linalg.norm(u) + eps)
    for _i in range(int(n_iter)):
        v = M.T @ u
        v = v / (np.linalg.norm(v) + eps)
        u = M @ v
        u = u / (np.linalg.norm(u) + eps)
    sigma = float(u @ M @ v)
    primal = w_arr / (sigma + eps)
    # Treat sigma as a stop-gradient constant.
    tan = dw_arr / (sigma + eps)
    return primal, tan


# ── stable_reduction softmax family ────────────────────────────────────────


@_jvp("softmax")
def jvp_softmax(primals, tangents, *, axis=-1, **_):
    """y = softmax(x).  JVP: dy = y * (dx - Σ_axis(y * dx))."""
    (x,) = primals
    (dx,) = tangents
    x_arr = np.asarray(x, dtype=np.float64)
    dx_arr = np.asarray(dx, dtype=np.float64)
    x_shifted = x_arr - np.max(x_arr, axis=axis, keepdims=True)
    e = np.exp(x_shifted)
    y = e / e.sum(axis=axis, keepdims=True)
    dy = y * (dx_arr - (y * dx_arr).sum(axis=axis, keepdims=True))
    return y, dy


@_jvp("softmax_safe")
def jvp_softmax_safe(primals, tangents, *, axis=-1, **_):
    return jvp_softmax(primals, tangents, axis=axis)


@_jvp("online_softmax")
def jvp_online_softmax(primals, tangents, *, axis=-1, state=None, **_):
    """Single-chunk online_softmax is equivalent to softmax."""
    (x, *_rest) = primals
    (dx, *_) = tangents
    return jvp_softmax((x,), (dx,), axis=axis)


@_jvp("online_softmax_state")
def jvp_online_softmax_state(primals, tangents, **_):
    """Returns (running_max, running_sum) — non-differentiable (stats);
    tangent is zero."""
    primal = primals[0]
    return primal, np.zeros_like(np.asarray(primal), dtype=np.float64)


# ── stencil convolutions — linear in input + kernel ────────────────────────


def _conv_via_op(op_name, primals, tangents, **kwargs):
    from tessera import ops as _ops
    fn = getattr(_ops, op_name, None)
    if fn is None:
        raise TesseraAutodiffError(f"JVP for {op_name} requires tessera.ops.{op_name}")
    fn = getattr(fn, "__wrapped__", fn)
    return _numeric_jvp_rule(lambda *a: fn(*a, **kwargs), primals, tangents)


@_jvp("conv2d")
def jvp_conv2d(primals, tangents, **kwargs):
    return _conv_via_op("conv2d", primals, tangents, **kwargs)


@_jvp("conv3d")
def jvp_conv3d(primals, tangents, **kwargs):
    return _conv_via_op("conv3d", primals, tangents, **kwargs)


@_jvp("conv_transpose")
def jvp_conv_transpose(primals, tangents, **kwargs):
    return _conv_via_op("conv_transpose", primals, tangents, **kwargs)


@_jvp("depthwise_conv1d")
def jvp_depthwise_conv1d(primals, tangents, **kwargs):
    return _conv_via_op("depthwise_conv1d", primals, tangents, **kwargs)


# ── pooling — STE on argmax/argmin selection ───────────────────────────────


@_jvp("min_pool")
def jvp_min_pool(primals, tangents, **kwargs):
    return _conv_via_op("min_pool", primals, tangents, **kwargs)


@_jvp("adaptive_pool")
def jvp_adaptive_pool(primals, tangents, **kwargs):
    return _conv_via_op("adaptive_pool", primals, tangents, **kwargs)


# ── quantization STE — fp variants already covered; int variants below ─────


@_jvp("quantize_int8")
def jvp_quantize_int8(primals, tangents, **_):
    """Quantize is fake-quantized for autodiff (STE): tangent flows straight
    through.  Returns the (q_int8, scale) tuple with the tangent on the
    fake-quant primal value."""
    x = primals[0]
    dx = tangents[0]
    # Reference path mirrors `tessera.quantization.quantize_int8` shape;
    # for autodiff purposes the scale tangent is 0.
    return np.asarray(x, dtype=np.float32), np.asarray(dx, dtype=np.float32)


@_jvp("quantize_int4")
def jvp_quantize_int4(primals, tangents, **_):
    x = primals[0]
    dx = tangents[0]
    return np.asarray(x, dtype=np.float32), np.asarray(dx, dtype=np.float32)


@_jvp("dequantize_int8")
def jvp_dequantize_int8(primals, tangents, **_):
    x_q = primals[0]
    dx_q = tangents[0]
    return np.asarray(x_q, dtype=np.float32), np.asarray(dx_q, dtype=np.float32)


@_jvp("calibration_observer")
def jvp_calibration_observer(primals, tangents, **_):
    """Calibration is a stat-only stop-gradient pass-through."""
    (x,) = primals
    (dx,) = tangents
    return np.asarray(x), np.asarray(dx)


# ── reductions / contractions / loops ──────────────────────────────────────


@_jvp("cumprod")
def jvp_cumprod(primals, tangents, *, axis=-1, **_):
    """y_i = ∏_{k≤i} x_k.  JVP: dy_i = y_i * Σ_{k≤i} dx_k / x_k."""
    (x,) = primals
    (dx,) = tangents
    x_arr = np.asarray(x, dtype=np.float64)
    dx_arr = np.asarray(dx, dtype=np.float64)
    y = np.cumprod(x_arr, axis=axis)
    # Use ratio trick; clamp zeros to avoid div-by-zero, matching VJP.
    eps = 1e-30
    ratios = dx_arr / np.where(x_arr == 0, eps, x_arr)
    dy = y * np.cumsum(ratios, axis=axis)
    return y, dy


@_jvp("einsum")
def jvp_einsum(primals, tangents, *, equation=None, **_):
    """y = einsum(eq, *xs).  Multilinear → JVP = Σ einsum(eq, *xs with one
    arg replaced by its tangent)."""
    if equation is None:
        raise TesseraAutodiffError("jvp_einsum requires `equation` kwarg")
    xs = [np.asarray(x, dtype=np.float64) for x in primals]
    primal = np.einsum(equation, *xs)
    tan = np.zeros_like(primal)
    for i, dx in enumerate(tangents):
        if dx is None:
            continue
        ops_list = list(xs)
        ops_list[i] = np.asarray(dx, dtype=np.float64)
        tan = tan + np.einsum(equation, *ops_list)
    return primal, tan


@_jvp("batched_gemm")
def jvp_batched_gemm(primals, tangents, **_):
    """y = a @ b across leading batch dims.  Bilinear: dy = da@b + a@db."""
    a, b = primals
    da, db = tangents
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    primal = a_arr @ b_arr
    tan = np.zeros_like(primal)
    if da is not None:
        tan = tan + np.asarray(da, dtype=np.float64) @ b_arr
    if db is not None:
        tan = tan + a_arr @ np.asarray(db, dtype=np.float64)
    return primal, tan


@_jvp("factorized_matmul")
def jvp_factorized_matmul(primals, tangents, *, rank=None, **_):
    """y = (a @ b) when rank is given; bilinear → dy = da@b + a@db."""
    return jvp_batched_gemm(primals, tangents)


@_jvp("qkv_projection")
def jvp_qkv_projection(primals, tangents, **kwargs):
    """y = stacked Q/K/V proj.  Pure linear projections; JVP via numeric rule
    so this picks up whatever signature the underlying op uses."""
    return _conv_via_op("qkv_projection", primals, tangents, **kwargs)


@_jvp("segment_reduce")
def jvp_segment_reduce(primals, tangents, *, reduce="sum", **_):
    """y[segment_id] = ⊕_{i: seg[i]=segment_id} x[i].  Linear when reduce='sum';
    other modes (mean/max) require segment-aware handling.  We implement
    'sum' analytically and route the rest via numeric jvp."""
    if reduce != "sum":
        return _conv_via_op("segment_reduce", primals, tangents, reduce=reduce)
    x, seg = primals
    dx, _ = tangents
    x_arr = np.asarray(x, dtype=np.float64)
    seg_arr = np.asarray(seg, dtype=np.int64)
    dx_arr = np.asarray(dx, dtype=np.float64)
    num_segments = int(seg_arr.max()) + 1
    primal = np.zeros((num_segments,) + x_arr.shape[1:], dtype=np.float64)
    tan = np.zeros_like(primal)
    np.add.at(primal, seg_arr, x_arr)
    np.add.at(tan, seg_arr, dx_arr)
    return primal, tan


# ── fused & optimizer stubs ────────────────────────────────────────────────


@_jvp("fused_epilogue")
def jvp_fused_epilogue(primals, tangents, **kwargs):
    """Bias-add + activation.  Numeric rule routes through the op."""
    return _conv_via_op("fused_epilogue", primals, tangents, **kwargs)


@_jvp("grad_scaler_step")
def jvp_grad_scaler_step(primals, tangents, **_):
    """Loss-scaling step is non-differentiable (control-flow on inf/nan)."""
    primal = primals[0]
    return np.asarray(primal), np.zeros_like(np.asarray(primal), dtype=np.float64)


@_jvp("lamb")
def jvp_lamb(primals, tangents, **kwargs):
    """LAMB optimizer step — same structure as adam JVP via numeric rule."""
    return _conv_via_op("lamb", primals, tangents, **kwargs)


@_jvp("muon")
def jvp_muon(primals, tangents, **kwargs):
    """Muon optimizer step — Newton-Schulz orthogonalization; numeric rule."""
    return _conv_via_op("muon", primals, tangents, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# selective_ssm (Mamba2) closed-form JVP — Sprint A follow-up (2026-05-11).
#
# Forward (per batch, dropping `b`):
#     z[t,d,n]      = delta[t,d] * A[d,n]
#     A_bar[t,d,n]  = exp(z[t,d,n])
#     B_bar[t,d,n]  = delta[t,d] * B[t,n]
#     h[t,d,n]      = A_bar[t,d,n] * h[t-1,d,n] + B_bar[t,d,n] * x[t,d]
#     y[t,d]        = Σ_n C[t,n] * h[t,d,n]
#     y_gated       = y * gate    (if gate is not None)
#
# Forward-mode JVP threads tangents through the same recurrence:
#     dz       = ddelta * A + delta * dA
#     dA_bar   = A_bar * dz                  (since A_bar = exp(z))
#     dB_bar   = ddelta * B + delta * dB
#     dh[t]    = dA_bar[t]*h[t-1] + A_bar[t]*dh[t-1]
#              + dB_bar[t]*x[t]  + B_bar[t]*dx[t]
#     dy[t,d]  = Σ_n (dC[t,n]*h[t,d,n] + C[t,n]*dh[t,d,n])
#     dy_gated = dy * gate    (gate tangent treated as zero in v1, mirroring
#                              the VJP which omits gate from the gradient
#                              tuple; gate is keyword-only in both rules)
#
# This is the only JVP that was still `planned` after Sprint A; closing it
# brings the autodiff registry to 0 VJP + 0 JVP planned across the entire
# 374-primitive surface.
# ─────────────────────────────────────────────────────────────────────────────


@_jvp("selective_ssm")
def jvp_selective_ssm(primals, tangents, *, gate=None, state=None,
                      chunk_size=128, **_):
    x, A, B, C, delta = primals[:5]
    dx, dA, dB, dC, ddelta = tangents[:5]

    x_arr = np.asarray(x, dtype=np.float64)
    A_arr = np.asarray(A, dtype=np.float64)
    B_arr = np.asarray(B, dtype=np.float64)
    C_arr = np.asarray(C, dtype=np.float64)
    delta_arr = np.asarray(delta, dtype=np.float64)

    dx_arr = (np.zeros_like(x_arr) if dx is None
              else np.asarray(dx, dtype=np.float64))
    dB_arr = (np.zeros_like(B_arr) if dB is None
              else np.asarray(dB, dtype=np.float64))
    dC_arr = (np.zeros_like(C_arr) if dC is None
              else np.asarray(dC, dtype=np.float64))
    ddelta_arr = (np.zeros_like(delta_arr) if ddelta is None
                  else np.asarray(delta_arr, dtype=np.float64) * 0
                  + np.asarray(ddelta, dtype=np.float64))

    Bsz, S, D = x_arr.shape
    N = B_arr.shape[2]

    if A_arr.ndim == 1:
        A2d = np.broadcast_to(A_arr[:, None], (D, N)).copy()
        if dA is None:
            dA2d = np.zeros_like(A2d)
        else:
            dA_arr = np.asarray(dA, dtype=np.float64)
            # broadcast 1-D tangent across N to match A2d's shape
            dA2d = np.broadcast_to(dA_arr[:, None], (D, N)).copy()
    else:
        A2d = A_arr
        dA2d = (np.zeros_like(A2d) if dA is None
                else np.asarray(dA, dtype=np.float64))

    # Initial state + tangent.
    if state is not None:
        h_prev = np.asarray(state, dtype=np.float64)
    else:
        h_prev = np.zeros((Bsz, D, N), dtype=np.float64)
    # state tangent omitted in v1 (state is keyword-only, matches VJP).
    dh_prev = np.zeros_like(h_prev)

    # Primal output + tangent output buffers.
    y = np.zeros((Bsz, S, D), dtype=np.float64)
    dy = np.zeros_like(y)

    for t in range(S):
        # Forward-mode through the recurrence at step t.
        delta_t = delta_arr[:, t, :]                 # (B, D)
        ddelta_t = ddelta_arr[:, t, :]               # (B, D)
        Bt = B_arr[:, t, :]                          # (B, N)
        dBt = dB_arr[:, t, :]                        # (B, N)
        Ct = C_arr[:, t, :]                          # (B, N)
        dCt = dC_arr[:, t, :]                        # (B, N)

        # z = delta_t * A2d  (B, D, N) — broadcast
        z = delta_t[:, :, None] * A2d[None, :, :]
        dz = ddelta_t[:, :, None] * A2d[None, :, :] + delta_t[:, :, None] * dA2d[None, :, :]

        A_bar = np.exp(z)                            # (B, D, N)
        dA_bar = A_bar * dz                          # (B, D, N)

        B_bar = delta_t[:, :, None] * Bt[:, None, :]     # (B, D, N)
        dB_bar = ddelta_t[:, :, None] * Bt[:, None, :] + delta_t[:, :, None] * dBt[:, None, :]

        # h_curr = A_bar * h_prev + B_bar * x_t   where x_t has shape (B, D)
        x_t = x_arr[:, t, :]                         # (B, D)
        dx_t = dx_arr[:, t, :]                       # (B, D)

        h_curr = A_bar * h_prev + B_bar * x_t[:, :, None]
        dh_curr = (
            dA_bar * h_prev
            + A_bar * dh_prev
            + dB_bar * x_t[:, :, None]
            + B_bar * dx_t[:, :, None]
        )

        # y[t,d] = sum_n C[t,n] * h_curr[d,n]
        y[:, t, :] = np.einsum("bn,bdn->bd", Ct, h_curr)
        dy[:, t, :] = (
            np.einsum("bn,bdn->bd", dCt, h_curr)
            + np.einsum("bn,bdn->bd", Ct, dh_curr)
        )

        h_prev = h_curr
        dh_prev = dh_curr

    # Gated path — gate tangent is treated as zero in v1 (gate is keyword-
    # only on both VJP and JVP, matching the VJP contract).  When gate is
    # provided, the output is y * gate and its tangent is dy * gate.
    if gate is not None:
        gate_arr = np.asarray(gate, dtype=np.float64)
        y = y * gate_arr
        dy = dy * gate_arr

    return y, dy


# ─────────────────────────────────────────────────────────────────────────
# Arch-7 (2026-05-22) — family-subpackage import hook (mirrors vjp.py).
# ─────────────────────────────────────────────────────────────────────────
from . import jvps  # noqa: F401, E402 — import-side-effect registration hook


__all__ = ["register_jvp", "get_jvp", "jvp"]
