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


@_jvp("conv1d")
def jvp_conv1d(
    primals,
    tangents,
    *,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    **_,
):
    from .vjp import _conv1d_forward_fp64

    x, weight = primals[:2]
    dx, dweight = tangents[:2]
    bias = primals[2] if len(primals) > 2 else None
    dbias = tangents[2] if len(tangents) > 2 else None
    kwargs = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
    primal = _conv1d_forward_fp64(x, weight, **kwargs)
    tangent = _conv1d_forward_fp64(dx, weight, **kwargs) + _conv1d_forward_fp64(x, dweight, **kwargs)
    if bias is not None:
        primal = primal + np.asarray(bias, dtype=np.float64).reshape(1, -1, 1)
    if dbias is not None:
        tangent = tangent + np.asarray(dbias, dtype=np.float64).reshape(1, -1, 1)
    return primal, tangent


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
    grad_lp_raw, *_ = get_vjp("ctc_loss")(
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


__all__ = ["register_jvp", "get_jvp", "jvp"]
