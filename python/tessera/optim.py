"""Functional optimizer, schedule, and gradient-transform references.

S10 keeps training-step semantics inside Tessera rather than relying on
PyTorch, Optax, or Flax. These functions operate over nested Python containers
of numpy arrays: dicts, lists, tuples, and leaves that coerce with
``np.asarray``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np


Tree = Any


def _asarray(x: Any) -> np.ndarray:
    if hasattr(x, "_data"):
        x = x._data
    if hasattr(x, "_data"):
        x = x._data
    return np.asarray(x)


def tree_map(fn: Callable[[Any], Any], tree: Tree) -> Tree:
    if isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    if isinstance(tree, tuple):
        return tuple(tree_map(fn, v) for v in tree)
    if isinstance(tree, list):
        return [tree_map(fn, v) for v in tree]
    return fn(tree)


def tree_map2(fn: Callable[[Any, Any], Any], a: Tree, b: Tree) -> Tree:
    if isinstance(a, dict):
        return {k: tree_map2(fn, a[k], b[k]) for k in a}
    if isinstance(a, tuple):
        return tuple(tree_map2(fn, x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, list):
        return [tree_map2(fn, x, y) for x, y in zip(a, b, strict=True)]
    return fn(a, b)


def tree_map3(fn: Callable[[Any, Any, Any], Any], a: Tree, b: Tree, c: Tree) -> Tree:
    if isinstance(a, dict):
        return {k: tree_map3(fn, a[k], b[k], c[k]) for k in a}
    if isinstance(a, tuple):
        return tuple(tree_map3(fn, x, y, z) for x, y, z in zip(a, b, c, strict=True))
    if isinstance(a, list):
        return [tree_map3(fn, x, y, z) for x, y, z in zip(a, b, c, strict=True)]
    return fn(a, b, c)


def zeros_like_tree(tree: Tree) -> Tree:
    return tree_map(lambda x: np.zeros_like(_asarray(x), dtype=np.float32), tree)


def tree_l2_norm(tree: Tree) -> float:
    total = 0.0

    def add(x):
        nonlocal total
        arr = _asarray(x).astype(np.float64, copy=False)
        total += float(np.sum(arr * arr))
        return x

    tree_map(add, tree)
    return math.sqrt(total)


def sgd(params: Tree, grads: Tree, lr: float) -> Tree:
    """Plain SGD update."""
    return tree_map2(lambda p, g: _asarray(p) - float(lr) * _asarray(g), params, grads)


def momentum(
    params: Tree,
    grads: Tree,
    state: dict[str, Tree] | None = None,
    *,
    lr: float,
    momentum: float = 0.9,
) -> tuple[Tree, dict[str, Tree]]:
    """SGD with classical momentum."""
    velocity = zeros_like_tree(params) if state is None else state["velocity"]
    new_velocity = tree_map2(
        lambda v, g: float(momentum) * _asarray(v) + _asarray(g),
        velocity,
        grads,
    )
    new_params = tree_map2(lambda p, v: _asarray(p) - float(lr) * _asarray(v), params, new_velocity)
    return new_params, {"velocity": new_velocity}


def nesterov(
    params: Tree,
    grads: Tree,
    state: dict[str, Tree] | None = None,
    *,
    lr: float,
    momentum: float = 0.9,
) -> tuple[Tree, dict[str, Tree]]:
    """Nesterov momentum update."""
    velocity = zeros_like_tree(params) if state is None else state["velocity"]
    new_velocity = tree_map2(
        lambda v, g: float(momentum) * _asarray(v) + _asarray(g),
        velocity,
        grads,
    )
    update = tree_map2(
        lambda g, v: _asarray(g) + float(momentum) * _asarray(v),
        grads,
        new_velocity,
    )
    return sgd(params, update, lr), {"velocity": new_velocity}


def adamw(
    params: Tree,
    grads: Tree,
    state: dict[str, Any] | None = None,
    *,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> tuple[Tree, dict[str, Any]]:
    """AdamW with decoupled weight decay."""
    if state is None:
        state = {"m": zeros_like_tree(params), "v": zeros_like_tree(params), "step": 0}
    step = int(state["step"]) + 1
    m = tree_map2(lambda m_, g: beta1 * _asarray(m_) + (1.0 - beta1) * _asarray(g), state["m"], grads)
    v = tree_map2(lambda v_, g: beta2 * _asarray(v_) + (1.0 - beta2) * (_asarray(g) ** 2), state["v"], grads)
    b1_corr = 1.0 - beta1 ** step
    b2_corr = 1.0 - beta2 ** step

    def update_param(p, m_, v_):
        p_arr = _asarray(p)
        update = (_asarray(m_) / b1_corr) / (np.sqrt(_asarray(v_) / b2_corr) + eps)
        if weight_decay:
            p_arr = p_arr * (1.0 - lr * weight_decay)
        return p_arr - lr * update

    return tree_map3(update_param, params, m, v), {"m": m, "v": v, "step": step}


def adafactor(
    params: Tree,
    grads: Tree,
    state: dict[str, Any] | None = None,
    *,
    lr: float = 1e-3,
    beta2: float = 0.999,
    eps: float = 1e-30,
) -> tuple[Tree, dict[str, Any]]:
    """Adafactor reference update.

    Matrix leaves store factored row/column second moments; lower-rank leaves
    fall back to full second moments.
    """
    if state is None:
        state = {"v": tree_map(lambda p: _adafactor_zero_state(_asarray(p)), params), "step": 0}
    new_v = _adafactor_tree_map(
        lambda s, g: _adafactor_update_state(s, _asarray(g), beta2),
        state["v"],
        grads,
    )
    updates = _adafactor_tree_map(
        lambda s, g: _adafactor_update_from_state(s, _asarray(g), eps),
        new_v,
        grads,
    )
    return sgd(params, updates, lr), {"v": new_v, "step": int(state["step"]) + 1}


def _is_adafactor_slot(x: Any) -> bool:
    return isinstance(x, dict) and "factored" in x


def _adafactor_tree_map(fn: Callable[[Any, Any], Any], slot_tree: Tree, grad_tree: Tree) -> Tree:
    if _is_adafactor_slot(slot_tree):
        return fn(slot_tree, grad_tree)
    if isinstance(slot_tree, dict):
        return {k: _adafactor_tree_map(fn, slot_tree[k], grad_tree[k]) for k in slot_tree}
    if isinstance(slot_tree, tuple):
        return tuple(_adafactor_tree_map(fn, s, g) for s, g in zip(slot_tree, grad_tree, strict=True))
    if isinstance(slot_tree, list):
        return [_adafactor_tree_map(fn, s, g) for s, g in zip(slot_tree, grad_tree, strict=True)]
    return fn(slot_tree, grad_tree)


def _adafactor_zero_state(arr: np.ndarray):
    if arr.ndim >= 2:
        return {
            "row": np.zeros(arr.shape[:-1], dtype=np.float32),
            "col": np.zeros(arr.shape[-1], dtype=np.float32),
            "factored": True,
        }
    return {"v": np.zeros_like(arr, dtype=np.float32), "factored": False}


def _adafactor_update_state(state, grad: np.ndarray, beta2: float):
    grad2 = grad.astype(np.float32, copy=False) ** 2
    if state["factored"]:
        return {
            "row": beta2 * state["row"] + (1.0 - beta2) * grad2.mean(axis=-1),
            "col": beta2 * state["col"] + (1.0 - beta2) * grad2.mean(axis=tuple(range(grad.ndim - 1))),
            "factored": True,
        }
    return {"v": beta2 * state["v"] + (1.0 - beta2) * grad2, "factored": False}


def _adafactor_update_from_state(state, grad: np.ndarray, eps: float):
    if state["factored"]:
        row = np.maximum(state["row"], eps)
        col = np.maximum(state["col"], eps)
        scale = row[..., None] * col / max(float(np.mean(row)), eps)
        return grad / (np.sqrt(scale) + eps)
    return grad / (np.sqrt(np.maximum(state["v"], eps)) + eps)


def lion(
    params: Tree,
    grads: Tree,
    state: dict[str, Any] | None = None,
    *,
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.0,
) -> tuple[Tree, dict[str, Any]]:
    if state is None:
        state = {"m": zeros_like_tree(params), "step": 0}
    update = tree_map2(lambda m, g: beta1 * _asarray(m) + (1.0 - beta1) * _asarray(g), state["m"], grads)
    new_m = tree_map2(lambda m, g: beta2 * _asarray(m) + (1.0 - beta2) * _asarray(g), state["m"], grads)

    def apply(p, u):
        p_arr = _asarray(p)
        if weight_decay:
            p_arr = p_arr * (1.0 - lr * weight_decay)
        return p_arr - lr * np.sign(_asarray(u))

    return tree_map2(apply, params, update), {"m": new_m, "step": int(state["step"]) + 1}


def muon(
    params: Tree,
    grads: Tree,
    state: dict[str, Any] | None = None,
    *,
    lr: float = 1e-3,
    momentum: float = 0.95,
) -> tuple[Tree, dict[str, Any]]:
    velocity = zeros_like_tree(params) if state is None else state["velocity"]
    new_velocity = tree_map2(lambda v, g: momentum * _asarray(v) + _asarray(g), velocity, grads)
    updates = tree_map(_orthogonalize_if_matrix, new_velocity)
    return sgd(params, updates, lr), {"velocity": new_velocity}


def _orthogonalize_if_matrix(x: Any) -> np.ndarray:
    arr = _asarray(x).astype(np.float32, copy=False)
    if arr.ndim < 2:
        norm = np.linalg.norm(arr)
        return arr / (norm + 1e-12)
    mat = arr.reshape(arr.shape[0], -1)
    u, _, vh = np.linalg.svd(mat, full_matrices=False)
    return (u @ vh).reshape(arr.shape)


def lamb(
    params: Tree,
    grads: Tree,
    state: dict[str, Any] | None = None,
    *,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-6,
    weight_decay: float = 0.0,
) -> tuple[Tree, dict[str, Any]]:
    next_params, adam_state = adamw(
        params,
        grads,
        state,
        lr=1.0,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=0.0,
    )
    adam_update = tree_map2(lambda p, p_next: _asarray(p) - _asarray(p_next), params, next_params)

    def apply(p, u):
        p_arr = _asarray(p)
        update = _asarray(u) + weight_decay * p_arr
        p_norm = np.linalg.norm(p_arr)
        u_norm = np.linalg.norm(update)
        trust = 1.0 if p_norm == 0.0 or u_norm == 0.0 else p_norm / u_norm
        return p_arr - lr * trust * update

    return tree_map2(apply, params, adam_update), adam_state


def constant_lr(value: float) -> Callable[[int], float]:
    return lambda step: float(value)


def cosine_lr(step: int, *, init_value: float, end_value: float = 0.0, decay_steps: int) -> float:
    t = min(max(int(step), 0), int(decay_steps)) / max(1, int(decay_steps))
    return float(end_value + 0.5 * (init_value - end_value) * (1.0 + math.cos(math.pi * t)))


def cosine_warmup_lr(step: int, *, peak_value: float, warmup_steps: int, decay_steps: int, end_value: float = 0.0) -> float:
    step = int(step)
    if step < warmup_steps:
        return float(peak_value * step / max(1, warmup_steps))
    return cosine_lr(step - warmup_steps, init_value=peak_value, end_value=end_value, decay_steps=max(1, decay_steps - warmup_steps))


def linear_warmup_lr(step: int, *, peak_value: float, warmup_steps: int) -> float:
    return float(peak_value * min(int(step), int(warmup_steps)) / max(1, int(warmup_steps)))


def polynomial_lr(step: int, *, init_value: float, end_value: float, decay_steps: int, power: float = 1.0) -> float:
    t = min(max(int(step), 0), int(decay_steps)) / max(1, int(decay_steps))
    return float((init_value - end_value) * ((1.0 - t) ** power) + end_value)


def inverse_sqrt_lr(step: int, *, init_value: float, warmup_steps: int = 1) -> float:
    step = max(1, int(step))
    return float(init_value * math.sqrt(max(1, warmup_steps)) / math.sqrt(step))


def clip_grad_norm(grads: Tree, max_norm: float, norm_type: float = 2.0) -> tuple[Tree, float]:
    if norm_type == float("inf"):
        max_abs = {"value": 0.0}
        tree_map(lambda g: _update_max_abs(g, max_abs), grads)
        total = max_abs["value"]
    else:
        total = tree_l2_norm(grads)
    scale = min(1.0, float(max_norm) / (total + 1e-12))
    return tree_map(lambda g: _asarray(g) * scale, grads), total


def _update_max_abs(g: Any, scope: dict[str, Any]):
    scope["value"] = max(float(scope["value"]), float(np.max(np.abs(_asarray(g)))))
    return g


def clip_grad_value(grads: Tree, clip_value: float) -> Tree:
    c = abs(float(clip_value))
    return tree_map(lambda g: np.clip(_asarray(g), -c, c), grads)


def centralize_grad(grads: Tree) -> Tree:
    def centralize(g):
        arr = _asarray(g)
        if arr.ndim <= 1:
            return arr
        axes = tuple(range(arr.ndim - 1))
        return arr - arr.mean(axis=axes, keepdims=True)

    return tree_map(centralize, grads)


def add_decoupled_weight_decay(grads: Tree, params: Tree, weight_decay: float) -> Tree:
    return tree_map2(lambda g, p: _asarray(g) + float(weight_decay) * _asarray(p), grads, params)


def ema_update(ema_params: Tree, params: Tree, decay: float) -> Tree:
    return tree_map2(lambda e, p: float(decay) * _asarray(e) + (1.0 - float(decay)) * _asarray(p), ema_params, params)


def polyak_avg(avg_params: Tree, params: Tree, step: int) -> Tree:
    step = int(step)
    return tree_map2(lambda a, p: (_asarray(a) * step + _asarray(p)) / (step + 1), avg_params, params)


@dataclass
class OptaxStyleChain:
    """Small transform chain for update trees.

    Each transform receives ``(updates, params)`` and returns a new update tree.
    """

    transforms: tuple[Callable[[Tree, Tree], Tree], ...]

    def __call__(self, updates: Tree, params: Tree) -> Tree:
        out = updates
        for transform in self.transforms:
            out = transform(out, params)
        return out


def chain(*transforms: Callable[[Tree, Tree], Tree]) -> OptaxStyleChain:
    return OptaxStyleChain(tuple(transforms))


__all__ = [
    "OptaxStyleChain",
    "adafactor",
    "adamw",
    "add_decoupled_weight_decay",
    "centralize_grad",
    "chain",
    "clip_grad_norm",
    "clip_grad_value",
    "constant_lr",
    "cosine_lr",
    "cosine_warmup_lr",
    "ema_update",
    "inverse_sqrt_lr",
    "lamb",
    "linear_warmup_lr",
    "lion",
    "momentum",
    "muon",
    "nesterov",
    "polyak_avg",
    "polynomial_lr",
    "sgd",
    "tree_l2_norm",
    "tree_map",
    "tree_map2",
    "zeros_like_tree",
]
