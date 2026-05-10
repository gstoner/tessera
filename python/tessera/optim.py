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


_DTYPE_ALIASES = {
    "f64": "fp64",
    "float64": "fp64",
    "f32": "fp32",
    "float32": "fp32",
    "f16": "fp16",
    "float16": "fp16",
    "bfloat16": "bf16",
}

_NUMPY_DTYPES = {
    "fp64": np.float64,
    "fp32": np.float32,
    "tf32": np.float32,
    "bf16": np.float32,  # numpy reference stores bf16 as fp32.
    "fp16": np.float16,
    "fp8_e4m3": np.float32,
    "fp8_e5m2": np.float32,
    "fp4_e2m1": np.float32,
    "nvfp4": np.float32,
}


def _asarray(x: Any) -> np.ndarray:
    if hasattr(x, "_data"):
        x = x._data
    if hasattr(x, "_data"):
        x = x._data
    return np.asarray(x)


def _normalize_dtype(dtype: str | None, *, default: str = "fp32") -> str:
    if dtype is None:
        return default
    normalized = _DTYPE_ALIASES.get(str(dtype), str(dtype))
    if normalized not in _NUMPY_DTYPES:
        raise ValueError(f"Unsupported optimizer dtype {dtype!r}")
    return normalized


def _np_dtype(dtype: str | None, *, default: str = "fp32"):
    return _NUMPY_DTYPES[_normalize_dtype(dtype, default=default)]


def _compute_array(x: Any, compute_dtype: str | None) -> np.ndarray:
    arr = _asarray(x)
    # Treat the default as "at least fp32" for reference math: fp16/bf16-style
    # storage promotes to fp32, while fp64 test/reference inputs keep fp64.
    if _normalize_dtype(compute_dtype) == "fp32" and arr.dtype == np.float64:
        return arr.astype(np.float64, copy=False)
    return arr.astype(_np_dtype(compute_dtype), copy=False)


def _state_array(x: Any, state_dtype: str | None) -> np.ndarray:
    return _asarray(x).astype(_np_dtype(state_dtype), copy=False)


def _cast_like_param(x: Any, param: Any, cast_updates_to_param_dtype: bool) -> np.ndarray:
    arr = np.asarray(x)
    if not cast_updates_to_param_dtype:
        return arr
    return arr.astype(_asarray(param).dtype, copy=False)


def _master_tree(params: Tree, state: dict[str, Any] | None, master_dtype: str | None) -> Tree:
    if master_dtype is None:
        return params
    if state is not None and "master_params" in state:
        return state["master_params"]
    return tree_map(lambda p: _asarray(p).astype(_np_dtype(master_dtype), copy=True), params)


def _attach_master_state(state: dict[str, Any], master_params: Tree, master_dtype: str | None) -> dict[str, Any]:
    if master_dtype is None:
        return state
    out = dict(state)
    out["master_params"] = master_params
    out["master_dtype"] = _normalize_dtype(master_dtype)
    return out


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


def zeros_like_tree(tree: Tree, dtype: str | None = "fp32") -> Tree:
    return tree_map(lambda x: np.zeros_like(_asarray(x), dtype=_np_dtype(dtype)), tree)


def tree_l2_norm(tree: Tree) -> float:
    total = 0.0

    def add(x):
        nonlocal total
        arr = _asarray(x).astype(np.float64, copy=False)
        total += float(np.sum(arr * arr))
        return x

    tree_map(add, tree)
    return math.sqrt(total)


def sgd(
    params: Tree,
    grads: Tree,
    lr: float,
    *,
    compute_dtype: str = "fp32",
    state_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> Tree | tuple[Tree, dict[str, Any]]:
    """Plain SGD update."""
    del state_dtype  # SGD has no optimizer slots but accepts the common dtype policy.
    base_params = _master_tree(params, None, master_dtype)
    new_master = tree_map2(
        lambda p, g: _compute_array(p, compute_dtype) - float(lr) * _compute_array(g, compute_dtype),
        base_params,
        grads,
    )
    new_params = tree_map2(lambda p_new, p_orig: _cast_like_param(p_new, p_orig, cast_updates_to_param_dtype), new_master, params)
    if master_dtype is None:
        return new_params
    return new_params, {"master_params": new_master, "master_dtype": _normalize_dtype(master_dtype)}


def momentum(
    params: Tree,
    grads: Tree,
    state: dict[str, Tree] | None = None,
    *,
    lr: float,
    momentum: float = 0.9,
    compute_dtype: str = "fp32",
    state_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> tuple[Tree, dict[str, Tree]]:
    """SGD with classical momentum."""
    base_params = _master_tree(params, state, master_dtype)
    velocity = zeros_like_tree(params, state_dtype) if state is None else state["velocity"]
    new_velocity = tree_map2(
        lambda v, g: _state_array(float(momentum) * _compute_array(v, compute_dtype) + _compute_array(g, compute_dtype), state_dtype),
        velocity,
        grads,
    )
    new_master = tree_map2(lambda p, v: _compute_array(p, compute_dtype) - float(lr) * _compute_array(v, compute_dtype), base_params, new_velocity)
    new_params = tree_map2(lambda p_new, p_orig: _cast_like_param(p_new, p_orig, cast_updates_to_param_dtype), new_master, params)
    return new_params, _attach_master_state({"velocity": new_velocity}, new_master, master_dtype)


def nesterov(
    params: Tree,
    grads: Tree,
    state: dict[str, Tree] | None = None,
    *,
    lr: float,
    momentum: float = 0.9,
    compute_dtype: str = "fp32",
    state_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> tuple[Tree, dict[str, Tree]]:
    """Nesterov momentum update."""
    base_params = _master_tree(params, state, master_dtype)
    velocity = zeros_like_tree(params, state_dtype) if state is None else state["velocity"]
    new_velocity = tree_map2(
        lambda v, g: _state_array(float(momentum) * _compute_array(v, compute_dtype) + _compute_array(g, compute_dtype), state_dtype),
        velocity,
        grads,
    )
    update = tree_map2(
        lambda g, v: _compute_array(g, compute_dtype) + float(momentum) * _compute_array(v, compute_dtype),
        grads,
        new_velocity,
    )
    new_master = tree_map2(lambda p, u: _compute_array(p, compute_dtype) - float(lr) * _compute_array(u, compute_dtype), base_params, update)
    new_params = tree_map2(lambda p_new, p_orig: _cast_like_param(p_new, p_orig, cast_updates_to_param_dtype), new_master, params)
    return new_params, _attach_master_state({"velocity": new_velocity}, new_master, master_dtype)


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
    compute_dtype: str = "fp32",
    state_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> tuple[Tree, dict[str, Any]]:
    """AdamW with decoupled weight decay."""
    base_params = _master_tree(params, state, master_dtype)
    if state is None:
        state = {"m": zeros_like_tree(params, state_dtype), "v": zeros_like_tree(params, state_dtype), "step": 0}
    step = int(state["step"]) + 1
    m = tree_map2(
        lambda m_, g: _state_array(beta1 * _compute_array(m_, compute_dtype) + (1.0 - beta1) * _compute_array(g, compute_dtype), state_dtype),
        state["m"],
        grads,
    )
    v = tree_map2(
        lambda v_, g: _state_array(beta2 * _compute_array(v_, compute_dtype) + (1.0 - beta2) * (_compute_array(g, compute_dtype) ** 2), state_dtype),
        state["v"],
        grads,
    )
    b1_corr = 1.0 - beta1 ** step
    b2_corr = 1.0 - beta2 ** step

    def update_param(p, m_, v_):
        p_arr = _compute_array(p, compute_dtype)
        update = (_compute_array(m_, compute_dtype) / b1_corr) / (np.sqrt(_compute_array(v_, compute_dtype) / b2_corr) + eps)
        if weight_decay:
            p_arr = p_arr * (1.0 - lr * weight_decay)
        return p_arr - lr * update

    new_master = tree_map3(update_param, base_params, m, v)
    new_params = tree_map2(lambda p_new, p_orig: _cast_like_param(p_new, p_orig, cast_updates_to_param_dtype), new_master, params)
    return new_params, _attach_master_state({"m": m, "v": v, "step": step}, new_master, master_dtype)


def adam(
    params: Tree,
    grads: Tree,
    state: dict[str, Any] | None = None,
    *,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    compute_dtype: str = "fp32",
    state_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> tuple[Tree, dict[str, Any]]:
    """Adam without decoupled weight decay."""
    return adamw(
        params,
        grads,
        state,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=0.0,
        compute_dtype=compute_dtype,
        state_dtype=state_dtype,
        master_dtype=master_dtype,
        cast_updates_to_param_dtype=cast_updates_to_param_dtype,
    )


def adafactor(
    params: Tree,
    grads: Tree,
    state: dict[str, Any] | None = None,
    *,
    lr: float = 1e-3,
    beta2: float = 0.999,
    eps: float = 1e-30,
    compute_dtype: str = "fp32",
    state_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> tuple[Tree, dict[str, Any]]:
    """Adafactor reference update.

    Matrix leaves store factored row/column second moments; lower-rank leaves
    fall back to full second moments.
    """
    base_params = _master_tree(params, state, master_dtype)
    if state is None:
        state = {"v": tree_map(lambda p: _adafactor_zero_state(_asarray(p), state_dtype=state_dtype), params), "step": 0}
    new_v = _adafactor_tree_map(
        lambda s, g: _adafactor_update_state(s, _compute_array(g, compute_dtype), beta2, state_dtype=state_dtype),
        state["v"],
        grads,
    )
    updates = _adafactor_tree_map(
        lambda s, g: _adafactor_update_from_state(s, _compute_array(g, compute_dtype), eps, compute_dtype=compute_dtype),
        new_v,
        grads,
    )
    new_master = tree_map2(lambda p, u: _compute_array(p, compute_dtype) - float(lr) * _compute_array(u, compute_dtype), base_params, updates)
    new_params = tree_map2(lambda p_new, p_orig: _cast_like_param(p_new, p_orig, cast_updates_to_param_dtype), new_master, params)
    return new_params, _attach_master_state({"v": new_v, "step": int(state["step"]) + 1}, new_master, master_dtype)


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


def _adafactor_zero_state(arr: np.ndarray, *, state_dtype: str = "fp32"):
    if arr.ndim >= 2:
        return {
            "row": np.zeros(arr.shape[:-1], dtype=_np_dtype(state_dtype)),
            "col": np.zeros(arr.shape[-1], dtype=_np_dtype(state_dtype)),
            "factored": True,
        }
    return {"v": np.zeros_like(arr, dtype=_np_dtype(state_dtype)), "factored": False}


def _adafactor_update_state(state, grad: np.ndarray, beta2: float, *, state_dtype: str = "fp32"):
    grad2 = grad.astype(_np_dtype(state_dtype), copy=False) ** 2
    if state["factored"]:
        return {
            "row": _state_array(beta2 * state["row"] + (1.0 - beta2) * grad2.mean(axis=-1), state_dtype),
            "col": _state_array(beta2 * state["col"] + (1.0 - beta2) * grad2.mean(axis=tuple(range(grad.ndim - 1))), state_dtype),
            "factored": True,
        }
    return {"v": _state_array(beta2 * state["v"] + (1.0 - beta2) * grad2, state_dtype), "factored": False}


def _adafactor_update_from_state(state, grad: np.ndarray, eps: float, *, compute_dtype: str = "fp32"):
    grad = _compute_array(grad, compute_dtype)
    if state["factored"]:
        row = np.maximum(_compute_array(state["row"], compute_dtype), eps)
        col = np.maximum(_compute_array(state["col"], compute_dtype), eps)
        scale = row[..., None] * col / max(float(np.mean(row)), eps)
        return grad / (np.sqrt(scale) + eps)
    return grad / (np.sqrt(np.maximum(_compute_array(state["v"], compute_dtype), eps)) + eps)


def lion(
    params: Tree,
    grads: Tree,
    state: dict[str, Any] | None = None,
    *,
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.0,
    compute_dtype: str = "fp32",
    state_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> tuple[Tree, dict[str, Any]]:
    base_params = _master_tree(params, state, master_dtype)
    if state is None:
        state = {"m": zeros_like_tree(params, state_dtype), "step": 0}
    update = tree_map2(
        lambda m, g: beta1 * _compute_array(m, compute_dtype) + (1.0 - beta1) * _compute_array(g, compute_dtype),
        state["m"],
        grads,
    )
    new_m = tree_map2(
        lambda m, g: _state_array(beta2 * _compute_array(m, compute_dtype) + (1.0 - beta2) * _compute_array(g, compute_dtype), state_dtype),
        state["m"],
        grads,
    )

    def apply(p, u):
        p_arr = _compute_array(p, compute_dtype)
        if weight_decay:
            p_arr = p_arr * (1.0 - lr * weight_decay)
        return p_arr - lr * np.sign(_asarray(u))

    new_master = tree_map2(apply, base_params, update)
    new_params = tree_map2(lambda p_new, p_orig: _cast_like_param(p_new, p_orig, cast_updates_to_param_dtype), new_master, params)
    return new_params, _attach_master_state({"m": new_m, "step": int(state["step"]) + 1}, new_master, master_dtype)


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


def cyclical_lr(
    step: int,
    *,
    base_value: float,
    max_value: float,
    step_size: int,
    mode: str = "triangular",
) -> float:
    """Triangular cyclical learning-rate schedule."""
    cycle = math.floor(1.0 + int(step) / (2.0 * max(1, step_size)))
    x = abs(int(step) / max(1, step_size) - 2.0 * cycle + 1.0)
    amplitude = max(0.0, 1.0 - x)
    if mode == "triangular2":
        amplitude /= 2.0 ** (cycle - 1)
    elif mode != "triangular":
        raise ValueError("cyclical_lr mode must be 'triangular' or 'triangular2'")
    return float(base_value + (max_value - base_value) * amplitude)


def chained_schedule(*schedules: Callable[[int], float]) -> Callable[[int], tuple[float, ...]]:
    """Return a schedule that evaluates each child schedule at the same step."""
    return lambda step: tuple(float(schedule(step)) for schedule in schedules)


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
    "adam",
    "adamw",
    "add_decoupled_weight_decay",
    "centralize_grad",
    "chain",
    "chained_schedule",
    "clip_grad_norm",
    "clip_grad_value",
    "constant_lr",
    "cosine_lr",
    "cosine_warmup_lr",
    "cyclical_lr",
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
