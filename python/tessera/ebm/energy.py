"""EBM1 — Energy primitive surface (Euclidean baseline).

Pure-Python, numpy-backed reference implementations of the five EBM
primitives specified in `docs/spec/EBM_SPEC.md` § 2.

All five primitives are pure functions of ``(state, RNGKey)`` per S4/S5
conventions. No mutation. Inner-loop iteration is left to
``tessera.control.scan`` at the call site; these primitives are the
per-step building blocks.

This is a CPU-reference implementation. EBM5 will add the
``tessera.ebm`` Graph IR dialect; EBM6 will add inner-loop fusion;
backend lowering follows the GA scope lock's backend-priority order.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from tessera.rng import RNGKey, normal


_DEFAULT_GRAD_EPS = 1e-3
_NoiseScalar = float | int

# Tessera canonical dtype strings → numpy dtypes. Mirrors `tessera.rng._np_dtype`;
# kept local so EBM doesn't depend on RNG internals.
_NUMPY_DTYPE_ALIASES: dict[str, np.dtype] = {
    "fp32": np.dtype(np.float32), "float32": np.dtype(np.float32), "f32": np.dtype(np.float32),
    "fp64": np.dtype(np.float64), "float64": np.dtype(np.float64), "f64": np.dtype(np.float64),
    "fp16": np.dtype(np.float16), "float16": np.dtype(np.float16), "f16": np.dtype(np.float16),
    "int32": np.dtype(np.int32), "i32": np.dtype(np.int32),
    "int64": np.dtype(np.int64), "i64": np.dtype(np.int64),
    "bool": np.dtype(np.bool_),
}


def _np_dtype(dtype: str | np.dtype) -> np.dtype:
    if isinstance(dtype, np.dtype):
        return dtype
    if dtype in _NUMPY_DTYPE_ALIASES:
        return _NUMPY_DTYPE_ALIASES[dtype]
    return np.dtype(dtype)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_array(x: Any, *, dtype: Optional[str] = None) -> np.ndarray:
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(_np_dtype(dtype), copy=False)
    return arr


def _numerical_grad(
    energy_fn: Callable[[np.ndarray], Any],
    y: np.ndarray,
    *,
    eps: float = _DEFAULT_GRAD_EPS,
) -> np.ndarray:
    """Central-difference gradient of a scalar-valued energy function.

    Used as a fallback when no analytic ``grad_fn`` is provided. EBM6
    will swap this for ``tessera.autodiff.vjp`` once the primitive
    surface ships an IR dialect.
    """
    base = y.astype(np.float64, copy=True)
    grad = np.zeros_like(base)
    it = np.nditer(base, flags=["multi_index"], op_flags=["readonly"])
    while not it.finished:
        idx = it.multi_index
        original = base[idx]
        base[idx] = original + eps
        e_plus = float(np.asarray(energy_fn(base)).sum())
        base[idx] = original - eps
        e_minus = float(np.asarray(energy_fn(base)).sum())
        base[idx] = original
        grad[idx] = (e_plus - e_minus) / (2.0 * eps)
        it.iternext()
    return grad.astype(y.dtype, copy=False)


# ---------------------------------------------------------------------------
# Primitive 1: energy
# ---------------------------------------------------------------------------

def energy(
    model_fn: Callable[..., Any],
    x: Any,
    y: Any,
    *,
    params: Any = None,
) -> np.ndarray:
    """Evaluate a user-provided energy function ``model_fn(x, y[, params])``.

    The output is whatever the user returns (scalar / per-batch /
    per-token); EBM does not normalize, softmax, or otherwise transform.
    The user owns the energy head — EBM owns everything that wraps it.
    """
    if params is None:
        out = model_fn(x, y)
    else:
        out = model_fn(x, y, params=params)
    return _as_array(out)


# ---------------------------------------------------------------------------
# Primitive 2: inner_step
# ---------------------------------------------------------------------------

def inner_step(
    y: Any,
    grad: Any,
    eta: float,
    *,
    rng_key: Optional[RNGKey] = None,
    noise_scale: _NoiseScalar = 0.0,
) -> np.ndarray:
    """Pluggable inner-loop update: ``y' = y - eta * grad [+ noise]``.

    When ``noise_scale > 0``, draws Gaussian noise from ``rng_key``.
    The key is consumed in-place at the caller's discretion; this
    primitive only reads from it. For functional key threading use
    ``langevin_step``.
    """
    y_arr = _as_array(y)
    grad_arr = _as_array(grad)
    if y_arr.shape != grad_arr.shape:
        raise ValueError(
            f"inner_step requires matching shapes; got y={y_arr.shape}, "
            f"grad={grad_arr.shape}."
        )
    out = y_arr - float(eta) * grad_arr
    if noise_scale > 0.0:
        if rng_key is None:
            raise ValueError(
                "inner_step requires rng_key when noise_scale > 0; "
                "pass an RNGKey or use noise_scale=0.0."
            )
        noise = normal(rng_key, shape=y_arr.shape, dtype=str(y_arr.dtype))
        out = out + float(noise_scale) * noise.astype(y_arr.dtype, copy=False)
    return out.astype(y_arr.dtype, copy=False)


# ---------------------------------------------------------------------------
# Primitive 3: langevin_step
# ---------------------------------------------------------------------------

def langevin_step(
    y: Any,
    energy_fn: Callable[[np.ndarray], Any],
    eta: float,
    temperature: float,
    rng_key: RNGKey,
    *,
    grad_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> tuple[np.ndarray, RNGKey]:
    """One Langevin step: ``y' = y - eta * ∂E/∂y + sqrt(2 * eta * T) * ξ``.

    ``temperature=0`` collapses to pure gradient descent. When
    ``grad_fn`` is omitted, the gradient is computed via central
    differences (slow but correct on any callable); supply ``grad_fn``
    when an analytic gradient is available.

    Returns ``(y', next_key)`` per S4 functional-RNG conventions.
    """
    if eta <= 0.0:
        raise ValueError(f"langevin_step requires eta > 0; got eta={eta}.")
    if temperature < 0.0:
        raise ValueError(
            f"langevin_step requires temperature >= 0; got temperature={temperature}."
        )
    y_arr = _as_array(y)
    if grad_fn is None:
        grad = _numerical_grad(energy_fn, y_arr)
    else:
        grad = _as_array(grad_fn(y_arr)).astype(y_arr.dtype, copy=False)
    sample_key, next_key = rng_key.split(2)
    noise_scale = math.sqrt(2.0 * float(eta) * float(temperature))
    out = inner_step(
        y_arr,
        grad,
        eta=eta,
        rng_key=sample_key if noise_scale > 0.0 else None,
        noise_scale=noise_scale,
    )
    return out, next_key


# ---------------------------------------------------------------------------
# Primitive 4: self_verify
# ---------------------------------------------------------------------------

def self_verify(
    energies: Any,
    candidates: Any,
    *,
    beta: Optional[float] = None,
) -> np.ndarray:
    """Reduce ``K`` candidates by energy.

    Shapes:
        ``energies.shape == (B, K)``,
        ``candidates.shape == (B, K, *event)``.

    ``beta is None`` ⇒ hard argmin: return ``candidates[b, argmin_k(energies[b, k])]``.
    ``beta > 0``     ⇒ soft-min: weights ``softmax(-beta * energies)`` over ``k``;
                        return the weighted sum over the candidate axis
                        (differentiable in ``candidates``).
    """
    e = _as_array(energies)
    c = _as_array(candidates)
    if e.ndim != 2:
        raise ValueError(
            f"self_verify requires energies of rank 2 (B, K); got rank {e.ndim}."
        )
    if c.shape[:2] != e.shape:
        raise ValueError(
            f"self_verify requires candidates.shape[:2] == energies.shape; "
            f"got energies={e.shape}, candidates={c.shape}."
        )
    if beta is None:
        idx = np.argmin(e, axis=1)
        b = np.arange(e.shape[0])
        return c[b, idx]
    if beta <= 0.0:
        raise ValueError(
            f"self_verify requires beta > 0 for soft-min; got beta={beta}. "
            f"Pass beta=None for hard argmin."
        )
    # Numerically stable softmax over -beta * e along axis=1.
    logits = -float(beta) * e.astype(np.float64, copy=False)
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / weights.sum(axis=1, keepdims=True)
    weights = weights.astype(c.dtype, copy=False)
    # Broadcast weights across the event dims and sum over K.
    expand = (slice(None), slice(None)) + (None,) * (c.ndim - 2)
    return (c * weights[expand]).sum(axis=1)


# ---------------------------------------------------------------------------
# Primitive 5: decode_init
# ---------------------------------------------------------------------------

def decode_init(
    x: Any,
    *,
    K: int,
    init_strategy: str = "noise",
    rng_key: Optional[RNGKey] = None,
    base_model_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    shape: Optional[Sequence[int]] = None,
    dtype: str = "fp32",
    std: float = 1.0,
) -> np.ndarray:
    """Initialize ``K`` candidate trajectories.

    Strategies:
        ``"noise"``      — draw Gaussian samples of ``shape`` from ``rng_key``.
        ``"base_model"`` — call ``base_model_fn(x)`` once and broadcast.
        ``"copy"``       — broadcast ``x`` itself ``K`` times along a new axis.

    Returns array of shape ``(B, K, *event)`` where ``B`` is the batch
    dimension of ``x`` (or 1 if ``x`` is scalar / unbatched) and
    ``event`` comes from ``shape`` (noise / base_model) or the trailing
    dims of ``x`` (copy).
    """
    if K <= 0:
        raise ValueError(f"decode_init requires K > 0; got K={K}.")
    x_arr = _as_array(x)
    batch = x_arr.shape[0] if x_arr.ndim >= 1 else 1
    if init_strategy == "noise":
        if rng_key is None:
            raise ValueError("decode_init(init_strategy='noise') requires rng_key.")
        if shape is None:
            raise ValueError(
                "decode_init(init_strategy='noise') requires explicit `shape` "
                "for the per-candidate event."
            )
        full_shape = (batch, K, *shape)
        return normal(rng_key, shape=full_shape, dtype=dtype, std=std)
    if init_strategy == "base_model":
        if base_model_fn is None:
            raise ValueError(
                "decode_init(init_strategy='base_model') requires base_model_fn."
            )
        single = _as_array(base_model_fn(x_arr)).astype(_np_dtype(dtype), copy=False)
        # Broadcast across the K axis.
        expanded = np.broadcast_to(
            single[:, None, ...] if single.ndim >= 1 else single[None, None, ...],
            (single.shape[0] if single.ndim >= 1 else 1, K, *single.shape[1:]),
        )
        return np.ascontiguousarray(expanded)
    if init_strategy == "copy":
        # Broadcast x itself across a new K axis.
        if x_arr.ndim == 0:
            base = x_arr.reshape(1)
            expanded = np.broadcast_to(base[None, None], (1, K, 1))
        else:
            expanded = np.broadcast_to(x_arr[:, None, ...], (batch, K, *x_arr.shape[1:]))
        return np.ascontiguousarray(expanded).astype(_np_dtype(dtype), copy=False)
    raise ValueError(
        f"decode_init: unknown init_strategy={init_strategy!r}; "
        f"expected one of 'noise', 'base_model', 'copy'."
    )


__all__ = [
    "decode_init",
    "energy",
    "inner_step",
    "langevin_step",
    "self_verify",
]
