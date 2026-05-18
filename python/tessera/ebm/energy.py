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
    # Apple GPU fast path — closes the integration gap (#1 of
    # docs/status/ga_ebm_milestone.md). Routes the no-noise f32 case
    # to `tessera_apple_gpu_ebm_inner_step_f32` when the runtime is up.
    # Bit-identical with the numpy path within fp32; falls back silently
    # on non-Darwin / no runtime / unsupported dtype.
    if noise_scale == 0.0 and y_arr.dtype == np.float32:
        gpu_out = _try_apple_gpu_inner_step(y_arr, grad_arr, float(eta))
        if gpu_out is not None:
            return gpu_out
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


def _try_apple_gpu_inner_step(
    y: np.ndarray, grad: np.ndarray, eta: float,
) -> Optional[np.ndarray]:
    """Try the Apple GPU dispatch path. Returns ``None`` when the
    runtime isn't available — callers fall back to the numpy path.

    Routed through ``tessera._apple_gpu_dispatch.bind_symbol`` so the
    runtime dylib is compiled once per process and the ctypes binding
    is cached. Only the no-noise, f32, contiguous case takes this path.
    """
    try:
        import ctypes
        from tessera._apple_gpu_dispatch import bind_symbol
    except ImportError:
        return None
    if not (y.flags["C_CONTIGUOUS"] and grad.flags["C_CONTIGUOUS"]):
        return None
    fn = bind_symbol(
        "tessera_apple_gpu_ebm_inner_step_f32",
        (ctypes.POINTER(ctypes.c_float),
         ctypes.POINTER(ctypes.c_float),
         ctypes.c_float,
         ctypes.POINTER(ctypes.c_float),
         ctypes.c_int32),
    )
    if fn is None:
        return None
    out = np.zeros_like(y)
    n = int(y.size)
    p = ctypes.POINTER(ctypes.c_float)
    fn(y.ctypes.data_as(p), grad.ctypes.data_as(p), ctypes.c_float(eta),
       out.ctypes.data_as(p), ctypes.c_int32(n))
    return out


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
        # Apple GPU fast path — hard argmin via the native MSL kernel.
        # Requires f32 + contiguous (B, K) / (B, K, D) inputs.
        gpu_out = _try_apple_gpu_self_verify_hard_argmin_f32(e, c)
        if gpu_out is not None:
            return gpu_out
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
    mean: Optional[Any] = None,
) -> np.ndarray:
    """Initialize ``K`` candidate trajectories.

    Strategies:
        ``"noise"``      — draw Gaussian samples of ``shape`` from ``rng_key``.
        ``"base_model"`` — call ``base_model_fn(x)`` once and broadcast.
        ``"copy"``       — broadcast ``x`` itself ``K`` times along a new axis.

    ``mean`` (optional, ``"noise"`` strategy only) — per-element offset
    that's broadcast across the K dim and added after scaling.  When
    provided + f32 + Apple-GPU runtime is up, the
    ``ebm_decode_init_noise_apply_f32`` kernel applies the
    ``base + std*noise`` combine on-device; otherwise the same is
    computed via numpy.

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
        # Generate noise with unit variance — caller-supplied `std` is
        # applied via the affine combine below so the GPU fast path
        # (which expects raw N(0,1) noise) and the numpy path share the
        # same arithmetic structure.
        noise = normal(rng_key, shape=full_shape, dtype=dtype, std=1.0)
        if mean is None:
            return (std * noise).astype(_np_dtype(dtype), copy=False)
        # Apple GPU fast path — `out[i] = base[i % base_len] + std * noise[i]`.
        # Activates for f32 inputs when the runtime is reachable.
        mean_arr = _as_array(mean).astype(_np_dtype(dtype), copy=False)
        if mean_arr.shape != full_shape:
            mean_arr = np.broadcast_to(mean_arr, full_shape).astype(
                _np_dtype(dtype), copy=False)
        gpu_out = _try_apple_gpu_decode_init_noise_apply_f32(
            mean_arr, noise, float(std))
        if gpu_out is not None:
            return gpu_out.reshape(full_shape)
        return (mean_arr + std * noise).astype(_np_dtype(dtype), copy=False)
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


# ---------------------------------------------------------------------------
# EBT-style multi-step refinement (Apple GPU fast path)
# ---------------------------------------------------------------------------

def refinement(
    y0: Any, grad: Any, *, eta: float, T: int,
) -> np.ndarray:
    """Run ``T`` inner-step iterations with a fixed gradient snapshot.

    Closed form: ``y_T = y0 - T * eta * grad``.  Equivalent to calling
    ``inner_step`` ``T`` times in a loop, but the Apple GPU fast path
    runs the whole T-step recurrence inside a single MSL kernel
    (each thread keeps ``y_i`` in a register and loops T times),
    eliminating the T-fold dispatch overhead.

    Falls back to the closed-form numpy expression when the runtime
    isn't available or the inputs aren't f32 / contiguous.
    """
    if T < 0:
        raise ValueError(f"refinement requires T >= 0; got T={T}.")
    y_arr = _as_array(y0)
    grad_arr = _as_array(grad)
    if y_arr.shape != grad_arr.shape:
        raise ValueError(
            f"refinement requires matching shapes; got y0={y_arr.shape}, "
            f"grad={grad_arr.shape}."
        )
    if T == 0:
        return y_arr.copy()
    if y_arr.dtype == np.float32:
        gpu_out = _try_apple_gpu_refinement_fused_f32(
            y_arr, grad_arr, float(eta), int(T))
        if gpu_out is not None:
            return gpu_out
    return (y_arr - float(T) * float(eta) * grad_arr).astype(
        y_arr.dtype, copy=False)


# ---------------------------------------------------------------------------
# Energy specialization: quadratic (Apple GPU fast path)
# ---------------------------------------------------------------------------

def energy_quadratic(x: Any, y: Any) -> np.ndarray:
    """Specialized quadratic energy ``E_b = 0.5 * ||x_b - y_b||^2``.

    The dominant energy form in EBT / diffusion training (reconstruction
    loss, Gaussian log-likelihood up to a constant).  Callers whose
    ``model_fn(x, y)`` is documented to match this shape can swap in
    ``energy_quadratic`` to opt into the
    ``tessera_apple_gpu_ebm_energy_quadratic_f32`` kernel.

    Falls back to numpy when the runtime isn't available or the inputs
    aren't f32 / contiguous / rank-2 with matching shape.
    """
    x_arr = _as_array(x)
    y_arr = _as_array(y)
    if x_arr.shape != y_arr.shape:
        raise ValueError(
            f"energy_quadratic requires matching shapes; "
            f"got x={x_arr.shape}, y={y_arr.shape}."
        )
    gpu_out = _try_apple_gpu_energy_quadratic_f32(x_arr, y_arr)
    if gpu_out is not None:
        return gpu_out
    if x_arr.ndim == 0:
        return np.asarray(0.5 * (x_arr - y_arr) ** 2)
    return 0.5 * np.sum((x_arr - y_arr) ** 2, axis=tuple(range(1, x_arr.ndim)))


# ---------------------------------------------------------------------------
# Apple GPU dispatch helpers — share the pattern with ga.ops fast paths.
# Each returns the GPU output or ``None``; callers fall back to numpy.
# ---------------------------------------------------------------------------

def _try_apple_gpu_refinement_fused_f32(
    y0: np.ndarray, grad: np.ndarray, eta: float, T: int,
) -> Optional[np.ndarray]:
    """Run T inner-step iterations on Apple GPU in a single dispatch."""
    if y0.dtype != np.float32 or grad.dtype != np.float32:
        return None
    if y0.shape != grad.shape:
        return None
    if not (y0.flags["C_CONTIGUOUS"] and grad.flags["C_CONTIGUOUS"]):
        return None
    try:
        import ctypes
        from tessera._apple_gpu_dispatch import bind_symbol
    except ImportError:
        return None
    fn = bind_symbol(
        "tessera_apple_gpu_ebm_refinement_f32",
        (ctypes.POINTER(ctypes.c_float),
         ctypes.POINTER(ctypes.c_float),
         ctypes.c_float, ctypes.c_int32,
         ctypes.POINTER(ctypes.c_float),
         ctypes.c_int32),
    )
    if fn is None:
        return None
    out = np.zeros_like(y0)
    n = int(y0.size)
    p = ctypes.POINTER(ctypes.c_float)
    fn(y0.ctypes.data_as(p), grad.ctypes.data_as(p),
       ctypes.c_float(eta), ctypes.c_int32(T),
       out.ctypes.data_as(p), ctypes.c_int32(n))
    return out


def _try_apple_gpu_self_verify_hard_argmin_f32(
    energies: np.ndarray, candidates: np.ndarray,
) -> Optional[np.ndarray]:
    """Hard-argmin self_verify on Apple GPU.  Requires
    ``energies.shape == (B, K)`` and ``candidates.shape == (B, K, D)``,
    both f32 + C-contiguous."""
    if energies.dtype != np.float32 or candidates.dtype != np.float32:
        return None
    if energies.ndim != 2 or candidates.ndim != 3:
        return None
    if candidates.shape[:2] != energies.shape:
        return None
    if not (energies.flags["C_CONTIGUOUS"] and candidates.flags["C_CONTIGUOUS"]):
        return None
    try:
        import ctypes
        from tessera._apple_gpu_dispatch import bind_symbol
    except ImportError:
        return None
    fn = bind_symbol(
        "tessera_apple_gpu_ebm_self_verify_hard_argmin_f32",
        (ctypes.POINTER(ctypes.c_float),
         ctypes.POINTER(ctypes.c_float),
         ctypes.POINTER(ctypes.c_float),
         ctypes.c_int32, ctypes.c_int32, ctypes.c_int32),
    )
    if fn is None:
        return None
    B, K, D = candidates.shape
    out = np.zeros((B, D), dtype=np.float32)
    p = ctypes.POINTER(ctypes.c_float)
    fn(energies.ctypes.data_as(p), candidates.ctypes.data_as(p),
       out.ctypes.data_as(p),
       ctypes.c_int32(B), ctypes.c_int32(K), ctypes.c_int32(D))
    return out


def _try_apple_gpu_energy_quadratic_f32(
    x: np.ndarray, y: np.ndarray,
) -> Optional[np.ndarray]:
    """``E_b = 0.5 * ||x_b - y_b||^2`` on Apple GPU.  Requires
    rank-2 f32 inputs with matching shape (B, D), contiguous."""
    if x.dtype != np.float32 or y.dtype != np.float32:
        return None
    if x.ndim != 2 or x.shape != y.shape:
        return None
    if not (x.flags["C_CONTIGUOUS"] and y.flags["C_CONTIGUOUS"]):
        return None
    try:
        import ctypes
        from tessera._apple_gpu_dispatch import bind_symbol
    except ImportError:
        return None
    fn = bind_symbol(
        "tessera_apple_gpu_ebm_energy_quadratic_f32",
        (ctypes.POINTER(ctypes.c_float),
         ctypes.POINTER(ctypes.c_float),
         ctypes.POINTER(ctypes.c_float),
         ctypes.c_int32, ctypes.c_int32),
    )
    if fn is None:
        return None
    B, D = x.shape
    out = np.zeros(B, dtype=np.float32)
    p = ctypes.POINTER(ctypes.c_float)
    fn(x.ctypes.data_as(p), y.ctypes.data_as(p), out.ctypes.data_as(p),
       ctypes.c_int32(B), ctypes.c_int32(D))
    return out


def _try_apple_gpu_decode_init_noise_apply_f32(
    base: np.ndarray, noise: np.ndarray, std: float,
) -> Optional[np.ndarray]:
    """``out = base + std * noise`` on Apple GPU.  Both buffers must
    be the same f32 shape; the result has the same shape too."""
    if base.dtype != np.float32 or noise.dtype != np.float32:
        return None
    if base.shape != noise.shape:
        return None
    base_c = np.ascontiguousarray(base)
    noise_c = np.ascontiguousarray(noise)
    try:
        import ctypes
        from tessera._apple_gpu_dispatch import bind_symbol
    except ImportError:
        return None
    fn = bind_symbol(
        "tessera_apple_gpu_ebm_decode_init_noise_apply_f32",
        (ctypes.POINTER(ctypes.c_float),
         ctypes.c_int32,
         ctypes.POINTER(ctypes.c_float),
         ctypes.c_float,
         ctypes.POINTER(ctypes.c_float),
         ctypes.c_int32),
    )
    if fn is None:
        return None
    n = int(noise_c.size)
    base_flat = base_c.reshape(-1)
    out = np.zeros(n, dtype=np.float32)
    p = ctypes.POINTER(ctypes.c_float)
    fn(base_flat.ctypes.data_as(p), ctypes.c_int32(int(base_flat.size)),
       noise_c.ctypes.data_as(p), ctypes.c_float(float(std)),
       out.ctypes.data_as(p), ctypes.c_int32(n))
    return out.reshape(noise.shape)


__all__ = [
    "decode_init",
    "energy",
    "energy_quadratic",
    "inner_step",
    "langevin_step",
    "refinement",
    "self_verify",
]
