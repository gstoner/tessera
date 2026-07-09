"""EBM7 — Manifold-aware Langevin integrators.

This is the merge point of the GA and EBM tracks. The Euclidean
Langevin step from EBM1 / EBM2 is extended to two non-flat manifolds
per the Q5 scope lock:

    bivector_langevin_step      State in Cl(p, 0) restricted to grade-2;
                                gradient and noise projected to the
                                bivector subspace each step. Used for
                                SO(n) sampling via its Lie algebra.

    sphere_langevin_step        State on S^{d-1} in ambient ℝ^d.
                                Riemannian Langevin via tangent-plane
                                projection + normalization retraction.

Both step primitives have chain wrappers (`*_sample`) that reuse the
existing `_collect_chain` harness from `tessera.rng`. Euclidean
Langevin is still served by `tessera.rng.langevin_sample` (no manifold
machinery needed).

See `docs/audit/domain/DOMAIN_AUDIT.md` § Q5 for the locked manifold set
and `docs/audit/domain/DOMAIN_AUDIT.md` § EBM7 for context.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

import numpy as np

from tessera.ga.multivector import Multivector
from tessera.ga.ops import grade_projection
from tessera.ga.signature import TesseraAlgebraError
from tessera.rng import RNGKey, _collect_chain, normal


_DEFAULT_GRAD_EPS = 1e-3


# ---------------------------------------------------------------------------
# Multivector gradient helper
# ---------------------------------------------------------------------------

def _numerical_grad_mv(
    energy_fn: Callable[[Multivector], Any],
    state: Multivector,
    *,
    eps: float = _DEFAULT_GRAD_EPS,
) -> np.ndarray:
    """Central-difference gradient of ``energy_fn(state)`` wrt every
    coefficient of ``state``."""
    algebra = state.algebra
    base_coeffs = state.coefficients.astype(np.float64, copy=True)
    grad_coeffs = np.zeros_like(base_coeffs)
    flat = base_coeffs.ravel()
    grad_flat = grad_coeffs.ravel()
    for idx in range(flat.size):
        original = flat[idx]
        flat[idx] = original + eps
        E_plus = float(energy_fn(Multivector(base_coeffs.copy(), algebra)))
        flat[idx] = original - eps
        E_minus = float(energy_fn(Multivector(base_coeffs.copy(), algebra)))
        flat[idx] = original
        grad_flat[idx] = (E_plus - E_minus) / (2.0 * eps)
    return grad_coeffs


# ---------------------------------------------------------------------------
# Bivector Langevin — state lives in the grade-2 subspace
# ---------------------------------------------------------------------------

def bivector_langevin_step(
    state: Multivector,
    energy_fn: Callable[[Multivector], Any],
    eta: float,
    temperature: float,
    rng_key: RNGKey,
    *,
    grade: int = 2,
    grad_fn: Optional[Callable[[Multivector], Multivector]] = None,
) -> Tuple[Multivector, RNGKey]:
    """One Langevin step on a grade-restricted multivector subspace.

    ``state`` must already be grade-``grade`` (typically a bivector for
    SO(n) sampling). Both the gradient and the noise are projected to
    the same grade to keep the state inside the restricted subspace
    over long chains. With grade=2 in Cl(3,0), this samples from
    distributions on the Lie algebra so(3) = grade-2 multivectors.

    Returns ``(new_state, next_key)``.
    """
    if eta <= 0.0:
        raise ValueError(f"bivector_langevin_step requires eta > 0; got {eta}.")
    if temperature < 0.0:
        raise ValueError(
            f"bivector_langevin_step requires temperature >= 0; got {temperature}."
        )
    algebra = state.algebra
    if grade < 0 or grade > algebra.n:
        raise ValueError(
            f"grade {grade} is out of range for {algebra!r}; "
            f"valid grades are {algebra.grades}."
        )

    if grad_fn is None:
        raw_grad_coeffs = _numerical_grad_mv(energy_fn, state)
        grad_mv = Multivector(raw_grad_coeffs, algebra)
    else:
        grad_mv = grad_fn(state)
        if grad_mv.algebra != algebra:
            raise TesseraAlgebraError(
                f"grad_fn returned algebra {grad_mv.algebra!r}; expected {algebra!r}."
            )
    grad_proj = grade_projection(grad_mv, grade)

    sub_key, next_key = rng_key.split(2)
    noise_scale = math.sqrt(2.0 * float(eta) * float(temperature))
    if noise_scale > 0.0:
        noise_coeffs = normal(
            sub_key, shape=state.coefficients.shape, dtype=str(state.dtype)
        )
        noise_mv = grade_projection(
            Multivector(noise_coeffs.astype(state.dtype, copy=False), algebra),
            grade,
        )
    else:
        noise_mv = None

    # Apple GPU fast path — the affine combination ``state - eta*grad +
    # noise_scale*noise`` is exactly the ``ebm_langevin_step`` kernel
    # operating on the grade-projected coefficient vectors.  Both
    # ``grad_proj`` and ``noise_mv`` are already grade-restricted, so
    # the result lives in the same subspace without a final
    # grade-projection pass.
    if state.dtype == np.float32:
        from tessera.ebm.energy import _try_apple_gpu_langevin_step_f32
        noise_c = (np.ascontiguousarray(noise_mv.coefficients,
                                          dtype=np.float32)
                    if noise_mv is not None
                    else np.zeros_like(state.coefficients, dtype=np.float32))
        state_c = np.ascontiguousarray(state.coefficients, dtype=np.float32)
        grad_c = np.ascontiguousarray(grad_proj.coefficients, dtype=np.float32)
        gpu_out = _try_apple_gpu_langevin_step_f32(
            state_c, grad_c, noise_c, float(eta), float(noise_scale),
            bridge_op_name="ebm_bivector_langevin",
        )
        if gpu_out is not None:
            return Multivector(gpu_out, algebra, grades=frozenset({grade})), next_key
        # Native x86 (AVX-512) / ROCm (gfx1151) affine-Langevin lanes — the same
        # affine combination on the grade-projected coefficient vectors, with the
        # host-drawn noise as an input (matches this numpy path exactly). Each
        # returns None off its silicon → the numpy fallback below.
        from tessera.ebm.energy import (
            _try_rocm_ebm_affine_langevin_step_f32,
            _try_x86_ebm_affine_langevin_step_f32,
        )
        for _dev in (_try_x86_ebm_affine_langevin_step_f32,
                     _try_rocm_ebm_affine_langevin_step_f32):
            dev_out = _dev(state_c, grad_c, noise_c,
                           float(eta), float(noise_scale))
            if dev_out is not None:
                mv = Multivector(dev_out, algebra, grades=frozenset({grade}))
                return mv, next_key

    # State + (-eta) * grad + noise_scale * noise.
    new_state = state + (-float(eta)) * grad_proj
    if noise_mv is not None:
        new_state = new_state + noise_scale * noise_mv
    # Final projection to clean up any float-noise leakage outside the
    # grade-restricted subspace.
    new_state = grade_projection(new_state, grade)
    return new_state, next_key


def bivector_langevin_sample(
    key: RNGKey,
    *,
    init: Multivector,
    energy_fn: Callable[[Multivector], Any],
    eta: float,
    temperature: float = 1.0,
    n_samples: int = 1,
    burn_in: int = 0,
    thin: int = 1,
    grade: int = 2,
    grad_fn: Optional[Callable[[Multivector], Multivector]] = None,
) -> Tuple[np.ndarray, RNGKey, dict]:
    """Run a bivector Langevin chain.

    Returns ``(samples, next_key, diagnostics)`` where ``samples`` has
    shape ``(n_samples, algebra.dim)`` — the multivector coefficient
    vectors in canonical blade order.
    """
    if not isinstance(init, Multivector):
        raise TypeError(
            f"bivector_langevin_sample.init must be a Multivector; "
            f"got {type(init).__name__}."
        )

    algebra = init.algebra
    init_coeffs = grade_projection(init, grade).coefficients

    def step_fn(y_coeffs: np.ndarray, k: RNGKey):
        state = Multivector(y_coeffs, algebra)
        new_state, k_next = bivector_langevin_step(
            state, energy_fn, eta=eta, temperature=temperature,
            rng_key=k, grade=grade, grad_fn=grad_fn,
        )
        return new_state.coefficients, k_next, {}

    return _collect_chain(init_coeffs, step_fn, key, n_samples, burn_in, thin)


# ---------------------------------------------------------------------------
# Sphere Langevin — Riemannian step on S^{d-1}
# ---------------------------------------------------------------------------

def _project_to_tangent_plane(v: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``P_x v = v − ⟨v, x⟩ · x`` — tangent at x on the unit sphere."""
    return v - float(np.dot(v, x)) * x


def sphere_langevin_step(
    x: np.ndarray,
    energy_fn: Callable[[np.ndarray], Any],
    eta: float,
    temperature: float,
    rng_key: RNGKey,
    *,
    grad_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, RNGKey]:
    """One Langevin step on S^{d-1} (the unit sphere in ℝ^d).

    Euler-Maruyama on the manifold:

        ξ ~ N(0, I_d)
        x' = retract(x − η · P_x(∇E) + √(2 η T) · P_x ξ)

    where ``P_x`` projects to the tangent plane and ``retract``
    normalizes to unit norm. For ``η`` small enough this is a faithful
    discretization of Brownian Langevin on the sphere.

    ``x`` is asserted near-unit-norm on entry; the caller is responsible
    for initializing on the sphere. Returns ``(x', next_key)``.
    """
    if eta <= 0.0:
        raise ValueError(f"sphere_langevin_step requires eta > 0; got {eta}.")
    if temperature < 0.0:
        raise ValueError(
            f"sphere_langevin_step requires temperature >= 0; got {temperature}."
        )
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError(
            f"sphere_langevin_step requires a rank-1 input; got shape {x_arr.shape}."
        )
    norm = float(np.linalg.norm(x_arr))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(
            f"sphere_langevin_step requires |x| = 1 on entry; got |x| = {norm:.6f}."
        )

    # Apple GPU fast path — the whole step (tangent projection +
    # Euler-Maruyama + retract) is one MSL kernel.  Activates when the
    # user passes f32 + an analytic ``grad_fn`` so the gradient query
    # itself is cheap.
    input_f32 = isinstance(x, np.ndarray) and x.dtype == np.float32
    if input_f32 and grad_fn is not None:
        sub_key, next_key = rng_key.split(2)
        noise_scale = math.sqrt(2.0 * float(eta) * float(temperature))
        if noise_scale > 0.0:
            noise = normal(sub_key, shape=(x_arr.shape[0],),
                            dtype="fp32").astype(np.float32, copy=False)
        else:
            noise = np.zeros(x_arr.shape[0], dtype=np.float32)
        from tessera.ebm.energy import _try_apple_gpu_sphere_langevin_step_f32
        grad_f32 = np.asarray(grad_fn(x_arr.astype(np.float32, copy=False)),
                                dtype=np.float32)
        gpu_out = _try_apple_gpu_sphere_langevin_step_f32(
            x_arr.astype(np.float32, copy=False), grad_f32, noise,
            float(eta), float(noise_scale),
        )
        if gpu_out is not None:
            return gpu_out, next_key
        # Native x86 (AVX-512) / ROCm (gfx1151) lane. Apple fuses the whole step;
        # here the tangent projection + retract (normalize) run on the host and the
        # affine core `x - eta*grad_tan + noise_scale*noise_tan` runs on the shared
        # affine-Langevin kernel — the same decomposition as the numpy path below,
        # matching it to f32 epsilon. Each helper returns None off its silicon.
        from tessera.ebm.energy import (
            _try_rocm_ebm_affine_langevin_step_f32,
            _try_x86_ebm_affine_langevin_step_f32,
        )
        x32 = x_arr.astype(np.float32, copy=False)
        grad_tan = _project_to_tangent_plane(grad_f32, x32).astype(np.float32)
        noise_tan = _project_to_tangent_plane(noise, x32).astype(np.float32)
        for _dev in (_try_x86_ebm_affine_langevin_step_f32,
                     _try_rocm_ebm_affine_langevin_step_f32):
            yv = _dev(x32, grad_tan, noise_tan, float(eta), float(noise_scale))
            if yv is not None:
                yv = np.asarray(yv, np.float32)
                ynorm = float(np.linalg.norm(yv))
                if ynorm < 1e-12:                # degenerate — keep the state
                    return x32, next_key
                return (yv / ynorm).astype(np.float32), next_key

    if grad_fn is None:
        # Numerical gradient via central differences.
        eps = _DEFAULT_GRAD_EPS
        grad = np.zeros_like(x_arr)
        for i in range(x_arr.shape[0]):
            base = x_arr.copy()
            base[i] = x_arr[i] + eps
            E_plus = float(energy_fn(base))
            base[i] = x_arr[i] - eps
            E_minus = float(energy_fn(base))
            grad[i] = (E_plus - E_minus) / (2.0 * eps)
    else:
        grad = np.asarray(grad_fn(x_arr), dtype=np.float64)

    grad_tan = _project_to_tangent_plane(grad, x_arr)
    sub_key, next_key = rng_key.split(2)
    noise_scale = math.sqrt(2.0 * float(eta) * float(temperature))
    if noise_scale > 0.0:
        ambient_noise = normal(sub_key, shape=x_arr.shape, dtype="fp64").astype(
            np.float64, copy=False
        )
        noise_tan = _project_to_tangent_plane(ambient_noise, x_arr)
        y = x_arr - float(eta) * grad_tan + noise_scale * noise_tan
    else:
        y = x_arr - float(eta) * grad_tan
    y_norm = float(np.linalg.norm(y))
    if y_norm < 1e-12:
        # Should not happen for sensible step sizes — defensively
        # return the original state.
        return x_arr, next_key
    return y / y_norm, next_key


def sphere_langevin_sample(
    key: RNGKey,
    *,
    init: np.ndarray,
    energy_fn: Callable[[np.ndarray], Any],
    eta: float,
    temperature: float = 1.0,
    n_samples: int = 1,
    burn_in: int = 0,
    thin: int = 1,
    grad_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, RNGKey, dict]:
    """Run a sphere Langevin chain on S^{d-1}.

    Returns ``(samples, next_key, diagnostics)``. ``samples`` has shape
    ``(n_samples, d)``; every sample lies on the unit sphere (to
    numerical precision).
    """
    init_raw = np.asarray(init)
    # Normalize in float64 for precision, but restore a float32 start when the
    # caller gave one so the per-step native (x86/ROCm) affine-Langevin fast path
    # can fire — sphere_langevin_step's device branch is gated on x.dtype==float32,
    # so a float64 chain would silently run the numpy reference every step.
    init_arr = init_raw.astype(np.float64)
    init_norm = float(np.linalg.norm(init_arr))
    if init_norm < 1e-12:
        raise ValueError("sphere_langevin_sample: init must be a non-zero vector.")
    init_arr = init_arr / init_norm
    if init_raw.dtype == np.float32:
        init_arr = init_arr.astype(np.float32)

    def step_fn(x: np.ndarray, k: RNGKey):
        new_x, k_next = sphere_langevin_step(
            x, energy_fn, eta=eta, temperature=temperature,
            rng_key=k, grad_fn=grad_fn,
        )
        return new_x, k_next, {}

    return _collect_chain(init_arr, step_fn, key, n_samples, burn_in, thin)


# ---------------------------------------------------------------------------
# von Mises-Fisher MLE — helper for the EBM7 vMF recovery test
# ---------------------------------------------------------------------------

def vmf_kappa_mle(samples: np.ndarray, dim: int) -> float:
    """Maximum-likelihood κ estimate for samples on S^{d-1} (Mardia-Jupp).

    For samples drawn from vMF(μ, κ), the resultant mean length
    ``r̄ = ‖(1/N) Σ x_i‖`` satisfies ``r̄ ≈ I_{d/2}(κ) / I_{d/2−1}(κ)``.
    Inverting this with the standard rational approximation:

        κ ≈ r̄ · (d − r̄²) / (1 − r̄²)
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(
            f"vmf_kappa_mle requires (N, d) samples; got shape {samples.shape}."
        )
    if samples.shape[1] != dim:
        raise ValueError(
            f"vmf_kappa_mle: dim={dim} but samples have {samples.shape[1]} columns."
        )
    r_bar = float(np.linalg.norm(samples.mean(axis=0)))
    if r_bar >= 1.0 - 1e-12:
        return float("inf")
    return r_bar * (dim - r_bar * r_bar) / (1.0 - r_bar * r_bar)


__all__ = [
    "bivector_langevin_sample",
    "bivector_langevin_step",
    "sphere_langevin_sample",
    "sphere_langevin_step",
    "vmf_kappa_mle",
]
