"""EBM3 — Partition function estimators.

Three methods for computing ``Z = ∫ exp(-E(y)) dy`` (or its discrete
analogue):

    method="exact"        Brute-force sum over a finite support.
    method="monte_carlo"  Importance-sampled estimate with a user-provided
                          proposal distribution.
    method="annealed"     Annealed Importance Sampling (Neal 2001) — ratio
                          Z_target / Z_ref via a tempering schedule.

All Euclidean. The manifold-aware partition function (Z over a
non-trivial measure) waits for EBM7.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Optional, Tuple

import numpy as np

from tessera.rng import RNGKey, _hmc_leapfrog, normal


def _temperature_schedule(n_steps: int, schedule: str = "linear") -> np.ndarray:
    """Return an array of β values from 0 to 1 (inclusive) of length ``n_steps``."""
    if n_steps < 2:
        raise ValueError(f"AIS requires n_steps >= 2; got {n_steps}.")
    if schedule == "linear":
        return np.linspace(0.0, 1.0, n_steps)
    if schedule == "sigmoid":
        # Sigmoid spacing concentrates more samples near the boundaries.
        t = np.linspace(-6.0, 6.0, n_steps)
        s = 1.0 / (1.0 + np.exp(-t))
        s = (s - s[0]) / (s[-1] - s[0])
        return s
    raise ValueError(f"Unknown schedule {schedule!r}; expected 'linear' or 'sigmoid'.")


# ---------------------------------------------------------------------------
# Exact (brute-force) partition sum
# ---------------------------------------------------------------------------

def partition_function_exact(
    energy_fn: Callable[[Any], float],
    states: Iterable[Any],
) -> float:
    """Exact ``Z = Σ_s exp(-E(s))`` over a finite iterable of states.

    For continuous distributions ``states`` should be a quadrature
    discretization; the caller is responsible for including the
    integration weights in ``energy_fn`` (i.e., return
    ``E(y) - log(w(y))``).
    """
    log_terms = []
    for s in states:
        log_terms.append(-float(energy_fn(s)))
    log_terms_arr = np.array(log_terms, dtype=np.float64)
    # Use logsumexp for numerical stability.
    max_lt = float(log_terms_arr.max())
    log_z = max_lt + math.log(float(np.exp(log_terms_arr - max_lt).sum()))
    return math.exp(log_z)


def partition_exact_from_energies(
    energies: Any, *, temperature: float = 1.0,
) -> float:
    """Compute ``Z = Σ_i exp(-energies[i] / T)`` from precomputed
    per-state energies.

    Closes the 8/9 → 9/9 native EBM gap.  When ``energies`` is a
    contiguous f32 numpy array and the Apple GPU runtime is
    available, routes through ``tessera_apple_gpu_ebm_partition_exact_f32``
    via the JIT bridge.  Otherwise falls back to a stable
    log-sum-exp on the host.

    Parameters
    ----------
    energies : array-like
        Per-state energies ``E_i``, any shape (treated as flat).
    temperature : float, default 1.0
        Temperature ``T > 0``; smaller T sharpens around the mode.

    Returns
    -------
    Z : float
        Partition value ``Σ_i exp(-E_i / T)``.
    """
    if temperature <= 0.0:
        raise ValueError(
            f"partition_exact_from_energies requires temperature > 0; "
            f"got temperature={temperature}.")
    E = np.ascontiguousarray(np.asarray(energies)).reshape(-1)
    if E.size == 0:
        return 0.0
    # Apple GPU fast path — f32 inputs only.
    if E.dtype == np.float32:
        z = _try_apple_gpu_partition_exact_f32(E, float(temperature))
        if z is not None:
            return float(z)
    # Stable host fallback.
    inv_t = 1.0 / float(temperature)
    neg = -E.astype(np.float64, copy=False) * inv_t
    max_neg = float(neg.max())
    return float(math.exp(max_neg + math.log(float(np.exp(neg - max_neg).sum()))))


def _try_apple_gpu_partition_exact_f32(
    energies: np.ndarray, temperature: float,
) -> Optional[float]:
    """Apple GPU fast path for stable logsumexp partition.  Routes
    through the JIT bridge so the route trace records the manifest-
    resolved symbol."""
    try:
        import ctypes
        from tessera.compiler import jit_bridge as _bridge
    except ImportError:
        return None
    if energies.dtype != np.float32 or not energies.flags["C_CONTIGUOUS"]:
        return None
    out = np.zeros(1, dtype=np.float32)
    p = ctypes.POINTER(ctypes.c_float)
    n = int(energies.size)
    argtypes = (ctypes.POINTER(ctypes.c_float),
                ctypes.c_int32,
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_float))
    args = (energies.ctypes.data_as(p), ctypes.c_int32(n),
            ctypes.c_float(temperature), out.ctypes.data_as(p))
    try:
        ok = _bridge.dispatch_via_manifest(
            "ebm_partition_exact", argtypes=argtypes, args=args,
            args_summary=_bridge.shaped_summary(energies),
        )
    except _bridge.JitBridgeMiss:
        return None
    if not ok:
        return None
    return float(out[0])


# ---------------------------------------------------------------------------
# Monte Carlo (importance-sampled) partition estimate
# ---------------------------------------------------------------------------

def partition_function_monte_carlo(
    energy_fn: Callable[[np.ndarray], float],
    *,
    key: RNGKey,
    proposal_sampler: Callable[[RNGKey], Tuple[np.ndarray, RNGKey]],
    proposal_log_density: Callable[[np.ndarray], float],
    n_samples: int = 1024,
) -> Tuple[float, dict]:
    """Importance-sampling estimate of ``Z``.

    ``Z ≈ (1/N) Σ_i exp(-E(y_i)) / q(y_i)`` where ``y_i ~ q``.

    Returns ``(Z_estimate, diagnostics)`` where diagnostics include
    ``ess`` (effective sample size) and ``log_var`` (log-variance of
    the importance weights — high values flag a bad proposal).
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive; got {n_samples}.")
    log_weights = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        y, key = proposal_sampler(key)
        log_weights[i] = -float(energy_fn(y)) - float(proposal_log_density(y))
    max_lw = float(log_weights.max())
    Z_estimate = math.exp(max_lw) * float(np.exp(log_weights - max_lw).mean())
    # Effective sample size.
    norm_weights = np.exp(log_weights - max_lw)
    ess = float(norm_weights.sum() ** 2 / (norm_weights ** 2).sum())
    return Z_estimate, {
        "ess": ess,
        "log_var": float(np.var(log_weights)),
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Annealed Importance Sampling (Neal 2001)
# ---------------------------------------------------------------------------

def partition_function_ais(
    energy_fn: Callable[[np.ndarray], float],
    *,
    key: RNGKey,
    ref_sampler: Callable[[RNGKey], Tuple[np.ndarray, RNGKey]],
    ref_log_density: Callable[[np.ndarray], float],
    grad_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ref_grad_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    Z_ref: float = 1.0,
    n_chains: int = 1024,
    n_steps: int = 32,
    schedule: str = "linear",
    mcmc_step_size: float = 0.1,
    mcmc_n_leapfrog: int = 5,
) -> Tuple[float, dict]:
    """Annealed Importance Sampling estimate of ``Z_target``.

    Constructs a sequence of bridging distributions
    ``p_β(y) ∝ p_ref(y)^{1−β} · p_target(y)^β`` for ``β`` ranging from
    0 (reference) to 1 (target). The ratio ``Z_target / Z_ref`` is
    estimated as the expectation of the product of step-importance
    weights ``exp((β_t − β_{t−1}) · (log p_ref(y) − log p_target(y)))``,
    with optional HMC transitions at each β to reduce variance.

    Args:
        energy_fn: target log-density is ``-energy_fn``.
        ref_sampler: draws ``y ~ p_ref`` (returns ``(y, next_key)``).
        ref_log_density: ``log p_ref(y)``.
        grad_fn: target gradient ``∇E(y)`` (optional; needed for HMC).
        ref_grad_fn: reference negative log-density gradient (optional).
        Z_ref: partition function of the bridge distribution at β=0.
            If ``ref_log_density`` already includes the normalizing
            constant (i.e., is the log of a normalized density), pass
            ``Z_ref=1.0`` (default) — the estimator then returns
            ``Z_target`` directly. If ``ref_log_density`` returns an
            unnormalized log-density and you know its partition function,
            pass it here so the returned ``Z_target`` is the true target
            partition function.
        n_chains: number of independent AIS chains.
        n_steps: number of temperature steps from β=0 to β=1.
        schedule: 'linear' or 'sigmoid'.
        mcmc_step_size: HMC step size used at each intermediate β.
        mcmc_n_leapfrog: leapfrog steps per HMC transition.

    Returns ``(Z_estimate, diagnostics)``.
    """
    if n_chains <= 0:
        raise ValueError(f"n_chains must be positive; got {n_chains}.")
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2; got {n_steps}.")
    betas = _temperature_schedule(n_steps, schedule=schedule)
    log_weights = np.zeros(n_chains, dtype=np.float64)

    def bridge_energy(y: np.ndarray, beta: float) -> float:
        if beta <= 0.0:
            return -float(ref_log_density(y))
        if beta >= 1.0:
            return float(energy_fn(y))
        return (1.0 - beta) * (-float(ref_log_density(y))) + beta * float(energy_fn(y))

    def bridge_grad(y: np.ndarray, beta: float) -> np.ndarray:
        if grad_fn is None and ref_grad_fn is None:
            return np.zeros_like(y)  # no MCMC moves possible
        target_g = np.asarray(grad_fn(y)) if grad_fn is not None else np.zeros_like(y)
        ref_g = np.asarray(ref_grad_fn(y)) if ref_grad_fn is not None else np.zeros_like(y)
        # ref_grad_fn is gradient of -log p_ref (i.e., the reference energy).
        # So bridge energy gradient = (1-β)·ref_g + β·target_g.
        return (1.0 - beta) * ref_g + beta * target_g

    do_mcmc = grad_fn is not None and ref_grad_fn is not None

    for c in range(n_chains):
        y, key = ref_sampler(key)
        y = np.asarray(y, dtype=np.float64)
        for t in range(1, n_steps):
            beta_prev = float(betas[t - 1])
            beta_curr = float(betas[t])
            # Importance weight increment: log p_{β_curr}(y) - log p_{β_prev}(y)
            # = -(bridge_energy(y, β_curr) - bridge_energy(y, β_prev)).
            log_weights[c] += bridge_energy(y, beta_prev) - bridge_energy(y, beta_curr)
            # MCMC step at the new β (leaves p_{β_curr} invariant).
            if do_mcmc:
                p_key, accept_key, key = key.split(3)
                p = normal(p_key, shape=y.shape, dtype="fp64").astype(np.float64, copy=False)
                H0 = bridge_energy(y, beta_curr) + 0.5 * float(np.sum(p * p))
                q_new, p_new = _hmc_leapfrog(
                    y, p,
                    lambda yy, b=beta_curr: bridge_grad(yy, b),
                    mcmc_step_size, mcmc_n_leapfrog,
                    np.ones_like(y),
                )
                H1 = bridge_energy(q_new, beta_curr) + 0.5 * float(np.sum(p_new * p_new))
                u = float(normal(accept_key, shape=(), dtype="fp64") * 0.0 + 0.5)  # placeholder
                # Use uniform for accept — split a fresh key.
                from tessera.rng import uniform as _uniform
                ukey, key = key.split(2)
                u = float(_uniform(ukey, shape=(), dtype="fp64"))
                if math.log(max(u, 1e-30)) < H0 - H1:
                    y = q_new

    max_lw = float(log_weights.max())
    z_ratio = math.exp(max_lw) * float(np.exp(log_weights - max_lw).mean())
    Z_target = float(Z_ref) * z_ratio
    log_var = float(np.var(log_weights))
    norm_weights = np.exp(log_weights - max_lw)
    ess = float(norm_weights.sum() ** 2 / (norm_weights ** 2).sum())
    return Z_target, {
        "ess": ess,
        "log_var": log_var,
        "n_chains": n_chains,
        "n_steps": n_steps,
        "Z_ref": float(Z_ref),
        "Z_ratio": z_ratio,
    }


# ---------------------------------------------------------------------------
# Dispatch wrapper
# ---------------------------------------------------------------------------

def partition_function(
    energy_fn: Callable[..., float],
    *,
    method: str = "exact",
    **kwargs: Any,
):
    """Dispatch to ``partition_function_{exact, monte_carlo, ais}``.

    See the individual function docstrings for method-specific kwargs.
    """
    if method == "exact":
        if "states" not in kwargs:
            raise TypeError("partition_function(method='exact') requires `states`.")
        return partition_function_exact(energy_fn, kwargs.pop("states"))
    if method == "monte_carlo":
        return partition_function_monte_carlo(energy_fn, **kwargs)
    if method in ("annealed", "ais"):
        return partition_function_ais(energy_fn, **kwargs)
    raise ValueError(
        f"Unknown method {method!r}; expected 'exact', 'monte_carlo', or 'annealed'."
    )


__all__ = [
    "partition_function",
    "partition_function_ais",
    "partition_function_exact",
    "partition_function_monte_carlo",
]
