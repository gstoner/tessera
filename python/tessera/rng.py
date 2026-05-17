"""Tessera-native RNG primitives — S-series sprint S4.

Replaces the implicit, process-global Python `numpy.random` source with a
typed, compiler-visible `RNGKey` whose state is split / fold_in / clone'd
explicitly. This is the single source of randomness for diffusion noise,
dropout masks, sampling, masking, and stochastic depth — every consumer
must take an RNGKey as input.

Design notes:
  - **Determinism.** Two `RNGKey(seed=k)` keys produce bit-identical samples
    on the same machine + numpy version. `split` is deterministic too: a
    given parent key + index pair always produces the same child.
  - **No global state.** Nothing here mutates `numpy.random.default_rng()`
    or the Python `random` module. Every sampler is a pure function of
    `(key, args)`.
  - **Counter-based.** Internally we use numpy's Philox bit generator
    (counter-based, splittable, parallel-safe). `split` advances the
    counter by hashing `(key.seed, indices)` so child streams don't
    overlap. `fold_in` mixes a 64-bit value into the seed without
    advancing the counter.
  - **Sampler surface (S4 acceptance):** `uniform`, `normal`,
    `truncated_normal`, `bernoulli`, `categorical`, `multinomial`,
    `randint`, `permutation`, `gamma`, `beta`, `dirichlet`, `poisson`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


# A 128-bit Philox key + counter. We pack the user-facing seed into the
# high 64 bits of the key and reserve the low 64 bits for `fold_in` mixing.
# The counter starts at zero; it's never incremented by the user — `split`
# spawns *new* keys instead.
_KEY_DTYPE = np.uint64


def _philox_seed(seed_high: int, seed_low: int) -> np.random.Generator:
    """Build a bit-identical Philox generator from two 64-bit halves."""
    bg = np.random.Philox(key=[seed_high & 0xFFFFFFFFFFFFFFFF,
                               seed_low & 0xFFFFFFFFFFFFFFFF])
    return np.random.Generator(bg)


def _hash_to_u64(*parts: object) -> int:
    """Deterministic, machine-independent hash of arbitrary parts to u64."""
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest(), "little")


@dataclass(frozen=True)
class RNGKey:
    """Typed, splittable RNG key.

    `seed_high`/`seed_low` together identify a 128-bit Philox key. Two
    `RNGKey` instances with equal halves are interchangeable.
    """

    seed_high: int
    seed_low: int = 0
    name: str = ""  # Optional debugging label, doesn't affect samples.
    algorithm: str = "philox"
    version: int = 1

    @staticmethod
    def from_seed(seed: int, *, name: str = "") -> "RNGKey":
        """Construct from a single user-facing integer seed."""
        # Mix the seed through a hash so adjacent seeds (0, 1, 2, ...) don't
        # produce correlated streams.
        high = _hash_to_u64("seed", int(seed))
        return RNGKey(seed_high=high, seed_low=0, name=name)

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> "RNGKey":
        """Restore a key from ``to_state`` replay metadata."""
        algorithm = str(state.get("algorithm", "philox"))
        version = int(state.get("version", 1))
        if algorithm != "philox" or version != 1:
            raise ValueError(
                f"unsupported RNGKey state: algorithm={algorithm!r} version={version}"
            )
        return RNGKey(
            seed_high=int(state["seed_high"]),
            seed_low=int(state.get("seed_low", 0)),
            name=str(state.get("name", "")),
            algorithm=algorithm,
            version=version,
        )

    def to_state(self) -> dict[str, int | str]:
        """Return deterministic replay metadata for checkpoint/state trees."""
        return {
            "algorithm": self.algorithm,
            "version": self.version,
            "seed_high": int(self.seed_high),
            "seed_low": int(self.seed_low),
            "name": self.name,
        }

    def split(self, num: int = 2) -> tuple["RNGKey", ...]:
        """Deterministically derive `num` independent child keys."""
        if num <= 0:
            raise ValueError("RNGKey.split requires num > 0")
        return tuple(
            RNGKey(
                seed_high=_hash_to_u64("split", self.seed_high, self.seed_low, i),
                seed_low=0,
                name=f"{self.name}.split[{i}]" if self.name else "",
                algorithm=self.algorithm,
                version=self.version,
            )
            for i in range(num)
        )

    def fold_in(self, data: int | str | bytes) -> "RNGKey":
        """Mix `data` into this key without advancing the counter.

        Common uses:
            key.fold_in(epoch)        -> per-epoch determinism
            key.fold_in("dropout_3")  -> per-layer determinism
            key.fold_in(rank)         -> per-shard determinism
        """
        return RNGKey(
            seed_high=self.seed_high,
            seed_low=_hash_to_u64("fold_in", self.seed_low, data),
            name=self.name,
            algorithm=self.algorithm,
            version=self.version,
        )

    def clone(self) -> "RNGKey":
        """Return an independent copy of this key with the same state.

        Useful when one routine wants to peek at the same stream a sibling
        is about to consume — neither side advances the other's counter.
        """
        return RNGKey(
            seed_high=self.seed_high,
            seed_low=self.seed_low,
            name=self.name,
            algorithm=self.algorithm,
            version=self.version,
        )

    def _generator(self) -> np.random.Generator:
        return _philox_seed(self.seed_high, self.seed_low)


# ───────────────────────────────────────────────────────────────────────────
# Samplers
# ───────────────────────────────────────────────────────────────────────────


def _shape_tuple(shape: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


def _np_dtype(dtype: str | np.dtype) -> np.dtype:
    """Map Tessera dtype strings to numpy dtypes."""
    if isinstance(dtype, np.dtype):
        return dtype
    aliases = {
        "fp32": np.float32, "float32": np.float32,
        "fp64": np.float64, "float64": np.float64,
        "fp16": np.float16, "float16": np.float16,
        "i32": np.int32, "int32": np.int32,
        "i64": np.int64, "int64": np.int64,
        "bool": np.bool_,
    }
    if dtype in aliases:
        return np.dtype(aliases[dtype])
    return np.dtype(dtype)


def uniform(
    key: RNGKey,
    shape: int | Sequence[int] = (),
    *,
    low: float = 0.0,
    high: float = 1.0,
    dtype: str | np.dtype = "fp32",
) -> np.ndarray:
    """Uniform[low, high) samples."""
    if high <= low:
        raise ValueError(f"uniform requires high > low, got low={low} high={high}")
    rng = key._generator()
    out = rng.uniform(low=low, high=high, size=_shape_tuple(shape))
    return np.asarray(out, dtype=_np_dtype(dtype))


def normal(
    key: RNGKey,
    shape: int | Sequence[int] = (),
    *,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: str | np.dtype = "fp32",
) -> np.ndarray:
    """Standard normal samples scaled by `std` and shifted by `mean`."""
    if std < 0:
        raise ValueError(f"normal requires std >= 0, got std={std}")
    rng = key._generator()
    out = rng.standard_normal(size=_shape_tuple(shape)) * std + mean
    return np.asarray(out, dtype=_np_dtype(dtype))


def truncated_normal(
    key: RNGKey,
    shape: int | Sequence[int] = (),
    *,
    lower: float = -2.0,
    upper: float = 2.0,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: str | np.dtype = "fp32",
) -> np.ndarray:
    """Truncated normal — samples outside [lower, upper] are resampled.

    Implementation: rejection-sample from `normal` until every entry lies
    in the allowed interval. For default ±2σ the rejection rate is ~5%
    so this is bounded-time in expectation.
    """
    if upper <= lower:
        raise ValueError(
            f"truncated_normal requires upper > lower, got lower={lower} upper={upper}"
        )
    rng = key._generator()
    target_shape = _shape_tuple(shape)
    out = np.empty(target_shape, dtype=np.float64)
    out_flat = out.reshape(-1)
    n = out_flat.size
    filled = 0
    while filled < n:
        chunk = rng.standard_normal(size=n - filled) * std + mean
        keep = chunk[(chunk >= lower) & (chunk <= upper)]
        take = min(keep.size, n - filled)
        out_flat[filled : filled + take] = keep[:take]
        filled += take
    return np.asarray(out.reshape(target_shape), dtype=_np_dtype(dtype))


def bernoulli(
    key: RNGKey,
    shape: int | Sequence[int] = (),
    *,
    p: float = 0.5,
    dtype: str | np.dtype = "bool",
) -> np.ndarray:
    """Bernoulli(p) samples; default returns bool, can cast to int/float."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"bernoulli requires 0 <= p <= 1, got p={p}")
    rng = key._generator()
    samples = rng.random(size=_shape_tuple(shape)) < p
    return np.asarray(samples, dtype=_np_dtype(dtype))


def categorical(
    key: RNGKey,
    logits: np.ndarray,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Sample category indices from `logits` along `axis`.

    Uses the Gumbel-max trick so the sampler is differentiable through
    `logits` if a downstream user wraps it in `custom_vjp`. (The forward
    pass is pure-numpy here.)
    """
    if logits.size == 0:
        raise ValueError("categorical requires non-empty logits")
    rng = key._generator()
    # Gumbel-max: argmax(logits + Gumbel(0,1)).
    g = -np.log(-np.log(rng.random(size=logits.shape) + 1e-20) + 1e-20)
    return np.argmax(logits + g, axis=axis)


def multinomial(
    key: RNGKey,
    n: int,
    p: np.ndarray,
) -> np.ndarray:
    """Multinomial(n trials, p) returning per-category counts."""
    if n < 0:
        raise ValueError("multinomial requires n >= 0")
    rng = key._generator()
    return rng.multinomial(n, p)


def randint(
    key: RNGKey,
    shape: int | Sequence[int] = (),
    *,
    low: int,
    high: int,
    dtype: str | np.dtype = "i32",
) -> np.ndarray:
    """Uniform integers from {low, low+1, ..., high-1}."""
    if high <= low:
        raise ValueError(f"randint requires high > low, got low={low} high={high}")
    rng = key._generator()
    out = rng.integers(low=low, high=high, size=_shape_tuple(shape))
    return np.asarray(out, dtype=_np_dtype(dtype))


def permutation(
    key: RNGKey,
    n_or_array: int | np.ndarray,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Uniformly random permutation of [0, n) or shuffled copy of `array`."""
    rng = key._generator()
    if isinstance(n_or_array, (int, np.integer)):
        return rng.permutation(int(n_or_array))
    return rng.permutation(np.asarray(n_or_array), axis=axis)


def gamma(
    key: RNGKey,
    shape: int | Sequence[int] = (),
    *,
    concentration: float,
    rate: float = 1.0,
    dtype: str | np.dtype = "fp32",
) -> np.ndarray:
    """Gamma samples with shape parameter `concentration` and `rate`."""
    if concentration <= 0:
        raise ValueError("gamma requires concentration > 0")
    if rate <= 0:
        raise ValueError("gamma requires rate > 0")
    rng = key._generator()
    out = rng.gamma(shape=concentration, scale=1.0 / rate, size=_shape_tuple(shape))
    return np.asarray(out, dtype=_np_dtype(dtype))


def beta(
    key: RNGKey,
    shape: int | Sequence[int] = (),
    *,
    alpha: float,
    beta_param: float,
    dtype: str | np.dtype = "fp32",
) -> np.ndarray:
    """Beta(alpha, beta) samples."""
    if alpha <= 0 or beta_param <= 0:
        raise ValueError("beta requires alpha > 0 and beta_param > 0")
    rng = key._generator()
    out = rng.beta(a=alpha, b=beta_param, size=_shape_tuple(shape))
    return np.asarray(out, dtype=_np_dtype(dtype))


def dirichlet(
    key: RNGKey,
    alpha: np.ndarray,
    *,
    shape: int | Sequence[int] = (),
    dtype: str | np.dtype = "fp32",
) -> np.ndarray:
    """Dirichlet(alpha) samples; result has trailing axis of len(alpha)."""
    alpha_arr = np.asarray(alpha, dtype=np.float64)
    if alpha_arr.ndim != 1 or np.any(alpha_arr <= 0):
        raise ValueError("dirichlet requires a 1-D alpha vector with all entries > 0")
    rng = key._generator()
    out = rng.dirichlet(alpha=alpha_arr, size=_shape_tuple(shape))
    return np.asarray(out, dtype=_np_dtype(dtype))


def poisson(
    key: RNGKey,
    shape: int | Sequence[int] = (),
    *,
    rate: float,
    dtype: str | np.dtype = "i32",
) -> np.ndarray:
    """Poisson(rate) integer samples."""
    if rate < 0:
        raise ValueError("poisson requires rate >= 0")
    rng = key._generator()
    out = rng.poisson(lam=rate, size=_shape_tuple(shape))
    return np.asarray(out, dtype=_np_dtype(dtype))


# ---------------------------------------------------------------------------
# EBM2 — iterative Markov-chain samplers
# ---------------------------------------------------------------------------
#
# These are the four primitives sequenced in `docs/audit/ga_ebm_roadmap.md`
# § EBM2. They share the same RNG-key threading discipline as the keyed
# point samplers above: take a key, internally split as needed, return
# the consumed key for downstream functional composition.
#
# All four target distributions of the form ``p(y) ∝ exp(-E(y) / T)``
# where ``T`` is a temperature parameter. ``temperature=1.0`` matches
# the conventional ``p(y) ∝ exp(-E(y))`` form.

import math as _math
from typing import Callable as _Callable, Optional as _Optional, Tuple as _Tuple


def _langevin_step_with_key(
    y: np.ndarray,
    grad_fn: _Callable[[np.ndarray], np.ndarray],
    eta: float,
    temperature: float,
    key: "RNGKey",
) -> _Tuple[np.ndarray, "RNGKey"]:
    grad = np.asarray(grad_fn(y)).astype(y.dtype, copy=False)
    sub_key, next_key = key.split(2)
    noise_scale = _math.sqrt(2.0 * eta * temperature)
    if noise_scale > 0.0:
        noise = normal(sub_key, shape=y.shape, dtype=str(y.dtype))
        out = y - eta * grad + noise_scale * noise.astype(y.dtype, copy=False)
    else:
        out = y - eta * grad
    return out.astype(y.dtype, copy=False), next_key


def _collect_chain(
    y0: np.ndarray,
    step_fn: _Callable[[np.ndarray, "RNGKey"], _Tuple[np.ndarray, "RNGKey", dict]],
    key: "RNGKey",
    n_samples: int,
    burn_in: int,
    thin: int,
) -> _Tuple[np.ndarray, "RNGKey", dict]:
    """Common chain-collection harness for Langevin / MALA / HMC / Gibbs.

    ``step_fn`` returns ``(y_next, next_key, step_info)``. ``step_info``
    is a dict of per-step diagnostics (e.g. ``{"accept": True}``).
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive; got {n_samples}.")
    if burn_in < 0:
        raise ValueError(f"burn_in must be non-negative; got {burn_in}.")
    if thin < 1:
        raise ValueError(f"thin must be >= 1; got {thin}.")
    y = np.array(y0, copy=True)
    samples: list[np.ndarray] = []
    accept_count = 0
    accept_total = 0
    total_steps = burn_in + n_samples * thin
    for step in range(total_steps):
        y, key, info = step_fn(y, key)
        if "accept" in info:
            accept_total += 1
            if info["accept"]:
                accept_count += 1
        if step >= burn_in and ((step - burn_in) % thin == 0):
            samples.append(y.copy())
    stacked = np.stack(samples, axis=0)
    diagnostics: dict = {}
    if accept_total > 0:
        diagnostics["accept_rate"] = accept_count / accept_total
    return stacked, key, diagnostics


def langevin_sample(
    key: "RNGKey",
    *,
    init: np.ndarray,
    grad_fn: _Callable[[np.ndarray], np.ndarray],
    eta: float,
    temperature: float = 1.0,
    n_samples: int = 1,
    burn_in: int = 0,
    thin: int = 1,
) -> _Tuple[np.ndarray, "RNGKey", dict]:
    """Unadjusted Langevin sampler (ULA).

    Iterates ``y ← y − η · ∇E(y) + √(2 η T) · ξ`` for
    ``burn_in + n_samples · thin`` steps. No MH correction — the chain
    has discretization bias proportional to ``η``; use ``mala_sample``
    for unbiased samples.

    Returns ``(samples, next_key, diagnostics)`` where ``samples`` has
    shape ``(n_samples, *init.shape)``.
    """
    if eta <= 0.0:
        raise ValueError(f"langevin_sample requires eta > 0; got {eta}.")
    if temperature < 0.0:
        raise ValueError(f"langevin_sample requires temperature >= 0; got {temperature}.")

    def step_fn(y: np.ndarray, k: "RNGKey") -> _Tuple[np.ndarray, "RNGKey", dict]:
        y_next, k_next = _langevin_step_with_key(y, grad_fn, eta, temperature, k)
        return y_next, k_next, {}

    return _collect_chain(np.asarray(init), step_fn, key, n_samples, burn_in, thin)


def mala_sample(
    key: "RNGKey",
    *,
    init: np.ndarray,
    energy_fn: _Callable[[np.ndarray], float],
    grad_fn: _Callable[[np.ndarray], np.ndarray],
    eta: float,
    temperature: float = 1.0,
    n_samples: int = 1,
    burn_in: int = 0,
    thin: int = 1,
) -> _Tuple[np.ndarray, "RNGKey", dict]:
    """Metropolis-Adjusted Langevin Algorithm.

    Each step proposes ``y' = y − η·∇E(y) + √(2 η T)·ξ`` then accepts
    with probability ``min(1, exp(log α))`` where

        log α = (E(y) - E(y')) / T
              + (||y' - y + η·∇E(y)||² - ||y - y' + η·∇E(y')||²) / (4 η T).

    Returns ``(samples, next_key, {"accept_rate": ...})``.
    """
    if eta <= 0.0:
        raise ValueError(f"mala_sample requires eta > 0; got {eta}.")
    if temperature <= 0.0:
        raise ValueError(
            f"mala_sample requires temperature > 0; got {temperature}. "
            f"Zero-temperature has degenerate MH ratio; use langevin_sample."
        )
    noise_scale = _math.sqrt(2.0 * eta * temperature)
    var_proposal = 2.0 * eta * temperature

    def step_fn(y: np.ndarray, k: "RNGKey") -> _Tuple[np.ndarray, "RNGKey", dict]:
        grad_y = np.asarray(grad_fn(y)).astype(y.dtype, copy=False)
        prop_key, accept_key, next_key = k.split(3)
        noise = normal(prop_key, shape=y.shape, dtype=str(y.dtype))
        y_prop = y - eta * grad_y + noise_scale * noise.astype(y.dtype, copy=False)
        grad_y_prop = np.asarray(grad_fn(y_prop)).astype(y.dtype, copy=False)
        # Log MH ratio.
        d_energy = float(energy_fn(y)) - float(energy_fn(y_prop))
        diff_fwd = y_prop - y + eta * grad_y
        diff_back = y - y_prop + eta * grad_y_prop
        log_q_ratio = (
            float(np.sum(diff_fwd ** 2)) - float(np.sum(diff_back ** 2))
        ) / (2.0 * var_proposal)
        log_alpha = d_energy / temperature + log_q_ratio
        u = uniform(accept_key, shape=(), dtype=str(y.dtype))
        accept = _math.log(max(float(u), 1e-30)) < log_alpha
        y_next = y_prop if accept else y
        return y_next, next_key, {"accept": bool(accept)}

    return _collect_chain(np.asarray(init), step_fn, key, n_samples, burn_in, thin)


def _hmc_leapfrog(
    q: np.ndarray,
    p: np.ndarray,
    grad_fn: _Callable[[np.ndarray], np.ndarray],
    step_size: float,
    n_leapfrog: int,
    mass_inv: np.ndarray,
) -> _Tuple[np.ndarray, np.ndarray]:
    """Standard leapfrog integrator for Hamiltonian dynamics.

    Volume-preserving and time-reversible (negate p; integrate forward;
    you recover the starting state). Tested in
    `test_ebm_samplers::test_hmc_leapfrog_is_reversible`.
    """
    p = p - 0.5 * step_size * np.asarray(grad_fn(q))
    for i in range(n_leapfrog):
        q = q + step_size * (mass_inv * p)
        if i < n_leapfrog - 1:
            p = p - step_size * np.asarray(grad_fn(q))
    p = p - 0.5 * step_size * np.asarray(grad_fn(q))
    return q, p


def hmc_sample(
    key: "RNGKey",
    *,
    init: np.ndarray,
    energy_fn: _Callable[[np.ndarray], float],
    grad_fn: _Callable[[np.ndarray], np.ndarray],
    step_size: float,
    n_leapfrog: int,
    n_samples: int = 1,
    burn_in: int = 0,
    thin: int = 1,
    mass: _Optional[np.ndarray] = None,
) -> _Tuple[np.ndarray, "RNGKey", dict]:
    """Hamiltonian Monte Carlo with leapfrog integration.

    Each iteration:
      1. Sample fresh momentum ``p ~ N(0, M)`` where ``M`` is the mass matrix.
      2. Integrate ``(q, p)`` via leapfrog for ``n_leapfrog`` steps of size
         ``step_size``.
      3. Accept with MH ratio on the Hamiltonian
         ``H(q, p) = E(q) + ½ pᵀ M⁻¹ p``.

    Returns ``(samples, next_key, {"accept_rate": ...})``.
    """
    if step_size <= 0.0:
        raise ValueError(f"hmc_sample requires step_size > 0; got {step_size}.")
    if n_leapfrog < 1:
        raise ValueError(f"hmc_sample requires n_leapfrog >= 1; got {n_leapfrog}.")
    init_arr = np.asarray(init)
    if mass is None:
        mass_arr = np.ones_like(init_arr, dtype=init_arr.dtype)
    else:
        mass_arr = np.asarray(mass).astype(init_arr.dtype, copy=False)
        if mass_arr.shape != init_arr.shape:
            raise ValueError(
                f"mass shape {mass_arr.shape} must match init shape {init_arr.shape}."
            )
    mass_inv = (1.0 / mass_arr).astype(init_arr.dtype, copy=False)
    sqrt_mass = np.sqrt(mass_arr).astype(init_arr.dtype, copy=False)

    def step_fn(q: np.ndarray, k: "RNGKey") -> _Tuple[np.ndarray, "RNGKey", dict]:
        p_key, accept_key, next_key = k.split(3)
        p = sqrt_mass * normal(p_key, shape=q.shape, dtype=str(q.dtype)).astype(
            q.dtype, copy=False
        )
        H0 = float(energy_fn(q)) + 0.5 * float(np.sum(mass_inv * p * p))
        q_new, p_new = _hmc_leapfrog(
            q, p, grad_fn, step_size, n_leapfrog, mass_inv
        )
        H1 = float(energy_fn(q_new)) + 0.5 * float(np.sum(mass_inv * p_new * p_new))
        u = uniform(accept_key, shape=(), dtype=str(q.dtype))
        log_alpha = H0 - H1
        accept = _math.log(max(float(u), 1e-30)) < log_alpha
        q_next = q_new if accept else q
        return q_next, next_key, {"accept": bool(accept)}

    return _collect_chain(init_arr, step_fn, key, n_samples, burn_in, thin)


def gibbs_sample(
    key: "RNGKey",
    *,
    init: np.ndarray,
    conditional_sample: _Callable[[int, np.ndarray, "RNGKey"], _Tuple[float, "RNGKey"]],
    n_samples: int = 1,
    burn_in: int = 0,
    thin: int = 1,
    sweep_order: _Optional[Sequence[int]] = None,
) -> _Tuple[np.ndarray, "RNGKey", dict]:
    """Generic coordinate-wise Gibbs sampler.

    ``conditional_sample(i, y, rng_key)`` is a user-provided callable
    that draws ``y_i ~ p(y_i | y_{-i})`` and returns ``(new_value,
    next_key)``. A "sweep" updates every coordinate once in the order
    given by ``sweep_order`` (default: ``range(len(init))``).

    Designed for the EBM8 RBM conformance test (Gibbs alternates
    between visible/hidden layers); generalizes to any factorized
    posterior.
    """
    init_arr = np.asarray(init)
    if init_arr.ndim != 1:
        raise ValueError(
            f"gibbs_sample currently requires a rank-1 init; got shape {init_arr.shape}."
        )
    if sweep_order is None:
        order = tuple(range(init_arr.shape[0]))
    else:
        order = tuple(sweep_order)
        if sorted(order) != list(range(init_arr.shape[0])):
            raise ValueError(
                f"sweep_order must be a permutation of range({init_arr.shape[0]}); got {order}."
            )

    def step_fn(y: np.ndarray, k: "RNGKey") -> _Tuple[np.ndarray, "RNGKey", dict]:
        y_next = y.copy()
        for i in order:
            new_value, k = conditional_sample(i, y_next, k)
            y_next[i] = new_value
        return y_next, k, {}

    return _collect_chain(init_arr, step_fn, key, n_samples, burn_in, thin)


__all__ = [
    "RNGKey",
    "uniform",
    "normal",
    "truncated_normal",
    "bernoulli",
    "categorical",
    "multinomial",
    "randint",
    "permutation",
    "gamma",
    "beta",
    "dirichlet",
    "poisson",
    # EBM2 iterative samplers
    "langevin_sample",
    "mala_sample",
    "hmc_sample",
    "gibbs_sample",
]
