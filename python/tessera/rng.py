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
from dataclasses import dataclass, field
from typing import Sequence

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

    @staticmethod
    def from_seed(seed: int, *, name: str = "") -> "RNGKey":
        """Construct from a single user-facing integer seed."""
        # Mix the seed through a hash so adjacent seeds (0, 1, 2, ...) don't
        # produce correlated streams.
        high = _hash_to_u64("seed", int(seed))
        return RNGKey(seed_high=high, seed_low=0, name=name)

    def split(self, num: int = 2) -> tuple["RNGKey", ...]:
        """Deterministically derive `num` independent child keys."""
        if num <= 0:
            raise ValueError("RNGKey.split requires num > 0")
        return tuple(
            RNGKey(
                seed_high=_hash_to_u64("split", self.seed_high, self.seed_low, i),
                seed_low=0,
                name=f"{self.name}.split[{i}]" if self.name else "",
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
]
