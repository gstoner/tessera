"""Portable member-keyed RNG contract for physical Evolution Strategies.

The general-purpose :mod:`tessera.rng` host stream intentionally uses NumPy's
Philox-4x64 implementation and BLAKE2-based ``fold_in``.  Neither primitive is
an appropriate physical ABI for a GPU kernel.  ES needs a stronger property:
every worker reconstructing ``(epoch, member_pair)`` must produce the same
factors without folding in its device rank.

This module defines that physical ABI.  Key derivation uses SplitMix64 (integer
operations only), then the existing device Philox-4x32-10 + Box--Muller
reference.  x86, ROCm, CUDA, and Apple emitters can implement it directly.
Changing any constant or ordering is an ABI version change.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .. import rng_device

ES_MEMBER_RNG_ALGORITHM = "splitmix64-philox4x32-boxmuller"
ES_MEMBER_RNG_VERSION = 1
_MASK64 = (1 << 64) - 1
_EPOCH_DOMAIN = 0x45535F45504F4348  # "ES_EPOCH"
_PAIR_DOMAIN = 0x45535F504149525F  # "ES_PAIR_"


class _RNGKeyLike(Protocol):
    @property
    def seed_high(self) -> int: ...

    @property
    def seed_low(self) -> int: ...


def _splitmix64(value: int) -> int:
    """SplitMix64 finalizer, expressed with portable unsigned operations."""
    z = (int(value) + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def member_seed(key: _RNGKeyLike, *, epoch: int, pair: int) -> int:
    """Derive the rank-invariant 64-bit Philox key for an ES member pair."""
    if epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {epoch}")
    if pair < 0:
        raise ValueError(f"pair must be non-negative, got {pair}")
    root = _splitmix64(int(key.seed_high) ^ _splitmix64(int(key.seed_low)))
    epoch_key = _splitmix64(root ^ _EPOCH_DOMAIN ^ int(epoch))
    return _splitmix64(epoch_key ^ _PAIR_DOMAIN ^ int(pair))


def member_normal(
    key: _RNGKeyLike,
    *,
    epoch: int,
    pair: int,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Generate the canonical f32 normal factors for one ES member pair."""
    if not shape or any(int(dim) <= 0 for dim in shape):
        raise ValueError(f"shape must contain positive extents, got {shape!r}")
    count = int(np.prod(shape, dtype=np.int64))
    seed = member_seed(key, epoch=epoch, pair=pair)
    return rng_device.normal(seed, count).reshape(shape)


__all__ = [
    "ES_MEMBER_RNG_ALGORITHM",
    "ES_MEMBER_RNG_VERSION",
    "member_normal",
    "member_seed",
]
