"""Device RNG reference — counter-based Philox-4x32-10 (P6).

A device RNG cannot bit-match numpy's ``Generator`` (numpy uses Philox-4x64 plus
its own Lemire/ziggurat transforms), so Tessera's *device* lane uses the
standard counter-based **Philox-4x32-10** (Salmon et al. 2011 — the algorithm
JAX and cuRAND use). This module is the numpy reference of the *identical*
algorithm: the x86 / gfx1151 kernels are validated BIT-EXACTLY (uniform) against
``philox_uniform`` here, and the normal / dropout transforms against the
matching helpers.

This is a SEPARATE deterministic stream from ``tessera.rng`` (the host
numpy-Generator path); RNG streams are implementation-defined per backend. The
device contract is "a correct, reproducible, parallel-safe stream" — proven by
(1) bit-exact agreement with this reference, (2) the right distribution
statistics, and (3) determinism (same key ⇒ same output).
"""

from __future__ import annotations

import numpy as np

_M0 = np.uint64(0xD2511F53)
_M1 = np.uint64(0xCD9E8D57)
_W0 = np.uint32(0x9E3779B9)   # golden ratio
_W1 = np.uint32(0xBB67AE85)   # sqrt(3) - 1
_MASK = np.uint64(0xFFFFFFFF)
_INV32 = np.float32(1.0 / 4294967296.0)   # 2^-32


def philox_uniform(seed: int, counter_base: int, n: int) -> np.ndarray:
    """``n`` uniform f32 in [0, 1) from Philox-4x32-10 keyed by ``seed``.

    Counter block ``b`` packs ``(b, 0, 0, 0)``; its 4 round outputs fill
    elements ``[4b, 4b+1, 4b+2, 4b+3]``. Bit-identical to the C / MLIR kernels.
    """
    nblocks = (int(n) + 3) // 4
    b = np.arange(nblocks, dtype=np.uint64) + np.uint64(int(counter_base))
    c0 = (b & _MASK).astype(np.uint32)
    c1 = (b >> np.uint64(32)).astype(np.uint32)
    c2 = np.zeros(nblocks, np.uint32)
    c3 = np.zeros(nblocks, np.uint32)
    k0 = np.full(nblocks, np.uint32(int(seed) & 0xFFFFFFFF))
    k1 = np.full(nblocks, np.uint32((int(seed) >> 32) & 0xFFFFFFFF))
    with np.errstate(over="ignore"):
        for r in range(10):
            if r > 0:
                k0 = k0 + _W0
                k1 = k1 + _W1
            p0 = c0.astype(np.uint64) * _M0
            p1 = c2.astype(np.uint64) * _M1
            hi0 = (p0 >> np.uint64(32)).astype(np.uint32)
            lo0 = (p0 & _MASK).astype(np.uint32)
            hi1 = (p1 >> np.uint64(32)).astype(np.uint32)
            lo1 = (p1 & _MASK).astype(np.uint32)
            n0 = hi1 ^ c1 ^ k0
            n2 = hi0 ^ c3 ^ k1
            c0, c1, c2, c3 = n0, lo1, n2, lo0
    out = np.stack([c0, c1, c2, c3], axis=1).reshape(-1)[:int(n)]
    return out.astype(np.float32) * _INV32


def uniform(seed: int, n: int, low: float = 0.0, high: float = 1.0,
            counter_base: int = 0) -> np.ndarray:
    return (np.float32(low)
            + np.float32(high - low) * philox_uniform(seed, counter_base, n))


def normal(seed: int, n: int, mean: float = 0.0, std: float = 1.0,
           counter_base: int = 0) -> np.ndarray:
    """Box-Muller from two independent uniform halves of the Philox stream."""
    m = (int(n) + 1) // 2
    u1 = philox_uniform(seed, counter_base, m)
    u2 = philox_uniform(seed, counter_base + (m + 3) // 4 + 1, m)
    u1 = np.clip(u1, np.float32(1e-7), np.float32(1.0))
    r = np.sqrt(-2.0 * np.log(u1)).astype(np.float32)
    theta = np.float32(2.0 * np.pi) * u2
    z = np.empty(2 * m, np.float32)
    z[0::2] = r * np.cos(theta)
    z[1::2] = r * np.sin(theta)
    return (np.float32(mean) + np.float32(std) * z[:int(n)]).astype(np.float32)


def dropout_mask(seed: int, n: int, p: float, counter_base: int = 0) -> np.ndarray:
    """Inverted-dropout multiplier: keep (u >= p) scaled by 1/(1-p)."""
    u = philox_uniform(seed, counter_base, n)
    keep = (u >= np.float32(p)).astype(np.float32)
    scale = np.float32(0.0) if p >= 1.0 else np.float32(1.0 / (1.0 - p))
    return keep * scale
