"""M6 Step 4 — Philox-4x32-10 reference (Python + MSL template).

Philox is a counter-based pseudorandom number generator from
Salmon et al. (2011) — "Parallel Random Numbers: As Easy as 1,
2, 3".  The key property: the same ``(key, counter)`` pair
produces the same 128-bit output regardless of which thread,
device, or order it ran in.

That's the property M6 Step 4 needs for on-device RNG.  The
existing Apple GPU kernels (``langevin_step``, ``decode_init``,
``sphere_langevin``) all take a host-supplied noise buffer
today; with Philox-4x32-10 emitted into the MSL shader source,
a kernel can compute its own noise from a 4-element key + a
counter that derives from the thread index.

This module ships:

  * :func:`philox_4x32_10` — pure-Python reference, verified
    against the canonical reference vectors from the Random123
    code base.
  * :func:`philox_normal_pair` — Box-Muller pair from two
    uniform Philox samples.
  * :func:`philox_msl_source` — MSL source template that mirrors
    the Python implementation byte-for-byte.

The MSL template + the Python reference are tied together by
:mod:`tests.unit.test_philox`, which:

  - locks the constants (any drift fails the test);
  - verifies known reference vectors;
  - checks the MSL source contains the same constants + the
    expected 10-round structure (so the C++ runtime side can
    drop the template into a dispatcher without re-deriving the
    algorithm).

Actual MSL compilation + Metal dispatch land in a follow-up
sprint that touches ``apple_gpu_runtime.mm``; this module is
the contract that follow-up will satisfy.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants — taken from the Random123 reference implementation.
# Any drift here breaks cross-platform determinism, so the tests
# pin them.
# ─────────────────────────────────────────────────────────────────────────────

PHILOX_M0 = np.uint32(0xD2511F53)
PHILOX_M1 = np.uint32(0xCD9E8D57)
PHILOX_W0 = np.uint32(0x9E3779B9)   # Weyl constant ≈ 2³² / φ
PHILOX_W1 = np.uint32(0xBB67AE85)   # Weyl constant ≈ 2³² / √2

PHILOX_ROUNDS = 10
PHILOX_OUTPUT_WORDS = 4   # 128-bit output
PHILOX_KEY_WORDS = 2      # 64-bit key
PHILOX_COUNTER_WORDS = 4  # 128-bit counter


# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python reference
# ─────────────────────────────────────────────────────────────────────────────

def _mulhilo(a: np.uint32, b: np.uint32) -> tuple[np.uint32, np.uint32]:
    """Return ``(lo, hi)`` of the 32×32 → 64 product ``a * b``."""
    p = np.uint64(a) * np.uint64(b)
    return np.uint32(p & np.uint64(0xFFFFFFFF)), np.uint32(p >> np.uint64(32))


def _philox_round(ctr: np.ndarray, key: np.ndarray) -> np.ndarray:
    """One round of Philox-4x32.  Bijective on a 128-bit counter."""
    c0, c1, c2, c3 = ctr[0], ctr[1], ctr[2], ctr[3]
    k0, k1 = key[0], key[1]
    lo0, hi0 = _mulhilo(PHILOX_M0, c0)
    lo1, hi1 = _mulhilo(PHILOX_M1, c2)
    out = np.array(
        [
            np.uint32(hi1 ^ c1 ^ k0),
            np.uint32(lo1),
            np.uint32(hi0 ^ c3 ^ k1),
            np.uint32(lo0),
        ],
        dtype=np.uint32,
    )
    return out


def _bump_key(key: np.ndarray) -> np.ndarray:
    """Weyl-sequence key update between rounds.

    Modular uint32 arithmetic is intentional — wrap-around at 2³²
    is part of the algorithm.  We do the add in uint64 to silence
    numpy's overflow warning, then truncate.
    """
    return np.array(
        [
            np.uint32(np.uint64(key[0]) + np.uint64(PHILOX_W0)),
            np.uint32(np.uint64(key[1]) + np.uint64(PHILOX_W1)),
        ],
        dtype=np.uint32,
    )


def philox_4x32_10(
    counter: np.ndarray, key: np.ndarray,
) -> np.ndarray:
    """Philox-4x32-10: 128-bit output from a 128-bit counter + 64-bit key.

    Parameters
    ----------
    counter
        4 × uint32 array.  Increment the lowest word between
        calls to advance the stream.
    key
        2 × uint32 array.

    Returns
    -------
    np.ndarray
        4 × uint32 — the pseudorandom output.
    """
    ctr = np.asarray(counter, dtype=np.uint32).copy()
    k = np.asarray(key, dtype=np.uint32).copy()
    if ctr.shape != (4,):
        raise ValueError(f"counter must be shape (4,) uint32; got {ctr.shape}")
    if k.shape != (2,):
        raise ValueError(f"key must be shape (2,) uint32; got {k.shape}")
    for _ in range(PHILOX_ROUNDS):
        ctr = _philox_round(ctr, k)
        k = _bump_key(k)
    return ctr


def philox_uniform(
    counter: np.ndarray, key: np.ndarray,
) -> np.ndarray:
    """Four uniform-(0, 1) floats from one Philox-4x32-10 call."""
    out = philox_4x32_10(counter, key)
    # Map uint32 → (0, 1] via ldexp(1.0, -32) = 2⁻³² scaling.
    # Adding 0.5 → centered, no zeros so log/exp are safe.
    return (out.astype(np.float64) + 0.5) * (2.0 ** -32)


def philox_normal_pair(
    counter: np.ndarray, key: np.ndarray,
) -> tuple[float, float]:
    """A pair of standard-normal samples via Box-Muller from
    Philox uniforms.

    Box-Muller: given u1, u2 ~ U(0, 1),
      z1 = sqrt(-2 ln u1) cos(2 π u2)
      z2 = sqrt(-2 ln u1) sin(2 π u2)
    """
    u = philox_uniform(counter, key)
    u1, u2 = u[0], u[1]
    r = np.sqrt(-2.0 * np.log(max(u1, 1e-300)))
    theta = 2.0 * np.pi * u2
    return (float(r * np.cos(theta)), float(r * np.sin(theta)))


# ─────────────────────────────────────────────────────────────────────────────
# MSL source template — the contract the C++ runtime side will
# satisfy when it lifts the kernel into apple_gpu_runtime.mm.
# ─────────────────────────────────────────────────────────────────────────────

_PHILOX_MSL_TEMPLATE = """\
// Philox-4x32-10 — auto-generated by tessera.compiler.philox.
// Constants and round structure match the Python reference in
// tessera/compiler/philox.py byte-for-byte.

constant constexpr uint PHILOX_M0 = 0xD2511F53u;
constant constexpr uint PHILOX_M1 = 0xCD9E8D57u;
constant constexpr uint PHILOX_W0 = 0x9E3779B9u;
constant constexpr uint PHILOX_W1 = 0xBB67AE85u;

inline void philox_mulhilo(uint a, uint b, thread uint &lo, thread uint &hi) {
    ulong p = (ulong)a * (ulong)b;
    lo = (uint)(p & 0xFFFFFFFFu);
    hi = (uint)(p >> 32);
}

inline void philox_round(thread uint ctr[4], thread const uint key[2]) {
    uint lo0, hi0, lo1, hi1;
    philox_mulhilo(PHILOX_M0, ctr[0], lo0, hi0);
    philox_mulhilo(PHILOX_M1, ctr[2], lo1, hi1);
    uint c0 = hi1 ^ ctr[1] ^ key[0];
    uint c1 = lo1;
    uint c2 = hi0 ^ ctr[3] ^ key[1];
    uint c3 = lo0;
    ctr[0] = c0;
    ctr[1] = c1;
    ctr[2] = c2;
    ctr[3] = c3;
}

inline void philox_bump_key(thread uint key[2]) {
    key[0] += PHILOX_W0;
    key[1] += PHILOX_W1;
}

// Produces 4 × uint32 of pseudorandom output from a 4-element
// counter + 2-element key.  10 rounds.
inline void philox_4x32_10(thread uint ctr[4], thread uint key[2],
                            thread uint out[4]) {
    uint c[4] = {ctr[0], ctr[1], ctr[2], ctr[3]};
    uint k[2] = {key[0], key[1]};
    for (int r = 0; r < 10; ++r) {
        philox_round(c, k);
        philox_bump_key(k);
    }
    out[0] = c[0];
    out[1] = c[1];
    out[2] = c[2];
    out[3] = c[3];
}

// Convert four Philox uint32 outputs to four uniform-(0, 1) floats.
inline void philox_to_uniform(thread const uint out[4], thread float u[4]) {
    constexpr float k = 0x1.0p-32f;   // 2^-32
    u[0] = ((float)out[0] + 0.5f) * k;
    u[1] = ((float)out[1] + 0.5f) * k;
    u[2] = ((float)out[2] + 0.5f) * k;
    u[3] = ((float)out[3] + 0.5f) * k;
}
"""


def philox_msl_source() -> str:
    """The MSL source template a future C++ kernel emitter
    drops into apple_gpu_runtime.mm.

    The Python reference + the MSL template share constants
    byte-for-byte — the test suite locks the cross-platform
    invariant so the C++ side can be lifted mechanically.
    """
    return _PHILOX_MSL_TEMPLATE


__all__ = [
    "PHILOX_M0", "PHILOX_M1", "PHILOX_W0", "PHILOX_W1",
    "PHILOX_ROUNDS", "PHILOX_OUTPUT_WORDS",
    "PHILOX_KEY_WORDS", "PHILOX_COUNTER_WORDS",
    "philox_4x32_10",
    "philox_uniform",
    "philox_normal_pair",
    "philox_msl_source",
]
