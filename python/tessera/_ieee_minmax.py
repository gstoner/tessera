"""IEEE-754-2019 elementwise maximum/minimum reference helpers.

Fleet-wide contract (decision 2026-08-23, rocm plan key
``IEEE-MINMAX-CONTRACT-2026-08-23``): ``tessera.maximum``/``minimum``
propagate NaN and ORDER signed zeros — maximum tie -> +0.0, minimum tie
-> -0.0 — on every execution route. ``np.maximum``/``np.minimum`` are NOT
valid references for the tie sign: numpy resolves a ±0 tie to whichever
operand the host ISA's min/max instruction returns (SSE: the second
operand; NEON: the IEEE-ordered zero), so a numpy-delegating eager or
fallback path would disagree with the compiled kernels depending on the
machine running it. These helpers are the single reference
implementation the eager op namespace and numpy fallback lanes share
(Decision #31 — one implementation per boundary).
"""

from __future__ import annotations


def ieee_maximum(x, y):
    """NaN-propagating maximum with IEEE signed-zero ordering (tie -> +0)."""
    import numpy as np

    out = np.maximum(x, y)
    # An ordered tie (False for NaN lanes) only changes the result for ±0:
    # equal non-zero values are bit-identical, so selecting either is a
    # no-op. For max, +0 must win: take y when x carries the sign bit.
    tie = x == y
    return np.where(tie, np.where(np.signbit(x), y, x), out)


def ieee_minimum(x, y):
    """NaN-propagating minimum with IEEE signed-zero ordering (tie -> -0)."""
    import numpy as np

    out = np.minimum(x, y)
    tie = x == y
    return np.where(tie, np.where(np.signbit(x), x, y), out)
