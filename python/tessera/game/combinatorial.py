"""Combinatorial game layer (G1): the segmented minimum excludant.

``mex`` is the one genuinely new combinatorial primitive the plan needs —
Grundy numbers are ``mex`` over reachable positions, and nim-sums ride the
existing ``bitwise_xor`` reduce. Integer-valued and non-differentiable by
construction: no VJP is registered, so a gradient path through ``mex`` raises
(the correct outcome — Decision #30's fail-closed reading, not a silent zero).
"""

from __future__ import annotations

import numpy as np

from ..custom import custom_primitive


def _mex_impl(values: np.ndarray, segment_ids: np.ndarray,
              *, num_segments: int) -> np.ndarray:
    """Per-segment smallest non-negative integer absent from the segment.

    ``values``/``segment_ids`` are flat, equal-length integer arrays (the
    ragged encoding shared with ``segment_reduce``); an empty segment's mex is
    0 by definition. Negative values are ignored (mex is over ℕ); a negative
    segment id fails closed.
    """
    values = np.asarray(values)
    segment_ids = np.asarray(segment_ids)
    if values.shape != segment_ids.shape or values.ndim != 1:
        raise ValueError(
            "mex expects flat, equal-length values and segment_ids; got "
            f"{values.shape} vs {segment_ids.shape}")
    if not (np.issubdtype(values.dtype, np.integer)
            and np.issubdtype(segment_ids.dtype, np.integer)):
        raise ValueError("mex operates on integer values and segment ids")
    if num_segments <= 0:
        raise ValueError(f"num_segments must be positive; got {num_segments}")
    if segment_ids.size and (segment_ids.min() < 0
                             or segment_ids.max() >= num_segments):
        raise ValueError("segment id out of range [0, num_segments)")
    out = np.zeros(num_segments, dtype=np.int64)
    for seg in range(num_segments):
        present = set(int(x) for x in values[segment_ids == seg] if x >= 0)
        g = 0
        while g in present:
            g += 1
        out[seg] = g
    return out


mex = custom_primitive("game_mex")(_mex_impl)
