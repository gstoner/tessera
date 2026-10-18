"""Semantic NVFP4 expansion reference and checked import boundary.

The reference is for eager/oracle use. Native lowering belongs to MLIR.
"""
from __future__ import annotations

import numpy as np


def nvfp4_expansion_policy():
    return {
        "execution_mode": "source_preserving_expansion",
        "source_format": "nvfp4_e2m1_e4m3_k16",
        "destination_format": "fp8_e4m3_k16_f32_scales",
        "source_scale_application": "projection_global_times_e4m3",
        "destination_code_selection": "exact_e2m1_signed_zero",
        "destination_scale_order": "k_group_n",
        "scale_rounding": "f64_product_to_f32_rne",
    }


def checked_expansion_inputs(codes, scales, projection_globals, *, row_offsets, numeric_policy, _scale_bounds=None):
    import ml_dtypes

    if numeric_policy != nvfp4_expansion_policy():
        raise ValueError("NVFP4 expansion conflicting or incomplete numeric policy")
    codes, scales, globals_ = map(np.asarray, (codes, scales, projection_globals))
    if codes.dtype != np.uint8 or codes.ndim != 2:
        raise ValueError("NVFP4 expansion requires packed uint8 rank-two codes")
    n, half_k = codes.shape
    if n <= 0 or half_k <= 0 or half_k % 8 or n > np.iinfo(np.intp).max // (2 * half_k):
        raise ValueError("NVFP4 expansion requires representable positive N/K16")
    k = 2 * half_k
    offsets = tuple(row_offsets)
    if (len(offsets) < 2 or any(type(x) is not int for x in offsets)
            or offsets[0] != 0 or offsets[-1] != n
            or any(a >= b for a, b in zip(offsets, offsets[1:]))):
        raise ValueError("NVFP4 expansion requires increasing complete projection boundaries")
    if scales.dtype != np.dtype(ml_dtypes.float8_e4m3fn) or scales.shape != (n, k // 16):
        raise ValueError("NVFP4 expansion requires E4M3 [N,K/16] scales")
    if globals_.dtype != np.float64 or globals_.shape != (len(offsets) - 1,):
        raise ValueError("NVFP4 expansion requires f64 projection globals")
    # Prepared native programs provide an encoded-byte admission scanner.
    # Legacy providers and eager/oracle callers retain the elementwise check.
    bounds = _scale_bounds(scales, offsets) if _scale_bounds is not None else None
    if bounds is not None and (
            not isinstance(bounds, tuple) or len(bounds) != len(offsets) - 1
            or any(not isinstance(pair, tuple) or len(pair) != 2
                   or any(type(v) is not int for v in pair) for pair in bounds)):
        raise ValueError("NVFP4 native scale bounds differ")
    decoded = None if bounds is not None else scales.astype(np.float64)
    if decoded is not None and (not np.all(np.isfinite(decoded)) or np.any(decoded < 0)):
        raise ValueError("NVFP4 expansion scales must be finite and nonnegative")
    if not np.all(np.isfinite(globals_)) or np.any(globals_ <= 0):
        raise ValueError("NVFP4 expansion globals must be finite and positive")
    # Positive multiplication and both rounding steps are monotone. A native
    # smallest-positive/largest-positive pair proves the original all-element
    # range predicate, retaining signed-zero and f64-product -> f32 behavior.
    for p, (start, stop) in enumerate(zip(offsets, offsets[1:])):
        if bounds is None:
            assert decoded is not None
            values = decoded[start:stop]
        else:
            low, high = bounds[p]
            if low == 127 and high == 0:
                continue
            if not 0 < low <= high < 127:
                raise ValueError("NVFP4 native scale bounds differ")
            values = np.array([low, high], np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float64)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            rounded = (values * globals_[p]).astype(np.float32)
        if (not np.all(np.isfinite(rounded))
                or np.any((values > 0) & (rounded == 0))):
            raise ValueError("NVFP4 expansion scale product overflows or underflows f32")
    return codes, scales, globals_, offsets


def reference_nvfp4_expand(codes, scales, projection_globals, *, row_offsets, numeric_policy):
    import ml_dtypes

    codes, scales, globals_, offsets = checked_expansion_inputs(
        codes, scales, projection_globals, row_offsets=row_offsets, numeric_policy=numeric_policy)
    # Encode exact E2M1 values directly as E4M3 bits, retaining both zeros.
    nibble = np.empty((codes.shape[0], 2 * codes.shape[1]), np.uint8)
    nibble[:, 0::2] = codes & 15
    nibble[:, 1::2] = codes >> 4
    mag = nibble & 7
    value_bits = np.where(mag == 0, 0, np.where(mag == 1, 48, (mag + 12) << 2))
    value_bits = np.asarray(value_bits | ((nibble & 8) << 4), dtype=np.uint8)
    scale_plane = np.empty((scales.shape[1], codes.shape[0]), np.float32)
    for p, (start, stop) in enumerate(zip(offsets, offsets[1:])):
        scale_plane[:, start:stop] = (scales[start:stop].astype(np.float64) * globals_[p]).T
    return value_bits.view(ml_dtypes.float8_e4m3fn), scale_plane
