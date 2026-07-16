"""Shared numerical contract for exact-device NVIDIA comparisons.

The values deliberately describe *storage* precision and accumulator semantics,
not a backend route.  CUDA and ROCm tests that exercise the same logical
operation can therefore use the same reference and document any physical-route
specific exception at its emission site.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Tolerance:
    """Absolute/relative comparison budget and whether storage is exact."""

    atol: float
    rtol: float
    exact: bool = False


# All floating MMA paths accumulate in f32.  The FP8 budgets reflect operand
# quantisation, not a permission to lose accumulator precision.  Integer and
# NVFP4 ABI fixtures have exact integer/reference semantics and must not use a
# floating tolerance to conceal a packing or scale-layout defect.
_BASE: dict[str, Tolerance] = {
    "f32": Tolerance(2e-5, 2e-5),
    "f16": Tolerance(5e-2, 2e-3),
    "bf16": Tolerance(2e-1, 4e-3),
    "tf32": Tolerance(1e-2, 2e-3),
    "fp8_e4m3": Tolerance(1.0, 2e-2),
    "fp8_e5m2": Tolerance(2.0, 4e-2),
    "int8": Tolerance(0.0, 0.0, exact=True),
    "nvfp4": Tolerance(0.0, 0.0, exact=True),
}

_ALIASES = {
    "float32": "f32", "float16": "f16", "bfloat16": "bf16",
    "e4m3": "fp8_e4m3", "float8_e4m3fn": "fp8_e4m3",
    "e5m2": "fp8_e5m2", "float8_e5m2": "fp8_e5m2",
    "s8": "int8", "i8": "int8", "fp4": "nvfp4",
}


def tolerance(dtype: str, *, reduction_length: int = 1) -> Tolerance:
    """Return the storage/accumulation comparison policy for ``dtype``.

    f32 accumulation grows a little with a very long dot/reduction; this keeps
    the policy scale-aware while retaining a non-zero near-zero absolute budget.
    """
    name = _ALIASES.get(dtype.lower(), dtype.lower())
    try:
        base = _BASE[name]
    except KeyError as exc:
        raise ValueError(f"no NVIDIA numerical policy for dtype {dtype!r}") from exc
    if base.exact:
        return base
    scale = max(1.0, float(reduction_length) / 256.0)
    return Tolerance(base.atol * scale, base.rtol * scale)


def assert_matches(actual: Any, expected: Any, dtype: str, *,
                   reduction_length: int = 1) -> None:
    """Compare with the declared budget, preserving NaN/Inf semantics."""
    policy = tolerance(dtype, reduction_length=reduction_length)
    if policy.exact:
        np.testing.assert_array_equal(actual, expected)
        return
    np.testing.assert_allclose(actual, expected, atol=policy.atol,
                               rtol=policy.rtol, equal_nan=True)
