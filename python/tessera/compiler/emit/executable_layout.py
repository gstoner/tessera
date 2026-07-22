"""Executable buffer-layout and dynamic-shape contracts for generic emitters.

CORE-COMPILER-2 turns layout metadata into a launch-time action.  A kernel
declares the physical order expected for each binding; the runner validates the
logical rank and materializes that order before entering native code.  The same
module owns the first guarded dynamic matmul envelope so a runtime-argument
kernel cannot receive a malformed or overflowing M/N/K tuple.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class LayoutOrder(Enum):
    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "col_major"


@dataclass(frozen=True)
class ExecutableLayout:
    binding: str
    order: LayoutOrder
    rank: int

    def __post_init__(self) -> None:
        if not self.binding:
            raise ValueError("executable layout binding must be non-empty")
        if self.rank < 0:
            raise ValueError("executable layout rank must be non-negative")


class DynamicShapeGuardError(ValueError):
    """A dynamic kernel call lies outside its declared runtime envelope."""


def materialize_layout(value: Any, contract: ExecutableLayout) -> Any:
    """Return an array in the contract's physical order.

    Import NumPy lazily so compiler metadata remains importable without the
    runtime dependency.  ``ascontiguousarray``/``asfortranarray`` are real
    copies when required and no-ops when the input already satisfies the ABI.
    """
    import numpy as np

    array = np.asarray(value)
    if array.ndim != contract.rank:
        raise ValueError(
            f"layout binding {contract.binding!r} requires rank "
            f"{contract.rank}, got {array.ndim}"
        )
    if contract.order is LayoutOrder.ROW_MAJOR:
        return np.ascontiguousarray(array)
    return np.asfortranarray(array)


def materialize_layouts(
    values: Mapping[str, Any], contracts: tuple[ExecutableLayout, ...]
) -> dict[str, Any]:
    result = dict(values)
    for contract in contracts:
        if contract.binding not in result:
            raise ValueError(
                f"missing executable layout binding {contract.binding!r}"
            )
        result[contract.binding] = materialize_layout(
            result[contract.binding], contract
        )
    return result


def guard_dynamic_matmul(
    a: Any,
    b: Any,
    *,
    bias: Any = None,
    residual: Any = None,
    require_bias: bool = False,
    require_residual: bool = False,
) -> tuple[int, int, int]:
    """Validate the first generic dynamic route and return ``(M, N, K)``.

    The emitted C/HIP/CUDA ABI uses signed 32-bit runtime dimensions.  Reject
    invalid ranks, empty/overflowing extents, contraction mismatches, and
    side-buffer shape mismatches before compilation or native launch.
    """
    import numpy as np

    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.ndim != 2 or bb.ndim != 2:
        raise DynamicShapeGuardError(
            f"dynamic matmul requires rank-2 A/B, got {aa.ndim}/{bb.ndim}"
        )
    m, k = (int(dim) for dim in aa.shape)
    kb, n = (int(dim) for dim in bb.shape)
    if min(m, n, k, kb) <= 0:
        raise DynamicShapeGuardError("dynamic matmul dimensions must be positive")
    if k != kb:
        raise DynamicShapeGuardError(
            f"dynamic matmul contracting dimensions differ: {k} vs {kb}"
        )
    if max(m, n, k) > 2_147_483_647:
        raise DynamicShapeGuardError(
            "dynamic matmul dimensions must fit the signed i32 launch ABI"
        )
    if require_bias and bias is None:
        raise DynamicShapeGuardError("dynamic fused matmul requires bias")
    if bias is not None and np.asarray(bias).shape != (n,):
        raise DynamicShapeGuardError(
            f"dynamic matmul bias must have shape ({n},), got "
            f"{np.asarray(bias).shape}"
        )
    if require_residual and residual is None:
        raise DynamicShapeGuardError("dynamic fused matmul requires residual")
    if residual is not None and np.asarray(residual).shape != (m, n):
        raise DynamicShapeGuardError(
            f"dynamic matmul residual must have shape ({m}, {n}), got "
            f"{np.asarray(residual).shape}"
        )
    return m, n, k


__all__ = [
    "DynamicShapeGuardError",
    "ExecutableLayout",
    "LayoutOrder",
    "guard_dynamic_matmul",
    "materialize_layout",
    "materialize_layouts",
]
