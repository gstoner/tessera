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


@dataclass(frozen=True)
class DynamicReductionContract:
    """Runtime dimensions for one contiguous last-axis reduction launch."""

    outer: int
    axis_extent: int
    output_shape: tuple[int, ...]


@dataclass(frozen=True)
class DynamicSoftmaxContract:
    """Runtime dimensions for one contiguous last-axis softmax launch."""

    outer: int
    axis_extent: int
    output_shape: tuple[int, ...]


@dataclass(frozen=True)
class DynamicAttentionContract:
    """Runtime dimensions for one dense/streaming attention launch."""

    batch_heads: int
    query_extent: int
    key_extent: int
    query_key_width: int
    value_width: int
    output_shape: tuple[int, ...]


@dataclass(frozen=True)
class DynamicKVCacheContract:
    """Runtime dimensions for one growing KV-cache movement launch."""

    max_sequence: int
    row_extent: int
    output_shape: tuple[int, ...]


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


def guard_dynamic_last_axis_reduction(
    value: Any, *, keepdims: bool = False
) -> DynamicReductionContract:
    """Validate a dynamic last-axis reduction before entering its native ABI."""
    import numpy as np

    array = np.asarray(value)
    if array.ndim < 1:
        raise DynamicShapeGuardError(
            "dynamic last-axis reduction requires rank >= 1"
        )
    shape = tuple(int(dim) for dim in array.shape)
    if any(dim <= 0 for dim in shape):
        raise DynamicShapeGuardError(
            "dynamic last-axis reduction dimensions must be positive"
        )
    axis_extent = shape[-1]
    outer = 1
    for dim in shape[:-1]:
        outer *= dim
    signed_i64_max = 9_223_372_036_854_775_807
    if outer > signed_i64_max or axis_extent > signed_i64_max:
        raise DynamicShapeGuardError(
            "dynamic last-axis reduction dimensions must fit the signed i64 launch ABI"
        )
    output_shape = shape[:-1] + ((1,) if keepdims else ())
    return DynamicReductionContract(outer, axis_extent, output_shape)


def guard_dynamic_last_axis_softmax(value: Any) -> DynamicSoftmaxContract:
    """Validate a dynamic last-axis softmax before entering its native ABI."""
    reduction = guard_dynamic_last_axis_reduction(value, keepdims=False)
    import numpy as np

    output_shape = tuple(int(dim) for dim in np.asarray(value).shape)
    return DynamicSoftmaxContract(
        reduction.outer, reduction.axis_extent, output_shape
    )


def guard_dynamic_attention(
    query: Any, key: Any, value: Any
) -> DynamicAttentionContract:
    """Validate runtime-sized ``[..., S, D]`` attention operands.

    Query heads may be a divisible multiple of KV heads (GQA/MQA).  All launch
    dimensions use signed i64 in the native x86 ABI.
    """
    import numpy as np

    q = np.asarray(query)
    k = np.asarray(key)
    v = np.asarray(value)
    if min(q.ndim, k.ndim, v.ndim) < 2:
        raise DynamicShapeGuardError(
            "dynamic attention requires rank >= 2 Q/K/V"
        )
    if q.ndim != k.ndim or k.ndim != v.ndim:
        raise DynamicShapeGuardError(
            "dynamic attention requires equal Q/K/V ranks"
        )
    if any(int(dim) <= 0 for array in (q, k, v) for dim in array.shape):
        raise DynamicShapeGuardError(
            "dynamic attention dimensions must be positive"
        )
    d = int(q.shape[-1])
    sq = int(q.shape[-2])
    sk = int(k.shape[-2])
    dv = int(v.shape[-1])
    if int(k.shape[-1]) != d:
        raise DynamicShapeGuardError(
            "dynamic attention Q/K widths must match"
        )
    if int(v.shape[-2]) != sk or v.shape[:-2] != k.shape[:-2]:
        raise DynamicShapeGuardError(
            "dynamic attention K/V leading dimensions and key extents must match"
        )
    if q.shape[:-2] != k.shape[:-2]:
        q_heads = int(q.shape[-3]) if q.ndim >= 3 else 1
        kv_heads = int(k.shape[-3]) if k.ndim >= 3 else 1
        if (
            q.shape[:-3] != k.shape[:-3]
            or kv_heads <= 0
            or q_heads % kv_heads != 0
        ):
            raise DynamicShapeGuardError(
                "dynamic attention requires matching leading dimensions or "
                "divisible GQA query/KV heads"
            )
    batch_heads = 1
    for dim in q.shape[:-2]:
        batch_heads *= int(dim)
    signed_i64_max = 9_223_372_036_854_775_807
    dimensions = (batch_heads, sq, sk, d, dv)
    if any(dim > signed_i64_max for dim in dimensions):
        raise DynamicShapeGuardError(
            "dynamic attention dimensions must fit the signed i64 launch ABI"
        )
    return DynamicAttentionContract(
        batch_heads, sq, sk, d, dv, tuple(q.shape[:-1]) + (dv,)
    )


def guard_dynamic_kv_cache(
    cache: Any,
    *,
    rows: Any = None,
    start: int = 0,
    end: int | None = None,
    current_sequence: int | None = None,
    limit: int | None = None,
) -> DynamicKVCacheContract:
    """Validate a runtime-sized KV-cache movement call before native entry."""
    import numpy as np

    array = np.asarray(cache)
    if array.ndim < 2:
        raise DynamicShapeGuardError(
            "dynamic KV-cache requires shape (max_sequence, ...)"
        )
    shape = tuple(int(dim) for dim in array.shape)
    if any(dim <= 0 for dim in shape):
        raise DynamicShapeGuardError(
            "dynamic KV-cache dimensions must be positive"
        )
    max_sequence = shape[0]
    row_extent = 1
    for dim in shape[1:]:
        row_extent *= dim
    signed_i64_max = 9_223_372_036_854_775_807
    if max_sequence > signed_i64_max or row_extent > signed_i64_max:
        raise DynamicShapeGuardError(
            "dynamic KV-cache dimensions must fit the signed i64 launch ABI"
        )
    if rows is not None:
        appended = np.asarray(rows)
        if appended.ndim != array.ndim or appended.shape[1:] != array.shape[1:]:
            raise DynamicShapeGuardError(
                "dynamic KV-cache appended rows must match the cache tail shape"
            )
        count = int(appended.shape[0])
        if start < 0 or start + count > max_sequence:
            raise DynamicShapeGuardError(
                "dynamic KV-cache append range is out of bounds"
            )
    if end is not None and not (0 <= start <= end <= max_sequence):
        raise DynamicShapeGuardError(
            "dynamic KV-cache read range is out of bounds"
        )
    if current_sequence is not None and not (
        0 <= current_sequence <= max_sequence
    ):
        raise DynamicShapeGuardError(
            "dynamic KV-cache current sequence is out of bounds"
        )
    if limit is not None and limit < 0:
        raise DynamicShapeGuardError(
            "dynamic KV-cache prune limit must be non-negative"
        )
    output_shape = shape
    if end is not None:
        output_shape = (end - start,) + shape[1:]
    return DynamicKVCacheContract(max_sequence, row_extent, output_shape)


__all__ = [
    "DynamicShapeGuardError",
    "DynamicReductionContract",
    "DynamicSoftmaxContract",
    "DynamicAttentionContract",
    "DynamicKVCacheContract",
    "ExecutableLayout",
    "LayoutOrder",
    "guard_dynamic_matmul",
    "guard_dynamic_last_axis_reduction",
    "guard_dynamic_last_axis_softmax",
    "guard_dynamic_attention",
    "guard_dynamic_kv_cache",
    "materialize_layout",
    "materialize_layouts",
]
