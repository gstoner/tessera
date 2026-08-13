"""Faithful host implementation of Block Attention Residuals.

This is Phase 2 of ``BLOCK-ATTNRES-1``.  It composes the public Phase-1
statistics operations and intentionally owns no Graph or target lowering.
Sublayer callables must *exclude* an internal residual; this module owns the
intra-block partial-sum recurrence.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from tessera import ops

Array = np.ndarray
Sublayer = Callable[[Array], Any]
Norm = Callable[[Array], Any]


def balanced_block_sizes(num_layers: int, num_blocks: int) -> tuple[int, ...]:
    """Partition layers into exactly ``num_blocks`` nonempty balanced blocks."""
    layers = int(num_layers)
    blocks = int(num_blocks)
    if layers <= 0 or blocks <= 0 or blocks > layers:
        raise ValueError("require 1 <= num_blocks <= num_layers")
    quotient, remainder = divmod(layers, blocks)
    return tuple(quotient + int(index < remainder) for index in range(blocks))


def zero_queries(
    num_layers: int,
    width: int,
) -> tuple[Array, ...]:
    """Create the required zero-initialized per-layer and output queries."""
    layers = int(num_layers)
    dimension = int(width)
    if layers <= 0 or dimension <= 0:
        raise ValueError("num_layers and width must be positive")
    return tuple(
        np.zeros((dimension,), dtype=np.float32) for _ in range(layers + 1)
    )


def _as_storage(value: Any, *, shape: tuple[int, ...], dtype: np.dtype) -> Array:
    if hasattr(value, "_data"):
        value = value._data
    result = np.asarray(value)
    if result.shape != shape:
        raise ValueError(
            f"sublayer output shape {result.shape} must match state shape {shape}"
        )
    if not np.issubdtype(result.dtype, np.floating):
        raise ValueError("Block AttnRes state must use a floating-point dtype")
    if not np.isfinite(result).all():
        raise ValueError("sublayer output must contain only finite values")
    return np.asarray(result, dtype=dtype)


def _validate_program(
    x: Any,
    sublayers: Sequence[Sublayer],
    queries: Sequence[Any],
    block_sizes: Sequence[int],
    norms: Sequence[Norm] | None,
) -> tuple[Array, tuple[Array, ...], tuple[int, ...], tuple[Norm, ...]]:
    state = np.asarray(x._data if hasattr(x, "_data") else x)
    if state.ndim < 1 or state.shape[-1] <= 0:
        raise ValueError("x must have shape [..., d] with positive d")
    if not np.issubdtype(state.dtype, np.floating):
        raise ValueError("x must use floating-point storage")
    if not np.isfinite(state).all():
        raise ValueError("x must contain only finite values")
    layer_count = len(sublayers)
    sizes = tuple(int(size) for size in block_sizes)
    if not sizes or any(size <= 0 for size in sizes) or sum(sizes) != layer_count:
        raise ValueError("block_sizes must be nonempty and partition all sublayers")
    if len(queries) != layer_count + 1:
        raise ValueError("queries must contain one query per sublayer plus output")
    query_values = tuple(np.asarray(query, dtype=np.float32) for query in queries)
    for query in query_values:
        if query.shape != (state.shape[-1],) or not np.isfinite(query).all():
            raise ValueError("every query must be finite with shape [d]")
    if norms is None:
        norm_values: tuple[Norm, ...] = tuple(lambda value: value for _ in sublayers)
    else:
        norm_values = tuple(norms)
        if len(norm_values) != layer_count:
            raise ValueError("norms must contain one callable per sublayer")
    return state, query_values, sizes, norm_values


def depth_attn(query: Any, sources: Any, *, eps: float = 1.0e-6) -> Array:
    """Compute depth attention through the canonical Graph primitive."""
    return ops.depth_attn(query, sources, eps=eps)


def block_attnres_forward(
    x: Any,
    sublayers: Sequence[Sublayer],
    queries: Sequence[Any],
    *,
    num_blocks: int | None = None,
    block_sizes: Sequence[int] | None = None,
    norms: Sequence[Norm] | None = None,
    output_norm: Norm | None = None,
    eps: float = 1.0e-6,
) -> Array:
    """Execute the direct Block AttnRes recurrence from plan algorithm A2."""
    if (num_blocks is None) == (block_sizes is None):
        raise ValueError("provide exactly one of num_blocks or block_sizes")
    if block_sizes is None:
        assert num_blocks is not None
        sizes = balanced_block_sizes(len(sublayers), num_blocks)
    else:
        sizes = tuple(block_sizes)
    state, query_values, sizes, norm_values = _validate_program(
        x, sublayers, queries, sizes, norms
    )
    storage_dtype = state.dtype
    blocks: list[Array] = [state]
    layer = 0
    for size in sizes:
        partial: Array | None = None
        for _ in range(size):
            # GAP-2: no zero-valued partial placeholder for a block's first op.
            sources = blocks if partial is None else [*blocks, partial]
            hidden = depth_attn(query_values[layer], np.stack(sources), eps=eps)
            hidden = np.asarray(hidden, dtype=storage_dtype)
            transformed = sublayers[layer](norm_values[layer](hidden))
            value = _as_storage(
                transformed, shape=state.shape, dtype=storage_dtype
            )
            partial = value if partial is None else partial + value
            layer += 1
        assert partial is not None
        blocks.append(np.asarray(partial, dtype=storage_dtype))
    output = depth_attn(query_values[-1], np.stack(blocks), eps=eps)
    output = np.asarray(output, dtype=storage_dtype)
    if output_norm is not None:
        output = _as_storage(
            output_norm(output), shape=state.shape, dtype=storage_dtype
        )
    return output


def block_attnres_two_phase(
    x: Any,
    sublayers: Sequence[Sublayer],
    queries: Sequence[Any],
    *,
    num_blocks: int | None = None,
    block_sizes: Sequence[int] | None = None,
    norms: Sequence[Norm] | None = None,
    output_norm: Norm | None = None,
    eps: float = 1.0e-6,
) -> Array:
    """Execute the exact query-hoisted two-phase recurrence from plan A4."""
    if (num_blocks is None) == (block_sizes is None):
        raise ValueError("provide exactly one of num_blocks or block_sizes")
    if block_sizes is None:
        assert num_blocks is not None
        sizes = balanced_block_sizes(len(sublayers), num_blocks)
    else:
        sizes = tuple(block_sizes)
    state, query_values, sizes, norm_values = _validate_program(
        x, sublayers, queries, sizes, norms
    )
    storage_dtype = state.dtype
    blocks: list[Array] = [state]
    layer = 0
    for size in sizes:
        base = layer
        completed = np.stack(blocks)
        phase_one = tuple(
            ops.attn_with_stats(query_values[base + index], completed, eps=eps)
            for index in range(size)
        )
        partial: Array | None = None
        for index in range(size):
            stats = phase_one[index]
            if partial is not None:
                partial_stats = ops.attn_with_stats(
                    query_values[layer], partial[None, ...], eps=eps
                )
                stats = ops.softmax_merge(*stats, *partial_stats)
            hidden = ops.softmax_finalize(stats[0], stats[2])
            hidden = np.asarray(hidden, dtype=storage_dtype)
            transformed = sublayers[layer](norm_values[layer](hidden))
            value = _as_storage(
                transformed, shape=state.shape, dtype=storage_dtype
            )
            partial = value if partial is None else partial + value
            layer += 1
        assert partial is not None
        blocks.append(np.asarray(partial, dtype=storage_dtype))
    output = depth_attn(query_values[-1], np.stack(blocks), eps=eps)
    output = np.asarray(output, dtype=storage_dtype)
    if output_norm is not None:
        output = _as_storage(
            output_norm(output), shape=state.shape, dtype=storage_dtype
        )
    return output


__all__ = [
    "balanced_block_sizes",
    "block_attnres_forward",
    "block_attnres_two_phase",
    "depth_attn",
    "zero_queries",
]
