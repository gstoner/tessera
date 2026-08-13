"""Host-reference primitives for Block Attention Residuals.

This module is the Phase-1 numerical authority from
``BLOCK_ATTNRES_ROCM_PLAN.md``.  It deliberately contains no target dispatch:
Graph/Schedule/Tile operations and architecture-owned packages land in later
phases.  All softmax statistics are produced and merged in fp32.
"""

from __future__ import annotations

from typing import Any

import numpy as np


BLOCK_ATTNRES_NUMERIC_POLICY = {
    "storage": "fp32",
    "softmax": "fp32",
    "accum": "fp32",
}


def _array(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "_data"):
        value = value._data
    result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def attn_with_stats(
    query: Any,
    sources: Any,
    *,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute an unnormalized, max-shifted depth-attention state.

    ``query`` has shape ``[d]`` and ``sources`` has shape ``[m, ..., d]``.
    Source axis zero is reduced independently for every intervening batch or
    token position.  The result is ``(o[..., d], m[...], l[...])`` and the
    finalized attention value is exactly ``o / l[..., None]``.
    """
    q = _array(query, name="query")
    values = _array(sources, name="sources")
    epsilon = float(eps)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("eps must be finite and positive")
    if q.ndim != 1:
        raise ValueError(f"query must have shape [d], got {q.shape}")
    if values.ndim < 2 or values.shape[0] == 0:
        raise ValueError(
            f"sources must have non-empty shape [m, ..., d], got {values.shape}"
        )
    if values.shape[-1] != q.shape[0]:
        raise ValueError(
            "query width must match the final source dimension; got "
            f"{q.shape[0]} and {values.shape[-1]}"
        )

    rms = np.sqrt(
        np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
        + np.float32(epsilon)
    )
    keys = values / rms
    logits = np.einsum("d,m...d->m...", q, keys, dtype=np.float32)
    maximum = np.max(logits, axis=0)
    weights = np.exp(logits - maximum[None, ...]).astype(np.float32)
    normalizer = np.sum(weights, axis=0, dtype=np.float32)
    output = np.einsum("m...,m...d->...d", weights, values, dtype=np.float32)
    return (
        np.asarray(output, dtype=np.float32),
        np.asarray(maximum, dtype=np.float32),
        np.asarray(normalizer, dtype=np.float32),
    )


def _validate_state(
    output: Any,
    maximum: Any,
    normalizer: Any,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = _array(output, name=f"{name}.output")
    max_value = _array(maximum, name=f"{name}.maximum")
    norm = _array(normalizer, name=f"{name}.normalizer")
    if out.ndim < 1 or out.shape[:-1] != max_value.shape:
        raise ValueError(
            f"{name} output prefix {out.shape[:-1]} must match maximum "
            f"shape {max_value.shape}"
        )
    if norm.shape != max_value.shape:
        raise ValueError(
            f"{name} normalizer shape {norm.shape} must match maximum "
            f"shape {max_value.shape}"
        )
    if np.any(norm <= 0.0):
        raise ValueError(f"{name} normalizer must be strictly positive")
    return out, max_value, norm


def softmax_merge(
    output_a: Any,
    maximum_a: Any,
    normalizer_a: Any,
    output_b: Any,
    maximum_b: Any,
    normalizer_b: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Associatively merge two max-shifted attention statistic states."""
    oa, ma, la = _validate_state(
        output_a, maximum_a, normalizer_a, name="state_a"
    )
    ob, mb, lb = _validate_state(
        output_b, maximum_b, normalizer_b, name="state_b"
    )
    if oa.shape != ob.shape or ma.shape != mb.shape:
        raise ValueError("softmax_merge states must have identical shapes")
    maximum = np.maximum(ma, mb)
    scale_a = np.exp(ma - maximum).astype(np.float32)
    scale_b = np.exp(mb - maximum).astype(np.float32)
    output = scale_a[..., None] * oa + scale_b[..., None] * ob
    normalizer = scale_a * la + scale_b * lb
    return (
        np.asarray(output, dtype=np.float32),
        np.asarray(maximum, dtype=np.float32),
        np.asarray(normalizer, dtype=np.float32),
    )


def softmax_finalize(output: Any, normalizer: Any) -> np.ndarray:
    """Finalize a max-shifted attention statistic state as ``output / l``."""
    out = _array(output, name="output")
    norm = _array(normalizer, name="normalizer")
    if out.ndim < 1 or out.shape[:-1] != norm.shape:
        raise ValueError(
            f"output prefix {out.shape[:-1]} must match normalizer shape "
            f"{norm.shape}"
        )
    if np.any(norm <= 0.0):
        raise ValueError("normalizer must be strictly positive")
    return np.asarray(out / norm[..., None], dtype=np.float32)


def depth_attn(
    query: Any,
    sources: Any,
    *,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """Canonical depth-attention Graph primitive.

    This intentionally decomposes through the statistics contract.  A future
    fused target implementation must compare against this exact boundary and
    may not reconstruct the operation from an unrelated backend recipe.
    """
    output, _, normalizer = attn_with_stats(query, sources, eps=eps)
    return softmax_finalize(output, normalizer)


def depth_attn_vjp(
    output_cotangent: Any,
    query: Any,
    sources: Any,
    *,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic reverse product for :func:`depth_attn`."""
    q = _array(query, name="query")
    values = _array(sources, name="sources")
    grad = _array(output_cotangent, name="output_cotangent")
    output = depth_attn(q, values, eps=eps)
    if grad.shape != output.shape:
        raise ValueError(
            f"output_cotangent shape {grad.shape} must match output {output.shape}"
        )

    epsilon = np.float32(eps)
    rms = np.sqrt(
        np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
        + epsilon
    )
    keys = values / rms
    logits = np.einsum("d,m...d->m...", q, keys, dtype=np.float32)
    logits -= np.max(logits, axis=0, keepdims=True)
    weights = np.exp(logits).astype(np.float32)
    weights /= np.sum(weights, axis=0, keepdims=True, dtype=np.float32)
    value_projection = np.sum(values * grad[None, ...], axis=-1)
    centered_projection = value_projection - np.sum(
        weights * value_projection, axis=0, keepdims=True, dtype=np.float32
    )
    score_cotangent = weights * centered_projection
    query_cotangent = np.sum(
        score_cotangent[..., None] * keys,
        axis=tuple(range(keys.ndim - 1)),
        dtype=np.float32,
    )
    key_query = np.einsum("m...d,d->m...", keys, q, dtype=np.float32)
    source_cotangent = weights[..., None] * grad[None, ...]
    source_cotangent += score_cotangent[..., None] * (
        q - keys * (key_query[..., None] / np.float32(values.shape[-1]))
    ) / rms
    return (
        np.asarray(query_cotangent, dtype=np.float32),
        np.asarray(source_cotangent, dtype=np.float32),
    )


def depth_attn_jvp(
    query: Any,
    sources: Any,
    query_tangent: Any,
    sources_tangent: Any,
    *,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the primal and exact directional product for depth attention."""
    q = _array(query, name="query")
    values = _array(sources, name="sources")
    dq = _array(query_tangent, name="query_tangent")
    dvalues = _array(sources_tangent, name="sources_tangent")
    if dq.shape != q.shape or dvalues.shape != values.shape:
        raise ValueError("depth_attn tangent shapes must match their primals")

    epsilon = np.float32(eps)
    rms = np.sqrt(
        np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
        + epsilon
    )
    keys = values / rms
    key_tangent = (
        dvalues
        - keys
        * np.mean(keys * dvalues, axis=-1, keepdims=True, dtype=np.float32)
    ) / rms
    logits = np.einsum("d,m...d->m...", q, keys, dtype=np.float32)
    weights = np.exp(logits - np.max(logits, axis=0, keepdims=True)).astype(
        np.float32
    )
    weights /= np.sum(weights, axis=0, keepdims=True, dtype=np.float32)
    score_tangent = np.einsum("d,m...d->m...", dq, keys, dtype=np.float32)
    score_tangent += np.einsum(
        "d,m...d->m...", q, key_tangent, dtype=np.float32
    )
    weight_tangent = weights * (
        score_tangent
        - np.sum(weights * score_tangent, axis=0, keepdims=True, dtype=np.float32)
    )
    primal = np.einsum("m...,m...d->...d", weights, values, dtype=np.float32)
    tangent = np.einsum(
        "m...,m...d->...d", weight_tangent, values, dtype=np.float32
    ) + np.einsum("m...,m...d->...d", weights, dvalues, dtype=np.float32)
    return np.asarray(primal, dtype=np.float32), np.asarray(tangent, dtype=np.float32)


__all__ = [
    "BLOCK_ATTNRES_NUMERIC_POLICY",
    "attn_with_stats",
    "depth_attn",
    "depth_attn_jvp",
    "depth_attn_vjp",
    "softmax_finalize",
    "softmax_merge",
]
