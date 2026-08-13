"""Phase-1 public reference contracts for Block Attention Residuals."""

from __future__ import annotations

import numpy as np
import pytest

import tessera


def _direct_attention(query: np.ndarray, sources: np.ndarray) -> np.ndarray:
    values = np.asarray(sources, dtype=np.float32)
    rms = np.sqrt(
        np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
        + np.float32(1.0e-6)
    )
    logits = np.einsum(
        "d,m...d->m...", np.asarray(query, np.float32), values / rms,
        dtype=np.float32,
    )
    weights = np.exp(logits - np.max(logits, axis=0, keepdims=True))
    weights /= np.sum(weights, axis=0, keepdims=True)
    return np.einsum("m...,m...d->...d", weights, values, dtype=np.float32)


def test_public_stats_finalize_matches_direct_batched_attention() -> None:
    rng = np.random.default_rng(10)
    query = rng.normal(size=(17,)).astype(np.float32)
    sources = rng.normal(size=(7, 2, 3, 17)).astype(np.float32)
    output, maximum, normalizer = tessera.ops.attn_with_stats(query, sources)
    actual = tessera.ops.softmax_finalize(output, normalizer)
    expected = _direct_attention(query, sources)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    assert output.dtype == maximum.dtype == normalizer.dtype == np.float32
    assert actual.dtype == np.float32


def test_merge_is_partition_invariant_associative_and_commutative() -> None:
    rng = np.random.default_rng(11)
    query = rng.normal(size=(13,)).astype(np.float32)
    sources = rng.normal(size=(9, 4, 13)).astype(np.float32)
    a = tessera.ops.attn_with_stats(query, sources[:2])
    b = tessera.ops.attn_with_stats(query, sources[2:6])
    c = tessera.ops.attn_with_stats(query, sources[6:])
    ab = tessera.ops.softmax_merge(*a, *b)
    ba = tessera.ops.softmax_merge(*b, *a)
    left = tessera.ops.softmax_merge(*ab, *c)
    bc = tessera.ops.softmax_merge(*b, *c)
    right = tessera.ops.softmax_merge(*a, *bc)
    whole = tessera.ops.attn_with_stats(query, sources)

    np.testing.assert_allclose(ab[0], ba[0], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(ab[1], ba[1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(ab[2], ba[2], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        tessera.ops.softmax_finalize(left[0], left[2]),
        tessera.ops.softmax_finalize(right[0], right[2]),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        tessera.ops.softmax_finalize(left[0], left[2]),
        tessera.ops.softmax_finalize(whole[0], whole[2]),
        rtol=2e-6,
        atol=2e-6,
    )


def test_block_attnres_ops_are_reference_only_registry_entries() -> None:
    for name in (
        "attn_with_stats", "depth_attn", "softmax_merge", "softmax_finalize"
    ):
        assert name in tessera.ops.registry.list()
        entry = tessera.ops.registry.get(name)
        assert entry is not None and entry.lowering is not None
        assert entry.runtime_kernel is None
        assert entry.lowering() == {"op": name, "status": "artifact_only"}


@pytest.mark.parametrize(
    "call,match",
    [
        (lambda: tessera.ops.attn_with_stats(np.ones(3), np.empty((0, 3))),
         "non-empty"),
        (lambda: tessera.ops.attn_with_stats(np.ones(4), np.ones((2, 3))),
         "width"),
        (lambda: tessera.ops.attn_with_stats(np.ones(3), np.ones((2, 3)), eps=0),
         "finite and positive"),
        (lambda: tessera.ops.softmax_finalize(np.ones((2, 3)), np.ones(3)),
         "prefix"),
    ],
)
def test_phase_one_contracts_fail_closed(call, match) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_depth_attn_analytic_products_match_decomposed_forward() -> None:
    from tessera._block_attnres_ops import depth_attn_jvp, depth_attn_vjp

    rng = np.random.default_rng(42)
    query = rng.normal(size=5).astype(np.float32)
    sources = rng.normal(size=(4, 2, 3, 5)).astype(np.float32)
    query_tangent = rng.normal(size=query.shape).astype(np.float32)
    sources_tangent = rng.normal(size=sources.shape).astype(np.float32)
    output_cotangent = rng.normal(size=(2, 3, 5)).astype(np.float32)

    primal, tangent = depth_attn_jvp(
        query, sources, query_tangent, sources_tangent
    )
    np.testing.assert_allclose(
        primal, tessera.ops.depth_attn(query, sources), rtol=2e-6, atol=2e-6
    )
    step = np.float32(2.0e-3)
    plus = tessera.ops.depth_attn(
        query + step * query_tangent,
        sources + step * sources_tangent,
    )
    minus = tessera.ops.depth_attn(
        query - step * query_tangent,
        sources - step * sources_tangent,
    )
    numerical_tangent = (plus - minus) / (2.0 * step)
    np.testing.assert_allclose(tangent, numerical_tangent, rtol=2e-3, atol=2e-3)

    query_cotangent, sources_cotangent = depth_attn_vjp(
        output_cotangent, query, sources
    )
    lhs = np.sum(output_cotangent * tangent, dtype=np.float64)
    rhs = np.sum(query_cotangent * query_tangent, dtype=np.float64)
    rhs += np.sum(sources_cotangent * sources_tangent, dtype=np.float64)
    np.testing.assert_allclose(lhs, rhs, rtol=2e-5, atol=2e-5)


def test_depth_attn_products_are_registered_with_public_autodiff() -> None:
    from tessera.autodiff.jvp import get_jvp
    from tessera.autodiff.vjp import get_vjp

    assert get_vjp("depth_attn") is not None
    assert get_jvp("depth_attn") is not None
