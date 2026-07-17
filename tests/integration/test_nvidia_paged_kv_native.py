"""PagedKVState consumer through the compiler-emitted NVIDIA Flash lane."""

from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import nvidia_mma_runtime_available
from tessera.cache import KVCacheHandle, paged_attention
from tessera.compiler.evaluator import paged_kv_native_equivalence


def _case(seed=3):
    rng = np.random.default_rng(seed)
    h = KVCacheHandle(num_heads=4, head_dim=32, max_seq=72, page_size=8)
    h.append(rng.standard_normal((48, 4, 32)).astype(np.float32),
             rng.standard_normal((48, 4, 32)).astype(np.float32))
    q = rng.standard_normal((4, 3, 32)).astype(np.float32)
    return h, q


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and a live NVIDIA GPU")
@pytest.mark.hardware_nvidia
def test_nvidia_paged_consumer_matches_reference_and_provenance():
    from tessera.compiler.emit import nvidia_cuda
    nvidia_cuda._nvidia_paged_attention_route_cache.clear()
    nvidia_cuda._nvidia_paged_attention_route_evidence.clear()
    state, q = _case()
    ref = paged_attention(q, state, backend="reference")
    out, execution = paged_attention(
        q, state, backend="nvidia", return_execution=True)
    assert execution == "native_gpu"
    np.testing.assert_allclose(out, ref, rtol=2e-5, atol=2e-5)
    evidence = next(iter(
        nvidia_cuda._nvidia_paged_attention_route_evidence.values()))
    assert set(evidence) == {"fused", "staged"}
    assert all(value > 0 for value in evidence.values())


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and a live NVIDIA GPU")
@pytest.mark.hardware_nvidia
def test_nvidia_paged_native_equivalence_oracle():
    state, q = _case(7)
    verdict = paged_kv_native_equivalence(
        state, q, backend="nvidia", rtol=2e-5, atol=2e-5)
    assert verdict.relation == "equivalent", verdict.detail
    assert "nvidia:native_gpu" in verdict.paths


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and a live NVIDIA GPU")
@pytest.mark.hardware_nvidia
def test_fused_paged_attention_honors_remap_and_causal_offset():
    from tessera.cache.paged_kv import _reference_attention
    from tessera.compiler.emit.nvidia_cuda import run_paged_attention_resident_f32

    rng = np.random.default_rng(1207)
    pages, page_size, heads, dim, tokens, q_len = 4, 4, 2, 16, 13, 3
    dense_k = (rng.standard_normal((pages * page_size, heads, dim)) * .1).astype(np.float32)
    dense_v = (rng.standard_normal(dense_k.shape) * .1).astype(np.float32)
    table = np.array([2, 0, 3, 1], np.int32)
    k_pages = np.empty((pages, page_size, heads, dim), np.float32)
    v_pages = np.empty_like(k_pages)
    for logical, physical in enumerate(table):
        begin = logical * page_size
        k_pages[physical] = dense_k[begin:begin + page_size]
        v_pages[physical] = dense_v[begin:begin + page_size]
    q = (rng.standard_normal((heads, q_len, dim)) * .1).astype(np.float32)
    idx = np.arange(tokens, dtype=np.int64)
    batch_medians = []
    actual, latency = run_paged_attention_resident_f32(
        q, k_pages, v_pages, table, idx, scale=dim ** -.5,
        causal=True, route="fused", device_batches=3,
        device_batch_medians=batch_medians)
    expected = _reference_attention(
        q, np.transpose(dense_k[:tokens], (1, 0, 2)),
        np.transpose(dense_v[:tokens], (1, 0, 2)), dim ** -.5, True)
    assert latency > 0
    assert len(batch_medians) == 3
    assert all(sample > 0 for sample in batch_medians)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("tokens,q_len", [
    (1, 1), (3, 1), (4, 3), (5, 3), (7, 3), (8, 3), (9, 3), (13, 3),
])
@pytest.mark.skipif(not nvidia_mma_runtime_available(),
                    reason="requires nvcc and a live NVIDIA GPU")
@pytest.mark.hardware_nvidia
def test_paged_routes_cover_permuted_pages_and_boundary_lengths(tokens, q_len):
    """Both routes consume logical token indices through the same remapped ABI."""
    from tessera.cache.paged_kv import _reference_attention
    from tessera.compiler.emit.nvidia_cuda import run_paged_attention_resident_f32

    rng = np.random.default_rng(1600 + tokens * 10 + q_len)
    page_size, heads, dim = 4, 2, 16
    logical_pages = (tokens + page_size - 1) // page_size
    physical_pages = max(4, logical_pages)
    table = np.roll(np.arange(physical_pages, dtype=np.int32), 1)
    dense_k = (rng.standard_normal(
        (physical_pages * page_size, heads, dim)) * .1).astype(np.float32)
    dense_v = (rng.standard_normal(dense_k.shape) * .1).astype(np.float32)
    k_pages = np.full(
        (physical_pages, page_size, heads, dim), np.nan, np.float32)
    v_pages = np.full_like(k_pages, np.nan)
    for logical, physical in enumerate(table):
        begin = logical * page_size
        k_pages[physical] = dense_k[begin:begin + page_size]
        v_pages[physical] = dense_v[begin:begin + page_size]
    # Non-monotonic logical indices prove that page lookup is not an identity
    # gather. The final Q positions retain the global causal offset T-Q.
    indices = np.arange(tokens, dtype=np.int64)
    if tokens > 2:
        indices[:-1] = indices[:-1][::-1]
    q = (rng.standard_normal((heads, min(q_len, tokens), dim)) * .1).astype(
        np.float32)
    selected_k = dense_k[indices]
    selected_v = dense_v[indices]
    expected = _reference_attention(
        q, np.transpose(selected_k, (1, 0, 2)),
        np.transpose(selected_v, (1, 0, 2)), dim ** -.5, True)
    for route in ("fused", "staged"):
        actual, latency = run_paged_attention_resident_f32(
            q, k_pages, v_pages, table, indices, scale=dim ** -.5,
            causal=True, route=route)
        assert latency > 0
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not nvidia_mma_runtime_available(),
                    reason="requires nvcc and a live NVIDIA GPU")
@pytest.mark.hardware_nvidia
def test_paged_routes_survive_physical_page_remap_and_reuse():
    from tessera.cache.paged_kv import _reference_attention
    from tessera.compiler.emit.nvidia_cuda import run_paged_attention_resident_f32

    rng = np.random.default_rng(1913)
    page_size, heads, dim, tokens = 4, 2, 16, 11
    logical_k = (rng.standard_normal((3, page_size, heads, dim)) * .1).astype(np.float32)
    logical_v = (rng.standard_normal(logical_k.shape) * .1).astype(np.float32)
    q = (rng.standard_normal((heads, 2, dim)) * .1).astype(np.float32)
    indices = np.arange(tokens, dtype=np.int64)
    expected = _reference_attention(
        q, np.transpose(logical_k.reshape(-1, heads, dim)[:tokens], (1, 0, 2)),
        np.transpose(logical_v.reshape(-1, heads, dim)[:tokens], (1, 0, 2)),
        dim ** -.5, True)
    # The second mapping reuses physical pages for different logical pages.
    for table in (np.array([2, 0, 3], np.int32),
                  np.array([1, 3, 0], np.int32)):
        k = np.full((4, page_size, heads, dim), np.nan, np.float32)
        v = np.full_like(k, np.nan)
        for logical, physical in enumerate(table):
            k[physical], v[physical] = logical_k[logical], logical_v[logical]
        for route in ("fused", "staged"):
            actual, _ = run_paged_attention_resident_f32(
                q, k, v, table, indices, scale=dim ** -.5,
                causal=True, route=route)
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
