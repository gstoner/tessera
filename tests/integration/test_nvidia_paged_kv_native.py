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
    actual, latency = run_paged_attention_resident_f32(
        q, k_pages, v_pages, table, idx, scale=dim ** -.5,
        causal=True, route="fused")
    expected = _reference_attention(
        q, np.transpose(dense_k[:tokens], (1, 0, 2)),
        np.transpose(dense_v[:tokens], (1, 0, 2)), dim ** -.5, True)
    assert latency > 0
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
