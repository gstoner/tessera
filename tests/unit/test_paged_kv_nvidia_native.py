"""PagedKVState consumer through the compiler-emitted NVIDIA Flash lane."""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from tessera.cache import KVCacheHandle, paged_attention
from tessera.compiler.evaluator import paged_kv_native_equivalence


def _live():
    if not (shutil.which("nvcc") or os.path.exists("/usr/local/cuda/bin/nvcc")):
        return False
    from tessera import runtime
    return runtime._nvidia_mma_runtime_available()


def _case(seed=3):
    rng = np.random.default_rng(seed)
    h = KVCacheHandle(num_heads=4, head_dim=32, max_seq=72, page_size=8)
    h.append(rng.standard_normal((48, 4, 32)).astype(np.float32),
             rng.standard_normal((48, 4, 32)).astype(np.float32))
    q = rng.standard_normal((4, 3, 32)).astype(np.float32)
    return h, q


@pytest.mark.skipif(not _live(), reason="requires nvcc and a live NVIDIA GPU")
def test_nvidia_paged_consumer_matches_reference_and_provenance():
    state, q = _case()
    ref = paged_attention(q, state, backend="reference")
    out, execution = paged_attention(
        q, state, backend="nvidia", return_execution=True)
    assert execution == "native_gpu"
    np.testing.assert_allclose(out, ref, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not _live(), reason="requires nvcc and a live NVIDIA GPU")
def test_nvidia_paged_native_equivalence_oracle():
    state, q = _case(7)
    verdict = paged_kv_native_equivalence(
        state, q, backend="nvidia", rtol=2e-5, atol=2e-5)
    assert verdict.relation == "equivalent", verdict.detail
    assert "nvidia:native_gpu" in verdict.paths


def test_nvidia_paged_fallback_cannot_earn_native_rung(monkeypatch):
    import tessera.cache.paged_kv as pk
    monkeypatch.setattr(
        pk, "_paged_attention_nvidia",
        lambda q, k, v, scale, causal:
            (pk._reference_attention(q, k, v, scale, causal), "reference"))
    state, q = _case(9)
    verdict = paged_kv_native_equivalence(state, q, backend="nvidia")
    assert verdict.relation == "inconclusive"
