import numpy as np

import tessera as ts
from tessera import runtime as R
from tessera.compiler import driver as _driver
from tessera.compiler.apple_gpu_envelope import lane_for
from tessera.nn import varlen as V


def test_composite_helpers_in_envelope():
    for op in (
        "tessera.memory_index_score",
        "tessera.msa_index_scores",
        "tessera.varlen_sdpa",
        "tessera.score_combine",
    ):
        assert op in _driver._APPLE_GPU_RUNTIME_OPS
        assert op in R._APPLE_GPU_RUNTIME_OPS
        assert lane_for(op) == "composite_helper"


def test_score_combine_composite_helper_matches_reference():
    base = np.arange(12, dtype=np.float32).reshape(3, 4) / 7.0
    delta = np.ones_like(base) * 0.25
    out = R._apple_gpu_dispatch_composite_helper(
        "tessera.score_combine", [base, delta], {"gamma": 1.5}, np)
    np.testing.assert_allclose(out, ts.ops.score_combine(base, delta, gamma=1.5))


def test_memory_index_score_composite_helper_matches_reference():
    rng = np.random.default_rng(7)
    keys = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    query = rng.standard_normal((2, 3, 6, 5)).astype(np.float32)
    out = R._apple_gpu_dispatch_composite_helper(
        "tessera.memory_index_score", [keys, query], {}, np)
    np.testing.assert_allclose(out, ts.ops.memory_index_score(keys, query), atol=1e-6)


def test_msa_index_scores_composite_helper_matches_reference():
    rng = np.random.default_rng(11)
    q = rng.standard_normal((2, 4, 5, 6)).astype(np.float32)
    k = rng.standard_normal((2, 2, 9, 6)).astype(np.float32)
    out = R._apple_gpu_dispatch_composite_helper(
        "tessera.msa_index_scores", [q, k], {"block_size": 4}, np)
    np.testing.assert_allclose(out, ts.ops.msa_index_scores(q, k, block_size=4), atol=1e-6)


def test_varlen_sdpa_composite_helper_matches_reference():
    rng = np.random.default_rng(13)
    q = rng.standard_normal((2, 5, 4)).astype(np.float32)
    k = rng.standard_normal((2, 7, 4)).astype(np.float32)
    v = rng.standard_normal((2, 7, 4)).astype(np.float32)
    cu_q = V.cu_seqlens_from_lengths([2, 3])
    cu_k = V.cu_seqlens_from_lengths([3, 4])
    out = R._apple_gpu_dispatch_composite_helper(
        "tessera.varlen_sdpa", [q, k, v, cu_q, cu_k], {"causal": True}, np)
    ref = ts.ops.varlen_sdpa(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, causal=True)
    np.testing.assert_allclose(out, ref, atol=1e-5)
