"""Apple GPU batched matmul (bmm) — Tier-2 keystone (2026-05-29).

Batched / rank-3+ matmul via the MetalPerformanceShadersGraph bmm lane:
  * A [..., M, K] @ B [..., K, N] with matching leading (batch) dims,
  * a shared / broadcast B operand ([K, N] or leading dims all 1) — the
    projection and GQA KV-sharing shape,
  * f32 / f16 native (fp32 compute), bf16 host-upcast.

This unblocks the Tier-2 attention family (per-head attention = batched matmul)
and the projection ops. See docs/apple_gpu_tier2_tier3_plan.md.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import tessera as ts
from tessera import runtime as R

DARWIN = sys.platform == "darwin"


def _dispatch(a, b, op="tessera.matmul"):
    return np.asarray(R._apple_gpu_dispatch_matmul(op, [a, b], np))


def test_bmm_rank3_batched_f32():
    rng = np.random.RandomState(0)
    a = rng.randn(4, 8, 16).astype(np.float32)
    b = rng.randn(4, 16, 32).astype(np.float32)
    out = _dispatch(a, b)
    assert out.shape == (4, 8, 32)
    np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-4)


def test_bmm_rank4_folds_to_batch():
    rng = np.random.RandomState(1)
    a = rng.randn(2, 3, 8, 16).astype(np.float32)
    b = rng.randn(2, 3, 16, 8).astype(np.float32)
    out = _dispatch(a, b, op="tessera.batched_gemm")
    assert out.shape == (2, 3, 8, 8)
    np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-4)


def test_bmm_broadcast_shared_weight():
    # [B, S, D] @ [D, N] — a single weight shared across the batch (projection).
    rng = np.random.RandomState(2)
    x = rng.randn(4, 10, 16).astype(np.float32)
    w = rng.randn(16, 32).astype(np.float32)
    out = _dispatch(x, w)
    assert out.shape == (4, 10, 32)
    np.testing.assert_allclose(out, x @ w, rtol=1e-5, atol=1e-4)


def test_bmm_broadcast_leading_one():
    # [B, M, K] @ [1, K, N] — explicit broadcast batch dim.
    rng = np.random.RandomState(3)
    a = rng.randn(5, 8, 12).astype(np.float32)
    b = rng.randn(1, 12, 7).astype(np.float32)
    out = _dispatch(a, b)
    assert out.shape == (5, 8, 7)
    np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-4)


def test_bmm_f16():
    rng = np.random.RandomState(4)
    a = (rng.randn(4, 8, 16) * 0.3).astype(np.float16)
    b = (rng.randn(4, 16, 32) * 0.3).astype(np.float16)
    out = _dispatch(a, b)
    assert out.dtype == np.float16
    ref = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=5e-2, atol=5e-2)


def test_bmm_bf16():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    bf16 = ml_dtypes.bfloat16
    rng = np.random.RandomState(5)
    a = (rng.randn(4, 8, 16) * 0.3).astype(bf16)
    b = (rng.randn(4, 16, 32) * 0.3).astype(bf16)
    out = _dispatch(a, b)
    assert out.dtype == bf16
    ref = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=8e-2, atol=8e-2)


def test_bmm_per_head_attention_scores():
    # The attention building block: per-head Q @ K^T (batch dim = heads).
    rng = np.random.RandomState(6)
    H, T, D = 8, 12, 16
    q = rng.randn(H, T, D).astype(np.float32)
    k = rng.randn(H, T, D).astype(np.float32)
    kt = np.ascontiguousarray(k.transpose(0, 2, 1))  # [H, D, T]
    out = _dispatch(q, kt)
    assert out.shape == (H, T, T)
    np.testing.assert_allclose(out, q @ kt, rtol=1e-5, atol=1e-4)


def test_bmm_symbols_exported():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_bmm_f32")
    assert hasattr(rt, "tessera_apple_gpu_bmm_f16")
    assert R._apple_gpu_bmm_f32() is not None
    assert R._apple_gpu_bmm_f16() is not None


# ── on-device dispatch gate via @jit ────────────────────────────────────────
@ts.jit(target="apple_gpu")
def _jit_bmm(a, b):
    return ts.ops.matmul(a, b)


def test_jit_rank3_matmul_runtime_executable():
    a = np.zeros((4, 8, 16), dtype=np.float32)
    b = np.zeros((4, 16, 32), dtype=np.float32)
    _jit_bmm(a, b)
    meta = _jit_bmm.runtime_artifact().metadata
    assert meta["compiler_path"] == "apple_gpu_mps"
    assert meta["runtime_status"] == "ready"
    assert meta["execution_mode"] in ("metal_runtime", "metal_artifact")


@pytest.mark.skipif(not DARWIN, reason="metal_runtime dispatch is Darwin-only")
def test_jit_rank3_matmul_metal_runtime_on_darwin():
    rng = np.random.RandomState(7)
    a = rng.randn(4, 8, 16).astype(np.float32)
    b = rng.randn(4, 16, 32).astype(np.float32)
    out = np.asarray(_jit_bmm(a, b))
    np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-4)
    assert _jit_bmm.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
