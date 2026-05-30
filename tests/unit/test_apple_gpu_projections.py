"""Apple GPU Tier-2 projection ops (2026-05-29).

`linear_general` (axis-flexible x @ W (+ bias)) and `qkv_projection`
(x @ W_qkv then split-3) now route through the GPU matmul / bmm lane on
`@jit(target="apple_gpu")`. The batched (rank-3) projection uses the bmm
broadcast path (shared weight across the batch). See
docs/apple_gpu_tier2_tier3_plan.md.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import tessera as ts
from tessera import runtime as R

DARWIN = sys.platform == "darwin"


# ── direct dispatcher tests ─────────────────────────────────────────────────
def test_linear_general_rank2():
    rng = np.random.RandomState(0)
    x = rng.randn(4, 16).astype(np.float32)
    w = rng.randn(16, 32).astype(np.float32)
    out = np.asarray(R._apple_gpu_dispatch_linear_general([x, w], {}, np))
    np.testing.assert_allclose(out, x @ w, rtol=1e-5, atol=1e-4)


def test_linear_general_rank3_batched_broadcast_weight():
    # [B, S, D] @ [D, N] — the projection shape; bmm broadcasts the weight.
    rng = np.random.RandomState(1)
    x = rng.randn(4, 10, 16).astype(np.float32)
    w = rng.randn(16, 32).astype(np.float32)
    out = np.asarray(R._apple_gpu_dispatch_linear_general([x, w], {}, np))
    assert out.shape == (4, 10, 32)
    np.testing.assert_allclose(out, x @ w, rtol=1e-5, atol=1e-4)


def test_linear_general_with_bias():
    rng = np.random.RandomState(2)
    x = rng.randn(8, 16).astype(np.float32)
    w = rng.randn(16, 32).astype(np.float32)
    bias = rng.randn(32).astype(np.float32)
    out = np.asarray(R._apple_gpu_dispatch_linear_general([x, w, bias], {}, np))
    np.testing.assert_allclose(out, x @ w + bias, rtol=1e-5, atol=1e-4)


def test_linear_general_general_axis_falls_back_correctly():
    # Contract a non-last axis with a rank-2 weight — numpy tensordot path;
    # must still be numerically correct.
    rng = np.random.RandomState(3)
    x = rng.randn(4, 16, 5).astype(np.float32)  # contract axis 1
    w = rng.randn(16, 7).astype(np.float32)
    out = np.asarray(R._apple_gpu_dispatch_linear_general([x, w], {"axis": 1}, np))
    ref = np.tensordot(x, w, axes=((1,), (0,)))
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-4)


def test_qkv_projection_splits_three():
    rng = np.random.RandomState(4)
    x = rng.randn(4, 16).astype(np.float32)
    wqkv = rng.randn(16, 48).astype(np.float32)
    out = R._apple_gpu_dispatch_qkv_projection([x, wqkv], np)
    assert isinstance(out, tuple) and len(out) == 3
    q, k, v = (np.asarray(t) for t in out)
    rq, rk, rv = np.split(x @ wqkv, 3, axis=-1)
    np.testing.assert_allclose(q, rq, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(k, rk, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(v, rv, rtol=1e-5, atol=1e-4)


def test_qkv_projection_rank3():
    rng = np.random.RandomState(5)
    x = rng.randn(2, 8, 16).astype(np.float32)
    wqkv = rng.randn(16, 48).astype(np.float32)
    out = R._apple_gpu_dispatch_qkv_projection([x, wqkv], np)
    q, k, v = (np.asarray(t) for t in out)
    assert q.shape == (2, 8, 16)
    rq, rk, rv = np.split(x @ wqkv, 3, axis=-1)
    np.testing.assert_allclose(q, rq, rtol=1e-5, atol=1e-4)


# ── on-device dispatch gates via @jit ───────────────────────────────────────
@ts.jit(target="apple_gpu")
def _jit_linear_general(x, w):
    return ts.ops.linear_general(x, w)


@ts.jit(target="apple_gpu")
def _jit_qkv(x, w):
    return ts.ops.qkv_projection(x, w)


def test_jit_projections_runtime_executable():
    x = np.zeros((4, 16), dtype=np.float32)
    w = np.zeros((16, 32), dtype=np.float32)
    _jit_linear_general(x, w)
    meta = _jit_linear_general.runtime_artifact().metadata
    assert meta["compiler_path"] == "apple_gpu_mps"
    assert meta["runtime_status"] == "ready"
    assert meta["execution_mode"] in ("metal_runtime", "metal_artifact")


@pytest.mark.skipif(not DARWIN, reason="metal_runtime dispatch is Darwin-only")
def test_jit_linear_general_metal_runtime_on_darwin():
    rng = np.random.RandomState(6)
    x = rng.randn(4, 10, 16).astype(np.float32)
    w = rng.randn(16, 32).astype(np.float32)
    out = np.asarray(_jit_linear_general(x, w))
    np.testing.assert_allclose(out, x @ w, rtol=1e-5, atol=1e-4)
    assert _jit_linear_general.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


@pytest.mark.skipif(not DARWIN, reason="metal_runtime dispatch is Darwin-only")
def test_jit_qkv_projection_metal_runtime_on_darwin():
    rng = np.random.RandomState(7)
    x = rng.randn(4, 16).astype(np.float32)
    w = rng.randn(16, 48).astype(np.float32)
    out = _jit_qkv(x, w)
    assert isinstance(out, tuple) and len(out) == 3
    rq, rk, rv = np.split(x @ w, 3, axis=-1)
    np.testing.assert_allclose(np.asarray(out[0]), rq, rtol=1e-5, atol=1e-4)
    assert _jit_qkv.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
