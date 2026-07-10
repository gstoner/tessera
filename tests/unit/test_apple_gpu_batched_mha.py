"""Apple GPU batched multi-head attention via the Tier-2 ops (2026-05-29).

A full multi-head attention block composed entirely from the new Tier-2
GPU ops — no per-head Python loop (unlike test_apple_gpu_mla_e2e.py, which
loops over heads calling the rank-2 matmul_softmax_matmul kernel):

    Q, K, V = qkv_projection(x, W_qkv)        # GPU matmul / bmm
    scores  = bmm(Q_h, K_h^T) * scale         # GPU batched bmm (batch = B*H)
    attn    = softmax(scores)                  # GPU MPSGraph softmax (no N limit)
    ctx     = bmm(attn, V_h)                    # GPU batched bmm
    out     = linear_general(concat_heads, W_o)  # GPU matmul / bmm

The per-head dimension is the batch axis of bmm, so all heads run in a single
dispatch per stage and there is no flash_attn head_dim<=256 envelope limit.
Validated against a float64 numpy reference; the heavy ops report
metal_runtime on Darwin. See docs/audit/backend/apple/archive/apple_gpu_tier2_tier3_plan.md.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

import tessera as ts

DARWIN = sys.platform == "darwin"


# ── jitted Tier-2 building blocks (module-level so @jit can inspect source) ──
@ts.jit(target="apple_gpu")
def _proj_qkv(x, w_qkv):
    return ts.ops.qkv_projection(x, w_qkv)


@ts.jit(target="apple_gpu")
def _bmm(a, b):
    return ts.ops.matmul(a, b)


@ts.jit(target="apple_gpu")
def _softmax(x):
    return ts.ops.softmax(x)


@ts.jit(target="apple_gpu")
def _out_proj(x, w_o):
    return ts.ops.linear_general(x, w_o)


def _np_mha(x, w_qkv, w_o, num_heads):
    """float64 reference: qkv-proj -> per-head SDPA -> out-proj."""
    x = x.astype(np.float64)
    w_qkv = w_qkv.astype(np.float64)
    w_o = w_o.astype(np.float64)
    B, T, Dm = x.shape
    D = Dm // num_heads
    qkv = x @ w_qkv
    Q, K, V = np.split(qkv, 3, axis=-1)

    def heads(t):
        return t.reshape(B, T, num_heads, D).transpose(0, 1, 2, 3)

    Qh, Kh, Vh = heads(Q), heads(K), heads(V)  # [B, T, H, D]
    scale = 1.0 / math.sqrt(D)
    # [B, T, H, D] -> [B, H, T, D]
    Qh = Qh.transpose(0, 2, 1, 3)
    Kh = Kh.transpose(0, 2, 1, 3)
    Vh = Vh.transpose(0, 2, 1, 3)
    scores = (Qh @ Kh.transpose(0, 1, 3, 2)) * scale  # [B, H, T, T]
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    attn = e / e.sum(axis=-1, keepdims=True)
    ctx = attn @ Vh  # [B, H, T, D]
    ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, Dm)  # concat heads
    return ctx @ w_o


def _run_mha(x, w_qkv, w_o, num_heads):
    """Compose the block from the jitted GPU ops; host glue = reshape/scale."""
    B, T, Dm = x.shape
    D = Dm // num_heads
    dtype = x.dtype

    Q, K, V = (np.asarray(t) for t in _proj_qkv(x, w_qkv))  # each [B, T, Dm]

    def fold(t):  # [B, T, Dm] -> [B*H, T, D]
        return np.ascontiguousarray(
            t.reshape(B, T, num_heads, D).transpose(0, 2, 1, 3).reshape(B * num_heads, T, D))

    Qf, Kf, Vf = fold(Q), fold(K), fold(V)
    scale = np.array(1.0 / math.sqrt(D), dtype=np.float32)
    Qf = (Qf.astype(np.float32) * np.float32(scale)).astype(dtype)
    Kt = np.ascontiguousarray(Kf.transpose(0, 2, 1))  # [B*H, D, T]

    scores = np.asarray(_bmm(Qf, Kt))          # [B*H, T, T]  GPU bmm
    attn = np.asarray(_softmax(scores))         # GPU softmax
    ctx = np.asarray(_bmm(attn.astype(dtype), Vf))  # [B*H, T, D]  GPU bmm

    concat = np.ascontiguousarray(
        ctx.reshape(B, num_heads, T, D).transpose(0, 2, 1, 3).reshape(B, T, Dm))
    return np.asarray(_out_proj(concat, w_o))


_SHAPES = [
    pytest.param(2, 8, 32, 4, id="B2_T8_Dm32_H4"),
    pytest.param(1, 16, 64, 8, id="B1_T16_Dm64_H8"),
    pytest.param(2, 12, 48, 6, id="B2_T12_Dm48_H6"),
]


@pytest.mark.parametrize("B,T,Dm,H", _SHAPES)
def test_batched_mha_matches_numpy(B, T, Dm, H):
    rng = np.random.RandomState(2026)
    x = (rng.randn(B, T, Dm) * 0.5).astype(np.float32)
    w_qkv = (rng.randn(Dm, 3 * Dm) * 0.3).astype(np.float32)
    w_o = (rng.randn(Dm, Dm) * 0.3).astype(np.float32)

    out = _run_mha(x, w_qkv, w_o, H)
    assert out.shape == (B, T, Dm)
    expected = _np_mha(x, w_qkv, w_o, H)
    np.testing.assert_allclose(out, expected, rtol=1.5e-3, atol=1.5e-3)


def test_batched_mha_ops_runtime_executable():
    """The four Tier-2 building blocks all dispatch through the apple_gpu
    runtime (qkv_projection / bmm / softmax / linear_general)."""
    _proj_qkv(np.zeros((2, 8, 32), np.float32), np.zeros((32, 96), np.float32))
    _bmm(np.zeros((8, 8, 8), np.float32), np.zeros((8, 8, 8), np.float32))
    _softmax(np.zeros((8, 8, 8), np.float32))
    _out_proj(np.zeros((2, 8, 32), np.float32), np.zeros((32, 32), np.float32))
    for fn in (_proj_qkv, _bmm, _softmax, _out_proj):
        meta = fn.runtime_artifact().metadata
        assert meta["compiler_path"] == "apple_gpu_mps"
        assert meta["runtime_status"] == "ready"
        assert meta["execution_mode"] in ("metal_runtime", "metal_artifact")


@pytest.mark.skipif(not DARWIN, reason="metal_runtime dispatch is Darwin-only")
def test_batched_mha_ops_metal_runtime_on_darwin():
    _bmm(np.zeros((8, 8, 8), np.float32), np.zeros((8, 8, 8), np.float32))
    _softmax(np.zeros((8, 8, 8), np.float32))
    assert _bmm.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
    assert _softmax.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


def test_batched_mha_large_head_dim_no_flash_limit():
    """bmm contracts over the head dim, so head_dim can exceed flash_attn's
    256 envelope — the batched-bmm attention path has no such limit."""
    B, T, H, D = 1, 6, 2, 320  # D=320 > flash_attn head_dim<=256
    Dm = H * D
    rng = np.random.RandomState(7)
    x = (rng.randn(B, T, Dm) * 0.3).astype(np.float32)
    w_qkv = (rng.randn(Dm, 3 * Dm) * 0.1).astype(np.float32)
    w_o = (rng.randn(Dm, Dm) * 0.1).astype(np.float32)
    out = _run_mha(x, w_qkv, w_o, H)
    assert out.shape == (B, T, Dm)
    np.testing.assert_allclose(out, _np_mha(x, w_qkv, w_o, H), rtol=2e-3, atol=2e-3)
