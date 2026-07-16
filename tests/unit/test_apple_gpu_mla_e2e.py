"""Apple GPU MLA-style end-to-end proof (Sprint M, 2026-05-22).

This is the first model-shaped end-to-end correctness test that
composes multiple `@tessera.jit(target="apple_gpu")` calls into a
single MLA-flavored attention decoder and validates the numerical
output against a faithful numpy reference.

What "MLA-flavored" means here is **not** the full DeepSeek MLA
(no decoupled-rope split, no compressed-KV cache, no decode-time
absorption — those are Phase G / Phase 8.4.8 follow-ups).  It's a
honest *single-layer attention block* that exercises the same
shapes a real MLA decoder uses for one decode step:

    1. project_q : x @ Wq  →  Q  (B, S, H * Dh)
    2. project_k : x @ Wk  →  K  (B, S, H * Dh)
    3. project_v : x @ Wv  →  V  (B, S, H * Dh)
    4. head-split + scale: Q ← Q / sqrt(Dh)
    5. scaled_dot_product : softmax(Q @ K^T) @ V  per head
    6. project_o : concat-heads @ Wo  →  Y  (B, S, D)

For a 2D batch x sequence collapse (B*S, D) the inner attention
math is exactly `matmul → softmax → matmul`, which Tessera fuses
into the `matmul_softmax_matmul_f32` MSL kernel on apple_gpu
(Phase 8.4.5).  The three projection matmuls and the output
projection take Tessera's `matmul` path (MPS for static rank-2
matmul).

Why this is the right "first MLA E2E proof":
  * It exercises real model shapes (multi-head dimension layout,
    transposes, head split/concat).
  * It composes 5+ jitted callables, proving the runtime artifact
    cache and the per-op dispatcher work across more than just
    a single fused kernel.
  * Numerical proof at fp32 rtol=1e-4 (matching the matching-shape
    fused-kernel test in test_apple_backend_roadmap.py).

When Phase 8.4.8 ships MLA-specific kernels (compressed-KV
decode, rotary-split, head absorption), this test extends to
those without changing the proof harness.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

import tessera as ts


# ─────────────────────────────────────────────────────────────────────────
# Numpy reference: faithful single-layer multi-head attention forward.
# ─────────────────────────────────────────────────────────────────────────


def _np_mha_reference(
    x: np.ndarray,
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    Wo: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    """Reference single-layer multi-head attention.

    x   : (T, D)        T = batch*seq (flattened); D = model dim
    Wq, Wk, Wv : (D, D) projection weights
    Wo  : (D, D)        output projection

    Implements the per-head causal-free attention used by MLA-flavored
    decoders for one decode step:
       Q, K, V = x @ Wq, x @ Wk, x @ Wv         # each (T, D)
       reshape to (T, H, Dh) and transpose to (H, T, Dh)
       scores = Q @ K.transpose / sqrt(Dh)       # (H, T, T)
       attn   = softmax(scores, axis=-1) @ V     # (H, T, Dh)
       concat heads → (T, D), then @ Wo
    """
    T, D = x.shape
    assert D % num_heads == 0
    Dh = D // num_heads

    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv

    # split heads: (T, D) -> (T, H, Dh) -> (H, T, Dh)
    def split(M):
        return M.reshape(T, num_heads, Dh).transpose(1, 0, 2)

    Qh, Kh, Vh = split(Q), split(K), split(V)

    scale = 1.0 / math.sqrt(Dh)
    # per head: scores (T, T), then softmax along last axis, then @ Vh
    out_heads = np.empty_like(Qh)
    for h in range(num_heads):
        s = (Qh[h] @ Kh[h].T) * scale          # (T, T)
        s = s - s.max(axis=-1, keepdims=True)  # numerical stability
        e = np.exp(s)
        p = e / e.sum(axis=-1, keepdims=True)
        out_heads[h] = p @ Vh[h]               # (T, Dh)

    # concat heads back: (H, T, Dh) -> (T, H, Dh) -> (T, D)
    out = out_heads.transpose(1, 0, 2).reshape(T, D)
    return out @ Wo


# ─────────────────────────────────────────────────────────────────────────
# Tessera Apple GPU path: each jitted block is an apple_gpu artifact.
# The composition of projections + fused attention + output projection
# is what's under test.
# ─────────────────────────────────────────────────────────────────────────


def _build_mla_decoder():
    """Build the four jitted callables (projections + per-head attention).

    Returning them as a tuple keeps the test driver explicit about the
    composition; a future refactor can collapse this into a single
    decorated callable once `@jit(target='apple_gpu')` accepts a body
    that calls itself or other jit fns.
    """

    @ts.jit(target="apple_gpu")
    def project(x, W):
        # 2D matmul; on apple_gpu this dispatches through MPS.
        return ts.ops.matmul(x, W)

    @ts.jit(target="apple_gpu")
    def per_head_attn(qh, kh_t, vh):
        # qh : (T, Dh)
        # kh_t : (Dh, T)
        # vh : (T, Dh)
        # The fused matmul -> softmax -> matmul collapses into one
        # MSL kernel on apple_gpu.  We pre-scale Q outside this call
        # to keep the fusion pattern matchable.
        return ts.ops.matmul(ts.ops.softmax(ts.ops.matmul(qh, kh_t)), vh)

    return project, per_head_attn


# ─────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────


# Each case: (T, D, num_heads) where T = batch*seq, D = model dim.
# Keep T small for fp32 fused-kernel envelope (Phase 8.4.5 single-row
# stack cap; the threadgroup-tiled f32 variant from Phase 8.4.6
# extends N to 8192 for matmul_softmax — we exercise modest shapes
# here so the proof stays in the well-trodden envelope).
_MLA_SHAPES = [
    pytest.param(8, 16, 2, id="T8_D16_H2"),
    pytest.param(16, 32, 4, id="T16_D32_H4"),
    pytest.param(32, 64, 8, id="T32_D64_H8"),
]


@pytest.mark.parametrize("T,D,H", _MLA_SHAPES)
def test_apple_gpu_mla_e2e_matches_numpy_reference(T: int, D: int, H: int):
    """Composed MLA decoder: 3 projections + per-head fused attention +
    output projection.  Output must match numpy reference at fp32
    rtol=1e-4 (matching the single-fused-kernel envelope from
    test_apple_backend_roadmap.py)."""

    Dh = D // H
    rng = np.random.RandomState(2026_05_22)

    # Keep amplitudes small so softmax saturates gracefully (avoids
    # near-degenerate softmax distributions that amplify fp differences).
    x = rng.randn(T, D).astype(np.float32) * 0.5
    Wq = rng.randn(D, D).astype(np.float32) * 0.5
    Wk = rng.randn(D, D).astype(np.float32) * 0.5
    Wv = rng.randn(D, D).astype(np.float32) * 0.5
    Wo = rng.randn(D, D).astype(np.float32) * 0.5

    project, per_head_attn = _build_mla_decoder()

    # 1. Q, K, V projections (Tessera apple_gpu path; MPS for matmul).
    Q = project(x, Wq)
    K = project(x, Wk)
    V = project(x, Wv)
    assert Q.shape == (T, D)
    assert Q.dtype == np.float32

    # 2. Head split: (T, D) -> (H, T, Dh).  Done in numpy on the host
    #    side — Tessera's `transpose` op is supported but the host
    #    reshape stays simpler for this proof.  The per-head attention
    #    kernel sees its operands as already-split.
    def split(M):
        return M.reshape(T, H, Dh).transpose(1, 0, 2)

    Qh = split(Q)
    Kh = split(K)
    Vh = split(V)

    # 3. Pre-scale Q so the fused kernel can stay as matmul→softmax→matmul.
    scale = 1.0 / math.sqrt(Dh)
    Qh = Qh * np.float32(scale)

    # 4. Per-head fused attention: matmul -> softmax -> matmul.
    out_heads = np.empty_like(Qh)
    for h in range(H):
        # Need kh^T as a contiguous matrix for the fused kernel.
        kh_t = np.ascontiguousarray(Kh[h].T)
        out_heads[h] = per_head_attn(Qh[h], kh_t, Vh[h])

    # 5. Concat heads back: (H, T, Dh) -> (T, D), then output projection.
    concat = np.ascontiguousarray(out_heads.transpose(1, 0, 2).reshape(T, D))
    Y = project(concat, Wo)
    assert Y.shape == (T, D)
    assert Y.dtype == np.float32

    # Reference.
    expected = _np_mha_reference(x, Wq, Wk, Wv, Wo, num_heads=H)
    np.testing.assert_allclose(Y, expected, rtol=1e-4, atol=1e-5)


def test_apple_gpu_mla_e2e_uses_fused_kernel_for_per_head_attention():
    """Sanity: the per_head_attn jitted fn must lower to the fused
    matmul_softmax_matmul MSL kernel on apple_gpu (not 3 separate ops)."""
    _, per_head_attn = _build_mla_decoder()

    # Trigger compilation with representative shapes (compile-time
    # only — no real dispatch needed for the IR check).
    T, Dh = 4, 8
    qh = np.zeros((T, Dh), dtype=np.float32)
    kh_t = np.zeros((Dh, T), dtype=np.float32)
    vh = np.zeros((T, Dh), dtype=np.float32)
    per_head_attn(qh, kh_t, vh)  # warm the cache

    # The Target IR carries one fused msl_kernel for the chain.
    target_ir = per_head_attn.target_ir
    assert "tessera_apple.gpu.msl_kernel" in target_ir, (
        "MLA per-head attention must lower to an MSL kernel on apple_gpu"
    )
    assert 'fusion = "matmul_softmax_matmul"' in target_ir, (
        "MLA per-head attention must fuse to matmul_softmax_matmul; "
        "if this regresses, the per-head fused-kernel envelope from "
        "Phase 8.4.5 has been disturbed"
    )
    # Exactly one MSL kernel emission (three Graph IR ops collapsed).
    assert target_ir.count('"tessera_apple.gpu.msl_kernel"') == 1, (
        "expected exactly 1 fused MSL kernel for 3-op chain; "
        "the fusion didn't fire"
    )


def test_apple_gpu_mla_e2e_artifact_metadata_consistent():
    """All four jitted callables in the MLA path produce metal_runtime
    artifacts on Darwin (or metal_artifact + reference fallback off
    Darwin). The metadata contract must stay uniform across them."""

    project, per_head_attn = _build_mla_decoder()
    # warm both
    project(np.zeros((4, 8), dtype=np.float32),
            np.zeros((8, 8), dtype=np.float32))
    per_head_attn(np.zeros((4, 8), dtype=np.float32),
                  np.zeros((8, 4), dtype=np.float32),
                  np.zeros((4, 8), dtype=np.float32))

    proj_meta = project.runtime_artifact().metadata
    attn_meta = per_head_attn.runtime_artifact().metadata

    # Both must declare apple_gpu compiler_path (the path label is
    # what release_gate.py keys on for Apple).
    assert proj_meta["compiler_path"] == "apple_gpu_mps"
    assert attn_meta["compiler_path"] == "apple_gpu_mps"

    # On Darwin (this Mac) both should be metal_runtime + ready.
    # On Linux/CI the apple_gpu portable-reference path runs, which
    # still reports ready but execution_mode may differ.  We accept
    # either to keep CI green.
    for meta in (proj_meta, attn_meta):
        assert meta["runtime_status"] == "ready"
        assert meta["execution_mode"] in ("metal_runtime", "metal_artifact")


@pytest.mark.hardware_apple_gpu
def test_apple_gpu_mla_e2e_darwin_uses_metal_runtime():
    """On Darwin specifically, the MLA path must hit metal_runtime
    (not the portable reference fallback)."""

    project, per_head_attn = _build_mla_decoder()
    project(np.zeros((4, 8), dtype=np.float32),
            np.zeros((8, 8), dtype=np.float32))

    meta = project.runtime_artifact().metadata
    assert meta["execution_mode"] == "metal_runtime", (
        "Darwin must dispatch through Metal, not the portable fallback"
    )
