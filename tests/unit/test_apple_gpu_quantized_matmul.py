"""P3 — Apple GPU packed-int4 quantized matmul.

Locks the packed-weight quantized-matmul lane (see
``docs/audit/backend/apple/archive/apple_backend_capability_roadmap.md`` P3). Unlike the existing
``dequant_matmul_f32`` (full-width f32 codes — 4 bytes/weight, no bandwidth win),
this lane stores weights as packed 4-bit codes (0.5 bytes/weight) and dequants
``w = scale·code + bias`` in-register.

The reference for correctness is ``X @ dequantize_int4_packed(W)^T`` — i.e. the
GPU kernel is compared against the *same* dequantized weights, so any difference
is pure float-accumulation order (tight tolerance), not quantization error.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera
import tessera.runtime as R
from tessera import ops
from tessera.quantization import (
    dequantize_fp4_packed,
    dequantize_int4_packed,
    quantize_fp4_packed,
    quantize_int4_packed,
)


def test_quant_matmul_symbol_present():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_quantized_matmul_i4_f32")


def test_pack_roundtrip_within_int4_tolerance():
    """Packing then unpacking a weight recovers it to int4 group resolution."""
    rng = np.random.RandomState(3)
    W = rng.randn(16, 128).astype(np.float32)
    packed, scales, biases = quantize_int4_packed(W, group_size=64)
    assert packed.dtype == np.uint8
    assert packed.shape == (16, 64)          # ceil(128/2)
    assert scales.shape == (16, 2)           # ceil(128/64)
    Wr = dequantize_int4_packed(packed, scales, biases, k=128, group_size=64)
    # int4 over a group has 15 steps spanning [min,max]; max error ≈ step/2.
    step = (W.max(axis=1) - W.min(axis=1)) / 15.0
    assert np.all(np.abs(Wr - W) <= (step[:, None] * 0.5 + 1e-5))


def test_packed_weight_bandwidth_is_8x_smaller():
    """The packed weight is ~8× smaller than full-width f32 codes — the point of
    the lane."""
    rng = np.random.RandomState(4)
    N, K = 32, 256
    W = rng.randn(N, K).astype(np.float32)
    packed, _, _ = quantize_int4_packed(W, group_size=64)
    f32_code_bytes = N * K * 4
    assert packed.nbytes == N * ((K + 1) // 2)
    assert f32_code_bytes // packed.nbytes == 8


@pytest.mark.parametrize(
    "M,N,K,gs",
    [
        (1, 32, 128, 64),    # QMV (decode)
        (4, 48, 128, 64),    # QMM (prefill)
        (3, 16, 70, 64),     # partial last group
        (2, 24, 65, 32),     # odd K (lone final nibble)
        (1, 8, 96, 96),      # single group (gs == K)
    ],
)
def test_gpu_quant_matmul_matches_dequant_reference(M, N, K, gs):
    rng = np.random.RandomState(M * 100 + N + K)
    X = rng.randn(M, K).astype(np.float32)
    W = rng.randn(N, K).astype(np.float32)  # nn.Linear convention [out, in]
    packed, scales, biases = quantize_int4_packed(W, group_size=gs)

    out = R.apple_gpu_quantized_matmul_i4(X, packed, scales, biases, K=K, group_size=gs)
    if out is None:
        pytest.skip("apple_gpu runtime unavailable")

    Wdq = dequantize_int4_packed(packed, scales, biases, k=K, group_size=gs)
    ref = X @ Wdq.T  # O[m,n] = sum_k X[m,k] * dequant(W[n,k])
    assert out.shape == (M, N)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_tiled_variant_present_and_matches_untiled():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_quantized_matmul_i4_tiled_f32")
    rng = np.random.RandomState(7)
    M, N, K, gs = 6, 40, 600, 64  # K > CHUNK(512) exercises multi-chunk tiling
    X = rng.randn(M, K).astype(np.float32)
    W = rng.randn(N, K).astype(np.float32)
    packed, scales, biases = quantize_int4_packed(W, group_size=gs)
    untiled = R.apple_gpu_quantized_matmul_i4(X, packed, scales, biases, K=K, group_size=gs)
    tiled = R.apple_gpu_quantized_matmul_i4(
        X, packed, scales, biases, K=K, group_size=gs, variant="tiled"
    )
    if untiled is None or tiled is None:
        pytest.skip("apple_gpu runtime unavailable")
    # Tiled is the same math with threadgroup-cached X — must match the untiled
    # kernel closely (only float reduction-order differences).
    np.testing.assert_allclose(tiled, untiled, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("M,N,K,gs", [(1, 32, 128, 64), (4, 48, 256, 64)])
def test_f16_variant_matches_dequant_reference(M, N, K, gs):
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_quantized_matmul_i4_f16")
    rng = np.random.RandomState(M + N + K)
    Xf16 = rng.randn(M, K).astype(np.float16)
    W = rng.randn(N, K).astype(np.float32)
    packed, scales, biases = quantize_int4_packed(W, group_size=gs)
    out = R.apple_gpu_quantized_matmul_i4(
        Xf16, packed, scales, biases, K=K, group_size=gs, variant="f16"
    )
    if out is None:
        pytest.skip("apple_gpu runtime unavailable")
    Wdq = dequantize_int4_packed(packed, scales, biases, k=K, group_size=gs)
    # Reference uses the same fp16 X (upcast to f32) so the only error is f16
    # rounding of X — tight tolerance, not a quant-error fudge.
    ref = Xf16.astype(np.float32) @ Wdq.T
    np.testing.assert_allclose(out, ref, rtol=2e-3, atol=2e-3)


# ── Full @jit artifact dispatch ─────────────────────────────────────────────

def test_driver_gating_recognizes_quantized_matmul():
    """A single-op apple_gpu plan with tessera.quantized_matmul is recognized as
    GPU-executable (envelope membership → `_is_apple_gpu_mps_executable`)."""
    from types import SimpleNamespace

    from tessera.compiler.driver import _is_apple_gpu_mps_executable

    plan = SimpleNamespace(
        target_kind="apple_gpu",
        ops=[SimpleNamespace(op_name="tessera.quantized_matmul")],
    )
    assert _is_apple_gpu_mps_executable(plan) is True


def test_lane_resolves_to_quant_matmul_handler():
    from tessera.compiler.apple_gpu_envelope import APPLE_GPU_LANE_BY_OP

    assert APPLE_GPU_LANE_BY_OP.get("tessera.quantized_matmul") == "quant_matmul"
    assert "quant_matmul" in R._apple_gpu_lane_handlers()


@pytest.mark.parametrize("M,N,K,gs", [(1, 32, 128, 64), (4, 48, 256, 64)])
def test_metadata_artifact_dispatches_to_gpu(M, N, K, gs):
    """The compiled-artifact form (metadata op list) routes
    tessera.quantized_matmul through the apple_gpu executor → lane handler → GPU
    kernel, producing the correct result. This is the end-to-end artifact
    dispatch (the runtime form a @jit(target='apple_gpu') program lowers to)."""
    rng = np.random.RandomState(M * 31 + N + K)
    X = rng.randn(M, K).astype(np.float32)
    W = rng.randn(N, K).astype(np.float32)
    packed, scales, biases = quantize_int4_packed(W, group_size=gs)

    metadata = {
        "arg_names": ["x", "wq", "scales", "biases"],
        "output_name": "out",
        "ops": [
            {
                "op_name": "tessera.quantized_matmul",
                "result": "out",
                "operands": ["x", "wq", "scales", "biases"],
                "kwargs": {"group_size": gs},
            }
        ],
    }
    out = R._execute_apple_gpu_mps_metadata(metadata, (X, packed, scales, biases))
    Wdq = dequantize_int4_packed(packed, scales, biases, k=K, group_size=gs)
    ref = X @ Wdq.T
    assert np.asarray(out).shape == (M, N)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@tessera.jit(target="apple_gpu")
def _qmm_jit(x, wq, sc, bi):
    return ops.quantized_matmul(x, wq, sc, bi, group_size=64)


def test_full_jit_apple_gpu_quantized_matmul():
    """End-to-end: a @jit(target='apple_gpu') function calling
    ops.quantized_matmul lowers (AST → tessera.quantized_matmul graph op →
    metadata artifact → apple_gpu executor → packed-int4 Metal kernel) and
    matches the dequant reference."""
    M, N, K, gs = 4, 32, 128, 64
    rng = np.random.RandomState(9)
    X = rng.randn(M, K).astype(np.float32)
    W = rng.randn(N, K).astype(np.float32)
    packed, scales, biases = quantize_int4_packed(W, group_size=gs)
    out = np.asarray(_qmm_jit(X, packed, scales, biases))
    ref = X @ dequantize_int4_packed(packed, scales, biases, k=K, group_size=gs).T
    assert out.shape == (M, N)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


# ── VJP (straight-through, frozen quantized weight) ─────────────────────────

def test_quantized_matmul_vjp_matches_analytic_and_finite_difference():
    from tessera.autodiff.vjp import get_vjp

    rng = np.random.RandomState(21)
    M, N, K, gs = 3, 8, 64, 32
    X = rng.randn(M, K).astype(np.float32)
    W = rng.randn(N, K).astype(np.float32)
    packed, scales, biases = quantize_int4_packed(W, group_size=gs)
    Wdq = dequantize_int4_packed(packed, scales, biases, k=K, group_size=gs)
    dout = rng.randn(M, N).astype(np.float32)

    vjp = get_vjp("quantized_matmul")
    assert vjp is not None
    dx, dwp, dsc, dbi = vjp(dout, X, packed, scales, biases, group_size=gs)

    # Frozen quantized weight → no grad to packed codes / scales / biases.
    assert dwp is None and dsc is None and dbi is None
    # Analytic: dx = dout @ Wdq (since y = X @ Wdq^T).
    np.testing.assert_allclose(dx, dout @ Wdq, rtol=1e-5, atol=1e-5)

    # Finite difference of f(X) = sum(dout * (X @ Wdq^T)).
    eps = 1e-3
    def f(x):
        return float(np.sum(dout * (x @ Wdq.T)))
    fd = np.zeros_like(X)
    for i in range(M):
        for j in range(K):
            xp = X.copy(); xp[i, j] += eps
            xm = X.copy(); xm[i, j] -= eps
            fd[i, j] = (f(xp) - f(xm)) / (2 * eps)
    np.testing.assert_allclose(dx, fd, rtol=1e-2, atol=1e-2)


def test_splitk_variant_present_and_matches_untiled():
    """Split-K parallelizes the K reduction (M·N·S threads + host partial-sum);
    same math as the untiled kernel. K large enough that the heuristic uses S>1."""
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_quantized_matmul_i4_splitk_f32")
    rng = np.random.RandomState(33)
    M, N, K, gs = 1, 12, 2048, 64  # K=2048 → S=4 splits; QMV decode shape
    X = rng.randn(M, K).astype(np.float32)
    W = rng.randn(N, K).astype(np.float32)
    packed, scales, biases = quantize_int4_packed(W, group_size=gs)
    base = R.apple_gpu_quantized_matmul_i4(X, packed, scales, biases, K=K, group_size=gs)
    sk = R.apple_gpu_quantized_matmul_i4(
        X, packed, scales, biases, K=K, group_size=gs, variant="splitk"
    )
    if base is None or sk is None:
        pytest.skip("apple_gpu runtime unavailable")
    Wdq = dequantize_int4_packed(packed, scales, biases, k=K, group_size=gs)
    ref = X @ Wdq.T
    np.testing.assert_allclose(sk, base, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(sk, ref, rtol=1e-3, atol=1e-3)


# ── MXFP4 / NVFP4 packed layout (FP4 e2m1 decode) ───────────────────────────

def test_fp4_pack_roundtrip_quantizes_to_e2m1_grid():
    """quantize→dequantize lands on the FP4 e2m1 magnitude grid scaled per group."""
    from tessera.quantization import _FP4_E2M1_LUT
    rng = np.random.RandomState(41)
    W = rng.randn(8, 64).astype(np.float32)
    packed, scales = quantize_fp4_packed(W, group_size=32, scale_mode="mx")
    assert packed.dtype == np.uint8 and packed.shape == (8, 32)
    Wr = dequantize_fp4_packed(packed, scales, k=64, group_size=32)
    # every reconstructed value is (per-group scale) × an e2m1 magnitude.
    for g in range(2):
        sl = Wr[:, g * 32:(g + 1) * 32]
        ratios = np.abs(sl) / scales[:, g:g + 1]
        # each ratio must be ~one of the LUT magnitudes
        nearest = np.min(np.abs(ratios[..., None] - _FP4_E2M1_LUT), axis=-1)
        assert np.max(nearest) < 1e-4


@pytest.mark.parametrize(
    "mode,gs", [("mx", 32), ("nv", 16)],  # MXFP4 (g32, pow2 scale) / NVFP4 (g16)
)
def test_gpu_fp4_matmul_matches_dequant_reference(mode, gs):
    rng = np.random.RandomState(hash((mode, gs)) & 0xFFFF)
    M, N, K = 4, 32, 128
    X = rng.randn(M, K).astype(np.float32)
    W = rng.randn(N, K).astype(np.float32)
    packed, scales = quantize_fp4_packed(W, group_size=gs, scale_mode=mode)
    out = R.apple_gpu_quantized_matmul_fp4(X, packed, scales, K=K, group_size=gs)
    if out is None:
        pytest.skip("apple_gpu runtime unavailable")
    Wdq = dequantize_fp4_packed(packed, scales, k=K, group_size=gs)
    ref = X @ Wdq.T
    assert out.shape == (M, N)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
