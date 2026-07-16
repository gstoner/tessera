"""P7 (2026-06-01) — fp16 matmul MTL4 routing (size-gated to M==1).

The fp16 GEMV decode step (M==1) is the one regime where the native MPP
``matmul2d`` tensor-op clears MPS (measured ~3.2-3.4x; MPS has a slow
fp16 M==1 path). At M>=2 and on square shapes MPS wins, so the default
route is strictly M==1. These tests pin the GATE + NUMERICS — they don't
benchmark (see benchmarks/apple_gpu/benchmark_mtl4_matmul_routing.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as rt


# ---- Gate logic (no GPU needed — exercises the routing predicate) -----

def test_f16_route_mode_default_is_auto():
    assert rt.apple_gpu_mtl4_f16_mode() in ("auto", "1")


def test_f16_route_gate_rejects_non_f16_and_non_2d():
    # The route function must return None (fall back to MPS) for inputs
    # outside its contract, regardless of Metal availability.
    a32 = np.ones((1, 4), np.float32)
    assert rt._mtl4_route_matmul2d_f16(a32, a32, np) is None
    a16 = np.ones((1, 4), np.float16)
    b16 = np.ones((4, 4), np.float16)
    a3d = np.ones((1, 1, 4), np.float16)
    assert rt._mtl4_route_matmul2d_f16(a3d, b16, np) is None  # not 2-D
    bad = np.ones((3, 4), np.float16)
    assert rt._mtl4_route_matmul2d_f16(a16, bad, np) is None  # K mismatch


def test_f16_route_off_mode_returns_none():
    prev = rt.apple_gpu_mtl4_f16_mode()
    try:
        rt.set_apple_gpu_mtl4_f16_mode("off")
        a = np.ones((1, 8), np.float16)
        b = np.ones((8, 8), np.float16)
        assert rt._mtl4_route_matmul2d_f16(a, b, np) is None
    finally:
        rt.set_apple_gpu_mtl4_f16_mode(prev)


def test_f16_auto_mode_gates_to_m_equals_one():
    """In 'auto', M>=2 must NOT route (returns None → MPS); only M==1
    is eligible. We assert the M>=2 rejection happens BEFORE any GPU
    dispatch, so this holds even without Metal."""
    prev = rt.apple_gpu_mtl4_f16_mode()
    try:
        rt.set_apple_gpu_mtl4_f16_mode("auto")
        a2 = np.ones((2, 8), np.float16)
        b = np.ones((8, 8), np.float16)
        assert rt._mtl4_route_matmul2d_f16(a2, b, np) is None  # M=2 → MPS
    finally:
        rt.set_apple_gpu_mtl4_f16_mode(prev)


# ---- Numerics (Metal-gated) -------------------------------------------

@pytest.mark.hardware_apple_gpu
@pytest.mark.metal4
def test_f16_m1_route_matches_reference():
    prev = rt.apple_gpu_mtl4_f16_mode()
    try:
        rt.set_apple_gpu_mtl4_f16_mode("auto")
        rng = np.random.default_rng(0xF16E)
        A = (rng.standard_normal((1, 256)) * 0.1).astype(np.float16)
        B = (rng.standard_normal((256, 128)) * 0.1).astype(np.float16)
        routed = rt._mtl4_route_matmul2d_f16(A, B, np)
        assert routed is not None, "M==1 fp16 should route in auto mode"
        assert routed.dtype == np.float16 and routed.shape == (1, 128)
        ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
        np.testing.assert_allclose(routed.astype(np.float32),
                                    ref.astype(np.float32),
                                    rtol=2e-2, atol=2e-2)
    finally:
        rt.set_apple_gpu_mtl4_f16_mode(prev)


@pytest.mark.hardware_apple_gpu
@pytest.mark.metal4
def test_f16_all_mode_routes_square():
    """'all' mode routes every fp16 2-D matmul (the benchmarking knob);
    numerically must still match the reference."""
    prev = rt.apple_gpu_mtl4_f16_mode()
    try:
        rt.set_apple_gpu_mtl4_f16_mode("all")
        rng = np.random.default_rng(0x5A)
        A = (rng.standard_normal((64, 64)) * 0.1).astype(np.float16)
        B = (rng.standard_normal((64, 64)) * 0.1).astype(np.float16)
        routed = rt._mtl4_route_matmul2d_f16(A, B, np)
        assert routed is not None and routed.shape == (64, 64)
        ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
        np.testing.assert_allclose(routed.astype(np.float32),
                                    ref.astype(np.float32),
                                    rtol=3e-2, atol=3e-2)
    finally:
        rt.set_apple_gpu_mtl4_f16_mode(prev)
