"""Register-blocked f32 GEMM device kernel on gfx1151 (generate-rocm-gemm-f32-
kernel). RDNA WMMA is f16/bf16-only, so the f32-exact expert GEMM (grouped
SwiGLU) rides this plain-VALU kernel. Each thread computes a TM×TN=4×4 output
tile, reusing loaded A/B in registers (the Stage-F register-budget lever) — a
~1.6× speedup over the naive one-thread-per-output at 1024³. This fixture locks
CORRECTNESS across shapes, especially non-multiples of the 4×4 tile (the bounds
guards). Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


_RNG = np.random.default_rng(5)


@pytest.mark.parametrize("M,N,K", [
    (4, 4, 4),        # exactly one tile
    (1, 1, 1),        # single element (heavy edge masking)
    (7, 5, 3),        # non-multiple of 4 in M, N, K
    (13, 17, 9),      # ragged both dims
    (3, 8, 2),        # M < TM
    (16, 4, 32),      # N == TN, deep K
    (2, 2, 100),      # deep K, tiny tile
    (64, 64, 64),     # aligned mid-size
    (65, 63, 31),     # ragged mid-size
])
def test_f32_gemm_matches_numpy(M, N, K):
    rt = _rocm_or_skip()
    a = _RNG.standard_normal((M, K)).astype(np.float32)
    b = _RNG.standard_normal((K, N)).astype(np.float32)
    got = rt._rocm_f32_gemm(a, b, np)
    assert got.shape == (M, N)
    np.testing.assert_allclose(got, a @ b, rtol=1e-4, atol=1e-4)


def test_f32_gemm_rejects_mismatched_k():
    rt = _rocm_or_skip()
    a = np.zeros((4, 8), np.float32)
    b = np.zeros((7, 4), np.float32)   # K mismatch (8 vs 7)
    with pytest.raises(ValueError):
        rt._rocm_f32_gemm(a, b, np)
