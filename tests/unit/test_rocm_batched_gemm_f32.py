"""Single-launch batched f32 GEMM device kernel on gfx1151
(generate-rocm-batched-gemm-f32-kernel).

`C[b] = A[b] @ B[b]` for a whole batch in ONE launch — the batch is folded into
the grid, so the caller does one hipMalloc/H2D/launch/D2H instead of the per-batch
round-trip of looping the single-GEMM kernel. Built to make the chunked-parallel
SSD scan's bmms single-launch (`_rocm_batched_gemm_f32`); it does — though the SSD
stays a reference rung, since its many per-chunk bmm CALLS are the residual
bottleneck (see STRIX_HALO_EXECUTION_PLAN). This fixture locks the kernel's
CORRECTNESS across batch/shape (incl. non-multiples of the 4×4 tile + NumPy
broadcast of a rank-2 operand). Skip-clean: tessera-opt not built / no GPU.
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


_RNG = np.random.default_rng(11)


@pytest.mark.parametrize("nb,m,k,n", [
    (4, 8, 6, 5),          # small, non-multiples of 4
    (1, 7, 3, 4),          # single batch, ragged
    (2, 13, 17, 9),        # ragged M/K/N
    (8, 64, 16, 32),       # aligned mid-size, deeper batch
    (3, 1, 1, 1),          # 1×1×1 per batch (heavy edge masking)
    (5, 2, 100, 3),        # deep K, tiny tile
])
def test_batched_gemm_matches_numpy(nb, m, k, n):
    rt = _rocm_or_skip()
    a = _RNG.standard_normal((nb, m, k)).astype(np.float32)
    b = _RNG.standard_normal((nb, k, n)).astype(np.float32)
    got = rt._rocm_batched_gemm_f32(a, b, np)
    assert got.shape == (nb, m, n)
    np.testing.assert_allclose(got, a @ b, rtol=1e-4, atol=1e-4)


def test_batched_gemm_broadcasts_rank2_operand():
    # np.matmul-style broadcast: a rank-2 shared operand against a batched one.
    rt = _rocm_or_skip()
    a = _RNG.standard_normal((3, 4)).astype(np.float32)          # (M, K)
    b = _RNG.standard_normal((5, 4, 6)).astype(np.float32)       # (Batch, K, N)
    got = rt._rocm_batched_gemm_f32(a, b, np)
    assert got.shape == (5, 3, 6)
    np.testing.assert_allclose(got, a @ b, rtol=1e-4, atol=1e-4)


def test_batched_gemm_rejects_mismatched_k():
    rt = _rocm_or_skip()
    a = np.zeros((2, 4, 8), np.float32)
    b = np.zeros((2, 7, 4), np.float32)     # K mismatch (8 vs 7)
    with pytest.raises(ValueError):
        rt._rocm_batched_gemm_f32(a, b, np)


def test_batched_gemm_matches_single_gemm_lane():
    # Per-batch, the batched kernel must equal the single-GEMM lane (same tile
    # logic, just batch-offset) — a cross-kernel consistency check.
    rt = _rocm_or_skip()
    a = _RNG.standard_normal((4, 12, 7)).astype(np.float32)
    b = _RNG.standard_normal((4, 7, 9)).astype(np.float32)
    batched = rt._rocm_batched_gemm_f32(a, b, np)
    for i in range(4):
        single = rt._rocm_f32_gemm(a[i], b[i], np)
        np.testing.assert_allclose(batched[i], single, rtol=1e-5, atol=1e-5)
