"""K-unrolled WMMA GEMM reference rung (tessera_rocm_wmma_gemm_f16_ku).

The K-unroll lever — process KU 16-wide K-panels per step — was measured on
gfx1151 and REGRESSES (the extra a/b load buffers blow the VGPR budget on this
occupancy-bound APU; see STRIX_HALO_EXECUTION_PLAN Stage H). It is kept as a
correctness-verified REFERENCE rung (like the LDS / pipelined rungs), not the
production path — production stays rung-1 register blocking (size-adaptive 2x4 /
3x4). This test locks the reference kernel's CORRECTNESS across shapes (incl.
ragged K where the tail loop runs) and KU factors. Skip-clean w/o the .so / GPU.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest


def _ku_or_skip():
    from tessera import runtime as rt
    lib = rt._load_rocm_gemm_runtime()
    if lib is None:
        pytest.skip("libtessera_rocm_gemm.so not loadable")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    fn = getattr(lib, "tessera_rocm_wmma_gemm_f16_ku", None)
    if fn is None:
        pytest.skip("libtessera_rocm_gemm.so lacks the _ku entry (rebuild)")
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 6
    fn.restype = ctypes.c_int
    return fn


_RNG = np.random.default_rng(3)


@pytest.mark.parametrize("M,N,K,mt,nt,ku", [
    (32, 32, 32, 2, 4, 2),
    (64, 48, 80, 3, 4, 2),
    (48, 64, 64, 4, 3, 2),
    (33, 17, 49, 2, 4, 2),      # ragged K (tail loop) + ragged M/N
    (64, 64, 64, 2, 4, 4),      # KU=4
    (80, 80, 96, 3, 4, 4),
    (16, 16, 17, 2, 4, 2),      # K=17: one main panel + a 1-wide tail
])
def test_ku_reference_matches_numpy(M, N, K, mt, nt, ku):
    fn = _ku_or_skip()
    a = (_RNG.standard_normal((M, K)) * 0.5).astype(np.float16)
    b = (_RNG.standard_normal((K, N)) * 0.5).astype(np.float16)
    d = np.zeros((M, N), np.float32)
    rc = fn(a.ctypes.data_as(ctypes.c_void_p),
            b.ctypes.data_as(ctypes.c_void_p),
            d.ctypes.data_as(ctypes.c_void_p), M, N, K, mt, nt, ku)
    assert rc == 0, f"kernel rc={rc}"
    ref = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(d, ref, rtol=5e-3, atol=5e-3)


def test_ku1_equals_production_semantics():
    # KU=1 is exactly the production register-blocked kernel — must be correct.
    fn = _ku_or_skip()
    a = (_RNG.standard_normal((48, 64)) * 0.5).astype(np.float16)
    b = (_RNG.standard_normal((64, 32)) * 0.5).astype(np.float16)
    d = np.zeros((48, 32), np.float32)
    rc = fn(a.ctypes.data_as(ctypes.c_void_p),
            b.ctypes.data_as(ctypes.c_void_p),
            d.ctypes.data_as(ctypes.c_void_p), 48, 32, 64, 2, 4, 1)
    assert rc == 0
    np.testing.assert_allclose(d, a.astype(np.float32) @ b.astype(np.float32),
                               rtol=5e-3, atol=5e-3)
