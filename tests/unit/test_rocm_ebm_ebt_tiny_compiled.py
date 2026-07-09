"""ROCm (gfx1151) native EBT-tiny lane — a compiler-generated fused inference
step (generate-rocm-ebm-ebt-tiny-kernel): one workgroup per batch, thread k
computes candidate k's closed-form energy, a shared-memory tree argmin over the
256 lanes (first-min tie-break) picks k*, then the lanes stride over D to write
the winner. The counterpart to the AVX-512 lane. Validated on real gfx1151.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest


def _rocm_or_skip():
    from tessera import runtime as rt
    if not (shutil.which("hipcc") or os.path.exists("/opt/rocm/bin/hipcc")):
        pytest.skip("no hipcc")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no live gfx1151")
    return rt


def _ref(y0, grad, eta, T, B, K, D):
    y_T = y0 - (T * eta) * grad
    energies = np.sum(y_T * y_T, axis=1).reshape(B, K)
    candidates = y_T.reshape(B, K, D)
    return candidates[np.arange(B), energies.argmin(axis=1)]


def test_rocm_ebt_tiny_kernel_matches_numpy():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(0)
    eta = 0.05
    for B, K, D, T in [(1, 1, 1, 1), (2, 4, 6, 3), (3, 8, 257, 2),
                       (4, 16, 128, 5), (2, 256, 64, 1), (5, 3, 999, 4)]:
        y0 = rng.standard_normal((B * K, D)).astype(np.float32)
        grad = rng.standard_normal((B * K, D)).astype(np.float32)
        out = rt._rocm_ebm_ebt_tiny(y0, grad, eta, T, B, K, D, np)
        assert out.shape == (B, D)
        np.testing.assert_allclose(
            out, _ref(y0, grad, eta, T, B, K, D), rtol=1e-4, atol=1e-4)
