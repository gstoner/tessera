"""ROCm (gfx1151) native quadratic-energy lane — a compiler-generated per-row
``0.5*||x-y||^2`` reduction (generate-rocm-ebm-energy-quadratic-kernel): one
workgroup per row, warp-shuffle sum-of-squares. The counterpart to the AVX-512
lane. Validated on real gfx1151.
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


def _ref(x, y):
    return 0.5 * np.sum((x - y) ** 2, axis=tuple(range(1, x.ndim)))


def test_rocm_energy_quadratic_kernel_matches_numpy():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(0)
    for shape in [(1, 1), (4, 6), (8, 257), (3, 5, 7), (64, 128), (2, 4096),
                  (7, 999)]:
        x = rng.standard_normal(shape).astype(np.float32)
        y = rng.standard_normal(shape).astype(np.float32)
        out = rt._rocm_ebm_energy_quadratic(x, y, np)
        np.testing.assert_allclose(out, _ref(x, y), rtol=1e-4, atol=1e-4)
        assert out.shape == (shape[0],)
