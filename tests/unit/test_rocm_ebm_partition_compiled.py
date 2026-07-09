"""ROCm (gfx1151) native exact-partition lane — a compiler-generated full-array
warp-shuffle log-sum-exp reduction (generate-rocm-ebm-partition-kernel), the
counterpart to the AVX-512 lane. Validated on real gfx1151.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from tessera.ebm.partition import partition_exact_from_energies


def _rocm_or_skip():
    from tessera import runtime as rt
    if not (shutil.which("hipcc") or os.path.exists("/opt/rocm/bin/hipcc")):
        pytest.skip("no hipcc")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no live gfx1151")
    return rt


def _ref(E, T):
    neg = -E.astype(np.float64) / T
    m = float(neg.max())
    return float(np.exp(m + np.log(float(np.exp(neg - m).sum()))))


def test_rocm_partition_kernel_matches_numpy():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(0)
    for n, T in [(1, 1.0), (13, 0.5), (256, 1.0), (4096, 0.3), (100000, 2.0),
                 (999, 1.7)]:
        E = (rng.standard_normal(n) * 3.0).astype(np.float32)
        z = rt._rocm_ebm_partition(E, T, np)
        np.testing.assert_allclose(z, _ref(E, T), rtol=1e-4)


def test_rocm_from_energies_routes_native():
    _rocm_or_skip()
    rng = np.random.default_rng(2)
    E = (rng.standard_normal(777) * 2.0).astype(np.float32)
    z = partition_exact_from_energies(E, temperature=1.3)
    np.testing.assert_allclose(z, _ref(E, 1.3), rtol=1e-4)
