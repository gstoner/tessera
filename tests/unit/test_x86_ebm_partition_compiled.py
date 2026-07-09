"""x86 native exact-partition lane (EBM3) — stable log-sum-exp reduction.

`tessera.ebm.partition_exact_from_energies` computes ``Z = Σ_i exp(-E_i/T)`` over
f32 per-state energies; this validates the native AVX-512 reduction kernel
(`tessera_x86_ebm_partition_exact_f32`, double-accumulated) that backs it, and
that the from-energies API routes through it.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from tessera.ebm.partition import partition_exact_from_energies


def _x86_or_skip():
    from tessera import runtime as rt
    if not (shutil.which("clang") or shutil.which("cc")
            or os.path.exists("/usr/bin/cc")):
        pytest.skip("no C toolchain")
    if rt._load_x86_elementwise() is None:
        pytest.skip("libtessera_x86_elementwise.so not built")
    return rt


def _ref(E, T):
    neg = -E.astype(np.float64) / T
    m = float(neg.max())
    return float(np.exp(m + np.log(float(np.exp(neg - m).sum()))))


def test_x86_partition_kernel_matches_numpy():
    rt = _x86_or_skip()
    rng = np.random.default_rng(0)
    for n, T in [(1, 1.0), (13, 0.5), (256, 1.0), (4096, 0.3), (100000, 2.0)]:
        E = (rng.standard_normal(n) * 3.0).astype(np.float32)
        z = rt._x86_ebm_partition_exact(E, T, np)
        np.testing.assert_allclose(z, _ref(E, T), rtol=1e-4)


def test_x86_from_energies_routes_native():
    _x86_or_skip()
    rng = np.random.default_rng(1)
    E = (rng.standard_normal(500) * 2.0).astype(np.float32)
    z = partition_exact_from_energies(E, temperature=0.7)
    np.testing.assert_allclose(z, _ref(E, 0.7), rtol=1e-4)
