"""x86 native quadratic-energy lane (EBM) — a dedicated fused per-row reduction.

`tessera.ebm.energy_quadratic(x, y)` computes the dominant EBT / diffusion energy
``out[b] = 0.5*||x_b - y_b||^2``; this validates the native AVX-512 kernel
(`tessera_x86_ebm_energy_quadratic_f32`, per-row double-accumulated) that backs
it, and that the high-level API routes through it. This is the concrete energy
shared by the `ebm_energy` / `ebm_energy_quadratic` manifest ops.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from tessera.ebm.energy import energy_quadratic


def _x86_or_skip():
    from tessera import runtime as rt
    if not (shutil.which("clang") or shutil.which("cc")
            or os.path.exists("/usr/bin/cc")):
        pytest.skip("no C toolchain")
    if rt._load_x86_elementwise() is None:
        pytest.skip("libtessera_x86_elementwise.so not built")
    return rt


def _ref(x, y):
    return 0.5 * np.sum((x - y) ** 2, axis=tuple(range(1, x.ndim)))


def test_x86_energy_quadratic_kernel_matches_numpy():
    rt = _x86_or_skip()
    rng = np.random.default_rng(0)
    for shape in [(1, 1), (4, 6), (8, 257), (3, 5, 7), (64, 128), (2, 4096)]:
        x = rng.standard_normal(shape).astype(np.float32)
        y = rng.standard_normal(shape).astype(np.float32)
        out = rt._x86_ebm_energy_quadratic(x, y, np)
        np.testing.assert_allclose(out, _ref(x, y), rtol=1e-4, atol=1e-4)
        assert out.shape == (shape[0],)


def test_x86_energy_quadratic_routes_native():
    _x86_or_skip()
    rng = np.random.default_rng(1)
    x = rng.standard_normal((16, 64)).astype(np.float32)
    y = rng.standard_normal((16, 64)).astype(np.float32)
    out = energy_quadratic(x, y)
    np.testing.assert_allclose(out, _ref(x, y), rtol=1e-4, atol=1e-4)
