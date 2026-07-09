"""ROCm (gfx1151) native on-device-Philox Langevin lane — the compiler-generated
``generate-rocm-ebm-langevin-kernel``: one thread per element, in-kernel
Philox-4x32-10 + Box-Muller noise from ``(key, counter)`` (no host noise buffer).
Counterpart to the AVX-512 lane; validated on real gfx1151 against a numpy
Philox oracle.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from tessera.compiler.philox import philox_4x32_10


def _rocm_or_skip():
    from tessera import runtime as rt
    if not (shutil.which("hipcc") or os.path.exists("/opt/rocm/bin/hipcc")):
        pytest.skip("no hipcc")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no live gfx1151")
    return rt


def _oracle(y, g, eta, ns, key, ctr):
    key = np.asarray(key, np.uint32)
    ctr = np.asarray(ctr, np.uint32)
    fy = y.reshape(-1).astype(np.float64)
    fg = g.reshape(-1).astype(np.float64)
    out = np.zeros(fy.size)
    for i in range(fy.size):
        ci = np.array([ctr[0] + np.uint32(i), ctr[1], ctr[2], ctr[3]], np.uint32)
        o = philox_4x32_10(ci, key)
        u0 = (float(o[0]) + 0.5) * 2.0 ** -32
        u1 = (float(o[1]) + 0.5) * 2.0 ** -32
        z = np.sqrt(-2.0 * np.log(u0)) * np.cos(2.0 * np.pi * u1)
        out[i] = fy[i] - eta * fg[i] + ns * z
    return out.reshape(y.shape)


def test_rocm_langevin_philox_kernel_matches_oracle():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(0)
    key, ctr, eta, ns = (11, 22), (9, 1, 2, 3), 0.1, 0.25
    for shape in [(8,), (37,), (256,), (4, 64), (7, 999)]:
        y = rng.standard_normal(shape).astype(np.float32)
        g = rng.standard_normal(shape).astype(np.float32)
        out = rt._rocm_ebm_langevin(y, g, eta, ns, key[0], key[1],
                                    ctr[0], ctr[1], ctr[2], ctr[3], np)
        np.testing.assert_allclose(out.reshape(shape),
                                   _oracle(y, g, eta, ns, key, ctr),
                                   rtol=1e-4, atol=1e-4)
