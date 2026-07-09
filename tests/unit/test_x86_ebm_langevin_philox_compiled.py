"""x86 native on-device-Philox Langevin lane (EBM).

`tessera.ebm.langevin_step_philox(y, grad, eta=…, noise_scale=…, key, counter)`
computes ``out[i] = y[i] - eta*grad[i] + noise_scale*z[i]`` where ``z`` is drawn
IN-KERNEL from ``(key, counter)`` via Philox-4x32-10 + Box-Muller (per-thread
counter ``(counter[0]+i, counter[1..3])``) — no host noise buffer. This validates
the native AVX-512 kernel (`tessera_x86_ebm_langevin_philox_f32`) and that the
high-level API routes through it, against an independent numpy Philox oracle.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from tessera.compiler.philox import philox_4x32_10
from tessera.ebm.energy import langevin_step_philox


def _x86_or_skip():
    from tessera import runtime as rt
    if not (shutil.which("clang") or shutil.which("cc")
            or os.path.exists("/usr/bin/cc")):
        pytest.skip("no C toolchain")
    if rt._load_x86_elementwise() is None:
        pytest.skip("libtessera_x86_elementwise.so not built")
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


def test_x86_langevin_philox_kernel_matches_oracle():
    rt = _x86_or_skip()
    rng = np.random.default_rng(0)
    key, ctr, eta, ns = (11, 22), (9, 1, 2, 3), 0.1, 0.25
    for shape in [(8,), (37,), (256,), (4, 64)]:
        y = rng.standard_normal(shape).astype(np.float32)
        g = rng.standard_normal(shape).astype(np.float32)
        out = rt._x86_ebm_langevin(y, g, eta, ns, key[0], key[1],
                                   ctr[0], ctr[1], ctr[2], ctr[3], np)
        np.testing.assert_allclose(out.reshape(shape), _oracle(y, g, eta, ns, key, ctr),
                                   rtol=1e-4, atol=1e-4)


def test_x86_langevin_philox_routes_native():
    _x86_or_skip()
    rng = np.random.default_rng(1)
    key, ctr, eta, ns = (7, 0), (5, 0, 0, 0), 0.15, 0.3
    y = rng.standard_normal((16, 32)).astype(np.float32)
    g = rng.standard_normal((16, 32)).astype(np.float32)
    out = langevin_step_philox(y, g, eta=eta, noise_scale=ns, key=key, counter=ctr)
    np.testing.assert_allclose(out, _oracle(y, g, eta, ns, key, ctr),
                               rtol=1e-4, atol=1e-4)
