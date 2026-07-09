"""x86 native EBT-tiny lane (EBM) — a dedicated fused inference step.

`tessera.ebm.ebt_tiny(y0, grad, eta=…, T=…, B, K, D)` fuses, per batch, the
closed-form refinement ``y_T = y0 - T*eta*grad``, the squared-norm energy
``e[k] = Σ_d y_T[k,d]^2``, a hard-argmin over K candidates (first-min
tie-break), and the gather of the winner into ``out[B, D]``. This validates the
native AVX-512 kernel (`tessera_x86_ebm_ebt_tiny_f32`, energies double-
accumulated) that backs it, and that the high-level API routes through it.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from tessera.ebm.energy import ebt_tiny, ebt_tiny_last_route


def _x86_or_skip():
    from tessera import runtime as rt
    if not (shutil.which("clang") or shutil.which("cc")
            or os.path.exists("/usr/bin/cc")):
        pytest.skip("no C toolchain")
    if rt._load_x86_elementwise() is None:
        pytest.skip("libtessera_x86_elementwise.so not built")
    return rt


def _ref(y0, grad, eta, T, B, K, D):
    y_T = y0 - (T * eta) * grad
    energies = np.sum(y_T * y_T, axis=1).reshape(B, K)
    candidates = y_T.reshape(B, K, D)
    return candidates[np.arange(B), energies.argmin(axis=1)]


def test_x86_ebt_tiny_kernel_matches_numpy():
    rt = _x86_or_skip()
    rng = np.random.default_rng(0)
    eta = 0.05
    for B, K, D, T in [(1, 1, 1, 1), (2, 4, 6, 3), (3, 8, 257, 2),
                       (4, 16, 128, 5), (2, 256, 64, 1), (5, 3, 999, 4)]:
        y0 = rng.standard_normal((B * K, D)).astype(np.float32)
        grad = rng.standard_normal((B * K, D)).astype(np.float32)
        out = rt._x86_ebm_ebt_tiny(y0, grad, eta, T, B, K, D, np)
        assert out.shape == (B, D)
        np.testing.assert_allclose(
            out, _ref(y0, grad, eta, T, B, K, D), rtol=1e-4, atol=1e-4)


def test_x86_ebt_tiny_routes_native():
    _x86_or_skip()
    rng = np.random.default_rng(1)
    B, K, D, T, eta = 4, 8, 64, 3, 0.05
    y0 = rng.standard_normal((B * K, D)).astype(np.float32)
    grad = rng.standard_normal((B * K, D)).astype(np.float32)
    out = ebt_tiny(y0, grad, eta=eta, T=T, B=B, K=K, D=D)
    assert ebt_tiny_last_route() == "x86"
    np.testing.assert_allclose(
        out, _ref(y0, grad, eta, T, B, K, D), rtol=1e-4, atol=1e-4)
