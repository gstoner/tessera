"""ROCm (gfx1151) native decode-init noise-apply lane — a compiler-generated
elementwise ``out = base + std*noise`` kernel (generate-rocm-ebm-decode-init-
kernel), the counterpart to the AVX-512 lane. Validated on real gfx1151.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from tessera.ebm import decode_init
from tessera.rng import RNGKey


def _rocm_or_skip():
    from tessera import runtime as rt
    if not (shutil.which("hipcc") or os.path.exists("/opt/rocm/bin/hipcc")):
        pytest.skip("no hipcc")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no live gfx1151")
    return rt


def test_rocm_decode_init_kernel_matches_numpy():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(0)
    for shape, std in [((1,), 1.0), ((256,), 0.5), ((3, 5, 7), 0.3),
                       ((10000,), 2.0), ((999,), 1.7)]:
        base = (rng.standard_normal(shape) * 2.0).astype(np.float32)
        noise = rng.standard_normal(shape).astype(np.float32)
        out = rt._rocm_ebm_decode_init(base, noise, std, np)
        np.testing.assert_allclose(out, base + std * noise, rtol=1e-4, atol=1e-5)
        assert out.shape == base.shape


def test_rocm_decode_init_routes_native():
    _rocm_or_skip()
    B, K, D = 2, 4, 8
    mean = np.arange(B * K * D, dtype=np.float32).reshape(B, K, D)
    out = decode_init(np.zeros((B, 1), dtype=np.float32), K=K,
                      init_strategy="noise", rng_key=RNGKey(0), shape=(D,),
                      dtype="fp32", std=0.5, mean=mean)
    assert out.shape == (B, K, D)
    from tessera.rng import normal
    noise = normal(RNGKey(0), shape=(B, K, D), dtype="fp32", std=1.0)
    np.testing.assert_allclose(out, mean + 0.5 * noise, rtol=1e-4, atol=1e-5)
