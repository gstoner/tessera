"""x86 native decode-init noise-apply lane (DFlash / EBM speculative decode).

`tessera.ebm.decode_init(init_strategy="noise", mean=…)` seeds K candidate
trajectories via ``out = base + std*noise``; this validates the native AVX-512
kernel (`tessera_x86_ebm_decode_init_noise_apply_f32`, double-accumulated) that
backs it, and that the noise-strategy API routes through it.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from tessera.ebm import decode_init
from tessera.rng import RNGKey


def _x86_or_skip():
    from tessera import runtime as rt
    if not (shutil.which("clang") or shutil.which("cc")
            or os.path.exists("/usr/bin/cc")):
        pytest.skip("no C toolchain")
    if rt._load_x86_elementwise() is None:
        pytest.skip("libtessera_x86_elementwise.so not built")
    return rt


def test_x86_decode_init_kernel_matches_numpy():
    rt = _x86_or_skip()
    rng = np.random.default_rng(0)
    for shape, std in [((1,), 1.0), ((256,), 0.5), ((3, 5, 7), 0.3),
                       ((10000,), 2.0)]:
        base = (rng.standard_normal(shape) * 2.0).astype(np.float32)
        noise = rng.standard_normal(shape).astype(np.float32)
        out = rt._x86_ebm_decode_init(base, noise, std, np)
        np.testing.assert_allclose(out, base + std * noise, rtol=1e-4, atol=1e-5)
        assert out.shape == base.shape


def test_x86_decode_init_routes_native():
    _x86_or_skip()
    B, K, D = 2, 4, 8
    mean = np.arange(B * K * D, dtype=np.float32).reshape(B, K, D)
    out = decode_init(np.zeros((B, 1), dtype=np.float32), K=K,
                      init_strategy="noise", rng_key=RNGKey(0), shape=(D,),
                      dtype="fp32", std=0.5, mean=mean)
    assert out.shape == (B, K, D)
    # The native lane computes base + std*noise; recompute the expected noise
    # from the same RNGKey to check the affine combine end-to-end.
    from tessera.rng import normal
    noise = normal(RNGKey(0), shape=(B, K, D), dtype="fp32", std=1.0)
    np.testing.assert_allclose(out, mean + 0.5 * noise, rtol=1e-4, atol=1e-5)
