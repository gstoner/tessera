"""ROCm (gfx1151) native affine-Langevin lane for the manifold EBM samplers.

Counterpart to test_x86_ebm_geo_langevin_compiled.py: validates the
compiler-generated ``generate-rocm-ebm-affine-langevin-kernel`` (noise as an
input) that backs `tessera.ebm.bivector_langevin_step`'s device fast path, on
real gfx1151 hardware.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from tessera.ga import Cl, Multivector
from tessera.rng import RNGKey
from tessera.ebm.geo_sampling import bivector_langevin_step, sphere_langevin_step


def _rocm_or_skip():
    from tessera import runtime as rt
    if not (shutil.which("hipcc") or os.path.exists("/opt/rocm/bin/hipcc")):
        pytest.skip("no hipcc")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no live gfx1151")
    return rt


def _quadratic(mv):
    return 0.5 * float((mv.coefficients ** 2).sum())


def test_rocm_affine_langevin_matches_numpy():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(0)
    y = rng.standard_normal(96).astype(np.float32)
    g = rng.standard_normal(96).astype(np.float32)
    z = rng.standard_normal(96).astype(np.float32)
    eta, ns = 0.15, 0.5
    out = rt._rocm_ebm_affine_langevin(y, g, z, eta, ns, np)
    np.testing.assert_allclose(out, y - eta * g + ns * z, rtol=0, atol=1e-5)


def test_rocm_bivector_step_f32_native_and_in_grade():
    _rocm_or_skip()
    a = Cl(3, 0)
    coeffs = np.zeros(a.dim, dtype=np.float32)
    coeffs[a.blade("e12").mask] = 0.7
    coeffs[a.blade("e13").mask] = -1.3
    coeffs[a.blade("e23").mask] = 0.4
    state = Multivector(coeffs, a, grades={2})
    key = RNGKey.from_seed(0)
    out, _ = bivector_langevin_step(state, _quadratic, eta=0.02,
                                    temperature=1.0, rng_key=key, grade=2)
    assert out.coefficients.dtype == np.float32
    for blade in a.blades():
        if blade.grade != 2:
            assert abs(float(out.coefficients[blade.mask])) < 1e-6


def test_rocm_sphere_step_f32_native_stays_on_sphere():
    _rocm_or_skip()
    d = 16
    rng = np.random.default_rng(1)
    x = rng.standard_normal(d).astype(np.float32)
    x = (x / np.linalg.norm(x)).astype(np.float32)

    def grad_fn(v):
        return np.asarray(v, np.float32)

    out, _ = sphere_langevin_step(
        x, lambda v: 0.5 * float((np.asarray(v) ** 2).sum()),
        eta=0.02, temperature=1.0, rng_key=RNGKey.from_seed(0), grad_fn=grad_fn)
    assert out.dtype == np.float32
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-4
