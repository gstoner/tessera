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

import importlib

from tessera.ga import Cl, Multivector
from tessera.rng import RNGKey
from tessera.ebm.geo_sampling import (
    bivector_langevin_sample,
    bivector_langevin_step,
    sphere_langevin_sample,
    sphere_langevin_step,
)


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


# ── the `_sample` chains compose the native per-step kernel in a host Markov ───
# loop. On this box the step's x86 lane wins first, so force it off to route the
# chain through the ROCm affine kernel; then force ROCm off too for the pure-numpy
# reference chain and require step-for-step agreement.

def test_rocm_bivector_sample_chain_matches_numpy(monkeypatch):
    _rocm_or_skip()
    energy = importlib.import_module("tessera.ebm.energy")
    monkeypatch.setattr(energy, "_try_x86_ebm_affine_langevin_step_f32",
                        lambda *a, **k: None)
    a = Cl(3, 0)
    coeffs = np.zeros(a.dim, dtype=np.float32)
    coeffs[a.blade("e12").mask] = 0.7
    coeffs[a.blade("e13").mask] = -1.3
    coeffs[a.blade("e23").mask] = 0.4
    init = Multivector(coeffs, a, grades={2})
    kw = dict(init=init, energy_fn=_quadratic, eta=0.02, temperature=1.0,
              n_samples=6, burn_in=2, grade=2)
    key = RNGKey.from_seed(1)
    native, _, _ = bivector_langevin_sample(key, **kw)      # ROCm native chain
    monkeypatch.setattr(energy, "_try_rocm_ebm_affine_langevin_step_f32",
                        lambda *a, **k: None)
    ref, _, _ = bivector_langevin_sample(key, **kw)         # pure-numpy chain
    np.testing.assert_allclose(native, ref, rtol=1e-4, atol=1e-4)
    nong = [b.mask for b in a.blades() if b.grade != 2]
    assert float(np.abs(native[:, nong]).max()) < 1e-6


def test_rocm_sphere_sample_chain_matches_numpy(monkeypatch):
    _rocm_or_skip()
    energy = importlib.import_module("tessera.ebm.energy")
    monkeypatch.setattr(energy, "_try_x86_ebm_affine_langevin_step_f32",
                        lambda *a, **k: None)
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal(16).astype(np.float32)

    def grad_fn(v):
        return np.asarray(v, np.float32)

    kw = dict(init=x0, energy_fn=lambda v: 0.5 * float((np.asarray(v) ** 2).sum()),
              eta=0.02, temperature=1.0, n_samples=6, burn_in=2, grad_fn=grad_fn)
    key = RNGKey.from_seed(1)
    native, _, _ = sphere_langevin_sample(key, **kw)        # ROCm native chain
    monkeypatch.setattr(energy, "_try_rocm_ebm_affine_langevin_step_f32",
                        lambda *a, **k: None)
    ref, _, _ = sphere_langevin_sample(key, **kw)           # pure-numpy chain
    np.testing.assert_allclose(native, ref, rtol=1e-4, atol=1e-4)
    norms = np.linalg.norm(native, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)
