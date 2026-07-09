"""x86 native affine-Langevin lane for the manifold EBM samplers (P7 follow-up).

`tessera.ebm.bivector_langevin_step` projects the gradient + a host-drawn Gaussian
onto a grade subspace, then takes the affine combination
``out = y - eta*grad + noise_scale*noise`` — the same shape as ``ebm_langevin_step``
but with the noise supplied FROM THE HOST (not device Philox). This validates the
native AVX-512 kernel (`tessera_x86_ebm_affine_langevin_f32`, noise as an input)
that backs the ROCm/x86 fast path, and that a float32 `bivector_langevin_step`
routes through it and matches the numpy reference.
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


def _x86_lib_or_skip():
    from tessera import runtime as rt
    if not (shutil.which("clang") or shutil.which("cc")
            or os.path.exists("/usr/bin/cc")):
        pytest.skip("no C toolchain")
    if rt._load_x86_elementwise() is None:
        pytest.skip("libtessera_x86_elementwise.so not built")
    return rt


def _quadratic(mv):
    return 0.5 * float((mv.coefficients ** 2).sum())


# ── the native affine kernel matches numpy ────────────────────────────────────

def test_x86_affine_langevin_matches_numpy():
    rt = _x86_lib_or_skip()
    rng = np.random.default_rng(0)
    y = rng.standard_normal(64).astype(np.float32)
    g = rng.standard_normal(64).astype(np.float32)
    z = rng.standard_normal(64).astype(np.float32)
    eta, ns = 0.1, 0.7
    out = rt._x86_ebm_affine_langevin(y, g, z, eta, ns, np)
    np.testing.assert_allclose(out, y - eta * g + ns * z, rtol=0, atol=1e-5)


# ── a float32 bivector step routes through the native lane + stays in grade ───

def test_x86_bivector_step_f32_native_and_in_grade():
    _x86_lib_or_skip()
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
    # No leakage outside grade 2 (the affine of grade-restricted inputs stays
    # in-grade — the native lane skips the numpy path's cleanup projection).
    for blade in a.blades():
        if blade.grade != 2:
            assert abs(float(out.coefficients[blade.mask])) < 1e-6


def test_x86_sphere_step_f32_native_stays_on_sphere():
    # sphere_langevin_step's f32 fast path: host tangent-projection + the native
    # affine kernel + host normalize (retract). The result must land back on the
    # unit sphere — proving the affine core ran and the retraction closed.
    _x86_lib_or_skip()
    d = 16
    rng = np.random.default_rng(1)
    x = rng.standard_normal(d).astype(np.float32)
    x = (x / np.linalg.norm(x)).astype(np.float32)

    def grad_fn(v):
        return np.asarray(v, np.float32)            # E = ||v||^2/2 → ∇E = v

    out, _ = sphere_langevin_step(
        x, lambda v: 0.5 * float((np.asarray(v) ** 2).sum()),
        eta=0.02, temperature=1.0, rng_key=RNGKey.from_seed(0), grad_fn=grad_fn)
    assert out.dtype == np.float32
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-4


# ── the `_sample` chains compose the native per-step kernel in a host Markov ───
# loop (like the spectral dct/stft composites over the device FFT executor). A
# native chain must reproduce the pure-numpy chain step-for-step (same host RNG
# draws + host projection; only the affine combine differs, and that's the kernel
# proven above). Forcing both device lanes off yields the numpy reference.

def test_x86_bivector_sample_chain_matches_numpy(monkeypatch):
    _x86_lib_or_skip()
    a = Cl(3, 0)
    coeffs = np.zeros(a.dim, dtype=np.float32)
    coeffs[a.blade("e12").mask] = 0.7
    coeffs[a.blade("e13").mask] = -1.3
    coeffs[a.blade("e23").mask] = 0.4
    init = Multivector(coeffs, a, grades={2})
    kw = dict(init=init, energy_fn=_quadratic, eta=0.02, temperature=1.0,
              n_samples=6, burn_in=2, grade=2)
    key = RNGKey.from_seed(1)
    native, _, _ = bivector_langevin_sample(key, **kw)      # x86 native chain
    energy = importlib.import_module("tessera.ebm.energy")
    monkeypatch.setattr(energy, "_try_x86_ebm_affine_langevin_step_f32",
                        lambda *a, **k: None)
    monkeypatch.setattr(energy, "_try_rocm_ebm_affine_langevin_step_f32",
                        lambda *a, **k: None)
    ref, _, _ = bivector_langevin_sample(key, **kw)         # pure-numpy chain
    np.testing.assert_allclose(native, ref, rtol=1e-4, atol=1e-4)
    nong = [b.mask for b in a.blades() if b.grade != 2]
    assert float(np.abs(native[:, nong]).max()) < 1e-6     # in-grade through chain


def test_x86_sphere_sample_chain_matches_numpy(monkeypatch):
    _x86_lib_or_skip()
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal(16).astype(np.float32)

    def grad_fn(v):
        return np.asarray(v, np.float32)

    kw = dict(init=x0, energy_fn=lambda v: 0.5 * float((np.asarray(v) ** 2).sum()),
              eta=0.02, temperature=1.0, n_samples=6, burn_in=2, grad_fn=grad_fn)
    key = RNGKey.from_seed(1)
    native, _, _ = sphere_langevin_sample(key, **kw)        # x86 native chain
    energy = importlib.import_module("tessera.ebm.energy")
    monkeypatch.setattr(energy, "_try_x86_ebm_affine_langevin_step_f32",
                        lambda *a, **k: None)
    monkeypatch.setattr(energy, "_try_rocm_ebm_affine_langevin_step_f32",
                        lambda *a, **k: None)
    ref, _, _ = sphere_langevin_sample(key, **kw)           # pure-numpy chain
    np.testing.assert_allclose(native, ref, rtol=1e-4, atol=1e-4)
    norms = np.linalg.norm(native, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)       # on-sphere through chain
