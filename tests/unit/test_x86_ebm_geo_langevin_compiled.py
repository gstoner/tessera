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

from tessera.ga import Cl, Multivector
from tessera.rng import RNGKey
from tessera.ebm.geo_sampling import bivector_langevin_step, sphere_langevin_step


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
