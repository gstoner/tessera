"""Autodiff + surface for clifford_exp / clifford_log (the GA exp/log tail).

Closes the "narrow exp/log GA autodiff tail": both ops are projected onto the
canonical tessera.ops surface (apple_gpu envelope → metal_runtime) and carry
exact, finite-difference-validated VJP+JVP:

  * exp — derivative of the truncated power series Σ aⁿ/n! is the
    non-commutative double sum Σ(1/n!)Σ_k aᵏ·da·aⁿ⁻¹⁻ᵏ; adjoint via the
    reverse-based geometric-product adjoint.
  * log — the Cl(3,0) closed-form rotor log (θ/2)·B̂ over the scalar+bivector
    subspace (grade-2 indices {3,5,6}), with the analytic adjoint.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.autodiff import get_jvp, get_vjp

O = ts.ops
_BIV = (3, 5, 6)


def _near_rotor(seed):
    r = np.random.default_rng(seed)
    a = np.zeros(8)
    a[0] = 1.0 + r.standard_normal() * 0.2
    for i in _BIV:
        a[i] = r.standard_normal() * 0.3
    return a


def _small(seed, shape=(8,)):
    return np.random.default_rng(seed).standard_normal(shape) * 0.25


# ── surface + registration ───────────────────────────────────────────────────
def test_on_namespace_and_registered():
    for n in ("clifford_exp", "clifford_log"):
        assert hasattr(O, n)
        assert get_vjp(n) is not None and get_jvp(n) is not None


def test_surface_matches_ga_lane():
    from tessera.ga import ops as G
    from tessera.ga.multivector import Multivector
    from tessera.ga.signature import Cl
    cl = Cl(3, 0, 0)
    a = _small(0).astype(np.float32)
    np.testing.assert_allclose(O.clifford_exp(a), G.exp_mv(Multivector(a, cl)).coefficients, atol=1e-5)
    r = _near_rotor(1).astype(np.float32)
    np.testing.assert_allclose(O.clifford_log(r), G.log_mv(Multivector(r, cl)).coefficients, atol=1e-5)


# ── JVP vs finite difference ─────────────────────────────────────────────────
def test_exp_jvp_matches_fd():
    a, da = _small(2), _small(3)
    _, tan = get_jvp("clifford_exp")((a,), (da,))
    eps = 1e-6
    fd = (np.asarray(O.clifford_exp(a + eps * da)) - np.asarray(O.clifford_exp(a - eps * da))) / (2 * eps)
    np.testing.assert_allclose(np.asarray(tan), fd, atol=1e-4)


def test_log_jvp_matches_fd():
    a, da = _near_rotor(4), _small(5)
    _, tan = get_jvp("clifford_log")((a,), (da,))
    eps = 1e-6
    fd = (np.asarray(O.clifford_log(a + eps * da)) - np.asarray(O.clifford_log(a - eps * da))) / (2 * eps)
    np.testing.assert_allclose(np.asarray(tan), fd, atol=1e-4)


# ── VJP adjoint consistency (<J da, g> == <da, J* g>) ────────────────────────
@pytest.mark.parametrize("name,make", [
    ("clifford_exp", _small), ("clifford_log", _near_rotor)])
def test_vjp_is_adjoint_of_jvp(name, make):
    rng = np.random.default_rng(hash(name) % 2**31)
    a = make(6)
    da = rng.standard_normal(8)
    g = rng.standard_normal(8)
    _, tan = get_jvp(name)((a,), (da,))
    grad = get_vjp(name)(g, a)[0]
    np.testing.assert_allclose(np.dot(np.asarray(tan), g), np.dot(da, grad), atol=1e-5)


# ── batched ──────────────────────────────────────────────────────────────────
def test_batched_exp_jvp():
    a = _small(7, (4, 8))
    da = np.random.default_rng(8).standard_normal((4, 8))
    _, tan = get_jvp("clifford_exp")((a,), (da,))
    eps = 1e-6
    fd = (np.asarray(O.clifford_exp(a + eps * da)) - np.asarray(O.clifford_exp(a - eps * da))) / (2 * eps)
    assert np.asarray(tan).shape == (4, 8)
    np.testing.assert_allclose(np.asarray(tan), fd, atol=1e-4)


# ── tape backward ─────────────────────────────────────────────────────────────
def test_tape_backward_log_matches_fd():
    a = _near_rotor(9)
    ap = ts.nn.Parameter(a.copy())
    with ts.autodiff.tape() as t:
        loss = O.reduce(O.clifford_log(ap), op="sum")
        t.backward(loss)

    def floss(x):
        return float(np.asarray(O.clifford_log(x)).sum())

    eps = 1e-6
    fd = np.zeros(8)
    for i in range(8):
        xp = a.copy(); xm = a.copy(); xp[i] += eps; xm[i] -= eps
        fd[i] = (floss(xp) - floss(xm)) / (2 * eps)
    np.testing.assert_allclose(ap.grad.numpy(), fd, atol=1e-4)
