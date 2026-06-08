"""Autodiff + surface tests for the canonical ``tessera.ops.clifford_*`` shim.

The GA Multivector lane (``tessera.ga.*``) gets a flat-coefficient projection
onto the canonical ``tessera.ops`` surface (Cl(3,0), last-axis length 8). This
suite locks three contracts:

  1. **Surface parity** — each ``ts.ops.clifford_*`` wrapper matches the GA lane.
  2. **Reverse-mode** — every differentiable wrapper flows through the autodiff
     tape and its registered VJP matches a finite-difference oracle.
  3. **Forward-mode** — every registered JVP matches a finite-difference oracle.

All adjoints are closed-form (geometric-product-via-reverse, self-adjoint
involutions, basis-probe for the bilinear products, metric-probe for the norms).
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.autodiff import get_jvp, get_vjp

O = ts.ops

_BILINEAR = [
    "clifford_geometric_product",
    "clifford_wedge",
    "clifford_left_contraction",
    "clifford_inner",
    "clifford_rotor_sandwich",  # quadratic in the rotor arg; composed adjoint
]
_UNARY = [
    "clifford_reverse",
    "clifford_grade_involution",
    "clifford_conjugate",
    "clifford_norm",
    "clifford_norm_squared",
]
_ALL = _BILINEAR + _UNARY + ["clifford_grade_projection"]


def _fd_grad(fn, x, *, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp.flat[i] += eps
        xm.flat[i] -= eps
        g.flat[i] = (fn(xp) - fn(xm)) / (2 * eps)
    return g


# ── surface parity ──────────────────────────────────────────────────────────
def test_all_clifford_ops_on_namespace():
    for name in _ALL:
        assert hasattr(O, name), name


def test_surface_matches_ga_lane():
    from tessera.ga import ops as G
    from tessera.ga.multivector import Multivector
    from tessera.ga.signature import Cl

    cl = Cl(3, 0, 0)
    rng = np.random.default_rng(0)
    a = rng.standard_normal(8).astype(np.float32)
    b = rng.standard_normal(8).astype(np.float32)
    ma, mb = Multivector(a, cl), Multivector(b, cl)
    np.testing.assert_allclose(
        O.clifford_geometric_product(a, b), G.geometric_product(ma, mb).coefficients, atol=1e-6)
    np.testing.assert_allclose(O.clifford_wedge(a, b), G.wedge(ma, mb).coefficients, atol=1e-6)
    np.testing.assert_allclose(
        O.clifford_left_contraction(a, b), G.left_contraction(ma, mb).coefficients, atol=1e-6)
    np.testing.assert_allclose(O.clifford_reverse(a), G.reverse(ma).coefficients, atol=1e-6)
    np.testing.assert_allclose(O.clifford_grade_projection(a, grade=2), G.grade_projection(ma, 2).coefficients, atol=1e-6)
    np.testing.assert_allclose(float(O.clifford_norm(a)), float(G.norm(ma)), atol=1e-6)


# ── registration completeness ────────────────────────────────────────────────
def test_every_op_has_vjp_and_jvp():
    for name in _ALL:
        assert get_vjp(name) is not None, f"missing VJP for {name}"
        assert get_jvp(name) is not None, f"missing JVP for {name}"


# ── reverse-mode via the tape ─────────────────────────────────────────────────
@pytest.mark.parametrize("name", _BILINEAR)
def test_tape_backward_bilinear(name):
    op = getattr(O, name)
    rng = np.random.default_rng(hash(name) % 2**31)
    a = rng.standard_normal(8)
    b = rng.standard_normal(8)
    ap = ts.nn.Parameter(a.copy())
    bp = ts.nn.Parameter(b.copy())
    with ts.autodiff.tape() as t:
        out = op(ap, bp)
        loss = O.reduce(out, op="sum") if np.asarray(out).ndim else out
        t.backward(loss)

    def scalar(x, y):
        r = np.asarray(op(x, y))
        return float(r.sum()) if r.ndim else float(r)

    np.testing.assert_allclose(ap.grad.numpy(), _fd_grad(lambda x: scalar(x, b), a), atol=1e-4)
    np.testing.assert_allclose(bp.grad.numpy(), _fd_grad(lambda y: scalar(a, y), b), atol=1e-4)


@pytest.mark.parametrize("name", _UNARY)
def test_tape_backward_unary(name):
    op = getattr(O, name)
    rng = np.random.default_rng(hash(name) % 2**31)
    a = rng.standard_normal(8) + 0.5  # keep norm away from 0
    ap = ts.nn.Parameter(a.copy())
    with ts.autodiff.tape() as t:
        out = op(ap)
        loss = O.reduce(out, op="sum") if np.asarray(out).ndim else out
        t.backward(loss)

    def scalar(x):
        r = np.asarray(op(x))
        return float(r.sum()) if r.ndim else float(r)

    np.testing.assert_allclose(ap.grad.numpy(), _fd_grad(scalar, a), atol=1e-4)


def test_tape_backward_grade_projection():
    rng = np.random.default_rng(11)
    a = rng.standard_normal(8)
    ap = ts.nn.Parameter(a.copy())
    with ts.autodiff.tape() as t:
        out = O.clifford_grade_projection(ap, grade=2)
        loss = O.reduce(out, op="sum")
        t.backward(loss)
    grad = _fd_grad(lambda x: float(np.asarray(O.clifford_grade_projection(x, grade=2)).sum()), a)
    np.testing.assert_allclose(ap.grad.numpy(), grad, atol=1e-4)


# ── forward-mode (JVP) ────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", _BILINEAR)
def test_jvp_bilinear(name):
    op = getattr(O, name)
    jvp = get_jvp(name)
    rng = np.random.default_rng(hash(name) % 2**31 + 1)
    a, b, da, db = (rng.standard_normal(8) for _ in range(4))
    _, tan = jvp((a, b), (da, db))
    eps = 1e-6
    fd = (np.asarray(op(a + eps * da, b + eps * db)) - np.asarray(op(a - eps * da, b - eps * db))) / (2 * eps)
    np.testing.assert_allclose(np.asarray(tan), fd, atol=1e-4)


@pytest.mark.parametrize("name", _UNARY)
def test_jvp_unary(name):
    op = getattr(O, name)
    jvp = get_jvp(name)
    rng = np.random.default_rng(hash(name) % 2**31 + 2)
    a = rng.standard_normal(8) + 0.5
    da = rng.standard_normal(8)
    _, tan = jvp((a,), (da,))
    eps = 1e-6
    fd = (np.asarray(op(a + eps * da)) - np.asarray(op(a - eps * da))) / (2 * eps)
    np.testing.assert_allclose(np.asarray(tan), fd, atol=1e-4)


def test_jvp_grade_projection():
    jvp = get_jvp("clifford_grade_projection")
    rng = np.random.default_rng(13)
    a = rng.standard_normal(8)
    da = rng.standard_normal(8)
    _, tan = jvp((a, 2), (da, None))
    eps = 1e-6
    fd = (np.asarray(O.clifford_grade_projection(a + eps * da, 2))
          - np.asarray(O.clifford_grade_projection(a - eps * da, 2))) / (2 * eps)
    np.testing.assert_allclose(np.asarray(tan), fd, atol=1e-4)


# ── batched leading dims ──────────────────────────────────────────────────────
def test_batched_geometric_product_vjp():
    vjp = get_vjp("clifford_geometric_product")
    rng = np.random.default_rng(21)
    a = rng.standard_normal((4, 8))
    b = rng.standard_normal((4, 8))
    dout = rng.standard_normal((4, 8))
    da, db = vjp(dout, a, b)
    assert da.shape == (4, 8) and db.shape == (4, 8)
    # per-row equals the unbatched VJP
    for i in range(4):
        da_i, db_i = vjp(dout[i], a[i], b[i])
        np.testing.assert_allclose(da[i], da_i, atol=1e-6)
        np.testing.assert_allclose(db[i], db_i, atol=1e-6)
