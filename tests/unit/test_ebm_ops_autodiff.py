"""Autodiff + surface tests for the canonical ``tessera.ops.ebm_*`` shim.

The tensor-clean EBM subset (``energy_quadratic`` / ``self_verify`` /
``refinement`` / ``inner_step``) is projected onto the canonical ``tessera.ops``
surface. This suite locks surface parity, reverse-mode (tape) gradients, and
forward-mode (JVP) — each adjoint validated against a finite-difference oracle.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.autodiff import get_jvp, get_vjp

O = ts.ops
_NAMES = ["ebm_energy_quadratic", "ebm_self_verify", "ebm_refinement", "ebm_inner_step"]


def _fd(f, x, dout, eps=1e-6):
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy(); xm = x.copy(); xp.flat[i] += eps; xm.flat[i] -= eps
        g.flat[i] = np.sum((np.asarray(f(xp)) - np.asarray(f(xm))) / (2 * eps) * np.asarray(dout, float))
    return g


# ── surface ───────────────────────────────────────────────────────────────────
def test_ops_present_and_match_lane():
    import tessera.ebm as E
    for n in _NAMES:
        assert hasattr(O, n), n
        assert get_vjp(n) is not None and get_jvp(n) is not None, n
    x = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    y = np.random.default_rng(1).standard_normal((3, 4)).astype(np.float32)
    np.testing.assert_allclose(O.ebm_energy_quadratic(x, y), E.energy_quadratic(x, y), atol=1e-5)


# ── reverse-mode (tape) ─────────────────────────────────────────────────────────
def test_energy_quadratic_tape_grad():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 4)); y = rng.standard_normal((3, 4))
    xp = ts.nn.Parameter(x.copy()); yp = ts.nn.Parameter(y.copy())
    with ts.autodiff.tape() as t:
        e = O.ebm_energy_quadratic(xp, yp)
        loss = O.reduce(e, op="sum")
        t.backward(loss)
    dout = np.ones(3)
    np.testing.assert_allclose(xp.grad.numpy(), _fd(lambda v: O.ebm_energy_quadratic(v, y), x, dout), atol=1e-4)
    np.testing.assert_allclose(yp.grad.numpy(), _fd(lambda v: O.ebm_energy_quadratic(x, v), y, dout), atol=1e-4)


def test_refinement_and_inner_step_tape_grad():
    rng = np.random.default_rng(3)
    y0 = rng.standard_normal((2, 3)); grad = rng.standard_normal((2, 3))
    yp = ts.nn.Parameter(y0.copy()); gp = ts.nn.Parameter(grad.copy())
    with ts.autodiff.tape() as t:
        out = O.ebm_refinement(yp, gp, eta=0.1, T=4)
        loss = O.reduce(out, op="sum")
        t.backward(loss)
    dout = np.ones((2, 3))
    np.testing.assert_allclose(yp.grad.numpy(), _fd(lambda v: O.ebm_refinement(v, grad, eta=0.1, T=4), y0, dout), atol=1e-4)
    np.testing.assert_allclose(gp.grad.numpy(), _fd(lambda v: O.ebm_refinement(y0, v, eta=0.1, T=4), grad, dout), atol=1e-4)

    yp2 = ts.nn.Parameter(y0.copy()); gp2 = ts.nn.Parameter(grad.copy())
    with ts.autodiff.tape() as t:
        out = O.ebm_inner_step(yp2, gp2, eta=0.2)
        loss = O.reduce(out, op="sum")
        t.backward(loss)
    np.testing.assert_allclose(yp2.grad.numpy(), _fd(lambda v: O.ebm_inner_step(v, grad, eta=0.2), y0, dout), atol=1e-4)
    np.testing.assert_allclose(gp2.grad.numpy(), _fd(lambda v: O.ebm_inner_step(y0, v, eta=0.2), grad, dout), atol=1e-4)


def test_self_verify_soft_tape_grad():
    rng = np.random.default_rng(4)
    e = rng.standard_normal((3, 4)); c = rng.standard_normal((3, 4, 2))
    ep = ts.nn.Parameter(e.copy()); cp = ts.nn.Parameter(c.copy())
    with ts.autodiff.tape() as t:
        out = O.ebm_self_verify(ep, cp, beta=1.5)
        loss = O.reduce(out, op="sum")
        t.backward(loss)
    dout = np.ones((3, 2))
    np.testing.assert_allclose(ep.grad.numpy(), _fd(lambda v: O.ebm_self_verify(v, c, beta=1.5), e, dout), atol=1e-4)
    np.testing.assert_allclose(cp.grad.numpy(), _fd(lambda v: O.ebm_self_verify(e, v, beta=1.5), c, dout), atol=1e-4)


def test_self_verify_hard_vjp_selects_argmin():
    vjp = get_vjp("ebm_self_verify")
    e = np.array([[3.0, 1.0, 2.0]])           # argmin = index 1
    c = np.arange(3.0).reshape(1, 3, 1)
    dout = np.array([[5.0]])
    de, dc = vjp(dout, e, c, beta=None)
    np.testing.assert_array_equal(de, np.zeros_like(e))            # argmin non-diff
    expect = np.zeros_like(c); expect[0, 1, 0] = 5.0
    np.testing.assert_array_equal(dc, expect)


# ── forward-mode (JVP) ──────────────────────────────────────────────────────────
def test_jvps_match_finite_difference():
    rng = np.random.default_rng(5)
    eps = 1e-6
    # energy_quadratic
    x = rng.standard_normal((3, 4)); y = rng.standard_normal((3, 4)); dx = rng.standard_normal((3, 4)); dy = rng.standard_normal((3, 4))
    _, tan = get_jvp("ebm_energy_quadratic")((x, y), (dx, dy))
    fd = (O.ebm_energy_quadratic(x + eps * dx, y + eps * dy) - O.ebm_energy_quadratic(x - eps * dx, y - eps * dy)) / (2 * eps)
    np.testing.assert_allclose(tan, fd, atol=1e-4)
    # self_verify soft
    e = rng.standard_normal((3, 4)); c = rng.standard_normal((3, 4, 2)); de = rng.standard_normal((3, 4)); dc = rng.standard_normal((3, 4, 2))
    _, tan = get_jvp("ebm_self_verify")((e, c), (de, dc), beta=1.2)
    fd = (np.asarray(O.ebm_self_verify(e + eps * de, c + eps * dc, beta=1.2)) - np.asarray(O.ebm_self_verify(e - eps * de, c - eps * dc, beta=1.2))) / (2 * eps)
    np.testing.assert_allclose(tan, fd, atol=1e-4)
    # refinement
    y0 = rng.standard_normal((2, 3)); g = rng.standard_normal((2, 3)); dy0 = rng.standard_normal((2, 3)); dg = rng.standard_normal((2, 3))
    _, tan = get_jvp("ebm_refinement")((y0, g), (dy0, dg), eta=0.1, T=3)
    fd = (np.asarray(O.ebm_refinement(y0 + eps * dy0, g + eps * dg, eta=0.1, T=3)) - np.asarray(O.ebm_refinement(y0 - eps * dy0, g - eps * dg, eta=0.1, T=3))) / (2 * eps)
    np.testing.assert_allclose(tan, fd, atol=1e-4)
