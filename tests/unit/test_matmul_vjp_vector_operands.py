"""`ops.matmul` with a 1-D operand: working forward, broken gradient.

`np.matmul` promotes a 1-D operand — a leading vector becomes a row, a
trailing one a column — and drops the added axis from the result. The VJP
did not repeat that promotion, so `np.swapaxes(B, -1, -2)` asked a 1-D
array for a second axis it does not have and raised `AxisError`.

This was not a corner case. `matmul(W, x)` for a weight matrix and a
feature VECTOR is the single most common shape in a network. Its forward
ran; only reverse mode died. Found while testing whether a jet-based
Laplacian composes with reverse mode over parameters (MSW-2b scoping): the
forward matched the reference to all digits and the parameter gradient
raised here. The same error had earlier made `hvp` look unusable.

Every case is checked against a central finite difference, and the JVP is
checked alongside the VJP for each shape: the two modes must move
together, and the JVP turned out never to have had the bug, which is
itself worth pinning.
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera
from tessera import autodiff as A

ops = tessera.ops

RNG = np.random.default_rng(0)
W = RNG.standard_normal((3, 2))
B = RNG.standard_normal((2, 4))
X = RNG.standard_normal(2)      # trailing vector operand
Y = RNG.standard_normal(3)      # leading vector operand
Z = RNG.standard_normal(2)      # a second, DISTINCT vector for dot products


def _fd_grad(fn, z, h=1e-6):
    g = np.zeros_like(z)
    it = np.nditer(z, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        up, dn = z.copy(), z.copy()
        up[i] += h
        dn[i] -= h
        g[i] = (float(np.asarray(fn(up))) - float(np.asarray(fn(dn)))) / (2 * h)
        it.iternext()
    return g


# Each case: (label, fn, primal). The captured operand is always a DIFFERENT
# object from the primal. A probe that aliased them differentiated `v.v` in
# one mode and `v.x` in the other and reported a 2x "defect" that was the
# probe's own — so distinctness is part of the fixture, not an accident.
CASES = [
    ("mat@vec wrt matrix", lambda M: ops.sum(ops.matmul(M, X)), W),
    ("mat@vec wrt vector", lambda v: ops.sum(ops.matmul(W, v)), X),
    ("vec@mat wrt vector", lambda v: ops.sum(ops.matmul(v, W)), Y),
    ("vec@vec wrt left",   lambda v: ops.sum(ops.matmul(v, Z)), X),
    ("vec@vec wrt right",  lambda v: ops.sum(ops.matmul(X, v)), Z),
    ("mat@mat wrt left",   lambda M: ops.sum(ops.matmul(M, B)), W),
]


@pytest.mark.parametrize("label,fn,primal", CASES, ids=[c[0] for c in CASES])
def test_matmul_vjp_matches_finite_differences(label, fn, primal):
    grad = np.asarray(A.grad(fn)(primal))
    assert grad.shape == primal.shape, label
    np.testing.assert_allclose(grad, _fd_grad(fn, primal), rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("label,fn,primal", CASES, ids=[c[0] for c in CASES])
def test_matmul_jvp_matches_finite_differences(label, fn, primal):
    """The forward-mode rule never had the bug; pin that it stays that way."""
    tangent = RNG.standard_normal(primal.shape)
    _, jvp_out = A.jvp(fn, (primal,), (tangent,))
    h = 1e-6
    fd = (float(np.asarray(fn(primal + h * tangent)))
          - float(np.asarray(fn(primal - h * tangent)))) / (2 * h)
    assert float(np.asarray(jvp_out)) == pytest.approx(fd, rel=1e-5, abs=1e-8)


def test_the_vector_cases_are_the_ones_that_used_to_raise():
    """Keeps the parametrisation honest about what it fixed.

    The 2-D case always worked; if only it were exercised the tests above
    would pass against the old code. Pin that the 1-D shapes are present and
    that the forward really does produce a 1-D result for them.
    """
    assert np.asarray(ops.matmul(W, X)).ndim == 1
    assert np.asarray(ops.matmul(Y, W)).ndim == 1
    assert np.asarray(ops.matmul(X, Z)).ndim == 0
    vector_cases = [c for c in CASES if np.asarray(c[2]).ndim == 1 or "vec" in c[0]]
    assert len(vector_cases) >= 4


def test_batched_matmul_is_unchanged():
    """The promotion must be a no-op for operands that were never 1-D."""
    Ab = RNG.standard_normal((5, 3, 2))
    Bb = RNG.standard_normal((5, 2, 4))
    fn = lambda M: ops.sum(ops.matmul(M, Bb))
    grad = np.asarray(A.grad(fn)(Ab))
    assert grad.shape == Ab.shape
    np.testing.assert_allclose(grad, _fd_grad(fn, Ab), rtol=1e-4, atol=1e-6)


def test_aliased_operands_agree_across_modes():
    """`v.v` with the same object in both slots is a legitimate program.

    Both modes must see one variable used twice — 2v — not one constant and
    one variable. This is the case a careless finite-difference oracle gets
    wrong, so it is stated as an analytic identity instead.
    """
    fn = lambda v: ops.sum(ops.matmul(v, v))
    tangent = RNG.standard_normal(X.shape)
    grad = np.asarray(A.grad(fn)(X))
    _, jvp_out = A.jvp(fn, (X,), (tangent,))
    np.testing.assert_allclose(grad, 2 * X, rtol=1e-12)
    assert float(np.asarray(jvp_out)) == pytest.approx(float(2 * X @ tangent), rel=1e-12)


def test_the_jet_laplacian_now_composes_with_parameter_gradients():
    """The reason this was found: MSW-2b's viability experiment, end to end.

    An order-2 jet whose coefficients ride on `ops.*` is on the tape, so
    `grad` over the network's parameters sees the whole Laplacian. Its value
    matches the exact jet path and its gradient matches finite differences.
    Before the fix the forward matched and the gradient raised here.
    """
    from tessera.autodiff import jet_trace, laplacian_exact

    W0 = np.random.default_rng(7).standard_normal((3, 2))
    x = np.array([0.3, -0.4])

    def jtanh(j):
        a0, a1, a2 = j
        t = ops.tanh(a0)
        s = ops.sub(np.float64(1.0), ops.mul(t, t))
        return [t, ops.mul(s, a1),
                ops.sub(ops.mul(s, a2), ops.mul(ops.mul(t, s), ops.mul(a1, a1)))]

    def laplacian_ops(point, M):
        total = None
        for i in range(point.size):
            e = np.zeros(point.size)
            e[i] = 1.0
            coeffs = [ops.sum(c) for c in jtanh([ops.matmul(M, c) for c in (point, e, np.zeros_like(point))])]
            term = ops.mul(coeffs[2], np.float64(2.0))
            total = term if total is None else ops.add(total, term)
        return total

    exact = laplacian_exact(jet_trace(lambda v: ops.sum(ops.tanh(ops.matmul(W0, v)))), x)
    assert float(np.asarray(laplacian_ops(x, W0))) == pytest.approx(exact, rel=1e-9)

    grad = np.asarray(A.grad(lambda M: laplacian_ops(x, M))(W0))
    np.testing.assert_allclose(grad, _fd_grad(lambda M: laplacian_ops(x, M), W0),
                               rtol=1e-4, atol=1e-7)
