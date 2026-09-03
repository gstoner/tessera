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


# ── The rank cross-product, generated rather than hand-picked ────────────────
#
# The hand-picked CASES above missed `(K,) @ (batch, K, N)`, and review caught
# it: numpy prepends A's promoted axis to A's OWN dimensions, which puts it at
# axis -2 once batch axes broadcast in front. Restoring it at axis 0 instead
# built `(1, batch, N)`, making `batch` the contraction dimension -- so the
# case raised for every batch size except 1, the one size that accidentally
# works. A picked list is exactly the wrong shape of test for a rank bug, so
# the ranks are enumerated here and the completeness is itself asserted.

K, M_, N_ = 3, 2, 4

#: label -> (A shape, B shape). `None` means the operand is 1-D.
RANK_SHAPES = {
    "1d @ 1d":            ((K,),           (K,)),
    "1d @ 2d":            ((K,),           (K, N_)),
    "2d @ 1d":            ((M_, K),        (K,)),
    "2d @ 2d":            ((M_, K),        (K, N_)),
    "1d @ batched":       ((K,),           (5, K, N_)),
    "batched @ 1d":       ((5, M_, K),     (K,)),
    "1d @ batch-of-1":    ((K,),           (1, K, N_)),
    "1d @ two batch dims": ((K,),          (3, 2, K, N_)),
    "two batch dims @ 1d": ((3, 2, M_, K), (K,)),
    "batched @ batched":  ((5, M_, K),     (5, K, N_)),
    "batched @ 2d":       ((5, M_, K),     (K, N_)),
    "2d @ batched":       ((M_, K),        (5, K, N_)),
}


def _fd_full(fn, z, h=1e-6):
    """Finite difference of a scalar-valued `fn` over every element of `z`."""
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


@pytest.mark.parametrize("label", list(RANK_SHAPES), ids=list(RANK_SHAPES))
@pytest.mark.parametrize("wrt", ["A", "B"])
def test_every_rank_combination_has_a_correct_gradient(label, wrt):
    a_shape, b_shape = RANK_SHAPES[label]
    rng = np.random.default_rng(abs(hash((label, wrt))) % (2**32))
    a, b = rng.standard_normal(a_shape), rng.standard_normal(b_shape)
    primal, fn = ((a, lambda M: ops.sum(ops.matmul(M, b))) if wrt == "A"
                  else (b, lambda M: ops.sum(ops.matmul(a, M))))

    grad = np.asarray(A.grad(fn)(primal))
    assert grad.shape == primal.shape, f"{label} d{wrt}: shape {grad.shape}"
    np.testing.assert_allclose(grad, _fd_full(fn, primal), rtol=1e-4, atol=1e-6)

    tangent = rng.standard_normal(primal.shape)
    _, jvp_out = A.jvp(fn, (primal,), (tangent,))
    h = 1e-6
    fd = (float(np.asarray(fn(primal + h * tangent)))
          - float(np.asarray(fn(primal - h * tangent)))) / (2 * h)
    assert float(np.asarray(jvp_out)) == pytest.approx(fd, rel=1e-5, abs=1e-7)


def test_the_rank_cross_product_is_actually_complete():
    """A generated matrix is only worth more than a picked list while it stays
    complete. Every (rank of A, rank of B) pair up to two batch dims, in both
    directions, must appear -- 1-D against a BATCHED operand especially, since
    that is the combination the picked list omitted."""
    seen = {(len(a), len(b)) for a, b in RANK_SHAPES.values()}
    for ra in (1, 2, 3):
        for rb in (1, 2, 3):
            if ra == 1 and rb == 1:
                continue
            assert (ra, rb) in seen, f"rank pair ({ra}, {rb}) is not covered"
    assert (1, 1) in seen and (4, 1) in seen and (1, 4) in seen


def test_the_batched_vector_case_needs_a_batch_bigger_than_one():
    """Pins the discriminator, because batch=1 passes under the wrong axis too.

    With A `(K,)` and B `(batch, K, N)` the forward is `(batch, N)`; inserting
    A's restored row axis at the FRONT gives `(1, batch, N)`, which is only a
    valid contraction when `batch == 1`. A regression that reinstates the old
    indexing therefore still passes a batch-of-1 fixture."""
    rng = np.random.default_rng(11)
    a, b = rng.standard_normal(K), rng.standard_normal((5, K, N_))
    assert np.asarray(ops.matmul(a, b)).shape == (5, N_)
    fn = lambda v: ops.sum(ops.matmul(v, b))
    np.testing.assert_allclose(np.asarray(A.grad(fn)(a)), _fd_full(fn, a),
                               rtol=1e-4, atol=1e-6)
