"""Phase 2 Sprint 2.2 — scf.for control flow (docs/spec/PRODUCTION_COMPILER_PLAN.md).

Real control flow: a bounded loop with a tensor carry, compiled as one function
through tessera→linalg→scf→cf→llvm. `GraphFn.for_loop(count, init, body)` builds
the loop; the body uses tessera ops and may close over outer values.

numpy oracle + invocation-counter (a loop is still ONE compiled function).
Skips when libtessera_jit is not built.
"""

import numpy as np
import pytest

from tessera import _jit_boundary as jb
from tessera._jit_boundary import GraphFn

pytestmark = pytest.mark.skipif(
    not jb.is_available(),
    reason="libtessera_jit not built; run `ninja -C build tessera_jit`",
)


def test_for_loop_accumulate():
    # init=x; 3 iterations of acc = acc + x  ->  4x.
    x = np.arange(4, dtype=np.float32) + 1.0  # [1,2,3,4]
    g = GraphFn()
    gx = g.arg((4,))
    r = g.for_loop(3, gx, lambda acc: g.add(acc, gx))
    g.ret(r)
    before = jb.invocation_count()
    out = g.run(x)
    assert jb.invocation_count() == before + 1  # a loop is ONE compiled function
    np.testing.assert_allclose(out, 4 * x, rtol=1e-6)


@pytest.mark.parametrize("n", [0, 1, 5, 10])
def test_for_loop_trip_counts(n):
    x = np.ones((3,), np.float32)
    g = GraphFn()
    gx = g.arg((3,))
    g.ret(g.for_loop(n, gx, lambda acc: g.add(acc, gx)))
    out = g.run(x)
    np.testing.assert_allclose(out, (n + 1) * x, rtol=1e-6)  # n=0 -> init unchanged


def test_for_loop_doubling():
    # init=x; k iterations of acc = acc + acc  ->  2^k * x.
    x = np.array([1.0, 2.0, 3.0], np.float32)
    g = GraphFn()
    gx = g.arg((3,))
    g.ret(g.for_loop(4, gx, lambda acc: g.add(acc, acc)))
    out = g.run(x)
    np.testing.assert_allclose(out, (2**4) * x, rtol=1e-6)


def test_for_loop_iterated_matmul_power():
    # Power iteration shape: repeatedly apply A (acc = A @ acc), 3 times -> A^3 @ x.
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3)).astype(np.float32) * 0.3
    x = rng.standard_normal((3, 1)).astype(np.float32)
    g = GraphFn()
    gA, gx = g.arg((3, 3)), g.arg((3, 1))
    g.ret(g.for_loop(3, gx, lambda acc: g.matmul(gA, acc)))
    out = g.run(A, x)
    np.testing.assert_allclose(out, A @ A @ A @ x, rtol=1e-4, atol=1e-4)


def test_for_loop_iterated_transformer_blocks():
    """Apply the same (bias-free, rmsnorm) FFN block N times in a loop — a
    'deep' stack with shared weights, compiled as one function."""
    rng = np.random.default_rng(3)
    T, D, F = 4, 8, 16

    def randn(*s):
        return rng.standard_normal(s).astype(np.float32)

    x = randn(T, D)
    W1, W2 = randn(D, F), randn(F, D)

    g = GraphFn()
    gx, gW1, gW2 = g.arg((T, D)), g.arg((D, F)), g.arg((F, D))

    def block(h):
        n = g.rmsnorm(h)
        ff = g.matmul(g.silu(g.matmul(n, gW1)), gW2)
        return g.add(h, ff)  # residual

    g.ret(g.for_loop(3, gx, block))
    out = g.run(x, W1, W2)

    # numpy oracle: same block, 3 times
    def rms(z, eps=1e-5):
        return z / np.sqrt(np.mean(z**2, -1, keepdims=True) + eps)

    def sig(z):
        return 1 / (1 + np.exp(-z))

    h = x
    for _ in range(3):
        n = rms(h)
        g1 = n @ W1
        ff = (g1 * sig(g1)) @ W2
        h = h + ff
    np.testing.assert_allclose(out, h, rtol=2e-4, atol=2e-4)
