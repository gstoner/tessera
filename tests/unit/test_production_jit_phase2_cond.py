"""Phase 2 Sprint 2.3 — scf.if conditional control flow
(docs/spec/PRODUCTION_COMPILER_PLAN.md).

Runtime-data-dependent divergent control: a shape-(1,) flag drives an scf.if so
only the taken branch executes (vs Sprint 2.1's data-parallel select, where both
sides are always computed). `GraphFn.cond(flag, then_fn, else_fn)`.

numpy oracle + invocation-counter. Skips when libtessera_jit is not built.
"""

import numpy as np
import pytest

from tessera import _jit_boundary as jb
from tessera._jit_boundary import GraphFn

pytestmark = pytest.mark.skipif(
    not jb.is_available(),
    reason="libtessera_jit not built; run `ninja -C build tessera_jit`",
)


def _build_cond_graph():
    # if flag>0:  x + x   else:  x * x   (elementwise)
    g = GraphFn()
    gx = g.arg((4,))
    gflag = g.arg((1,))
    r = g.cond(gflag, lambda: g.add(gx, gx), lambda: g.mul(gx, gx))
    g.ret(r)
    return g


@pytest.mark.parametrize("flag,expect_double", [(1.0, True), (-1.0, False), (0.0, False)])
def test_cond_selects_branch(flag, expect_double):
    g = _build_cond_graph()
    x = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    out = g.run(x, np.array([flag], np.float32))
    expect = (x + x) if expect_double else (x * x)
    np.testing.assert_allclose(out, expect, rtol=1e-6)


def test_cond_is_one_compiled_function():
    g = _build_cond_graph()
    x = np.ones((4,), np.float32)
    before = jb.invocation_count()
    g.run(x, np.array([1.0], np.float32))
    assert jb.invocation_count() == before + 1


def test_cond_branches_can_use_matmul():
    # if flag>0:  A @ x   else:  x  (identity passthrough)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3)).astype(np.float32)
    x = rng.standard_normal((3, 2)).astype(np.float32)

    g = GraphFn()
    gA, gx, gflag = g.arg((3, 3)), g.arg((3, 2)), g.arg((1,))
    r = g.cond(gflag, lambda: g.matmul(gA, gx), lambda: gx)
    g.ret(r)

    np.testing.assert_allclose(g.run(A, x, np.array([1.0], np.float32)), A @ x, rtol=1e-4)
    np.testing.assert_allclose(g.run(A, x, np.array([-1.0], np.float32)), x, rtol=1e-6)


def test_cond_inside_loop():
    # 3 iterations; each iter, if flag>0 add x else subtract x. flag>0 -> +3x.
    x = np.array([1.0, 1.0], np.float32)

    g = GraphFn()
    gx, gflag = g.arg((2,)), g.arg((1,))

    def body(acc):
        return g.cond(gflag, lambda: g.add(acc, gx), lambda: g.sub(acc, gx))

    g.ret(g.for_loop(3, gx, body))
    out = g.run(x, np.array([1.0], np.float32))
    np.testing.assert_allclose(out, 4 * x, rtol=1e-6)  # x + 3x
    out2 = g.run(x, np.array([-1.0], np.float32))
    np.testing.assert_allclose(out2, x - 3 * x, rtol=1e-6)  # x - 3x = -2x
