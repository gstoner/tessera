"""Phase 3 G-A.2 — GraphFn cond (if/else) on the Apple GPU
(docs/spec/PRODUCTION_COMPILER_PLAN.md, docs/apple_gpu_control_flow_lowering.md).

`GraphFn(target="apple_gpu").cond(flag, then_fn, else_fn)` authors the divergent
branch as ONE MPSGraph `if` (predicate = flag[0] > 0) and runs it in a single
dispatch — only the taken branch executes. Each branch is the recorded
straight-line op-list over the args; both produce the same shape. v1: f32, flag
is a function arg.

Oracle (D4): the SAME graph built target="cpu" (compiled tessera→linalg→scf.if→
LLVM), which matches numpy.

Skips off-Darwin / when the Apple GPU runtime or libtessera_jit is unavailable.
"""

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera._jit_boundary import GraphFn, TesseraJitError

pytestmark = pytest.mark.skipif(
    not (agb.is_available() and jb.is_available()),
    reason="Apple GPU runtime or libtessera_jit unavailable",
)


def _np_silu(z):
    return z / (1.0 + np.exp(-z))


def _block(g):
    f = g.arg((1,))
    a, b = g.arg((4, 8)), g.arg((4, 8))
    # then: silu(a) ; else: relu(b)
    g.ret(g.cond(f, lambda: g.silu(a), lambda: g.relu(b)))


@pytest.mark.parametrize("flag_val,branch", [(1.0, "then"), (-1.0, "else"), (0.0, "else")])
def test_cond_selects_branch_matches_cpu_and_numpy(flag_val, branch):
    rng = np.random.default_rng(abs(hash((flag_val, branch))) & 0xFFFF)
    a = (rng.standard_normal((4, 8)) * 1.5).astype(np.float32)
    b = (rng.standard_normal((4, 8)) * 1.5).astype(np.float32)
    flag = np.array([flag_val], np.float32)

    gg = GraphFn(target="apple_gpu")
    _block(gg)
    gc = GraphFn(target="cpu")
    _block(gc)
    og = gg.run(flag, a, b)
    np.testing.assert_allclose(og, gc.run(flag, a, b), rtol=1e-4, atol=1e-4)
    expect = _np_silu(a) if branch == "then" else np.maximum(b, 0.0)
    np.testing.assert_allclose(og, expect, rtol=1e-4, atol=1e-4)
    assert gg.last_dispatch() == ["cond"]  # one MPSGraph `if` dispatch


def test_cond_multi_op_branches_matmul():
    """Branches with their own multi-op computations: then = a@W, else = b@W."""
    rng = np.random.default_rng(3)
    M, K, N = 4, 8, 8
    a = (rng.standard_normal((M, K)) / np.sqrt(K)).astype(np.float32)
    b = (rng.standard_normal((M, K)) / np.sqrt(K)).astype(np.float32)
    w = rng.standard_normal((K, N)).astype(np.float32)

    def build(g):
        f = g.arg((1,))
        ai, bi, wi = g.arg((M, K)), g.arg((M, K)), g.arg((K, N))
        g.ret(g.cond(f, lambda: g.matmul(ai, wi), lambda: g.matmul(bi, wi)))

    gg = GraphFn(target="apple_gpu")
    build(gg)
    gc = GraphFn(target="cpu")
    build(gc)
    for flag_val, ref in [(1.0, a @ w), (-1.0, b @ w)]:
        flag = np.array([flag_val], np.float32)
        og = gg.run(flag, a, b, w)
        np.testing.assert_allclose(og, gc.run(flag, a, b, w), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(og, ref, rtol=1e-4, atol=1e-4)


def test_cond_branch_returns_arg_directly():
    """A branch that returns an arg unchanged (no ops) — exercises the
    branch-output id resolving to an arg id."""
    rng = np.random.default_rng(5)
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)

    def build(g):
        f = g.arg((1,))
        ai, bi = g.arg((4, 8)), g.arg((4, 8))
        g.ret(g.cond(f, lambda: ai, lambda: g.add(bi, bi)))

    gg = GraphFn(target="apple_gpu")
    build(gg)
    gc = GraphFn(target="cpu")
    build(gc)
    np.testing.assert_allclose(gg.run(np.array([1.0], np.float32), a, b),
                               gc.run(np.array([1.0], np.float32), a, b),
                               rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gg.run(np.array([-1.0], np.float32), a, b), b + b,
                               rtol=1e-4, atol=1e-4)


# ── envelope: v1 restrictions rejected ───────────────────────────────────────


def test_cond_rejects_bf16():
    g = GraphFn(target="apple_gpu", elem="bf16")
    f = g.arg((1,))
    a, b = g.arg((4, 8)), g.arg((4, 8))
    g.ret(g.cond(f, lambda: g.silu(a), lambda: g.relu(b)))
    import ml_dtypes
    bf16 = ml_dtypes.bfloat16
    with pytest.raises(TesseraJitError, match="f32-only"):
        g.run(np.ones((1,), bf16), np.ones((4, 8), bf16), np.ones((4, 8), bf16))


def test_cond_rejects_multiple_conds():
    g = GraphFn(target="apple_gpu")
    f = g.arg((1,))
    a, b = g.arg((4, 8)), g.arg((4, 8))
    x = g.cond(f, lambda: g.silu(a), lambda: g.relu(b))
    g.cond(f, lambda: g.add(x, x), lambda: g.mul(x, x))
    g.ret(x)
    with pytest.raises(TesseraJitError, match="more than one cond"):
        g.run(np.ones((1,), np.float32), np.ones((4, 8), np.float32),
              np.ones((4, 8), np.float32))
