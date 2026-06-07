"""Phase-F F3 — general Layer-2 (concrete interpreter over a traced function).

`execute_traced` lifts the GraphFn-executor constraints (return == construct
result, loop init == function arg) that bound F2: it walks the traced op list with
a concrete env, running straight-line ops as per-op Apple GPU kernels and each
control region as ONE fused `run_graph_*` dispatch over its live concrete inputs.
So a control construct can sit anywhere — with straight-line code before
(computing its carry/inputs) and after (consuming its result) — the case neither
the AST bridge nor F2's `to_graphfn` can execute.

Needs the Apple GPU runtime; skips otherwise.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera.compiler.trace import run_traced

pytestmark = pytest.mark.skipif(
    not (agb.is_available() and jb.is_available()),
    reason="apple_gpu runtime / libtessera_jit unavailable")


def _silu(z):
    return z / (1.0 + np.exp(-z))


def _rms(z, eps=1e-5):
    return z / np.sqrt(np.mean(z * z, axis=-1, keepdims=True) + eps)


def test_loop_with_pre_and_post_straight_line_code():
    """The carry init is COMPUTED (not a function arg) and there is post-loop code
    + a residual to a pre-loop value — impossible for the AST bridge / to_graphfn."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((2, 8)) / 8).astype(np.float32)
    w1 = (rng.standard_normal((8, 8)) / 3).astype(np.float32)
    w2 = (rng.standard_normal((8, 8)) / 3).astype(np.float32)

    def f(x, w1, w2):
        h = ts.ops.silu(ts.ops.matmul(x, w1))                      # pre-loop
        y = ts.control.fori_loop(
            0, 3, lambda i, c: ts.ops.silu(ts.ops.matmul(c, w2)), h)
        return ts.ops.rmsnorm(ts.ops.add(y, h))                    # post-loop + residual

    out = run_traced(f, x, w1, w2)
    h = _silu(x @ w1)
    y = h.copy()
    for _ in range(3):
        y = _silu(y @ w2)
    ref = _rms(y + h)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("flagv", [1.0, -1.0])
def test_cond_with_surrounding_code(flagv):
    rng = np.random.default_rng(int(abs(flagv)) + 4)
    x = (rng.standard_normal((1, 8)) / 8).astype(np.float32)
    w = (rng.standard_normal((8, 8)) / 3).astype(np.float32)

    def f(flag, x, w):
        h = ts.ops.silu(ts.ops.matmul(x, w))
        y = ts.control.cond(flag, lambda: ts.ops.matmul(h, w), lambda: h)
        return ts.ops.rmsnorm(y)

    out = run_traced(f, np.array([flagv], np.float32), x, w)
    h = _silu(x @ w)
    y = (h @ w) if flagv > 0 else h
    np.testing.assert_allclose(out, _rms(y), rtol=1e-4, atol=1e-4)


def test_while_with_surrounding_code():
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((1, 4)) / 4).astype(np.float32)
    w = (rng.standard_normal((4, 4)) / 2).astype(np.float32)

    def f(x, w, thr):
        h = ts.ops.silu(ts.ops.matmul(x, w))
        y = ts.control.while_loop(
            lambda c: thr, lambda c: ts.ops.matmul(c, w), h, max_steps=3)
        return ts.ops.add(y, h)

    out = run_traced(f, x, w, np.array([1.0], np.float32))
    h = _silu(x @ w)
    y = h.copy()
    for _ in range(3):
        y = y @ w
    np.testing.assert_allclose(out, y + h, rtol=1e-4, atol=1e-4)


def test_loop_const_is_computed_pre_loop():
    """A loop CONST (not just the carry) is computed before the loop."""
    rng = np.random.default_rng(6)
    x = (rng.standard_normal((1, 8)) / 8).astype(np.float32)
    w = (rng.standard_normal((8, 8)) / 3).astype(np.float32)

    def f(x, w):
        wn = ts.ops.rmsnorm(w)                       # computed const used in body
        return ts.control.fori_loop(
            0, 3, lambda i, c: ts.ops.silu(ts.ops.matmul(c, wn)), x)

    out = run_traced(f, x, w)
    wn = _rms(w)
    y = x.copy()
    for _ in range(3):
        y = _silu(y @ wn)
    np.testing.assert_allclose(out, y, rtol=1e-4, atol=1e-4)


def test_straight_line_still_routes_to_graphfn():
    """No control flow → run_traced uses the GraphFn (fusion) path, still correct."""
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((2, 8)) / 8).astype(np.float32)
    w = (rng.standard_normal((8, 8)) / 3).astype(np.float32)

    def f(x, w):
        return ts.ops.rmsnorm(ts.ops.silu(ts.ops.matmul(x, w)))

    out = run_traced(f, x, w)
    np.testing.assert_allclose(out, _rms(_silu(x @ w)), rtol=1e-4, atol=1e-4)
