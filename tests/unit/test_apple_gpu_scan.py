"""Phase-H H3 — fused scan (`run_graph_scan_f32` + `jit_scan`).

`(carry, ys) = scan(body, init, xs)` runs as ONE MPSGraph forLoop carrying
`[carry, ys]`: per step `x_t = xs[i]` (gather), `(carry, y) = body(carry, x_t)`,
`ys[i] = y` (scatterND). `jit_scan(trip, body, init=, xs=, consts=)` is the
front-end; `body` is `(g, carry, x_t, *consts) -> (carry, y)` on GraphFn handles.

Needs the Apple GPU runtime; skips otherwise.
"""

import ctypes

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera._jit_boundary import TesseraJitError, build_scan, jit_scan

pytestmark = pytest.mark.skipif(
    not (agb.is_available() and jb.is_available()),
    reason="Apple GPU runtime / libtessera_jit unavailable")


def _silu(z):
    return z / (1.0 + np.exp(-z))


def test_scan_cumsum_exact():
    def body(g, c, xt, *_):
        s = g.add(c, xt)
        return s, s

    xs = np.arange(1, 6, dtype=np.float32).reshape(5, 1)
    carry, ys = jit_scan(5, body, init=np.zeros((1,), np.float32), xs=xs)
    assert carry.shape == (1,) and ys.shape == (5, 1)
    np.testing.assert_array_equal(carry.ravel(), [15.0])
    np.testing.assert_array_equal(ys.ravel(), np.cumsum([1, 2, 3, 4, 5]))


def test_scan_rnn_style_with_const_matches_numpy():
    rng = np.random.default_rng(0)
    d, T = 4, 6
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    xs = (rng.standard_normal((T, 1, d)) / d).astype(np.float32)
    c0 = np.zeros((1, d), np.float32)

    def body(g, c, xt, w_):
        c2 = g.silu(g.matmul(g.add(c, xt), w_))
        return c2, c2

    carry, ys = jit_scan(T, body, init=c0, xs=xs, consts=[w])
    c = c0.copy()
    ref = []
    for t in range(T):
        c = _silu((c + xs[t]) @ w)
        ref.append(c)
    ref = np.stack(ref)
    np.testing.assert_allclose(carry, c, rtol=1e-4, atol=1e-4)
    assert ys.shape == ref.shape
    np.testing.assert_allclose(ys, ref, rtol=1e-4, atol=1e-4)


def test_scan_carry_and_y_differ():
    """y need not equal the next carry — a moving sum carry with a per-step
    squared output."""
    def body(g, c, xt, *_):
        c2 = g.add(c, xt)        # running sum
        y = g.mul(xt, xt)        # per-step x^2
        return c2, y

    xs = np.array([[1.0], [2.0], [3.0]], np.float32)
    carry, ys = jit_scan(3, body, init=np.zeros((1,), np.float32), xs=xs)
    np.testing.assert_array_equal(carry.ravel(), [6.0])
    np.testing.assert_array_equal(ys.ravel(), [1.0, 4.0, 9.0])


def test_scan_runtime_symbol_present():
    from tessera._apple_gpu_dispatch import apple_gpu_runtime

    lib = apple_gpu_runtime()
    assert lib is not None
    fn = getattr(lib, "tessera_apple_gpu_run_graph_scan_f32")
    assert isinstance(fn, ctypes._CFuncPtr)


def test_build_scan_captures_single_scan_record():
    g = build_scan(
        3, lambda g, c, xt, *_: (g.add(c, xt), g.add(c, xt)),
        init_shape=(1,), x_shape=(1,))
    assert g._scan is not None and "_unsupported" not in g._scan
    assert g._scan["trip"] == 3


def test_second_scan_rejected():
    g = build_scan(
        2, lambda g, c, xt, *_: (g.add(c, xt), g.add(c, xt)),
        init_shape=(1,), x_shape=(1,))
    # a second scan on the same graph disables the GPU scan path
    xt = g.arg((1,))
    g.scan(2, init=g.arg((1,)), xs=g.arg((2, 1)),
           body=lambda c, x: (g.add(c, x), g.add(c, x)))
    with pytest.raises(TesseraJitError, match="more than one scan"):
        g._serialize_scan_spec()
