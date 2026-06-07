"""Phase-F F2 — control-flow trace-awareness.

Under a trace, `tessera.control.fori_loop` / `cond` / `while_loop` run their
body/branches in a nested trace and emit a `tessera.control_for` / `control_if` /
`control_while` op (vs a host Python loop). `to_graphfn` replays each region
through `GraphFn.for_loop`/`cond`/`while_loop`, so the traced control flow
executes on Apple GPU. This reaches AST-bridge parity via the cleaner
run-by-tracing mechanism (carry captured by Python variable flow, full op set).

trace-shape checks are pure; execute-vs-numpy cases need the Apple GPU runtime.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera.compiler.trace import TesseraTraceError, run_traced, trace

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")


def _silu(z):
    return z / (1.0 + np.exp(-z))


# --- trace shape (no runtime) ----------------------------------------------- #
def test_fori_loop_traces_to_control_for():
    def f(x, w):
        return ts.control.fori_loop(
            0, 4, lambda i, c: ts.ops.silu(ts.ops.matmul(c, w)), x)

    tf = trace(f, np.zeros((1, 8), np.float32), np.zeros((8, 8), np.float32))
    assert [op.op_name for op in tf.body] == ["tessera.control_for"]
    cf = tf.body[0]
    assert cf.kwargs["_trip"] == 4
    assert [b.op_name for b in cf.kwargs["_body"]] == ["tessera.matmul", "tessera.silu"]


def test_cond_traces_to_control_if():
    def f(flag, x, w):
        return ts.control.cond(
            flag,
            lambda: ts.ops.silu(ts.ops.matmul(x, w)),
            lambda: ts.ops.relu(ts.ops.matmul(x, w)))

    tf = trace(f, np.zeros((1,), np.float32), np.zeros((1, 8), np.float32),
               np.zeros((8, 8), np.float32))
    assert [op.op_name for op in tf.body] == ["tessera.control_if"]
    assert tf.body[0].kwargs["_then_body"][-1].op_name == "tessera.silu"
    assert tf.body[0].kwargs["_else_body"][-1].op_name == "tessera.relu"


def test_while_traces_to_control_while():
    def f(x, w, thr):
        return ts.control.while_loop(
            lambda c: thr, lambda c: ts.ops.matmul(c, w), x, max_steps=3)

    tf = trace(f, np.zeros((1, 4), np.float32), np.zeros((4, 4), np.float32),
               np.zeros((1,), np.float32))
    assert [op.op_name for op in tf.body] == ["tessera.control_while"]
    assert tf.body[0].kwargs["_max_iters"] == 3


def test_traced_while_requires_max_steps():
    def f(x, w):
        return ts.control.while_loop(
            lambda c: c, lambda c: ts.ops.matmul(c, w), x)  # no max_steps

    with pytest.raises(TesseraTraceError, match="max_steps"):
        trace(f, np.zeros((1, 4), np.float32), np.zeros((4, 4), np.float32))


def test_host_control_flow_still_works_outside_trace():
    # No trace context → fori_loop is the host Python loop.
    out = ts.control.fori_loop(0, 3, lambda i, c: c + 1, 0)
    assert out == 3


# --- execute vs numpy (apple_gpu) ------------------------------------------- #
@gpu
def test_traced_fori_loop_executes():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1, 8)) / 8).astype(np.float32)
    w = (rng.standard_normal((8, 8)) / 3).astype(np.float32)

    def f(x, w):
        return ts.control.fori_loop(
            0, 4, lambda i, c: ts.ops.silu(ts.ops.matmul(c, w)), x)

    out = run_traced(f, x, w)
    ref = x.copy()
    for _ in range(4):
        ref = _silu(ref @ w)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@gpu
@pytest.mark.parametrize("flagv", [1.0, -1.0])
def test_traced_cond_executes(flagv):
    rng = np.random.default_rng(int(abs(flagv)) + 1)
    x = (rng.standard_normal((1, 8)) / 8).astype(np.float32)
    w = (rng.standard_normal((8, 8)) / 3).astype(np.float32)

    def f(flag, x, w):
        return ts.control.cond(
            flag,
            lambda: ts.ops.silu(ts.ops.matmul(x, w)),
            lambda: ts.ops.relu(ts.ops.matmul(x, w)))

    out = run_traced(f, np.array([flagv], np.float32), x, w)
    z = x @ w
    ref = _silu(z) if flagv > 0 else np.maximum(z, 0.0)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@gpu
def test_traced_while_executes():
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((1, 4)) / 4).astype(np.float32)
    w = (rng.standard_normal((4, 4)) / 2).astype(np.float32)

    def f(x, w, thr):
        return ts.control.while_loop(
            lambda c: thr, lambda c: ts.ops.matmul(c, w), x, max_steps=3)

    out = run_traced(f, x, w, np.array([1.0], np.float32))
    ref = x.copy()
    for _ in range(3):
        ref = ref @ w
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
