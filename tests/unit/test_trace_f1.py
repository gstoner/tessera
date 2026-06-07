"""Phase-F F1 — abstract-interpretation tracing lift (straight-line core).

`trace(fn, *specs)` interprets a `tessera.ops` function once by RUNNING it with
abstract Tracer values (via the autodiff op-wrapper chokepoint), recording
graph_ir; `to_graphfn`/`run_traced` lower the straight-line trace to an executable
GraphFn. Unlike the AST bridge, no source inspection / op-name pattern-matching is
needed — the op list falls out of Python execution.

trace/shape/diagnostic checks are pure; the execute-vs-numpy cases need the Apple
GPU runtime and skip otherwise.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera.compiler.trace import (
    TesseraTraceError,
    Tracer,
    register_shape_rule,
    run_traced,
    to_graphfn,
    trace,
)

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")


def _silu(z):
    return z / (1.0 + np.exp(-z))


def _rms(z, eps=1e-5):
    return z / np.sqrt(np.mean(z * z, axis=-1, keepdims=True) + eps)


# --- trace shape (no runtime) ----------------------------------------------- #
def test_trace_records_op_list_by_running():
    def f(x, w):
        return ts.ops.silu(ts.ops.matmul(x, w))

    tf = trace(f, np.zeros((2, 8), np.float32), np.zeros((8, 8), np.float32))
    assert [op.op_name for op in tf.body] == ["tessera.matmul", "tessera.silu"]
    assert len(tf.args) == 2 and tf.outputs == ["v1"]


def test_trace_residual_cross_reference():
    """A residual that references the original input AFTER other ops — the tracer
    captures the data flow naturally (Python variable reuse), no SSA heuristics."""
    def f(x, w):
        return ts.ops.add(ts.ops.matmul(x, w), x)

    tf = trace(f, np.zeros((2, 8), np.float32), np.zeros((8, 8), np.float32))
    add = tf.body[-1]
    assert add.op_name == "tessera.add"
    # add's second operand is the original arg %a0 (the residual), first is matmul.
    assert add.operands[1] == "%a0"


def test_trace_matmul_shape_rule():
    def f(x, w):
        return ts.ops.matmul(x, w)

    tf = trace(f, (np.zeros((4, 16), np.float32)), np.zeros((16, 32), np.float32))
    out = tf.body[-1]
    assert out.result_type == "tensor<4x32xf32>"


def test_trace_broadcast_add_shape():
    def f(x, b):
        return ts.ops.add(x, b)

    tf = trace(f, np.zeros((4, 8), np.float32), np.zeros((1, 8), np.float32))
    assert tf.body[-1].result_type == "tensor<4x8xf32>"


def test_trace_non_tracer_positional_raises():
    def f(x):
        return ts.ops.matmul(x, np.ones((8, 8), np.float32))  # const positional

    with pytest.raises(TesseraTraceError, match="non-Tracer positional"):
        trace(f, np.zeros((2, 8), np.float32))


def test_trace_must_return_tracer():
    def f(x):
        ts.ops.silu(x)
        return 5  # not a Tracer

    with pytest.raises(TesseraTraceError, match="must return Tracer"):
        trace(f, np.zeros((2, 8), np.float32))


def test_to_graphfn_multi_output_is_f6():
    def f(x, w):
        return ts.ops.matmul(x, w), ts.ops.silu(x)

    tf = trace(f, np.zeros((2, 8), np.float32), np.zeros((8, 8), np.float32))
    with pytest.raises(Exception, match="single output"):
        to_graphfn(tf)


def test_register_shape_rule_extends_vocab():
    # A made-up unary op gets a shape rule; trace records it without numpy.
    register_shape_rule("relu", lambda ins, kw: ins[0])  # idempotent re-register
    def f(x):
        return ts.ops.relu(x)
    tf = trace(f, np.zeros((3, 5), np.float32))
    assert tf.body[-1].result_type == "tensor<3x5xf32>"


# --- execute vs numpy (apple_gpu) ------------------------------------------- #
@gpu
def test_run_traced_mlp_residual_matches_numpy():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((2, 8)) / 8).astype(np.float32)
    w1 = (rng.standard_normal((8, 8)) / 3).astype(np.float32)
    w2 = (rng.standard_normal((8, 8)) / 3).astype(np.float32)

    def f(x, w1, w2):
        h = ts.ops.silu(ts.ops.matmul(x, w1))
        y = ts.ops.add(ts.ops.matmul(h, w2), x)
        return ts.ops.rmsnorm(y)

    out = run_traced(f, x, w1, w2, target="apple_gpu")
    h = _silu(x @ w1)
    ref = _rms(h @ w2 + x)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@gpu
@pytest.mark.parametrize("d", [8, 16])
def test_run_traced_chain_matches_numpy(d):
    rng = np.random.default_rng(d)
    x = (rng.standard_normal((3, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)

    def f(x, w):
        return ts.ops.gelu(ts.ops.matmul(ts.ops.silu(ts.ops.matmul(x, w)), w))

    out = run_traced(f, x, w, target="apple_gpu")
    g = lambda z: 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))  # noqa: E731
    ref = g(_silu(x @ w) @ w)
    np.testing.assert_allclose(out, ref, rtol=2e-3, atol=2e-3)


def test_trace_does_not_disturb_eager_ops():
    """Outside a trace context, ops still compute numpy normally."""
    a = np.ones((2, 2), np.float32)
    assert isinstance(ts.ops.silu(a), np.ndarray)
    assert not isinstance(ts.ops.silu(a), Tracer)
