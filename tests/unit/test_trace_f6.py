"""Phase-F F6 — execution-vocab parity (concrete tracing + smart dispatch).

The prerequisite for the F5 default-flip: the tracer must not break the broad
apple_gpu @jit vocab (rope/flash_attn/qkv/mla/MPSGraph/...). F6 achieves parity
without re-implementing every op's GPU execution:

* **Concrete tracing** — the op wrapper passes the original numpy op to
  `record_op`, which runs it on the inputs' concrete values, so shape/dtype come
  from real execution (any op, no per-op shape rule). `Tracer` carries the value.
* **Smart dispatch** — `run_jit_traced` traces (full vocab), then routes control
  flow to `execute_traced` and straight-line to the canonical apple_gpu path. The
  control-flow-ness is cached (structural).

Needs the Apple GPU runtime for the execution cases.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera.compiler.trace import jit_trace, trace

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.hardware_apple_gpu


def _silu(z):
    return z / (1.0 + np.exp(-z))


@ts.jit(target="apple_gpu")
def broad_flash(q, k, v):           # vocab outside the tracer's executable subset
    return ts.ops.flash_attn(q, k, v)


@ts.jit(target="apple_gpu")
def narrow_straightline(x, w):
    return ts.ops.softmax(ts.ops.matmul(x, w))


@ts.jit(target="apple_gpu")
def control_loop(x, w):
    return ts.control.fori_loop(
        0, 3, lambda i, c: ts.ops.silu(ts.ops.matmul(c, w)), x)


# --- concrete tracing (no runtime) ------------------------------------------ #
def test_concrete_tracing_handles_broad_vocab():
    """flash_attn (no shape rule) traces because numpy gives the output shape."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal((1, 2, 4, 8)).astype(np.float32)
    tf = trace(broad_flash._fn, q, q.copy(), q.copy())
    assert [op.op_name for op in tf.body] == ["tessera.flash_attn"]
    # shape came from real execution, not a rule
    assert tf.body[-1].result_type.startswith("tensor<")


def test_tracer_carries_concrete_value():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 8)).astype(np.float32)
    w = rng.standard_normal((8, 8)).astype(np.float32)
    tf = trace(narrow_straightline._fn, x, w)
    # matmul → softmax recorded with real shapes
    assert [op.op_name for op in tf.body] == ["tessera.matmul", "tessera.softmax"]


# --- smart dispatch + deferral (apple_gpu) ---------------------------------- #
@gpu
def test_straightline_defers_to_canonical_bit_identical():
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((2, 8)) / 8).astype(np.float32)
    w = (rng.standard_normal((8, 8)) / 3).astype(np.float32)
    base = narrow_straightline(x, w)                 # canonical path
    with jit_trace(True):
        traced = narrow_straightline(x, w)           # tracer → detects straight-line → canonical
    np.testing.assert_array_equal(base, traced)
    assert narrow_straightline._needs_trace is False


@gpu
def test_broad_vocab_straightline_defers():
    rng = np.random.default_rng(3)
    q = (rng.standard_normal((1, 2, 4, 8)) / 8).astype(np.float32)
    base = broad_flash(q, q.copy(), q.copy())
    with jit_trace(True):
        traced = broad_flash(q, q.copy(), q.copy())
    np.testing.assert_array_equal(base, traced)
    assert broad_flash._needs_trace is False


@gpu
def test_control_flow_routes_to_execute_traced():
    rng = np.random.default_rng(4)
    x = (rng.standard_normal((1, 8)) / 8).astype(np.float32)
    w = (rng.standard_normal((8, 8)) / 3).astype(np.float32)
    with jit_trace(True):
        out = control_loop(x, w)
    assert control_loop._needs_trace is True
    ref = x.copy()
    for _ in range(3):
        ref = _silu(ref @ w)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@gpu
def test_control_flow_ness_is_cached():
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((2, 8)) / 8).astype(np.float32)
    w = (rng.standard_normal((8, 8)) / 3).astype(np.float32)
    with jit_trace(True):
        narrow_straightline(x, w)
        assert narrow_straightline._needs_trace is False
        narrow_straightline(x, w)  # cached → no re-trace, straight to canonical
        assert narrow_straightline._needs_trace is False
