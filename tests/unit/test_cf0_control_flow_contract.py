"""CF0 — control-flow contract acceptance (the negative-test matrix of
docs/spec/CONTROL_FLOW_CONTRACT.md §6).

Every form *outside* the device-lowerable envelope (§2) must be rejected at
trace time with a TesseraTraceError — never silently unrolled into a host loop
inside a device_verified_jit-backend claim. The positive trace-emission cases live in
test_trace_*.py; this file pins the rejections so the envelope cannot quietly
widen. Pure trace-shape checks — no GPU runtime needed.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.trace import TesseraTraceError, trace


def _f32(*shape):
    return np.zeros(shape, np.float32)


# ── while_loop must be bounded under trace ──────────────────────────────────
def test_while_loop_without_max_steps_rejected():
    def fn(x):
        return ts.control.while_loop(
            lambda c: ts.ops.add(c, c), lambda c: ts.ops.add(c, c), x)
    with pytest.raises(TesseraTraceError, match="max_steps"):
        trace(fn, _f32(2))


def test_while_loop_with_max_steps_traces():
    def fn(x):
        return ts.control.while_loop(
            lambda c: c, lambda c: ts.ops.add(c, c), x, max_steps=4)
    tf = trace(fn, _f32(2))
    assert [op.op_name for op in tf.body] == ["tessera.control_while"]
    assert tf.body[0].kwargs["_max_iters"] == 4


# ── carry shape must be preserved (no data-dependent reshape) ───────────────
def test_fori_loop_body_changing_carry_shape_rejected():
    def fn(x, w):
        # body returns matmul(c, w): (4,4)@(4,2)=(4,2) ≠ the (4,4) carry.
        return ts.control.fori_loop(0, 3, lambda i, c: ts.ops.matmul(c, w), x)
    with pytest.raises(TesseraTraceError, match="carry shape"):
        trace(fn, _f32(4, 4), _f32(4, 2))


def test_scan_body_changing_carry_shape_rejected():
    def fn(init, xs, w):
        def body(c, xt):
            c2 = ts.ops.matmul(c, w)   # (1,4)@(4,2)=(1,2) breaks carry shape
            return c2, c2
        return ts.control.scan(body, init, xs)
    with pytest.raises(TesseraTraceError, match="carry shape"):
        trace(fn, _f32(1, 4), _f32(5, 1, 4), _f32(4, 2))


# ── cond branches must agree in shape ───────────────────────────────────────
def test_cond_branches_mismatched_shape_rejected():
    def fn(pred, x, w):
        return ts.control.cond(
            pred,
            lambda a: a,                    # shape (4, 4)
            lambda a: ts.ops.matmul(a, w),  # (4,4)@(4,2) = (4, 2)
            x)
    with pytest.raises(TesseraTraceError, match="share a shape"):
        trace(fn, _f32(1), _f32(4, 4), _f32(4, 2))


def test_cond_matching_branches_traces():
    def fn(pred, x):
        return ts.control.cond(
            pred, lambda a: ts.ops.add(a, a), lambda a: a, x)
    tf = trace(fn, _f32(1), _f32(4))
    assert [op.op_name for op in tf.body] == ["tessera.control_if"]


# ── body must return a Tracer (no host-object / data-dependent capture) ─────
def test_fori_loop_body_returning_host_object_rejected():
    def fn(x):
        # body discards the traced carry and returns a Python float — a host
        # value captured into the loop, invisible to the device body.
        return ts.control.fori_loop(0, 3, lambda i, c: 1.0, x)
    with pytest.raises(TesseraTraceError, match="must return a Tracer"):
        trace(fn, _f32(2))


def test_scan_body_not_returning_pair_rejected():
    def fn(init, xs):
        # scan body must return (carry, y); returning a single Tracer is invalid.
        return ts.control.scan(lambda c, xt: ts.ops.add(c, xt), init, xs)
    with pytest.raises(TesseraTraceError, match="must return"):
        trace(fn, _f32(1), _f32(5, 1))


# ── the eager reference is unchanged outside a trace (the oracle) ───────────
def test_eager_reference_unchanged_outside_trace():
    carry, ys = ts.control.scan(
        lambda c, x: (c + x, c + x), 0.0, np.array([1.0, 2.0, 3.0]))
    assert carry == 6.0
    np.testing.assert_array_equal(np.asarray(ys).ravel(), [1.0, 3.0, 6.0])

    assert ts.control.fori_loop(0, 4, lambda i, c: c + i, 0) == 6
    assert ts.control.while_loop(lambda c: c < 5, lambda c: c + 2, 0) == 6
    assert ts.control.cond(True, lambda: 1, lambda: 2) == 1
    assert ts.control.cond(False, lambda: 1, lambda: 2) == 2
