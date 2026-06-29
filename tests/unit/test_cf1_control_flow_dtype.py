"""CF1 — trace-time carry/branch DTYPE contract for the control-flow ops.

CF0 pinned the *shape* half of the envelope (test_cf0_control_flow_contract.py).
CF1 tightens the trace front-end to also enforce the *dtype* half, matching the
C++ verifiers: ControlForOp / ControlWhileOp require the loop result type to
equal the carried iter_arg type, and ControlIfOp merges a single result type —
all of which include the element dtype, not just the shape. Before CF1 the trace
checked shape only, so a body that changed the carry dtype (e.g. an in-body
cast) slipped past the trace and only failed later at the MLIR verifier. Now it
fails early with a clear TesseraTraceError. Pure trace-shape checks — no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.trace import TesseraTraceError, trace


def _f32(*shape):
    return np.zeros(shape, np.float32)


# ── loops: body must preserve the carry dtype ───────────────────────────────
def test_fori_loop_body_changing_carry_dtype_rejected():
    def fn(x):
        # cast keeps the shape but flips f32 -> f16: a carry-dtype change.
        return ts.control.fori_loop(
            0, 3, lambda i, c: ts.ops.cast(c, dtype="fp16"), x)
    with pytest.raises(TesseraTraceError, match="carry dtype"):
        trace(fn, _f32(4))


def test_while_loop_body_changing_carry_dtype_rejected():
    def fn(x):
        return ts.control.while_loop(
            lambda c: c, lambda c: ts.ops.cast(c, dtype="fp16"), x,
            max_steps=4)
    with pytest.raises(TesseraTraceError, match="carry dtype"):
        trace(fn, _f32(4))


def test_scan_body_changing_carry_dtype_rejected():
    def fn(init, xs):
        def body(c, xt):
            c2 = ts.ops.cast(c, dtype="fp16")   # carry dtype drift
            return c2, c2
        return ts.control.scan(body, init, xs)
    with pytest.raises(TesseraTraceError, match="carry dtype"):
        trace(fn, _f32(1), _f32(5, 1))


# ── cond: branches must share a dtype ───────────────────────────────────────
def test_cond_branches_mismatched_dtype_rejected():
    def fn(pred, x):
        return ts.control.cond(
            pred,
            lambda a: a,                          # f32
            lambda a: ts.ops.cast(a, dtype="fp16"),  # f16, same shape
            x)
    with pytest.raises(TesseraTraceError, match="share a dtype"):
        trace(fn, _f32(1), _f32(4))


# ── positive: a dtype-stable body still traces (the cast is not in the carry) ─
def test_fori_loop_dtype_stable_traces():
    def fn(x):
        return ts.control.fori_loop(0, 3, lambda i, c: ts.ops.add(c, c), x)
    tf = trace(fn, _f32(4))
    assert [op.op_name for op in tf.body] == ["tessera.control_for"]


def test_cond_same_dtype_traces():
    def fn(pred, x):
        return ts.control.cond(
            pred, lambda a: ts.ops.add(a, a), lambda a: a, x)
    tf = trace(fn, _f32(1), _f32(4))
    assert [op.op_name for op in tf.body] == ["tessera.control_if"]
