"""Phase-H H3b — `tessera.control.scan` trace-awareness.

Under a trace, a forward `tessera.control.scan(fn, init, xs)` over a single Tracer
`xs` lowers to a `tessera.control_scan` IROp; `execute_traced` dispatches the fused
`run_graph_scan_f32` (or host-orchestrates when the scan body nests control). So a
`@jit(target="apple_gpu")` function using `ts.control.scan` routes through the
tracer like `fori_loop`/`cond`/`while_loop`. `jit_scan` is the explicit front-end.

trace-shape checks are pure; execution cases need the Apple GPU runtime.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera.compiler.trace import run_traced, trace

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.hardware_apple_gpu


def _silu(z):
    return z / (1.0 + np.exp(-z))


@ts.jit(target="apple_gpu")
def scan_decode(init, xs, w):
    def body(c, xt):
        c2 = ts.ops.silu(ts.ops.matmul(ts.ops.add(c, xt), w))
        return c2, c2
    return ts.control.scan(body, init, xs)


@ts.jit(target="apple_gpu")
def scan_with_inner_loop(init, xs, w):
    def body(c, xt):
        c2 = ts.control.fori_loop(
            0, 2, lambda i, cc: ts.ops.silu(ts.ops.matmul(cc, w)),
            ts.ops.add(c, xt))
        return c2, c2
    return ts.control.scan(body, init, xs)


def _cumsum_fn(init, xs):
    def body(c, xt):
        s = ts.ops.add(c, xt)
        return s, s
    return ts.control.scan(body, init, xs)


# --- trace shape + detection (no runtime) ----------------------------------- #
def test_scan_traces_to_control_scan():
    tf = trace(_cumsum_fn, np.zeros((1,), np.float32), np.zeros((5, 1), np.float32))
    assert [op.op_name for op in tf.body] == ["tessera.control_scan"]
    assert len(tf.outputs) == 2  # (carry, ys)
    assert tf.body[0].kwargs["_trip"] == 5


def test_scan_function_flagged_needs_trace():
    assert scan_decode._needs_trace is True
    assert scan_with_inner_loop._needs_trace is True


# --- execution (apple_gpu) -------------------------------------------------- #
@gpu
def test_cumsum_scan_via_tracer_exact():
    xs = np.arange(1, 6, dtype=np.float32).reshape(5, 1)
    carry, ys = run_traced(_cumsum_fn, np.zeros((1,), np.float32), xs)
    np.testing.assert_array_equal(carry.ravel(), [15.0])
    np.testing.assert_array_equal(ys.ravel(), np.cumsum([1, 2, 3, 4, 5]))


@gpu
def test_jit_scan_decode_matches_numpy():
    rng = np.random.default_rng(0)
    d, T = 4, 5
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    xs = (rng.standard_normal((T, 1, d)) / d).astype(np.float32)
    c0 = np.zeros((1, d), np.float32)
    carry, ys = scan_decode(c0, xs, w)
    c = c0.copy()
    ref = []
    for t in range(T):
        c = _silu((c + xs[t]) @ w)
        ref.append(c)
    ref = np.stack(ref)
    np.testing.assert_allclose(carry, c, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ys, ref, rtol=1e-4, atol=1e-4)


@gpu
def test_jit_nested_scan_host_orchestrates_and_matches():
    """A scan whose body nests a fori_loop host-orchestrates the scan; the inner
    loop still fuses."""
    rng = np.random.default_rng(1)
    d, T = 4, 5
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    xs = (rng.standard_normal((T, 1, d)) / d).astype(np.float32)
    c0 = np.zeros((1, d), np.float32)
    carry, ys = scan_with_inner_loop(c0, xs, w)
    c = c0.copy()
    ref = []
    for t in range(T):
        cc = c + xs[t]
        for _ in range(2):
            cc = _silu(cc @ w)
        c = cc
        ref.append(c)
    ref = np.stack(ref)
    np.testing.assert_allclose(carry, c, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ys, ref, rtol=1e-4, atol=1e-4)


def test_host_scan_still_works_outside_trace():
    """No trace context → control.scan is the host reference loop."""
    carry, ys = ts.control.scan(
        lambda c, x: (c + x, c + x), 0.0, np.array([1.0, 2.0, 3.0]))
    assert carry == 6.0
    np.testing.assert_array_equal(np.asarray(ys).ravel(), [1.0, 3.0, 6.0])
