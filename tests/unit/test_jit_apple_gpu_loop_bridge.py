"""Phase-F F5 — @jit(target="apple_gpu") control flow via the tracer.

The AST bridge (Phase-G close-out A/C) was retired in F5: a control-flow apple_gpu
@jit function is detected at decoration (``_needs_trace``) and executes through the
abstract-interp tracer (``compiler/trace.py``) by default. A raw Python
``for _ in range(N)`` unrolls; explicit ``tessera.control.*`` lowers to the fused
control ops; a raw data-dependent ``if`` raises ("use tessera.control.cond").

Detection (``_needs_trace``) + the hard-diagnostic path are pure; the numeric
end-to-end cases need the Apple GPU runtime and skip otherwise.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera._jit_boundary import TesseraJitError

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.hardware_apple_gpu


# --- module-level decorated functions (source must be file-inspectable) ----- #
@ts.jit(target="apple_gpu")
def linear_loop(x, w):                 # raw static-for → unrolls under the tracer
    for _ in range(5):
        x = ts.ops.matmul(x, w)
    return x


@ts.jit(target="apple_gpu")
def silu_loop(x, w):
    for _ in range(4):
        x = ts.ops.silu(ts.ops.matmul(x, w))
    return x


@ts.jit(target="apple_gpu")
def prenorm_residual_loop(x, w):
    for _ in range(5):
        x = ts.ops.add(x, ts.ops.matmul(ts.ops.rmsnorm(x), w))
    return x


@ts.jit(target="apple_gpu")
def straightline(x, w):                # pure straight-line → NOT _needs_trace
    return ts.ops.silu(ts.ops.matmul(x, w))


@ts.jit(target="apple_gpu")
def untranslatable_loop(x, w):         # tessera.sqrt has no GraphFn builder
    for _ in range(2):
        x = ts.ops.sqrt(ts.ops.matmul(x, w))
    return x


@ts.jit(target="apple_gpu")
def ctrl_branch(flag, x, w):           # explicit ts.control.cond (data-dependent)
    return ts.control.cond(
        flag,
        lambda: ts.ops.silu(ts.ops.matmul(x, w)),
        lambda: ts.ops.relu(ts.ops.matmul(x, w)))


def _silu(z):
    return z / (1.0 + np.exp(-z))


def _rms(z, eps=1e-5):
    return z / np.sqrt(np.mean(z * z, axis=-1, keepdims=True) + eps)


# --- surgical gate detection (no runtime) ----------------------------------- #
def test_control_flow_functions_flagged_needs_trace():
    assert silu_loop._needs_trace is True          # raw for → scf markers
    assert prenorm_residual_loop._needs_trace is True
    assert ctrl_branch._needs_trace is True         # ts.control.cond call
    assert straightline._needs_trace is False       # pure straight-line → untouched


def test_untranslatable_op_raises_hard_diagnostic():
    """A raw-for body op outside the GraphFn-executable subset raises at run."""
    x = np.ones((1, 8), np.float32)
    w = np.eye(8, dtype=np.float32)
    with pytest.raises(TesseraJitError, match="cannot be lowered to a GraphFn"):
        untranslatable_loop(x, w)


# --- apple_gpu execution via the tracer ------------------------------------- #
@gpu
@pytest.mark.parametrize("d", [8, 16])
def test_linear_recurrence_matches_numpy(d):
    rng = np.random.default_rng(d)
    x = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    out = linear_loop(x, w)
    ref = x.copy()
    for _ in range(5):
        ref = ref @ w
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@gpu
def test_silu_matmul_loop_matches_numpy():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((1, 16)) / 16).astype(np.float32)
    w = (rng.standard_normal((16, 16)) / 4).astype(np.float32)
    out = silu_loop(x, w)
    ref = x.copy()
    for _ in range(4):
        ref = _silu(ref @ w)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@gpu
def test_prenorm_residual_loop_matches_numpy():
    rng = np.random.default_rng(2026)
    x = (rng.standard_normal((1, 32)) * 0.3).astype(np.float32)
    w = (rng.standard_normal((32, 32)) / np.sqrt(32)).astype(np.float32)
    out = prenorm_residual_loop(x, w)
    ref = x.copy()
    for _ in range(5):
        ref = ref + _rms(ref) @ w
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@gpu
@pytest.mark.parametrize("flagv", [1.0, -1.0])
def test_ctrl_cond_selects_branch_matches_numpy(flagv):
    rng = np.random.default_rng(int(abs(flagv)) + 3)
    d = 8
    x = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    out = ctrl_branch(np.array([flagv], np.float32), x, w)
    z = x @ w
    ref = _silu(z) if flagv > 0 else np.maximum(z, 0.0)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
