"""Phase-F F4 — wire @jit(target="apple_gpu") to trace-by-running.

Behind a flag (`TESSERA_JIT_TRACE` / `jit_trace()` context manager) while parity is
oracled against the AST bridge. The tracer's domain: straight-line `tessera.ops`,
a Python `for _ in range(N)` over a static N (unrolls), and explicit
`tessera.control.*` for data-dependent control flow. A raw Python `if`/`while` on a
traced value raises (the abstract-trace hazard) — use `tessera.control.cond` /
`while_loop`.

Decoration still emits AST graph_ir, so the @jit functions here are AST-valid;
F5 makes the trace path the default and relaxes decoration for trace-only bodies.

Needs the Apple GPU runtime; skips otherwise.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera.compiler.trace import (
    TesseraTraceError,
    jit_trace,
    jit_trace_enabled,
)

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")


def _silu(z):
    return z / (1.0 + np.exp(-z))


def _rms(z, eps=1e-5):
    return z / np.sqrt(np.mean(z * z, axis=-1, keepdims=True) + eps)


@ts.jit(target="apple_gpu")
def loop_whole(x, w):           # raw static-range for → unrolls under trace
    for _ in range(4):
        x = ts.ops.silu(ts.ops.matmul(x, w))
    return x


@ts.jit(target="apple_gpu")
def straightline(x, w):
    return ts.ops.rmsnorm(ts.ops.silu(ts.ops.matmul(x, w)))


@ts.jit(target="apple_gpu")
def ctrl_cond(flag, x, w):      # explicit ts.control.cond → fused control_if
    return ts.control.cond(
        flag,
        lambda: ts.ops.silu(ts.ops.matmul(x, w)),
        lambda: ts.ops.relu(ts.ops.matmul(x, w)))


@ts.jit(target="apple_gpu")
def raw_if(flag, x, w):         # raw `if` on data → must raise under trace
    if flag:
        y = ts.ops.silu(x)
    else:
        y = ts.ops.relu(x)
    return y


# --- flag plumbing (no runtime) --------------------------------------------- #
def test_jit_trace_default_on():
    # F5: the tracer is on by default (control-flow apple_gpu @jit routes through
    # it). The context manager / set_jit_trace still toggle it.
    assert jit_trace_enabled() is True
    with jit_trace(False):
        assert jit_trace_enabled() is False
    assert jit_trace_enabled() is True


@gpu
def test_raw_if_on_traced_value_raises():
    x = np.ones((1, 8), np.float32)
    w = np.eye(8, dtype=np.float32)
    with jit_trace(True):
        with pytest.raises(TesseraTraceError, match="cannot branch on a traced"):
            raw_if(np.array([1.0], np.float32), x, w)


# --- parity + numpy (apple_gpu) --------------------------------------------- #
@gpu
def test_loop_whole_trace_parity_with_ast_bridge():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1, 8)) / 8).astype(np.float32)
    w = (rng.standard_normal((8, 8)) / 3).astype(np.float32)
    ast = loop_whole(x, w)                     # AST bridge → control_for
    with jit_trace(True):
        tr = loop_whole(x, w)                  # tracer → unrolled straight-line
    ref = x.copy()
    for _ in range(4):
        ref = _silu(ref @ w)
    np.testing.assert_allclose(ast, ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(tr, ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ast, tr, rtol=1e-5, atol=1e-5)


@gpu
def test_straightline_trace_matches_numpy():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((2, 8)) / 8).astype(np.float32)
    w = (rng.standard_normal((8, 8)) / 3).astype(np.float32)
    with jit_trace(True):
        out = straightline(x, w)
    np.testing.assert_allclose(out, _rms(_silu(x @ w)), rtol=1e-4, atol=1e-4)


@gpu
@pytest.mark.parametrize("flagv", [1.0, -1.0])
def test_ctrl_cond_trace_selects_branch(flagv):
    rng = np.random.default_rng(int(abs(flagv)) + 2)
    x = (rng.standard_normal((1, 8)) / 8).astype(np.float32)
    w = (rng.standard_normal((8, 8)) / 3).astype(np.float32)
    with jit_trace(True):
        out = ctrl_cond(np.array([flagv], np.float32), x, w)
    z = x @ w
    ref = _silu(z) if flagv > 0 else np.maximum(z, 0.0)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@gpu
def test_control_flow_function_is_flagged_for_tracer():
    """F5: a control-flow function (raw static-for) is flagged ``_needs_trace`` and
    executes through the tracer by default; the AST bridge is retired."""
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((1, 8)) / 8).astype(np.float32)
    w = (rng.standard_normal((8, 8)) / 3).astype(np.float32)
    assert loop_whole._needs_trace is True           # raw `for` → scf markers
    assert straightline._needs_trace is False        # pure straight-line
    out = loop_whole(x, w)  # default-on tracer → unrolled execution
    ref = x.copy()
    for _ in range(4):
        ref = _silu(ref @ w)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
