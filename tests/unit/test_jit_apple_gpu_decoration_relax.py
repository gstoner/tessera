"""Phase-F follow-on — relaxed apple_gpu decoration defers to the tracer.

An AST Graph-IR emission failure no longer hard-fails ``@jit(target="apple_gpu")``
decoration: the tracer executes the function by RUNNING it (it never reads the AST
graph_ir), so a body the AST can't emit — e.g. surrounding straight-line code +
a residual + a shape-conditional around a raw loop — still decorates and runs via
the tracer at call time. Non-apple_gpu targets still raise (they depend on the IR).
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera.compiler.jit import TesseraJitError  # the class @jit decoration raises

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")


# This body made AST emission fail ("unresolved operand") before the relaxation:
# a raw for-loop with a post-loop residual + a shape-conditional.
@ts.jit(target="apple_gpu")
def ast_unemittable(x, w1, w2):
    h = ts.ops.silu(ts.ops.matmul(x, w1))
    for _ in range(3):
        h = ts.ops.silu(ts.ops.matmul(h, w2))
    return ts.ops.rmsnorm(ts.ops.add(h, x if x.shape == h.shape else h))


def _silu(z):
    return z / (1.0 + np.exp(-z))


def _rms(z, eps=1e-5):
    return z / np.sqrt(np.mean(z * z, axis=-1, keepdims=True) + eps)


def test_ast_unemittable_apple_gpu_decorates_and_defers():
    # Decoration must SUCCEED (was a hard TesseraJitError) and force the tracer.
    assert ast_unemittable._needs_trace is True
    codes = [d.code for d in (ast_unemittable.lowering_diagnostics or [])]
    assert "JIT_APPLE_GPU_TRACE_DEFERRED" in codes


def test_non_apple_gpu_still_raises_on_ast_failure():
    def bad(x, w):
        h = ts.ops.silu(ts.ops.matmul(x, w))
        for _ in range(2):
            h = ts.ops.silu(ts.ops.matmul(h, w))
        return ts.ops.add(h, x if x.shape == h.shape else h)

    with pytest.raises(TesseraJitError):
        ts.jit(target="cpu")(bad)


@gpu
def test_ast_unemittable_executes_via_tracer_matches_numpy():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1, 8)) / 8).astype(np.float32)
    w1 = (rng.standard_normal((8, 8)) / 3).astype(np.float32)
    w2 = (rng.standard_normal((8, 8)) / 3).astype(np.float32)
    out = ast_unemittable(x, w1, w2)
    h = _silu(x @ w1)
    for _ in range(3):
        h = _silu(h @ w2)
    ref = _rms(h + x)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
