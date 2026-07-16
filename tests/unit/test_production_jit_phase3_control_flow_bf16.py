"""Phase-G close-out, Phase B — bf16 control flow (host upcast).

A bf16 bounded loop (via ``jit_fori_loop`` or the AST ``@jit`` bridge) runs the
f32 ``run_graph_loop_f32`` executor with host upcast: inputs bf16 → f32 once,
the loop computes in f32 (f32 carry — more accurate than per-step bf16 rounding),
the result downcasts back to bf16. A native ``run_graph_loop_bf16`` is a
perf/exact-rounding follow-on.

Needs the Apple GPU runtime + a built ``tessera-opt`` + ml_dtypes; skips otherwise.
"""

import numpy as np
import pytest

ml_dtypes = pytest.importorskip("ml_dtypes")

import tessera as ts  # noqa: E402
from tessera import _apple_gpu_backend as agb  # noqa: E402
from tessera import _jit_boundary as jb  # noqa: E402
from tessera._jit_boundary import (  # noqa: E402
    TesseraJitError,
    _find_tessera_opt,
    jit_fori_loop,
)

bf16 = ml_dtypes.bfloat16
_GPU = agb.is_available() and jb.is_available() and _find_tessera_opt() is not None
gpu = pytest.mark.hardware_apple_gpu


def _silu(z):
    return z / (1.0 + np.exp(-z))


def _f32_ref(x32, w32, trip):
    ref = x32.copy()
    for _ in range(trip):
        ref = _silu(ref @ w32)
    return ref


@ts.jit(target="apple_gpu")
def silu_loop_bf16(x, w):
    for _ in range(4):
        x = ts.ops.silu(ts.ops.matmul(x, w))
    return x


@gpu
def test_jit_fori_bf16_matches_f32_reference():
    rng = np.random.default_rng(0)
    x32 = (rng.standard_normal((1, 16)) / 16).astype(np.float32)
    w32 = (rng.standard_normal((16, 16)) / 4).astype(np.float32)
    out = jit_fori_loop(
        4, lambda g, c, w_: g.silu(g.matmul(c, w_)),
        init=x32.astype(bf16), consts=[w32.astype(bf16)])
    assert out.dtype == bf16
    ref = _f32_ref(x32, w32, 4)
    # bf16 storage tolerance (~2^-8 relative).
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=2e-2, atol=2e-3)


@gpu
def test_jit_fori_bf16_direct_and_ir_paths_agree():
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((1, 16)) / 16).astype(np.float32).astype(bf16)
    w = (rng.standard_normal((16, 16)) / 4).astype(np.float32).astype(bf16)
    body = lambda g, c, w_: g.silu(g.matmul(c, w_))  # noqa: E731
    via = jit_fori_loop(4, body, init=x, consts=[w], via_target_ir=True)
    direct = jit_fori_loop(4, body, init=x, consts=[w], via_target_ir=False)
    assert via.dtype == bf16 and direct.dtype == bf16
    np.testing.assert_array_equal(via.astype(np.float32), direct.astype(np.float32))


@gpu
def test_bf16_loop_via_tracer_executes():
    """F5: a bf16 raw-`for` @jit function executes through the tracer (the AST
    bridge is retired) and returns bf16, matching the f32 reference."""
    rng = np.random.default_rng(1)
    x32 = (rng.standard_normal((1, 16)) / 16).astype(np.float32)
    w32 = (rng.standard_normal((16, 16)) / 4).astype(np.float32)
    assert silu_loop_bf16._needs_trace is True
    out = silu_loop_bf16(x32.astype(bf16), w32.astype(bf16))
    assert out.dtype == bf16
    ref = _f32_ref(x32, w32, 4)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=2e-2, atol=2e-3)


@gpu
def test_bf16_loop_rejects_mismatched_arg_dtype():
    """A bf16 GraphFn requires bf16 args (the carry dtype sets _elem)."""
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((1, 8)) / 8).astype(np.float32).astype(bf16)
    w_f32 = (rng.standard_normal((8, 8)) / 4).astype(np.float32)  # wrong dtype
    with pytest.raises(TesseraJitError, match="dtype/shape mismatch"):
        jit_fori_loop(3, lambda g, c, w_: g.matmul(c, w_), init=x, consts=[w_f32])
