"""Phase-H H2 — native f16 control flow.

MPSGraph supports f16 natively, so `tessera_apple_gpu_run_graph_{loop,cond,while}_f16`
run f16 control flow without the host-upcast bf16 requires (MPSGraph has no bf16
type, confirmed in the runtime + SDK headers — bf16 stays host-upcast). f16 control
flow was previously unsupported; this adds it. `jit_fori_loop`/`jit_while_loop`
infer `f16` from an `np.float16` carry and route to the native f16 lane.

Needs the Apple GPU runtime; skips otherwise.
"""

import ctypes

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera._jit_boundary import GraphFn, jit_fori_loop, jit_while_loop

pytestmark = [
    pytest.mark.hardware_apple_gpu,
    pytest.mark.usefixtures("apple_gpu_jit_runtime"),
]


def _silu(z):
    return z / (1.0 + np.exp(-z))


def _f32_ref_loop(x32, w32, trip):
    ref = x32.copy()
    for _ in range(trip):
        ref = _silu(ref @ w32)
    return ref


def test_f16_loop_native_matches_f32_reference():
    rng = np.random.default_rng(0)
    x32 = (rng.standard_normal((1, 8)) / 8).astype(np.float32)
    w32 = (rng.standard_normal((8, 8)) / 3).astype(np.float32)
    out = jit_fori_loop(
        4, lambda g, c, w_: g.silu(g.matmul(c, w_)),
        init=x32.astype(np.float16), consts=[w32.astype(np.float16)])
    assert out.dtype == np.float16
    np.testing.assert_allclose(out.astype(np.float32), _f32_ref_loop(x32, w32, 4),
                               rtol=2e-2, atol=2e-3)


def test_f16_while_native_matches_f32_reference():
    rng = np.random.default_rng(7)
    x32 = (rng.standard_normal((1, 4)) / 4).astype(np.float32)
    w32 = (rng.standard_normal((4, 4)) / 2).astype(np.float32)
    out = jit_while_loop(
        3, cond=lambda g, c, w_, t: t, body=lambda g, c, w_, t: g.matmul(c, w_),
        init=x32.astype(np.float16),
        consts=[w32.astype(np.float16), np.array([1.0], np.float16)])
    assert out.dtype == np.float16
    ref = x32.copy()
    for _ in range(3):
        ref = ref @ w32
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("flagv", [1.0, -1.0])
def test_f16_cond_native_selects_branch(flagv):
    rng = np.random.default_rng(int(abs(flagv)) + 3)
    x32 = (rng.standard_normal((1, 8)) / 8).astype(np.float32)
    w32 = (rng.standard_normal((8, 8)) / 3).astype(np.float32)
    g = GraphFn(target="apple_gpu", elem="f16")
    flag, x, w = g.arg((1,)), g.arg((1, 8)), g.arg((8, 8))
    g.ret(g.cond(flag,
                 then_fn=lambda: g.silu(g.matmul(x, w)),
                 else_fn=lambda: g.relu(g.matmul(x, w))))
    out = g.run(np.array([flagv], np.float16), x32.astype(np.float16),
                w32.astype(np.float16))
    assert out.dtype == np.float16
    z = x32 @ w32
    ref = _silu(z) if flagv > 0 else np.maximum(z, 0.0)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=2e-2, atol=2e-3)


def test_f16_runtime_symbols_present():
    """The 3 native f16 control-flow C ABI symbols are exported + bindable."""
    from tessera._apple_gpu_dispatch import apple_gpu_runtime

    lib = apple_gpu_runtime()
    assert lib is not None
    for sym in ("tessera_apple_gpu_run_graph_loop_f16",
                "tessera_apple_gpu_run_graph_cond_f16",
                "tessera_apple_gpu_run_graph_while_f16"):
        fn = getattr(lib, sym)
        assert isinstance(fn, ctypes._CFuncPtr)


def test_bf16_stays_host_upcast_not_native_f16():
    """bf16 control flow remains correct via host-upcast (no native MPSGraph
    bf16 type) — a regression guard on the no-native-bf16 finding."""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    bf16 = ml_dtypes.bfloat16
    rng = np.random.default_rng(1)
    x32 = (rng.standard_normal((1, 8)) / 8).astype(np.float32)
    w32 = (rng.standard_normal((8, 8)) / 3).astype(np.float32)
    out = jit_fori_loop(
        4, lambda g, c, w_: g.silu(g.matmul(c, w_)),
        init=x32.astype(bf16), consts=[w32.astype(bf16)])
    assert out.dtype == bf16
    np.testing.assert_allclose(out.astype(np.float32), _f32_ref_loop(x32, w32, 4),
                               rtol=2e-2, atol=2e-3)
