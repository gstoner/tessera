"""Phase 3 G-A — GraphFn bounded for-loop on the Apple GPU
(docs/spec/PRODUCTION_COMPILER_PLAN.md, docs/audit/backend/apple/archive/apple_gpu_control_flow_lowering.md).

`GraphFn(target="apple_gpu").for_loop(count, init, body)` authors the bounded
loop as ONE MPSGraph `forLoop` and runs it in a single dispatch (vs the host
per-iteration interpreter). The body is the recorded straight-line op-list; the
carry threads through the loop iteration argument. v1: f32, single carry, init is
a function arg, body references only args + carry.

Oracle (D4): the SAME graph built target="cpu" (compiled tessera→linalg→scf→LLVM),
which itself matches numpy.

Skips off-Darwin / when the Apple GPU runtime or libtessera_jit is unavailable.
"""

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera._jit_boundary import GraphFn, TesseraJitError

pytestmark = [
    pytest.mark.hardware_apple_gpu,
    pytest.mark.usefixtures("apple_gpu_jit_runtime"),
]


def _np_silu(z):
    return z / (1.0 + np.exp(-z))


# ── linear recurrence: carry = carry @ W ─────────────────────────────────────


@pytest.mark.parametrize("d,trip", [(8, 4), (16, 8), (32, 3)])
def test_loop_linear_recurrence_matches_cpu_and_numpy(d, trip):
    rng = np.random.default_rng(d + trip)
    c0 = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)

    def build(g):
        ci, wi = g.arg((1, d)), g.arg((d, d))
        g.ret(g.for_loop(trip, init=ci, body=lambda c: g.matmul(c, wi)))

    gg = GraphFn(target="apple_gpu")
    build(gg)
    gc = GraphFn(target="cpu")
    build(gc)
    og = gg.run(c0, w)
    np.testing.assert_allclose(og, gc.run(c0, w), rtol=1e-4, atol=1e-4)
    ref = c0.copy()
    for _ in range(trip):
        ref = ref @ w
    np.testing.assert_allclose(og, ref, rtol=1e-4, atol=1e-4)
    assert gg.last_dispatch() == ["forloop"]  # one MPSGraph forLoop dispatch


# ── 2-op body: carry = silu(carry @ W) ───────────────────────────────────────


def test_loop_silu_matmul_body_matches_cpu():
    rng = np.random.default_rng(1)
    d, trip = 16, 6
    c0 = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)

    def build(g):
        ci, wi = g.arg((1, d)), g.arg((d, d))
        g.ret(g.for_loop(trip, init=ci, body=lambda c: g.silu(g.matmul(c, wi))))

    gg = GraphFn(target="apple_gpu")
    build(gg)
    gc = GraphFn(target="cpu")
    build(gc)
    og = gg.run(c0, w)
    np.testing.assert_allclose(og, gc.run(c0, w), rtol=1e-4, atol=1e-4)
    ref = c0.copy()
    for _ in range(trip):
        ref = _np_silu(ref @ w)
    np.testing.assert_allclose(og, ref, rtol=1e-4, atol=1e-4)


# ── pre-norm + projection + residual decode-step body (carry reused) ─────────


def test_loop_prenorm_residual_block_matches_cpu():
    """A decode-step recurrence: carry = carry + rmsnorm(carry) @ W. The carry is
    referenced twice in the body (residual + the norm input) — exercises the
    carry-id threading through the forLoop body."""
    rng = np.random.default_rng(2026)
    d, trip = 32, 5
    c0 = (rng.standard_normal((1, d)) * 0.3).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)

    def build(g):
        ci, wi = g.arg((1, d)), g.arg((d, d))
        g.ret(g.for_loop(
            trip, init=ci,
            body=lambda c: g.add(c, g.matmul(g.rmsnorm(c), wi))))

    gg = GraphFn(target="apple_gpu")
    build(gg)
    gc = GraphFn(target="cpu")
    build(gc)
    np.testing.assert_allclose(gg.run(c0, w), gc.run(c0, w), rtol=1e-3, atol=1e-3)


# ── body using multiple args: carry = (carry @ W1) @ W2 ──────────────────────


def test_loop_multi_arg_body_matches_cpu():
    rng = np.random.default_rng(7)
    d, trip = 16, 4
    c0 = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w1 = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    w2 = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)

    def build(g):
        ci, w1i, w2i = g.arg((1, d)), g.arg((d, d)), g.arg((d, d))
        g.ret(g.for_loop(
            trip, init=ci, body=lambda c: g.matmul(g.matmul(c, w1i), w2i)))

    gg = GraphFn(target="apple_gpu")
    build(gg)
    gc = GraphFn(target="cpu")
    build(gc)
    np.testing.assert_allclose(gg.run(c0, w1, w2), gc.run(c0, w1, w2),
                               rtol=1e-4, atol=1e-4)


# ── envelope: v1 restrictions are rejected with clear diagnostics ────────────


def test_loop_bf16_now_supported_via_host_upcast():
    """Phase-G close-out B: a bf16 loop no longer raises 'f32-only' — it runs the
    f32 executor via host upcast and returns bf16. (Numeric correctness is covered
    by test_production_jit_phase3_control_flow_bf16.py; here we only assert the
    f32-only gate is gone and the elem is honored.)"""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    g = GraphFn(target="apple_gpu", elem="bf16")
    c = g.arg((1, 8))
    w = g.arg((8, 8))
    g.ret(g.for_loop(3, init=c, body=lambda x: g.matmul(x, w)))
    # Serialization (the former f32-only gate) must accept bf16 now.
    spec = g._serialize_loop_spec()
    assert spec[1] == 3  # trip
    out = g.run(np.ones((1, 8), ml_dtypes.bfloat16),
                np.ones((8, 8), ml_dtypes.bfloat16))
    assert out.dtype == ml_dtypes.bfloat16


def test_loop_rejects_init_not_an_arg():
    g = GraphFn(target="apple_gpu")
    x = g.arg((1, 8))
    w = g.arg((8, 8))
    init = g.matmul(x, w)  # a computed (non-arg) init
    g.ret(g.for_loop(3, init=init, body=lambda c: g.matmul(c, w)))
    with pytest.raises(TesseraJitError, match="must be a function arg"):
        g.run(np.ones((1, 8), np.float32), np.ones((8, 8), np.float32))


def test_loop_rejects_multiple_loops():
    g = GraphFn(target="apple_gpu")
    c = g.arg((1, 8))
    w = g.arg((8, 8))
    a = g.for_loop(2, init=c, body=lambda x: g.matmul(x, w))
    g.for_loop(2, init=a, body=lambda x: g.matmul(x, w))
    g.ret(a)
    with pytest.raises(TesseraJitError, match="more than one for_loop"):
        g.run(np.ones((1, 8), np.float32), np.ones((8, 8), np.float32))
