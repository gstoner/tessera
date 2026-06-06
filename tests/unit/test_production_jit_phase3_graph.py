"""Phase 3 Sprint 3.3 — GraphFn(target="apple_gpu") graph-level dispatch
(docs/spec/PRODUCTION_COMPILER_PLAN.md).

Sprint 3.2 wired per-op GPU kernels. This sprint routes a WHOLE multi-op graph
to the Apple GPU back-half: ``GraphFn(target="apple_gpu").run(...)`` interprets
the recorded graph against the bespoke Metal kernels and auto-fuses the canonical
chains (matmul→softmax→matmul, matmul→softmax, matmul→gelu, matmul→rmsnorm) into
single fused Metal kernels.

Oracle (D4): the SAME graph built with ``target="cpu"`` (compiled linalg→LLVM→ORC,
which matches numpy). ``g.last_dispatch()`` lets us prove fusion actually fired.

Skips on non-Darwin / when the Apple GPU runtime or libtessera_jit can't load.
"""

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera._jit_boundary import GraphFn, TesseraJitError

pytestmark = pytest.mark.skipif(
    not (agb.is_available() and jb.is_available()),
    reason="Apple GPU runtime or libtessera_jit unavailable",
)


def _cpu(build):
    g = GraphFn(target="cpu")
    args = build(g)
    return g, args


def _gpu(build):
    g = GraphFn(target="apple_gpu")
    args = build(g)
    return g, args


# ── a plain matmul graph runs on GPU and matches the CPU lane ────────────────


def test_graph_matmul_gpu_matches_cpu():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((8, 16)).astype(np.float32)
    b = rng.standard_normal((16, 4)).astype(np.float32)

    def build(g):
        x, y = g.arg((8, 16)), g.arg((16, 4))
        g.ret(g.matmul(x, y))
        return (a, b)

    gg, ar = _gpu(build)
    gc, _ = _cpu(build)
    np.testing.assert_allclose(gg.run(*ar), gc.run(*ar), rtol=1e-4, atol=1e-4)
    assert gg.last_dispatch() == ["matmul"]


# ── attention graph fuses to ONE matmul_softmax_matmul kernel ────────────────


@pytest.mark.parametrize("T,d", [(8, 16), (16, 32), (4, 64)])
def test_graph_attention_fuses_and_matches_cpu(T, d):
    rng = np.random.default_rng(T * 10 + d)
    q = (rng.standard_normal((T, d)) / np.sqrt(d)).astype(np.float32)
    k = rng.standard_normal((T, d)).astype(np.float32)
    v = rng.standard_normal((T, d)).astype(np.float32)

    def build(g):
        qi, ki, vi = g.arg((T, d)), g.arg((T, d)), g.arg((T, d))
        s = g.matmul(qi, ki, transpose_b=True)  # Q Kᵀ
        p = g.softmax(s)
        g.ret(g.matmul(p, vi))
        return (q, k, v)

    gg, ar = _gpu(build)
    gc, _ = _cpu(build)
    np.testing.assert_allclose(gg.run(*ar), gc.run(*ar), rtol=1e-4, atol=1e-4)
    # The whole 3-op chain collapsed to a SINGLE fused Metal kernel.
    assert gg.last_dispatch() == ["matmul_softmax_matmul"]


# ── matmul→gelu and matmul→rmsnorm fuse ──────────────────────────────────────


def test_graph_matmul_gelu_fuses():
    rng = np.random.default_rng(2)
    a = (rng.standard_normal((8, 16)) / 4).astype(np.float32)
    b = rng.standard_normal((16, 32)).astype(np.float32)

    def build(g):
        x, y = g.arg((8, 16)), g.arg((16, 32))
        g.ret(g.gelu(g.matmul(x, y)))
        return (a, b)

    gg, ar = _gpu(build)
    gc, _ = _cpu(build)
    np.testing.assert_allclose(gg.run(*ar), gc.run(*ar), rtol=2e-2, atol=2e-2)
    assert gg.last_dispatch() == ["matmul_gelu"]


def test_graph_matmul_rmsnorm_fuses():
    rng = np.random.default_rng(3)
    a = (rng.standard_normal((8, 16)) / 4).astype(np.float32)
    b = rng.standard_normal((16, 32)).astype(np.float32)

    def build(g):
        x, y = g.arg((8, 16)), g.arg((16, 32))
        g.ret(g.rmsnorm(g.matmul(x, y)))
        return (a, b)

    gg, ar = _gpu(build)
    gc, _ = _cpu(build)
    np.testing.assert_allclose(gg.run(*ar), gc.run(*ar), rtol=1e-4, atol=1e-4)
    assert gg.last_dispatch() == ["matmul_rmsnorm"]


# ── a full pre-norm attention block routes through the GPU graph ─────────────


def test_graph_prenorm_attention_block_matches_cpu():
    """rmsnorm → QKV proj → softmax(QKᵀ)V → residual, the whole block built as
    ONE GraphFn and run end-to-end on the GPU back-half, vs the CPU lane.

    The Sprint-3.3 milestone: a transformer sub-block is expressed as a single
    graph, routed to apple_gpu, and the attention chain auto-fuses to one kernel
    while the projections + residual run as their own GPU kernels."""
    rng = np.random.default_rng(2026)
    T, d = 16, 32
    scale = np.float32(1.0 / np.sqrt(d))
    x = (rng.standard_normal((T, d)) * 0.5).astype(np.float32)
    wq = ((rng.standard_normal((d, d)) / np.sqrt(d)) * scale).astype(np.float32)
    wk = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    wv = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)

    def build(g):
        xi = g.arg((T, d))
        wqi, wki, wvi = g.arg((d, d)), g.arg((d, d)), g.arg((d, d))
        xn = g.rmsnorm(xi)
        q = g.matmul(xn, wqi)  # scale folded into wq
        k = g.matmul(xn, wki)
        v = g.matmul(xn, wvi)
        s = g.matmul(q, k, transpose_b=True)
        p = g.softmax(s)
        attn = g.matmul(p, v)
        g.ret(g.add(xi, attn))  # residual
        return (x, wq, wk, wv)

    gg, ar = _gpu(build)
    gc, _ = _cpu(build)
    np.testing.assert_allclose(gg.run(*ar), gc.run(*ar), rtol=1e-4, atol=1e-4)
    disp = gg.last_dispatch()
    # attention collapsed to one fused kernel; 3 standalone projection matmuls;
    # rmsnorm + residual add ran as their own GPU kernels.
    assert "matmul_softmax_matmul" in disp
    assert disp.count("matmul") == 3
    assert "rmsnorm" in disp and "add" in disp


# ── fusion is conservative: a reused intermediate is NOT fused ───────────────


def test_graph_no_fusion_when_intermediate_returned():
    """If the matmul result is also a function output, the matmul→softmax chain
    must NOT fuse (fusion only collapses single-use, non-returned intermediates),
    so both the matmul and the softmax fire as separate kernels."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((8, 8)).astype(np.float32)

    def build(g):
        x, y = g.arg((4, 8)), g.arg((8, 8))
        s = g.matmul(x, y)
        p = g.softmax(s)
        g.ret(p, s)  # s is also returned → cannot fuse
        return (a, b)

    gg, ar = _gpu(build)
    gc, _ = _cpu(build)
    pg, sg = gg.run(*ar)
    pc, sc = gc.run(*ar)
    np.testing.assert_allclose(pg, pc, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(sg, sc, rtol=1e-4, atol=1e-4)
    assert gg.last_dispatch() == ["matmul", "softmax"]


# ── envelope: control flow + dtype rejected on apple_gpu ─────────────────────


def test_graph_gpu_rejects_control_flow():
    g = GraphFn(target="apple_gpu")
    x = g.arg((4, 4))
    out = g.for_loop(3, x, lambda c: g.add(c, c))
    g.ret(out)
    with pytest.raises(TesseraJitError, match="control flow"):
        g.run(np.ones((4, 4), np.float32))


def test_graph_gpu_rejects_arg_dtype_mismatch():
    g = GraphFn(target="apple_gpu")
    x, y = g.arg((4, 4)), g.arg((4, 4))
    g.ret(g.add(x, y))
    with pytest.raises(TesseraJitError, match="dtype/shape"):
        g.run(np.ones((4, 4), np.float64), np.ones((4, 4), np.float64))
