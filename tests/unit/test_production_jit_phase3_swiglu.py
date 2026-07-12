"""Phase 3 Sprint 3.3 follow-on — SwiGLU DAG fusion + the full transformer block
on the Apple GPU graph lane (docs/spec/PRODUCTION_COMPILER_PLAN.md).

The SwiGLU MLP ``O = (silu(X@Wg) ⊙ (X@Wu)) @ Wd`` is written in a GraphFn as five
primitive ops (two gate/up matmuls, silu, elementwise mul, down matmul). The
GPU graph executor recognizes that DAG and collapses it to a SINGLE fused Metal
kernel (`swiglu_f32`). Combined with the attention fusion from Sprint 3.3, a
**full pre-norm transformer block (attention + SwiGLU MLP + residuals)** now
routes end-to-end on the Apple GPU back-half — the Phase 3 block milestone.

Oracle (D4): the same graph built ``target="cpu"`` (device_verified_jit linalg→LLVM→ORC).

Skips on non-Darwin / when the Apple GPU runtime or libtessera_jit can't load.
"""

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera._jit_boundary import GraphFn

pytestmark = pytest.mark.skipif(
    not (agb.is_available() and jb.is_available()),
    reason="Apple GPU runtime or libtessera_jit unavailable",
)


def _swiglu_graph(g, T, K, H, Kout):
    x = g.arg((T, K))
    wg, wu, wd = g.arg((K, H)), g.arg((K, H)), g.arg((H, Kout))
    gate = g.silu(g.matmul(x, wg))
    up = g.matmul(x, wu)
    g.ret(g.matmul(g.mul(gate, up), wd))


# ── standalone gpu_swiglu kernel matches the CPU composition ─────────────────


@pytest.mark.parametrize("T,K,H,Kout", [(4, 8, 16, 8), (8, 16, 32, 16), (2, 32, 64, 32)])
def test_gpu_swiglu_kernel_matches_cpu(T, K, H, Kout):
    rng = np.random.default_rng(T * 11 + H)
    x = (rng.standard_normal((T, K)) / np.sqrt(K)).astype(np.float32)
    wg = (rng.standard_normal((K, H)) / np.sqrt(K)).astype(np.float32)
    wu = (rng.standard_normal((K, H)) / np.sqrt(K)).astype(np.float32)
    wd = (rng.standard_normal((H, Kout)) / np.sqrt(H)).astype(np.float32)

    gpu = agb.gpu_swiglu(x, wg, wu, wd)
    # CPU composition oracle
    gate = jb.jit_silu(jb.jit_matmul(x, wg))
    up = jb.jit_matmul(x, wu)
    cpu = jb.jit_matmul(jb.jit_mul(gate, up), wd)
    np.testing.assert_allclose(gpu, cpu, rtol=2e-4, atol=2e-4)
    # numpy oracle
    gnp = x @ wg
    ref = ((gnp / (1.0 + np.exp(-gnp))) * (x @ wu)) @ wd
    np.testing.assert_allclose(gpu, ref, rtol=2e-4, atol=2e-4)


# ── the SwiGLU DAG fuses to ONE kernel in a GraphFn ──────────────────────────


@pytest.mark.parametrize("T,K,H,Kout", [(4, 8, 16, 8), (8, 16, 32, 16)])
def test_graph_swiglu_fuses_and_matches_cpu(T, K, H, Kout):
    rng = np.random.default_rng(K * 7 + H)
    x = (rng.standard_normal((T, K)) / np.sqrt(K)).astype(np.float32)
    wg = (rng.standard_normal((K, H)) / np.sqrt(K)).astype(np.float32)
    wu = (rng.standard_normal((K, H)) / np.sqrt(K)).astype(np.float32)
    wd = (rng.standard_normal((H, Kout)) / np.sqrt(H)).astype(np.float32)
    ar = (x, wg, wu, wd)

    gg = GraphFn(target="apple_gpu")
    _swiglu_graph(gg, T, K, H, Kout)
    gc = GraphFn(target="cpu")
    _swiglu_graph(gc, T, K, H, Kout)

    np.testing.assert_allclose(gg.run(*ar), gc.run(*ar), rtol=2e-4, atol=2e-4)
    # The whole 5-op SwiGLU DAG collapsed to a SINGLE fused Metal kernel.
    assert gg.last_dispatch() == ["swiglu"]


# ── full pre-norm transformer block: attention + SwiGLU MLP + residuals ──────


def test_graph_full_transformer_block_matches_cpu():
    """The Phase-3 block milestone. One GraphFn expresses:

        h = x + attention(rmsnorm(x))
        out = h + swiglu(rmsnorm(h))

    routed end-to-end on the Apple GPU back-half. The attention chain auto-fuses
    to one matmul_softmax_matmul kernel and the SwiGLU MLP to one swiglu kernel;
    everything is oracle-matched against the same graph on the CPU lane."""
    rng = np.random.default_rng(31337)
    T, d, H = 16, 32, 64
    scale = np.float32(1.0 / np.sqrt(d))
    x = (rng.standard_normal((T, d)) * 0.5).astype(np.float32)
    wq = ((rng.standard_normal((d, d)) / np.sqrt(d)) * scale).astype(np.float32)
    wk = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    wv = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    wg = (rng.standard_normal((d, H)) / np.sqrt(d)).astype(np.float32)
    wu = (rng.standard_normal((d, H)) / np.sqrt(d)).astype(np.float32)
    wd = (rng.standard_normal((H, d)) / np.sqrt(H)).astype(np.float32)
    ar = (x, wq, wk, wv, wg, wu, wd)

    def build(g):
        xi = g.arg((T, d))
        wqi, wki, wvi = g.arg((d, d)), g.arg((d, d)), g.arg((d, d))
        wgi, wui, wdi = g.arg((d, H)), g.arg((d, H)), g.arg((H, d))
        # ── attention sub-block ──
        xn = g.rmsnorm(xi)
        q, k, v = g.matmul(xn, wqi), g.matmul(xn, wki), g.matmul(xn, wvi)
        attn = g.matmul(g.softmax(g.matmul(q, k, transpose_b=True)), v)
        h = g.add(xi, attn)
        # ── SwiGLU MLP sub-block ──
        hn = g.rmsnorm(h)
        gate = g.silu(g.matmul(hn, wgi))
        mlp = g.matmul(g.mul(gate, g.matmul(hn, wui)), wdi)
        g.ret(g.add(h, mlp))

    gg = GraphFn(target="apple_gpu")
    build(gg)
    gc = GraphFn(target="cpu")
    build(gc)

    np.testing.assert_allclose(gg.run(*ar), gc.run(*ar), rtol=1e-3, atol=1e-3)
    disp = gg.last_dispatch()
    # attention pre-norm + QKV projections → one qkv_concat_prenorm kernel;
    # attention → one matmul_softmax_matmul; MLP → one swiglu. Only the MLP
    # pre-norm remains a standalone rmsnorm (it feeds the swiglu kernel).
    assert "qkv_concat_prenorm" in disp     # attn pre-norm + 3 projections fused
    assert "matmul_softmax_matmul" in disp  # attention fused to one kernel
    assert "swiglu" in disp                 # MLP fused to one kernel
    assert disp.count("rmsnorm") == 1       # only the MLP pre-norm stands alone
    assert disp.count("add") == 2           # two residuals ran on GPU


# ── fusion is conservative: a reused SwiGLU intermediate is NOT fused ────────


def test_graph_swiglu_not_fused_when_gate_reused():
    """If the gate (silu output) is also returned, the SwiGLU DAG must NOT fuse —
    the ops fire individually so the extra observable value stays correct."""
    rng = np.random.default_rng(5)
    T, K, H, Kout = 4, 8, 16, 8
    x = (rng.standard_normal((T, K)) / np.sqrt(K)).astype(np.float32)
    wg = (rng.standard_normal((K, H)) / np.sqrt(K)).astype(np.float32)
    wu = (rng.standard_normal((K, H)) / np.sqrt(K)).astype(np.float32)
    wd = (rng.standard_normal((H, Kout)) / np.sqrt(H)).astype(np.float32)
    ar = (x, wg, wu, wd)

    def build(g):
        xi = g.arg((T, K))
        wgi, wui, wdi = g.arg((K, H)), g.arg((K, H)), g.arg((H, Kout))
        gate = g.silu(g.matmul(xi, wgi))
        up = g.matmul(xi, wui)
        out = g.matmul(g.mul(gate, up), wdi)
        g.ret(out, gate)  # gate reused as a second output → no fusion

    gg = GraphFn(target="apple_gpu")
    build(gg)
    gc = GraphFn(target="cpu")
    build(gc)

    og, gg_gate = gg.run(*ar)
    oc, gc_gate = gc.run(*ar)
    np.testing.assert_allclose(og, oc, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(gg_gate, gc_gate, rtol=2e-4, atol=2e-4)
    assert "swiglu" not in gg.last_dispatch()
    assert "silu" in gg.last_dispatch()  # ran as individual kernels instead
