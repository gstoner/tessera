"""Phase 3 Sprint 3.3 — whole-graph compile: GraphFn → ONE MPSGraph dispatch
(docs/spec/PRODUCTION_COMPILER_PLAN.md).

`GraphFn.run()` interprets the graph op-by-op against bespoke Metal kernels (with
hand fusions). `GraphFn.run_mlpkg()` instead authors the WHOLE straight-line graph
into one serialized MPSGraph package (the new `mlpkg_author_graph` C ABI / PK8c)
and dispatches it as a SINGLE Metal ML pass — MPSGraph fuses the entire graph
globally. The device_verified_jit package is cached on the instance.

Oracle (D4): the same graph built target="cpu" (device_verified_jit linalg→LLVM→ORC).

Needs the packaged-ML dispatch path (newer macOS / MTL4 ML encoder); skips cleanly
otherwise, and off-Darwin / when the runtime can't load.
"""

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import apple_mlpkg as mp
from tessera._jit_boundary import GraphFn, TesseraJitError

pytestmark = pytest.mark.skipif(
    not (agb.is_available() and jb.is_available() and mp.packaged_ml_available()),
    reason="Apple GPU runtime / libtessera_jit / packaged-ML dispatch unavailable",
)


# ── a 2-op graph compiles+dispatches as one package ──────────────────────────


@pytest.mark.parametrize("M,K,N", [(4, 8, 4), (8, 16, 8)])
def test_mlpkg_matmul_softmax_one_dispatch(M, K, N):
    rng = np.random.default_rng(M + N)
    a = (rng.standard_normal((M, K)) / np.sqrt(K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    ar = (a, b)

    def build(g):
        ai, bi = g.arg((M, K)), g.arg((K, N))
        g.ret(g.softmax(g.matmul(ai, bi)))

    gg = GraphFn(target="apple_gpu")
    build(gg)
    gc = GraphFn(target="cpu")
    build(gc)
    try:
        out = gg.run_mlpkg(*ar)
        np.testing.assert_allclose(out, gc.run(*ar), rtol=2e-4, atol=2e-4)
        s = a @ b
        e = np.exp(s - s.max(-1, keepdims=True))
        np.testing.assert_allclose(out, e / e.sum(-1, keepdims=True),
                                   rtol=2e-4, atol=2e-4)
    finally:
        gg.close()


# ── the FULL transformer block compiles to ONE MPSGraph dispatch ─────────────


def _build_block(g, T, d, H):
    xi = g.arg((T, d))
    wqi, wki, wvi = g.arg((d, d)), g.arg((d, d)), g.arg((d, d))
    wgi, wui, wdi = g.arg((d, H)), g.arg((d, H)), g.arg((H, d))
    xn = g.rmsnorm(xi)
    q, k, v = g.matmul(xn, wqi), g.matmul(xn, wki), g.matmul(xn, wvi)
    attn = g.matmul(g.softmax(g.matmul(q, k, transpose_b=True)), v)
    h = g.add(xi, attn)
    hn = g.rmsnorm(h)
    mlp = g.matmul(g.mul(g.silu(g.matmul(hn, wgi)), g.matmul(hn, wui)), wdi)
    g.ret(g.add(h, mlp))


def test_mlpkg_full_transformer_block_one_dispatch():
    """The whole pre-norm transformer block (~13 ops: rmsnorm + QKV + attention
    + residual + rmsnorm + SwiGLU + residual) authored into ONE MPSGraph package
    and dispatched as a SINGLE Metal pass — matched against the CPU lane AND the
    per-kernel interpreter (`run()`)."""
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

    gg = GraphFn(target="apple_gpu")
    _build_block(gg, T, d, H)
    gc = GraphFn(target="cpu")
    _build_block(gc, T, d, H)
    gi = GraphFn(target="apple_gpu")
    _build_block(gi, T, d, H)
    try:
        one_dispatch = gg.run_mlpkg(*ar)
        cpu = gc.run(*ar)
        interp = gi.run(*ar)  # per-kernel interpreter path
        np.testing.assert_allclose(one_dispatch, cpu, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(one_dispatch, interp, rtol=1e-3, atol=1e-3)
    finally:
        gg.close()


# ── the device_verified_jit package is cached across runs ───────────────────────────────


def test_mlpkg_pipeline_cached_across_runs():
    rng = np.random.default_rng(2)
    M, K, N = 8, 16, 8

    def build(g):
        ai, bi = g.arg((M, K)), g.arg((K, N))
        g.ret(g.gelu(g.matmul(ai, bi)))

    gg = GraphFn(target="apple_gpu")
    build(gg)
    try:
        a1 = (rng.standard_normal((M, K)) / 4).astype(np.float32)
        b1 = rng.standard_normal((K, N)).astype(np.float32)
        gg.run_mlpkg(a1, b1)
        pipe1 = gg._mlpkg_pipe
        # second run with DIFFERENT data must reuse the device_verified_jit pipeline
        a2 = (rng.standard_normal((M, K)) / 4).astype(np.float32)
        b2 = rng.standard_normal((K, N)).astype(np.float32)
        out2 = gg.run_mlpkg(a2, b2)
        pipe2 = gg._mlpkg_pipe
        assert pipe1 is pipe2 and pipe1 is not None  # author+compile happened once
        ref = jb.jit_gelu(jb.jit_matmul(a2, b2))
        np.testing.assert_allclose(out2, ref, rtol=2e-2, atol=2e-2)
    finally:
        gg.close()


# ── envelope: control flow / multi-output / unexpressible op rejected ────────


def test_mlpkg_rejects_control_flow():
    g = GraphFn(target="apple_gpu")
    x = g.arg((4, 4))
    g.ret(g.for_loop(2, x, lambda c: g.add(c, c)))
    with pytest.raises(TesseraJitError, match="control flow"):
        g.run_mlpkg(np.ones((4, 4), np.float32))


def test_mlpkg_rejects_multi_output():
    g = GraphFn(target="apple_gpu")
    a, b = g.arg((4, 4)), g.arg((4, 4))
    g.ret(g.add(a, b), g.mul(a, b))
    with pytest.raises(TesseraJitError, match="one return value"):
        g.run_mlpkg(np.ones((4, 4), np.float32), np.ones((4, 4), np.float32))


def test_mlpkg_rejects_unexpressible_op():
    g = GraphFn(target="apple_gpu")
    a = g.arg((4, 8))
    g.ret(g.transpose(a))  # standalone transpose not in the whole-graph op set
    with pytest.raises(TesseraJitError, match="cannot express op"):
        g.run_mlpkg(np.ones((4, 8), np.float32))
