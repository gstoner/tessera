"""Phase 3 Sprint 3.3 perf-fusion — fused pre-norm + projection on the Apple GPU
(docs/spec/PRODUCTION_COMPILER_PLAN.md).

A new custom MPSGraph kernel `tessera_apple_gpu_rmsnorm_matmul_f32` computes
``O = rmsnorm(X) @ W`` in ONE dispatch — the hottest chain in a pre-norm
transformer (a norm feeding a projection). The GraphFn GPU lane recognizes a
``rmsnorm(x) → matmul`` pattern (single-use, plain matmul) and collapses the two
ops into this fused kernel.

Oracle (D4): the compiled CPU lane ``matmul(rmsnorm(x), W)``, which matches numpy.

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


# ── the fused kernel matches the CPU composition + numpy ─────────────────────


@pytest.mark.parametrize("M,K,N", [(8, 16, 4), (1, 32, 32), (16, 64, 48), (4, 8, 128)])
def test_gpu_rmsnorm_matmul_kernel_matches_cpu(M, K, N):
    rng = np.random.default_rng(M * 17 + N)
    x = (rng.standard_normal((M, K)) * 2.0).astype(np.float32)
    w = (rng.standard_normal((K, N)) / np.sqrt(K)).astype(np.float32)

    gpu = agb.gpu_rmsnorm_matmul(x, w)
    cpu = jb.jit_matmul(jb.jit_rmsnorm(x), w)
    np.testing.assert_allclose(gpu, cpu, rtol=2e-4, atol=2e-4)
    # numpy oracle
    ms = (x * x).mean(-1, keepdims=True)
    ref = (x / np.sqrt(ms + 1e-5)) @ w
    np.testing.assert_allclose(gpu, ref, rtol=2e-4, atol=2e-4)


# ── GraphFn collapses rmsnorm→matmul to ONE fused kernel ─────────────────────


@pytest.mark.parametrize("T,d,N", [(8, 16, 32), (16, 32, 16)])
def test_graph_rmsnorm_matmul_fuses_and_matches_cpu(T, d, N):
    rng = np.random.default_rng(d * 5 + N)
    x = (rng.standard_normal((T, d)) * 1.5).astype(np.float32)
    w = (rng.standard_normal((d, N)) / np.sqrt(d)).astype(np.float32)
    ar = (x, w)

    def build(g):
        xi, wi = g.arg((T, d)), g.arg((d, N))
        g.ret(g.matmul(g.rmsnorm(xi), wi))

    gg = GraphFn(target="apple_gpu")
    build(gg)
    gc = GraphFn(target="cpu")
    build(gc)

    np.testing.assert_allclose(gg.run(*ar), gc.run(*ar), rtol=2e-4, atol=2e-4)
    assert gg.last_dispatch() == ["rmsnorm_matmul"]


# ── conservative: a norm feeding two matmuls is NOT fused ────────────────────


def test_graph_rmsnorm_not_fused_when_shared_by_two_projections():
    """When the rmsnorm output feeds more than one matmul (the QKV shape), the
    single-call rmsnorm_matmul kernel can't represent it, so fusion is declined
    and the norm + both matmuls run as separate GPU kernels."""
    rng = np.random.default_rng(9)
    T, d, N = 8, 16, 16
    x = (rng.standard_normal((T, d)) * 1.5).astype(np.float32)
    w1 = (rng.standard_normal((d, N)) / np.sqrt(d)).astype(np.float32)
    w2 = (rng.standard_normal((d, N)) / np.sqrt(d)).astype(np.float32)
    ar = (x, w1, w2)

    def build(g):
        xi = g.arg((T, d))
        w1i, w2i = g.arg((d, N)), g.arg((d, N))
        xn = g.rmsnorm(xi)
        g.ret(g.add(g.matmul(xn, w1i), g.matmul(xn, w2i)))

    gg = GraphFn(target="apple_gpu")
    build(gg)
    gc = GraphFn(target="cpu")
    build(gc)

    np.testing.assert_allclose(gg.run(*ar), gc.run(*ar), rtol=2e-4, atol=2e-4)
    disp = gg.last_dispatch()
    assert "rmsnorm_matmul" not in disp
    assert disp.count("rmsnorm") == 1 and disp.count("matmul") == 2


# ── pre-norm attention block: the norm→Wq fuses, K/V projections stay split ──


def test_graph_prenorm_block_fuses_first_projection():
    """In a block where rmsnorm output is consumed once (a single projection),
    the norm folds into that projection's kernel — proving the fusion fires in a
    realistic position and stays oracle-clean."""
    rng = np.random.default_rng(123)
    T, d = 16, 32
    x = (rng.standard_normal((T, d)) * 0.5).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    b = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    ar = (x, w, b)

    def build(g):
        xi, wi, bi = g.arg((T, d)), g.arg((d, d)), g.arg((d, d))
        proj = g.matmul(g.rmsnorm(xi), wi)  # norm → single projection: fuses
        g.ret(g.matmul(proj, bi))           # downstream op consumes the result

    gg = GraphFn(target="apple_gpu")
    build(gg)
    gc = GraphFn(target="cpu")
    build(gc)

    np.testing.assert_allclose(gg.run(*ar), gc.run(*ar), rtol=2e-4, atol=2e-4)
    disp = gg.last_dispatch()
    assert disp.count("rmsnorm_matmul") == 1
    assert "rmsnorm" not in disp  # the norm folded in — no standalone norm kernel
