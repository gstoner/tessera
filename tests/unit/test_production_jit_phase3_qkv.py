"""Phase 3 Sprint 3.3 perf-fusion — QKV-concat fusion on the Apple GPU graph lane
(docs/spec/PRODUCTION_COMPILER_PLAN.md).

When ≥2 plain matmuls share one input X (the Q/K/V projection shape), the GraphFn
GPU executor concatenates their weights `[Wq|Wk|Wv]`, issues ONE matmul, and splits
the columns back — collapsing 3 GEMM dispatches into 1 (no new Metal kernel; a
host-side weight concat + existing gpu_matmul + column split). If X is a single-use
pre-norm of the group, the rmsnorm folds in too (one gpu_rmsnorm_matmul on the
concat weight) — so a full `rmsnorm → QKV` collapses to ONE kernel.

Oracle (D4): the same graph built target="cpu" (device_verified_jit linalg→LLVM→ORC).

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


def _run_both(build, ar):
    gg = GraphFn(target="apple_gpu")
    build(gg)
    gc = GraphFn(target="cpu")
    build(gc)
    return gg, gc.run(*ar), gg.run(*ar)


# ── 3 projections sharing X concat into one matmul (no norm) ─────────────────


@pytest.mark.parametrize("T,d,N", [(8, 16, 16), (16, 32, 32), (4, 8, 24)])
def test_qkv_concat_no_norm(T, d, N):
    rng = np.random.default_rng(T + d + N)
    x = (rng.standard_normal((T, d))).astype(np.float32)
    wq = (rng.standard_normal((d, N)) / np.sqrt(d)).astype(np.float32)
    wk = (rng.standard_normal((d, N)) / np.sqrt(d)).astype(np.float32)
    wv = (rng.standard_normal((d, N)) / np.sqrt(d)).astype(np.float32)
    ar = (x, wq, wk, wv)

    def build(g):
        xi = g.arg((T, d))
        wqi, wki, wvi = g.arg((d, N)), g.arg((d, N)), g.arg((d, N))
        q, k, v = g.matmul(xi, wqi), g.matmul(xi, wki), g.matmul(xi, wvi)
        g.ret(g.add(g.add(q, k), v))  # use all three

    gg, out_c, out_g = _run_both(build, ar)
    np.testing.assert_allclose(out_g, out_c, rtol=2e-4, atol=2e-4)
    disp = gg.last_dispatch()
    assert disp.count("qkv_concat") == 1       # one fused projection
    assert disp.count("matmul") == 0           # no standalone projection matmuls


# ── pre-norm + QKV collapses to ONE qkv_concat_prenorm kernel ────────────────


def test_qkv_concat_with_prenorm_fold():
    rng = np.random.default_rng(77)
    T, d, N = 16, 32, 32
    x = (rng.standard_normal((T, d)) * 0.5).astype(np.float32)
    wq = (rng.standard_normal((d, N)) / np.sqrt(d)).astype(np.float32)
    wk = (rng.standard_normal((d, N)) / np.sqrt(d)).astype(np.float32)
    wv = (rng.standard_normal((d, N)) / np.sqrt(d)).astype(np.float32)
    ar = (x, wq, wk, wv)

    def build(g):
        xi = g.arg((T, d))
        wqi, wki, wvi = g.arg((d, N)), g.arg((d, N)), g.arg((d, N))
        xn = g.rmsnorm(xi)
        q, k, v = g.matmul(xn, wqi), g.matmul(xn, wki), g.matmul(xn, wvi)
        g.ret(g.add(g.add(q, k), v))

    gg, out_c, out_g = _run_both(build, ar)
    np.testing.assert_allclose(out_g, out_c, rtol=2e-4, atol=2e-4)
    disp = gg.last_dispatch()
    assert disp.count("qkv_concat_prenorm") == 1   # norm + 3 projections → 1 kernel
    assert "rmsnorm" not in disp
    assert "qkv_concat" not in disp                # plain variant did NOT fire
    assert disp.count("matmul") == 0


# ── GQA shape: unequal projection widths split correctly ─────────────────────


def test_qkv_concat_unequal_widths_gqa():
    """GQA/MQA: Wq is full width, Wk/Wv are narrower. The concat over the N axis
    must split back to the right per-projection widths."""
    rng = np.random.default_rng(1234)
    T, d = 8, 32
    nq, nk, nv = 32, 8, 8
    x = (rng.standard_normal((T, d))).astype(np.float32)
    wq = (rng.standard_normal((d, nq)) / np.sqrt(d)).astype(np.float32)
    wk = (rng.standard_normal((d, nk)) / np.sqrt(d)).astype(np.float32)
    wv = (rng.standard_normal((d, nv)) / np.sqrt(d)).astype(np.float32)
    ar = (x, wq, wk, wv)

    def build(g):
        xi = g.arg((T, d))
        wqi, wki, wvi = g.arg((d, nq)), g.arg((d, nk)), g.arg((d, nv))
        q = g.matmul(xi, wqi)
        k = g.matmul(xi, wki)
        v = g.matmul(xi, wvi)
        g.ret(q, k, v)  # distinct widths nq=32, nk=nv=8 → split must be exact

    gg = GraphFn(target="apple_gpu")
    build(gg)
    gc = GraphFn(target="cpu")
    build(gc)
    qg, kg, vg = gg.run(*ar)
    qc, kc, vc = gc.run(*ar)
    np.testing.assert_allclose(qg, qc, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(kg, kc, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(vg, vc, rtol=2e-4, atol=2e-4)
    # numpy oracle for the split widths
    np.testing.assert_allclose(qg, x @ wq, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(kg, x @ wk, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(vg, x @ wv, rtol=2e-4, atol=2e-4)
    assert gg.last_dispatch().count("qkv_concat") == 1


# ── a lone projection (no sharing) is not grouped ────────────────────────────


def test_single_projection_not_concat_fused():
    rng = np.random.default_rng(3)
    T, d, N = 8, 16, 16
    x = rng.standard_normal((T, d)).astype(np.float32)
    w = (rng.standard_normal((d, N)) / np.sqrt(d)).astype(np.float32)
    ar = (x, w)

    def build(g):
        xi, wi = g.arg((T, d)), g.arg((d, N))
        g.ret(g.matmul(xi, wi))

    gg, out_c, out_g = _run_both(build, ar)
    np.testing.assert_allclose(out_g, out_c, rtol=2e-4, atol=2e-4)
    disp = gg.last_dispatch()
    assert "qkv_concat" not in disp and "qkv_concat_prenorm" not in disp
    assert disp == ["matmul"]
