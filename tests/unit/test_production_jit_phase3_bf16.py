"""Phase 3 Sprint 3.4 — bf16 across the Apple GPU back-half + GraphFn
(docs/spec/PRODUCTION_COMPILER_PLAN.md).

The back-half kernels are dtype-polymorphic: an ``ml_dtypes.bfloat16`` input routes
to the native bf16 Metal kernel (matmul / softmax / gelu / attention / fused-MLP
chains / swiglu) or to an f32-compute-then-round path (rmsnorm / layer_norm /
silu / elementwise — no native bf16 kernel). ``GraphFn(target="apple_gpu",
elem="bf16")`` runs the whole graph in bf16 through the interpreter.

Oracle: the SAME computation in f32 (the f32 lane is itself oracle-clean vs numpy),
compared at bf16 tolerance — bf16 carries ~8 mantissa bits, so ~1e-2 relative.

Skips off-Darwin / when the runtime or ml_dtypes is unavailable.
"""

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera._jit_boundary import GraphFn, TesseraJitError

ml_dtypes = pytest.importorskip("ml_dtypes")
bf16 = ml_dtypes.bfloat16

pytestmark = [
    pytest.mark.hardware_apple_gpu,
    pytest.mark.usefixtures("apple_gpu_jit_runtime"),
]


# ── back-half kernels: bf16 output, matches the f32 kernel at bf16 tol ───────


def test_bf16_matmul_native():
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((8, 16)) / 4).astype(np.float32)
    b = rng.standard_normal((16, 8)).astype(np.float32)
    out = agb.gpu_matmul(a.astype(bf16), b.astype(bf16))
    assert out.dtype == bf16
    np.testing.assert_allclose(out.astype(np.float32), a @ b, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("fn,extra", [
    ("gpu_softmax", None), ("gpu_gelu", None), ("gpu_silu", None),
    ("gpu_rmsnorm", None), ("gpu_layer_norm", None),
])
def test_bf16_unary_kernels_match_f32(fn, extra):
    rng = np.random.default_rng(abs(hash(fn)) & 0xFFFF)
    x = (rng.standard_normal((8, 32)) * 1.5).astype(np.float32)
    f = getattr(agb, fn)
    out_bf = f(x.astype(bf16))
    out_f32 = f(x)
    assert out_bf.dtype == bf16
    np.testing.assert_allclose(out_bf.astype(np.float32), out_f32, rtol=3e-2, atol=3e-2)


def test_bf16_binary_add_round_trips():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    out = agb.gpu_binary("add", a.astype(bf16), b.astype(bf16))
    assert out.dtype == bf16
    np.testing.assert_allclose(out.astype(np.float32), a + b, rtol=3e-2, atol=3e-2)


def test_bf16_attention_native():
    rng = np.random.default_rng(2)
    M, K, N, P = 8, 16, 16, 8
    a = (rng.standard_normal((M, K)) / np.sqrt(K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    c = rng.standard_normal((N, P)).astype(np.float32)
    out = agb.gpu_attention(a.astype(bf16), b.astype(bf16), c.astype(bf16))
    assert out.dtype == bf16
    ref = agb.gpu_attention(a, b, c)  # f32 lane
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=5e-2, atol=5e-2)


def test_bf16_swiglu_native():
    rng = np.random.default_rng(3)
    T, K, H, Kout = 8, 16, 32, 16
    x = (rng.standard_normal((T, K)) / np.sqrt(K)).astype(np.float32)
    wg = (rng.standard_normal((K, H)) / np.sqrt(K)).astype(np.float32)
    wu = (rng.standard_normal((K, H)) / np.sqrt(K)).astype(np.float32)
    wd = (rng.standard_normal((H, Kout)) / np.sqrt(H)).astype(np.float32)
    out = agb.gpu_swiglu(x.astype(bf16), wg.astype(bf16), wu.astype(bf16), wd.astype(bf16))
    assert out.dtype == bf16
    ref = agb.gpu_swiglu(x, wg, wu, wd)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=5e-2, atol=5e-2)


def test_bf16_rejects_f64():
    with pytest.raises(agb.AppleGpuError):
        agb.gpu_matmul(np.ones((4, 4), np.float64), np.ones((4, 4), np.float64))


# ── GraphFn(elem="bf16") interpreter: whole graphs in bf16 ───────────────────


def _block(g, T, d, H, scale):
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


def test_graph_bf16_full_block_matches_f32_lane():
    """The whole pre-norm transformer block in bf16 through the GraphFn
    interpreter, vs the same graph in f32, at bf16 tolerance. Fusion still fires
    (qkv_concat_prenorm + matmul_softmax_matmul + swiglu)."""
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
    f32args = (x, wq, wk, wv, wg, wu, wd)
    bf16args = tuple(t.astype(bf16) for t in f32args)

    gf = GraphFn(target="apple_gpu")
    _block(gf, T, d, H, scale)
    gb = GraphFn(target="apple_gpu", elem="bf16")
    _block(gb, T, d, H, scale)

    out_f32 = gf.run(*f32args)
    out_bf16 = gb.run(*bf16args)
    assert out_bf16.dtype == bf16
    # bf16 mantissa ~8 bits; a deep block accumulates rounding → loose rel tol.
    np.testing.assert_allclose(
        out_bf16.astype(np.float32), out_f32, rtol=6e-2, atol=6e-2)
    assert "qkv_concat_prenorm" in gb.last_dispatch()
    assert "swiglu" in gb.last_dispatch()


def test_graph_bf16_attention_fuses():
    rng = np.random.default_rng(9)
    T, dd = 8, 16
    q = (rng.standard_normal((T, dd)) / np.sqrt(dd)).astype(np.float32)
    k = rng.standard_normal((T, dd)).astype(np.float32)
    v = rng.standard_normal((T, dd)).astype(np.float32)

    def build(g):
        qi, ki, vi = g.arg((T, dd)), g.arg((T, dd)), g.arg((T, dd))
        g.ret(g.matmul(g.softmax(g.matmul(qi, ki, transpose_b=True)), vi))

    gb = GraphFn(target="apple_gpu", elem="bf16")
    build(gb)
    gf = GraphFn(target="apple_gpu")
    build(gf)
    out = gb.run(q.astype(bf16), k.astype(bf16), v.astype(bf16))
    assert out.dtype == bf16 and gb.last_dispatch() == ["matmul_softmax_matmul"]
    np.testing.assert_allclose(
        out.astype(np.float32), gf.run(q, k, v), rtol=5e-2, atol=5e-2)


def test_graph_bf16_arg_dtype_checked():
    g = GraphFn(target="apple_gpu", elem="bf16")
    a, b = g.arg((4, 4)), g.arg((4, 4))
    g.ret(g.add(a, b))
    # passing f32 to a bf16 graph must be rejected
    with pytest.raises(TesseraJitError, match="dtype/shape"):
        g.run(np.ones((4, 4), np.float32), np.ones((4, 4), np.float32))


def test_run_mlpkg_bf16_whole_graph():
    """bf16 in the whole-graph lane: GraphFn(elem='bf16').run_mlpkg() returns bf16
    and matches the f32 package at bf16 tolerance. The package is f32 internally
    (bf16 converted at the Python boundary — the mlpkg reflection path can't bind
    bf16 tensors yet), so this is bf16-storage / f32-compute."""
    import tessera.apple_mlpkg as mp
    if not mp.packaged_ml_available():
        pytest.skip("packaged-ML dispatch unavailable (needs macOS 26+)")
    rng = np.random.default_rng(5)
    M, K, N = 8, 16, 8
    a = (rng.standard_normal((M, K)) / np.sqrt(K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)

    def build(g):
        ai, bi = g.arg((M, K)), g.arg((K, N))
        g.ret(g.softmax(g.matmul(ai, bi)))

    gb = GraphFn(target="apple_gpu", elem="bf16")
    build(gb)
    gf = GraphFn(target="apple_gpu")
    build(gf)
    try:
        out = gb.run_mlpkg(a.astype(bf16), b.astype(bf16))
        assert out.dtype == bf16
        np.testing.assert_allclose(
            out.astype(np.float32), gf.run_mlpkg(a, b), rtol=3e-2, atol=3e-2)
    finally:
        gb.close()
        gf.close()
