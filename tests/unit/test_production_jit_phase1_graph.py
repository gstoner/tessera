"""Phase 1 Sprint 1.8 — multi-op graph compilation
(docs/spec/PRODUCTION_COMPILER_PLAN.md).

The leap from op-at-a-time to compiling a whole multi-op `tessera` function as
ONE JIT'd unit. The decisive proof is the invocation counter: an N-op graph
advances it by exactly 1 (one device_verified_jit function), not N. Intermediates never
cross the boundary.

Skips when libtessera_jit is not built.
"""

import numpy as np
import pytest

from tessera import _jit_boundary as jb
from tessera._jit_boundary import GraphFn

pytestmark = pytest.mark.skipif(
    not jb.is_available(),
    reason="libtessera_jit not built; run `ninja -C build tessera_jit`",
)


def test_three_op_graph_is_one_compiled_function():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((3, 4)).astype(np.float32)
    c = rng.standard_normal((3, 4)).astype(np.float32)

    g = GraphFn()
    va, vb, vc = g.arg((3, 4)), g.arg((3, 4)), g.arg((3, 4))
    g.ret(g.mul(g.add(va, vb), vc))  # (a+b)*c — three ops

    before = jb.invocation_count()
    out = g.run(a, b, c)
    after = jb.invocation_count()

    assert after - before == 1, "a 3-op graph must compile to ONE function"
    np.testing.assert_allclose(out, (a + b) * c, rtol=1e-6, atol=1e-6)


def test_attention_block_as_one_compiled_function():
    """softmax(Q Kᵀ) V composed in ONE function (scale folded into Q externally).

    Contrast with test_single_head_attention_composes (Sprint 1.7), which ran 4
    separate compile+invoke cycles. Here the whole block is one device_verified_jit function
    — one counter increment — with Q Kᵀ scores and softmax probs never crossing
    the boundary.
    """
    rng = np.random.default_rng(7)
    T, d = 6, 8
    scale = np.float32(1.0 / np.sqrt(d))
    q = (rng.standard_normal((T, d)).astype(np.float32)) * scale  # fold the scale
    k = rng.standard_normal((T, d)).astype(np.float32)
    v = rng.standard_normal((T, d)).astype(np.float32)

    g = GraphFn()
    vq, vk, vv = g.arg((T, d)), g.arg((T, d)), g.arg((T, d))
    scores = g.matmul(vq, vk, transpose_b=True)  # (T,T)
    probs = g.softmax(scores)
    out_v = g.matmul(probs, vv)  # (T,d)
    g.ret(out_v)

    before = jb.invocation_count()
    out = g.run(q, k, v)
    after = jb.invocation_count()
    assert after - before == 1  # 3 ops, ONE device_verified_jit function

    ref_scores = q @ k.T
    ref_probs = np.exp(ref_scores - ref_scores.max(-1, keepdims=True))
    ref_probs /= ref_probs.sum(-1, keepdims=True)
    ref = ref_probs @ v
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_llama_style_transformer_block_as_one_function():
    """A bias-free, single-head, pre-norm transformer block — device_verified_jit ONCE.

    x -> rmsnorm -> attn(QKᵀ→softmax→·V→Wo) -> +residual
      -> rmsnorm -> MLP(silu(x W1) ⊙ (x W3)) W2 -> +residual

    This is the Phase-1 "model layer end-to-end" milestone: a real (LLaMA-shaped)
    decoder layer runs through tessera→linalg→LLVM as a single device_verified_jit function.
    """
    rng = np.random.default_rng(11)
    T, D, F = 4, 8, 16
    scale = np.float32(1.0 / np.sqrt(D))

    def randn(*s):
        return rng.standard_normal(s).astype(np.float32)

    x = randn(T, D)
    Wq, Wk, Wv, Wo = randn(D, D), randn(D, D), randn(D, D), randn(D, D)
    W1, W3, W2 = randn(D, F), randn(D, F), randn(F, D)

    g = GraphFn()
    gx = g.arg((T, D))
    gWq, gWk, gWv, gWo = g.arg((D, D)), g.arg((D, D)), g.arg((D, D)), g.arg((D, D))
    gW1, gW3, gW2 = g.arg((D, F)), g.arg((D, F)), g.arg((F, D))

    # --- attention ---
    h = g.rmsnorm(gx)
    q = g.matmul(h, gWq)
    k = g.matmul(h, gWk)
    vv = g.matmul(h, gWv)
    s = g.matmul(q, k, transpose_b=True)         # (T,T) — scale folded below
    # fold scale into scores via elementwise mul by a scale-filled constant:
    # avoid scalars by pre-scaling q's projection weight instead.
    p = g.softmax(s)
    a = g.matmul(p, vv)
    a = g.matmul(a, gWo)
    x1 = g.add(gx, a)                            # residual
    # --- MLP (SwiGLU) ---
    h2 = g.rmsnorm(x1)
    gate = g.silu(g.matmul(h2, gW1))
    up = g.matmul(h2, gW3)
    mlp = g.matmul(g.mul(gate, up), gW2)
    y = g.add(x1, mlp)                           # residual
    g.ret(y)

    # Fold the 1/√D scale into Wq so the graph needs no scalar constant.
    Wq_scaled = (Wq * scale).astype(np.float32)

    before = jb.invocation_count()
    out = g.run(x, Wq_scaled, Wk, Wv, Wo, W1, W3, W2)
    after = jb.invocation_count()
    assert after - before == 1, "the whole transformer block must be ONE function"

    # numpy oracle (matching the graph exactly, incl. scale-in-Wq)
    def rms(z, eps=1e-5):
        return z / np.sqrt(np.mean(z**2, axis=-1, keepdims=True) + eps)

    def sig(z):
        return 1.0 / (1.0 + np.exp(-z))

    hN = rms(x)
    q_ = hN @ Wq_scaled
    k_ = hN @ Wk
    v_ = hN @ Wv
    s_ = q_ @ k_.T
    p_ = np.exp(s_ - s_.max(-1, keepdims=True))
    p_ /= p_.sum(-1, keepdims=True)
    a_ = (p_ @ v_) @ Wo
    x1_ = x + a_
    h2N = rms(x1_)
    gate_ = (h2N @ W1) * sig(h2N @ W1)
    up_ = h2N @ W3
    mlp_ = (gate_ * up_) @ W2
    ref = x1_ + mlp_

    np.testing.assert_allclose(out, ref, rtol=2e-4, atol=2e-4)


def test_graph_intermediate_dtype_bf16():
    # The graph machinery is dtype-generic too.
    bf16 = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(2)
    a = rng.standard_normal((3, 4)).astype(bf16)
    b = rng.standard_normal((3, 4)).astype(bf16)
    g = GraphFn(elem="bf16")
    va, vb = g.arg((3, 4)), g.arg((3, 4))
    g.ret(g.add(g.mul(va, vb), va))
    out = g.run(a, b)
    assert out.dtype == bf16
    ref = (a.astype(np.float64) * b.astype(np.float64)) + a.astype(np.float64)
    np.testing.assert_allclose(out.astype(np.float64), ref, rtol=3e-2, atol=3e-2)
