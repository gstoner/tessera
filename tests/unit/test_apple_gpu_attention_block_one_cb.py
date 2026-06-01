"""Single-command-buffer attention block — Stage 2 of audit Action 6.

The scaffold (PR before this) shipped layer_norm + bmm encoded ops.
Stage 2 lands ``flash_attn_dev_f32_enc`` — the hardest encode-session
op because it uses a custom MSL compute kernel rather than the
MPSGraph ``encodeToCommandBuffer:`` lane the other ops live on.

With flash_attn in place, a complete transformer attention block
encodes onto ONE command buffer:

    x → layer_norm → bmm(x, Wq) → Q          ┐
                  └→ bmm(x, Wk) → K          ├→ flash_attn(Q, K, V) → A → bmm(A, Wo) → out
                  └→ bmm(x, Wv) → V          ┘

That's 5 encoded ops (norm + 3 projections + flash_attn + out_proj =
6 actually) running in one cb with no per-op CPU↔GPU roundtrip.

This is the headline single-cb proof: the structural goal of audit
Action 6 / Pattern table row #6 is met for the *attention* sub-tree
of a decoder layer. MLP-on-one-cb is the next follow-on (needs
silu/gelu/rmsnorm encode-session variants).

Tests pin:

* **Symbol availability** — the new flash_attn_dev_f32_enc binds.
* **Numerical correctness** — a single flash_attn_enc call matches
  the existing non-encoded ``tessera_apple_gpu_flash_attn_f32``
  (which we already tested numerically against numpy).
* **Single command buffer** — a full attention block (6 ops)
  commits exactly 1 cb. Drift gate for the architecture goal.
* **End-to-end numerical** — the attention block's output matches
  a numpy reference (``layer_norm → 3 projections → softmax(QKᵀ/√d) →
  attention @ V → out_proj``) at fp32 tolerance.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol
from tessera.apple_gpu_batched import (
    batched_session,
    bmm_enc,
    device_tensor,
    flash_attn_enc,
    layer_norm_enc,
    session_available,
    session_commit_count,
)


def _np_layer_norm(x, gamma, beta, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps) * gamma + beta).astype(np.float32)


def _np_flash_attn(Q, K, V, scale, causal=False):
    """Reference flash-attention forward (math-equivalent, not the
    online-softmax form). Shapes: Q,K,V = (B, S*, D)."""
    B, Sq, D = Q.shape
    _, Sk, _ = K.shape
    # scores: (B, Sq, Sk)
    scores = np.einsum("bqd,bkd->bqk", Q, K) * scale
    if causal:
        mask = np.triu(np.ones((Sq, Sk), dtype=bool), k=1)
        scores = np.where(mask[None], -np.inf, scores)
    # Numerically-stable softmax
    scores = scores - scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    # output: (B, Sq, D)
    return np.einsum("bqk,bkd->bqd", attn, V).astype(np.float32)


# ---- Symbol availability ------------------------------------------------

def test_flash_attn_dev_f32_enc_symbol_resolves():
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    fn = bind_symbol(
        "tessera_apple_gpu_flash_attn_dev_f32_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_float, ctypes.c_int32),
        ctypes.c_int32)
    assert fn is not None


# ---- Single-op numerical correctness -----------------------------------

def test_flash_attn_enc_matches_numpy():
    """One-shot encoded flash_attn produces correct output."""
    if not session_available():
        pytest.skip("Apple GPU encode-session not available")
    B, Sq, Sk, D = 2, 8, 8, 16
    scale = 1.0 / np.sqrt(D)
    rng = np.random.default_rng(0xA77)
    Q = rng.standard_normal((B, Sq, D), dtype=np.float32) * 0.1
    K = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1
    V = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1

    q_dev = device_tensor(Q)
    k_dev = device_tensor(K)
    v_dev = device_tensor(V)
    try:
        with batched_session() as s:
            o_dev = flash_attn_enc(s, q_dev, k_dev, v_dev,
                                    B=B, Sq=Sq, Sk=Sk, D=D, scale=scale,
                                    causal=False)
        try:
            gpu_out = o_dev.download(np.float32, (B, Sq, D))
        finally:
            o_dev.free()
        expected = _np_flash_attn(Q, K, V, scale, causal=False)
        np.testing.assert_allclose(gpu_out, expected, rtol=1e-3, atol=1e-3)
    finally:
        q_dev.free()
        k_dev.free()
        v_dev.free()


def test_flash_attn_enc_causal_mask():
    """Causal masking propagates through the encoded variant."""
    if not session_available():
        pytest.skip("Apple GPU encode-session not available")
    B, Sq, Sk, D = 1, 6, 6, 8
    scale = 1.0 / np.sqrt(D)
    rng = np.random.default_rng(0xCA5A1)
    Q = rng.standard_normal((B, Sq, D), dtype=np.float32) * 0.1
    K = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1
    V = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1

    q_dev = device_tensor(Q)
    k_dev = device_tensor(K)
    v_dev = device_tensor(V)
    try:
        with batched_session() as s:
            o_dev = flash_attn_enc(s, q_dev, k_dev, v_dev,
                                    B=B, Sq=Sq, Sk=Sk, D=D, scale=scale,
                                    causal=True)
        gpu_out = o_dev.download(np.float32, (B, Sq, D))
        o_dev.free()
        expected = _np_flash_attn(Q, K, V, scale, causal=True)
        np.testing.assert_allclose(gpu_out, expected, rtol=1e-3, atol=1e-3)
    finally:
        q_dev.free()
        k_dev.free()
        v_dev.free()


# ---- Full attention block on ONE command buffer ------------------------

def test_full_attention_block_runs_on_one_command_buffer():
    """The headline Stage-2 proof: a complete transformer attention
    block — layer_norm → 3 projections → flash_attn → out projection —
    runs on EXACTLY 1 command buffer with correct numerical output.

    This is what audit Action 6 / Pattern row #6 has been driving
    toward. Without this proof, the encode-session ABI is just a
    BMM trick; with it, the attention sub-tree of a decoder layer is
    fully on the GPU timeline with no CPU turnarounds."""
    if not session_available():
        pytest.skip("Apple GPU encode-session not available")

    # Tiny decoder shapes — same recipe regardless of size.
    B, S, D = 1, 8, 16
    H = 1   # one head — fold head into batch dim for the projections
    scale = 1.0 / np.sqrt(D)
    eps = 1e-5
    rng = np.random.default_rng(0xA77BAD)

    # Input + per-input norm params.
    X = rng.standard_normal((B * S, D), dtype=np.float32) * 0.1
    gamma = rng.standard_normal((D,), dtype=np.float32)
    beta = rng.standard_normal((D,), dtype=np.float32)

    # Q / K / V projection weights (D → D, shared across head).
    Wq = rng.standard_normal((D, D), dtype=np.float32) * 0.05
    Wk = rng.standard_normal((D, D), dtype=np.float32) * 0.05
    Wv = rng.standard_normal((D, D), dtype=np.float32) * 0.05
    # Output projection.
    Wo = rng.standard_normal((D, D), dtype=np.float32) * 0.05

    # bmm needs (batch, K, N) shape; we use batch=1 broadcast.
    Wq_b = Wq.reshape(1, D, D)
    Wk_b = Wk.reshape(1, D, D)
    Wv_b = Wv.reshape(1, D, D)
    Wo_b = Wo.reshape(1, D, D)

    x_dev = device_tensor(X)
    g_dev = device_tensor(gamma)
    b_dev = device_tensor(beta)
    wq_dev = device_tensor(Wq_b)
    wk_dev = device_tensor(Wk_b)
    wv_dev = device_tensor(Wv_b)
    wo_dev = device_tensor(Wo_b)
    try:
        commits_before = session_commit_count()
        with batched_session() as s:
            # Op 1: pre-attention layer norm
            x_n = layer_norm_enc(s, x_dev, g_dev, b_dev,
                                  rows=B * S, cols=D, eps=eps)
            # Reshape (B*S, D) → bmm input (1, B*S, D) by treating
            # batch=1 in the bmm call.
            # Op 2-4: Q / K / V projections
            q_proj = bmm_enc(s, x_n, wq_dev, batch=1, M=B * S, N=D, K=D)
            k_proj = bmm_enc(s, x_n, wk_dev, batch=1, M=B * S, N=D, K=D)
            v_proj = bmm_enc(s, x_n, wv_dev, batch=1, M=B * S, N=D, K=D)
            # Reshape projections from (1, B*S, D) to (B, S, D) for
            # flash_attn — the underlying buffer layout is identical
            # (row-major, contiguous), so we just declare different
            # B/S in the flash_attn call.
            # Op 5: flash attention
            attn = flash_attn_enc(s, q_proj, k_proj, v_proj,
                                   B=B, Sq=S, Sk=S, D=D, scale=scale,
                                   causal=False)
            # Op 6: output projection (B, S, D) @ (D, D) → (B, S, D).
            # bmm needs (batch, M, K) @ (batch, K, N); the attn output's
            # buffer is (B, S, D), so M=B*S, K=D, N=D, batch=1.
            out_dev = bmm_enc(s, attn, wo_dev, batch=1, M=B * S, N=D, K=D)
        commits_after = session_commit_count()

        # Single-cb invariant — exactly one commit.
        delta = commits_after - commits_before
        assert delta == 1, (
            f"expected exactly 1 commit for full attention block, "
            f"got delta={delta} — the chain may have been split")

        # Read back the output.
        try:
            gpu_out = out_dev.download(
                np.float32, (1, B * S, D)).reshape(B, S, D)
        finally:
            for t in (x_n, q_proj, k_proj, v_proj, attn, out_dev):
                t.free()

        # CPU reference: replay the same recipe.
        x_n_ref = _np_layer_norm(X, gamma, beta, eps)
        q_ref = (x_n_ref @ Wq).reshape(B, S, D)
        k_ref = (x_n_ref @ Wk).reshape(B, S, D)
        v_ref = (x_n_ref @ Wv).reshape(B, S, D)
        attn_ref = _np_flash_attn(q_ref, k_ref, v_ref, scale,
                                   causal=False)
        out_ref = (attn_ref.reshape(B * S, D) @ Wo).reshape(B, S, D)

        np.testing.assert_allclose(gpu_out, out_ref, rtol=2e-3, atol=2e-3)
    finally:
        for d in (x_dev, g_dev, b_dev, wq_dev, wk_dev, wv_dev, wo_dev):
            d.free()


def test_full_attention_block_with_causal_mask_on_one_command_buffer():
    """Same block, but with a causal mask on the attention. Verifies
    the causal flag survives the encode path."""
    if not session_available():
        pytest.skip("Apple GPU encode-session not available")

    B, S, D = 1, 4, 8
    scale = 1.0 / np.sqrt(D)
    eps = 1e-5
    rng = np.random.default_rng(0xA77CA5)

    X = rng.standard_normal((B * S, D), dtype=np.float32) * 0.1
    gamma = rng.standard_normal((D,), dtype=np.float32)
    beta = rng.standard_normal((D,), dtype=np.float32)
    Wq = rng.standard_normal((D, D), dtype=np.float32) * 0.05
    Wk = rng.standard_normal((D, D), dtype=np.float32) * 0.05
    Wv = rng.standard_normal((D, D), dtype=np.float32) * 0.05
    Wo = rng.standard_normal((D, D), dtype=np.float32) * 0.05

    devs = [device_tensor(X), device_tensor(gamma), device_tensor(beta),
            device_tensor(Wq.reshape(1, D, D)),
            device_tensor(Wk.reshape(1, D, D)),
            device_tensor(Wv.reshape(1, D, D)),
            device_tensor(Wo.reshape(1, D, D))]
    x_dev, g_dev, b_dev, wq_dev, wk_dev, wv_dev, wo_dev = devs
    try:
        commits_before = session_commit_count()
        with batched_session() as s:
            x_n = layer_norm_enc(s, x_dev, g_dev, b_dev,
                                  rows=B * S, cols=D, eps=eps)
            q_proj = bmm_enc(s, x_n, wq_dev, batch=1, M=B * S, N=D, K=D)
            k_proj = bmm_enc(s, x_n, wk_dev, batch=1, M=B * S, N=D, K=D)
            v_proj = bmm_enc(s, x_n, wv_dev, batch=1, M=B * S, N=D, K=D)
            attn = flash_attn_enc(s, q_proj, k_proj, v_proj,
                                   B=B, Sq=S, Sk=S, D=D, scale=scale,
                                   causal=True)
            out_dev = bmm_enc(s, attn, wo_dev, batch=1, M=B * S, N=D, K=D)
        commits_after = session_commit_count()
        assert (commits_after - commits_before) == 1

        gpu_out = out_dev.download(np.float32, (1, B * S, D)).reshape(B, S, D)
        for t in (x_n, q_proj, k_proj, v_proj, attn, out_dev):
            t.free()

        x_n_ref = _np_layer_norm(X, gamma, beta, eps)
        q_ref = (x_n_ref @ Wq).reshape(B, S, D)
        k_ref = (x_n_ref @ Wk).reshape(B, S, D)
        v_ref = (x_n_ref @ Wv).reshape(B, S, D)
        attn_ref = _np_flash_attn(q_ref, k_ref, v_ref, scale, causal=True)
        out_ref = (attn_ref.reshape(B * S, D) @ Wo).reshape(B, S, D)
        np.testing.assert_allclose(gpu_out, out_ref, rtol=2e-3, atol=2e-3)
    finally:
        for d in devs:
            d.free()
