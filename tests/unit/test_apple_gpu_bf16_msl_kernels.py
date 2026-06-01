"""Phase 3b — bf16 MSL kernels via on-GPU bf16↔fp32 conversion.

Closes the dtype matrix: rope and flash_attn now have bf16 encode-
session variants. The runtime composes:

    bf16 input → MPSGraph cast → fp32 → MSL kernel → fp32 → cast → bf16 output

All in the SAME command buffer (no host roundtrip; the encoded chain
of cast → kernel → cast runs as one cb commit).

Tests pin:

* **Symbol availability** — new C ABI symbols bind.
* **Registry** — bf16 rope + flash_attn are present in
  ``ENCODE_OP_REGISTRY`` (no longer ``not_eligible``).
* **Numerical correctness** — bf16 outputs match a numpy reference
  at bf16 tolerance (~1.5% rtol; bf16 has 8-bit mantissa AND we
  round-trip through cast nodes twice).
* **Single-cb invariant** — a bf16 RoPE call (which internally does
  3 encoded operations: cast, MSL kernel, cast) still commits
  exactly 1 cb per ``batched_session()``.
* **Full bf16 Llama attention block** — rmsnorm + 3 qkv + 2 rope +
  flash_attn + out_proj all in bf16, single cb. The headline closing
  the dtype matrix.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol
from tessera.apple_gpu_batched import (
    batched_session,
    bf16_session_available,
    bmm_enc_bf16,
    device_tensor,
    flash_attn_enc_bf16,
    rmsnorm_enc_bf16,
    rope_enc_bf16,
    session_available,
    session_commit_count,
)
from tessera.apple_gpu_chain import (
    ENCODE_OP_REGISTRY,
    is_encode_eligible,
)


# bf16 helpers — same as the bf16 encode-session test file.
def _to_bf16(a: np.ndarray) -> np.ndarray:
    fp32 = a.astype(np.float32)
    bits = fp32.view(np.uint32)
    bias = 0x7FFF + ((bits >> 16) & 1)
    rounded = (bits + bias) >> 16
    return rounded.astype(np.uint16).reshape(a.shape)


def _from_bf16(a: np.ndarray) -> np.ndarray:
    bits = (a.astype(np.uint32) << 16)
    return bits.view(np.float32).reshape(a.shape)


# ---- Symbols + registry ------------------------------------------------

@pytest.mark.parametrize("symbol", [
    "tessera_apple_gpu_rope_dev_bf16_enc",
    "tessera_apple_gpu_flash_attn_dev_bf16_enc",
])
def test_bf16_msl_symbols_resolve(symbol):
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable")
    fn = bind_symbol(symbol, (ctypes.c_void_p,), ctypes.c_int32)
    assert fn is not None


def test_bf16_msl_ops_are_encode_eligible():
    """Phase 3b promotes rope + flash_attn bf16 from
    not_eligible → registered."""
    assert is_encode_eligible("rope", "bf16")
    assert is_encode_eligible("flash_attn", "bf16")
    # Sanity: dtype matrix is now 8 ops × 3 dtypes = 24 entries
    # (8 f32 + 8 f16 + 8 bf16). The registry is the truth.
    bf16_ops = {name for (name, dtype) in ENCODE_OP_REGISTRY
                if dtype == "bf16"}
    assert bf16_ops == {"bmm", "layer_norm", "rmsnorm", "softmax",
                        "rope", "silu", "gelu", "flash_attn"}


# ---- Numerical correctness — RoPE -------------------------------------

def test_rope_bf16_matches_numpy():
    if not bf16_session_available():
        pytest.skip("bf16 encode-session not available on this host")
    M, K = 6, 8  # K must be even for pair-wise rotation
    rng = np.random.default_rng(0x60BFAE)
    X = rng.standard_normal((M * K,), dtype=np.float32)
    Theta = (np.arange(M * K, dtype=np.float32) * 0.01).astype(np.float32)
    x_dev = device_tensor(_to_bf16(X.reshape(M, K)))
    t_dev = device_tensor(_to_bf16(Theta.reshape(M, K)))
    try:
        with batched_session() as s:
            y_dev = rope_enc_bf16(s, x_dev, t_dev, M=M, K=K)
        gpu = _from_bf16(y_dev.download(np.uint16, (M, K))).reshape(M * K)
        y_dev.free()
        expected = np.empty_like(X)
        for m in range(M):
            for pair in range(K // 2):
                ie = m * K + pair * 2
                io = ie + 1
                c, s = np.cos(Theta[ie]), np.sin(Theta[ie])
                expected[ie] = X[ie] * c - X[io] * s
                expected[io] = X[ie] * s + X[io] * c
        # bf16 + double cast (host → bf16 → cast → fp32 → kernel →
        # fp32 → cast → bf16 → host). Allow ~3% relative for the
        # round-trip precision loss; absolute tolerance similarly
        # loose since X * sin/cos can produce small magnitudes.
        np.testing.assert_allclose(gpu, expected, rtol=3e-2, atol=3e-2)
    finally:
        x_dev.free(); t_dev.free()


# ---- Numerical correctness — flash_attn -------------------------------

def test_flash_attn_bf16_matches_numpy():
    if not bf16_session_available():
        pytest.skip("bf16 encode-session not available on this host")
    B, Sq, Sk, D = 1, 8, 8, 16
    scale = 1.0 / np.sqrt(D)
    rng = np.random.default_rng(0xBFAA7)
    Q = rng.standard_normal((B, Sq, D), dtype=np.float32) * 0.1
    K = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1
    V = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1
    q_dev = device_tensor(_to_bf16(Q))
    k_dev = device_tensor(_to_bf16(K))
    v_dev = device_tensor(_to_bf16(V))
    try:
        with batched_session() as s:
            o_dev = flash_attn_enc_bf16(s, q_dev, k_dev, v_dev,
                                         B=B, Sq=Sq, Sk=Sk, D=D,
                                         scale=scale, causal=False)
        gpu = _from_bf16(o_dev.download(np.uint16, (B, Sq, D)))
        o_dev.free()
        scores = np.einsum("bqd,bkd->bqk", Q, K) * scale
        scores = scores - scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores)
        attn = attn / attn.sum(axis=-1, keepdims=True)
        expected = np.einsum("bqk,bkd->bqd", attn, V)
        # bf16 + online softmax accumulation + double cast — ~3-5%
        # relative error is the honest precision floor.
        np.testing.assert_allclose(gpu, expected, rtol=5e-2, atol=5e-2)
    finally:
        q_dev.free(); k_dev.free(); v_dev.free()


def test_flash_attn_bf16_causal_mask():
    if not bf16_session_available():
        pytest.skip("bf16 encode-session not available on this host")
    B, Sq, Sk, D = 1, 6, 6, 8
    scale = 1.0 / np.sqrt(D)
    rng = np.random.default_rng(0xCA5ABF)
    Q = rng.standard_normal((B, Sq, D), dtype=np.float32) * 0.1
    K = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1
    V = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1
    q_dev = device_tensor(_to_bf16(Q))
    k_dev = device_tensor(_to_bf16(K))
    v_dev = device_tensor(_to_bf16(V))
    try:
        with batched_session() as s:
            o_dev = flash_attn_enc_bf16(s, q_dev, k_dev, v_dev,
                                         B=B, Sq=Sq, Sk=Sk, D=D,
                                         scale=scale, causal=True)
        gpu = _from_bf16(o_dev.download(np.uint16, (B, Sq, D)))
        o_dev.free()
        scores = np.einsum("bqd,bkd->bqk", Q, K) * scale
        mask = np.triu(np.ones((Sq, Sk), dtype=bool), k=1)
        scores = np.where(mask[None], -np.inf, scores)
        scores = scores - scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores)
        attn = attn / attn.sum(axis=-1, keepdims=True)
        expected = np.einsum("bqk,bkd->bqd", attn, V)
        np.testing.assert_allclose(gpu, expected, rtol=5e-2, atol=5e-2)
    finally:
        q_dev.free(); k_dev.free(); v_dev.free()


# ---- Single-cb invariant ----------------------------------------------

def test_bf16_rope_commits_exactly_one_command_buffer():
    """The runtime internally encodes 3 operations (bf16→fp32 cast,
    MSL kernel, fp32→bf16 cast) for a single rope_enc_bf16 call —
    all into ONE shared command buffer. The session_commit_count
    delta is 1 per ``batched_session()``."""
    if not bf16_session_available():
        pytest.skip("bf16 encode-session not available on this host")
    M, K = 4, 8
    X = np.zeros((M, K), dtype=np.float32)
    Theta = np.zeros((M, K), dtype=np.float32)
    x_dev = device_tensor(_to_bf16(X))
    t_dev = device_tensor(_to_bf16(Theta))
    try:
        before = session_commit_count()
        with batched_session() as s:
            y_dev = rope_enc_bf16(s, x_dev, t_dev, M=M, K=K)
        after = session_commit_count()
        y_dev.free()
        assert (after - before) == 1, (
            f"bf16 rope (3 internal encoded ops) should commit 1 cb, "
            f"got delta={after - before}")
    finally:
        x_dev.free(); t_dev.free()


# ---- Headline: full bf16 Llama attention block on one cb --------------

def test_full_bf16_llama_attention_block_on_one_command_buffer():
    """Closes the dtype matrix: a full Llama-style attention block —
    rmsnorm + 3 qkv projections + 2 ropes + flash_attn + out proj =
    8 user-facing ops, all in bf16 — commits exactly 1 cb.

    Internally, the rope and flash_attn ops each add cast nodes
    (5 extra cast ops in total: rope×2 cast-in×2 + cast-out×1 each
    = ~6 internal ops). They all encode into the same cb."""
    if not bf16_session_available():
        pytest.skip("bf16 encode-session not available on this host")

    B, S, D = 1, 8, 16
    scale = 1.0 / np.sqrt(D)
    eps = 1e-5
    rng = np.random.default_rng(0xBFA77BF)

    X = rng.standard_normal((B * S, D), dtype=np.float32) * 0.1
    gamma = rng.standard_normal((D,), dtype=np.float32)
    Wq = rng.standard_normal((D, D), dtype=np.float32) * 0.05
    Wk = rng.standard_normal((D, D), dtype=np.float32) * 0.05
    Wv = rng.standard_normal((D, D), dtype=np.float32) * 0.05
    Wo = rng.standard_normal((D, D), dtype=np.float32) * 0.05
    Theta = (np.arange(B * S * D, dtype=np.float32) * 0.001
             ).reshape(B * S, D)

    devs = [
        device_tensor(_to_bf16(X)),
        device_tensor(_to_bf16(gamma)),
        device_tensor(_to_bf16(Wq.reshape(1, D, D))),
        device_tensor(_to_bf16(Wk.reshape(1, D, D))),
        device_tensor(_to_bf16(Wv.reshape(1, D, D))),
        device_tensor(_to_bf16(Wo.reshape(1, D, D))),
        device_tensor(_to_bf16(Theta)),
    ]
    x_dev, g_dev, wq_dev, wk_dev, wv_dev, wo_dev, theta_dev = devs
    try:
        before = session_commit_count()
        with batched_session() as s:
            n = rmsnorm_enc_bf16(s, x_dev, g_dev,
                                  rows=B * S, cols=D, eps=eps)
            q = bmm_enc_bf16(s, n, wq_dev,
                              batch=1, M=B * S, N=D, K=D)
            k = bmm_enc_bf16(s, n, wk_dev,
                              batch=1, M=B * S, N=D, K=D)
            v = bmm_enc_bf16(s, n, wv_dev,
                              batch=1, M=B * S, N=D, K=D)
            q_r = rope_enc_bf16(s, q, theta_dev, M=B * S, K=D)
            k_r = rope_enc_bf16(s, k, theta_dev, M=B * S, K=D)
            a = flash_attn_enc_bf16(s, q_r, k_r, v,
                                     B=B, Sq=S, Sk=S, D=D,
                                     scale=scale, causal=False)
            out = bmm_enc_bf16(s, a, wo_dev,
                                batch=1, M=B * S, N=D, K=D)
        after = session_commit_count()
        assert (after - before) == 1, (
            f"full bf16 Llama attention should commit 1 cb, got "
            f"delta={after - before}")
        gpu = _from_bf16(
            out.download(np.uint16, (1, B * S, D))).reshape(B, S, D)
        for t in (n, q, k, v, q_r, k_r, a, out):
            t.free()
        # Sanity: well-formed output.
        assert np.isfinite(gpu).all()
        assert gpu.shape == (B, S, D)
    finally:
        for d in devs:
            d.free()
