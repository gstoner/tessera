"""Project-3 f16 encode-session ABIs — fp16 LLM-decode coverage.

Stage-2 closed the f32 encode envelope for a decoder layer; Project 3
adds the f16 variants so real LLM workloads (which typically run in
fp16 with f32 master accumulators) can use the single-cb path.

Tests pin:

* **Symbol availability** for all 7 new f16 C ABIs.
* **Numerical correctness** of each f16 op against a numpy reference
  at fp16 tolerance. The ml_dtypes.float16 byte layout is what's
  passed across the C ABI; we round-trip through it to verify the
  runtime decodes / re-encodes the bit pattern correctly.
* **Batched bmm > 1** — the existing f32 bmm_enc already accepts a
  ``batch`` arg; verify it actually executes for batch=2/4 and the
  f16 variant honors batch too.
* **f16 attention block on one cb** — repeat the Llama-style block
  from the f32 tests in fp16 and prove single-cb behavior holds.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

# numpy has native fp16 (np.float16) — ml_dtypes adds bfloat16 / fp8
# but float16 is IEEE-754 half so it lives in numpy. Use np.float16
# directly; the C ABI just consumes the 16-bit bit pattern via
# uint16 view.
_ML_DTYPES = True

from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol
from tessera.apple_gpu_batched import (
    batched_session,
    bmm_enc,
    bmm_enc_f16,
    device_tensor,
    flash_attn_enc_f16,
    gelu_enc_f16,
    layer_norm_enc_f16,
    rmsnorm_enc_f16,
    rope_enc_f16,
    session_available,
    session_commit_count,
    silu_enc_f16,
    softmax_enc_f16,
)


def _to_f16(a: np.ndarray) -> np.ndarray:
    """Convert fp32 → fp16 bit pattern (np.float16 view as uint16).
    Caller passes the uint16 bytes to the C ABI."""
    return np.ascontiguousarray(a.astype(np.float16)).view(np.uint16)


def _from_f16(a: np.ndarray) -> np.ndarray:
    return a.view(np.float16).astype(np.float32)


# ---- Symbol availability -----------------------------------------------

@pytest.mark.parametrize("symbol", [
    "tessera_apple_gpu_bmm_dev_f16_enc",
    "tessera_apple_gpu_layer_norm_dev_f16_enc",
    "tessera_apple_gpu_rmsnorm_dev_f16_enc",
    "tessera_apple_gpu_softmax_dev_f16_enc",
    "tessera_apple_gpu_rope_dev_f16_enc",
    "tessera_apple_gpu_unary_dev_f16_enc",
    "tessera_apple_gpu_flash_attn_dev_f16_enc",
])
def test_f16_encode_symbols_resolve(symbol):
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    fn = bind_symbol(symbol, (ctypes.c_void_p,), ctypes.c_int32)
    assert fn is not None, f"missing f16 ABI: {symbol}"


# ---- Per-op f16 numerical correctness -----------------------------------

def test_rmsnorm_f16_matches_numpy():
    if not session_available() or not _ML_DTYPES:
        pytest.skip("encode-session or ml_dtypes unavailable")
    rows, cols, eps = 8, 32, 1e-5
    rng = np.random.default_rng(0xF16D8E1)
    X = rng.standard_normal((rows, cols), dtype=np.float32) * 0.5
    gamma = rng.standard_normal((cols,), dtype=np.float32)
    X16 = _to_f16(X)
    g16 = _to_f16(gamma)
    x_dev = device_tensor(X16)
    g_dev = device_tensor(g16)
    try:
        with batched_session() as s:
            y_dev = rmsnorm_enc_f16(s, x_dev, g_dev,
                                     rows=rows, cols=cols, eps=eps)
        gpu_out = _from_f16(y_dev.download(np.uint16, (rows, cols)))
        y_dev.free()
        # numpy reference (fp32 math).
        var = (X * X).mean(axis=-1, keepdims=True)
        expected = X / np.sqrt(var + eps) * gamma
        # fp16 has ~3 decimal digits of precision; rtol=5e-3 is the
        # loose-but-fair tolerance for a single-pass norm.
        np.testing.assert_allclose(gpu_out, expected, rtol=5e-3, atol=5e-3)
    finally:
        x_dev.free(); g_dev.free()


def test_layer_norm_f16_matches_numpy():
    if not session_available() or not _ML_DTYPES:
        pytest.skip("encode-session or ml_dtypes unavailable")
    rows, cols, eps = 8, 32, 1e-5
    rng = np.random.default_rng(0xF16178)
    X = rng.standard_normal((rows, cols), dtype=np.float32) * 0.5
    gamma = rng.standard_normal((cols,), dtype=np.float32)
    beta = rng.standard_normal((cols,), dtype=np.float32)
    x_dev = device_tensor(_to_f16(X))
    g_dev = device_tensor(_to_f16(gamma))
    b_dev = device_tensor(_to_f16(beta))
    try:
        with batched_session() as s:
            y_dev = layer_norm_enc_f16(s, x_dev, g_dev, b_dev,
                                        rows=rows, cols=cols, eps=eps)
        gpu_out = _from_f16(y_dev.download(np.uint16, (rows, cols)))
        y_dev.free()
        mean = X.mean(axis=-1, keepdims=True)
        var = X.var(axis=-1, keepdims=True)
        expected = (X - mean) / np.sqrt(var + eps) * gamma + beta
        np.testing.assert_allclose(gpu_out, expected, rtol=5e-3, atol=5e-3)
    finally:
        x_dev.free(); g_dev.free(); b_dev.free()


def test_softmax_f16_matches_numpy():
    if not session_available() or not _ML_DTYPES:
        pytest.skip("encode-session or ml_dtypes unavailable")
    rows, cols = 6, 12
    rng = np.random.default_rng(0xF1650A)
    X = rng.standard_normal((rows, cols), dtype=np.float32)
    x_dev = device_tensor(_to_f16(X))
    try:
        with batched_session() as s:
            y_dev = softmax_enc_f16(s, x_dev, rows=rows, cols=cols)
        gpu_out = _from_f16(y_dev.download(np.uint16, (rows, cols)))
        y_dev.free()
        # numpy softmax reference.
        m = X.max(axis=-1, keepdims=True)
        e = np.exp(X - m)
        expected = e / e.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(gpu_out, expected, rtol=5e-3, atol=5e-3)
        np.testing.assert_allclose(gpu_out.sum(axis=-1), 1.0, atol=1e-2)
    finally:
        x_dev.free()


def test_silu_f16_matches_numpy():
    if not session_available() or not _ML_DTYPES:
        pytest.skip("encode-session or ml_dtypes unavailable")
    n = 128
    rng = np.random.default_rng(0xF165B5)
    X = rng.standard_normal((n,), dtype=np.float32)
    x_dev = device_tensor(_to_f16(X))
    try:
        with batched_session() as s:
            y_dev = silu_enc_f16(s, x_dev, n=n)
        gpu_out = _from_f16(y_dev.download(np.uint16, (n,)))
        y_dev.free()
        expected = X / (1.0 + np.exp(-X))
        np.testing.assert_allclose(gpu_out, expected, rtol=5e-3, atol=5e-3)
    finally:
        x_dev.free()


def test_gelu_f16_matches_numpy():
    if not session_available() or not _ML_DTYPES:
        pytest.skip("encode-session or ml_dtypes unavailable")
    n = 128
    rng = np.random.default_rng(0xF16E1)
    X = rng.standard_normal((n,), dtype=np.float32)
    x_dev = device_tensor(_to_f16(X))
    try:
        with batched_session() as s:
            y_dev = gelu_enc_f16(s, x_dev, n=n)
        gpu_out = _from_f16(y_dev.download(np.uint16, (n,)))
        y_dev.free()
        c = np.sqrt(2.0 / np.pi)
        expected = 0.5 * X * (1.0 + np.tanh(c * (X + 0.044715 * X ** 3)))
        np.testing.assert_allclose(gpu_out, expected, rtol=1e-2, atol=1e-2)
    finally:
        x_dev.free()


def test_rope_f16_matches_numpy():
    if not session_available() or not _ML_DTYPES:
        pytest.skip("encode-session or ml_dtypes unavailable")
    M, K = 6, 8
    rng = np.random.default_rng(0xF16E0)
    X = rng.standard_normal((M * K,), dtype=np.float32)
    Theta = (np.arange(M * K, dtype=np.float32) * 0.01).astype(np.float32)
    x_dev = device_tensor(_to_f16(X.reshape(M, K)))
    t_dev = device_tensor(_to_f16(Theta.reshape(M, K)))
    try:
        with batched_session() as s:
            y_dev = rope_enc_f16(s, x_dev, t_dev, M=M, K=K)
        gpu_out = _from_f16(y_dev.download(np.uint16, (M, K))).reshape(M * K)
        y_dev.free()
        expected = np.empty_like(X)
        for m in range(M):
            for pair in range(K // 2):
                ie = m * K + pair * 2
                io = ie + 1
                c, s = np.cos(Theta[ie]), np.sin(Theta[ie])
                expected[ie] = X[ie] * c - X[io] * s
                expected[io] = X[ie] * s + X[io] * c
        np.testing.assert_allclose(gpu_out, expected, rtol=5e-3, atol=5e-3)
    finally:
        x_dev.free(); t_dev.free()


def test_flash_attn_f16_matches_numpy():
    if not session_available() or not _ML_DTYPES:
        pytest.skip("encode-session or ml_dtypes unavailable")
    B, Sq, Sk, D = 1, 8, 8, 16
    scale = 1.0 / np.sqrt(D)
    rng = np.random.default_rng(0xF1AA7)
    Q = rng.standard_normal((B, Sq, D), dtype=np.float32) * 0.1
    K = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1
    V = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1
    q_dev = device_tensor(_to_f16(Q))
    k_dev = device_tensor(_to_f16(K))
    v_dev = device_tensor(_to_f16(V))
    try:
        with batched_session() as s:
            o_dev = flash_attn_enc_f16(s, q_dev, k_dev, v_dev,
                                        B=B, Sq=Sq, Sk=Sk, D=D,
                                        scale=scale, causal=False)
        gpu_out = _from_f16(o_dev.download(np.uint16, (B, Sq, D)))
        o_dev.free()
        # numpy reference.
        scores = np.einsum("bqd,bkd->bqk", Q, K) * scale
        scores = scores - scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores)
        attn = attn / attn.sum(axis=-1, keepdims=True)
        expected = np.einsum("bqk,bkd->bqd", attn, V)
        # fp16 + online softmax — relax to ~1% tolerance.
        np.testing.assert_allclose(gpu_out, expected, rtol=1e-2, atol=1e-2)
    finally:
        q_dev.free(); k_dev.free(); v_dev.free()


# ---- Batched bmm (batch > 1) -------------------------------------------

@pytest.mark.parametrize("batch", [2, 4])
def test_bmm_enc_f32_handles_batch_greater_than_one(batch):
    if not session_available():
        pytest.skip("encode-session unavailable")
    M, N, K = 4, 8, 6
    rng = np.random.default_rng(0xBA7CED ^ batch)
    A = rng.standard_normal((batch, M, K), dtype=np.float32)
    B_mat = rng.standard_normal((batch, K, N), dtype=np.float32)
    a_dev = device_tensor(A)
    b_dev = device_tensor(B_mat)
    try:
        with batched_session() as s:
            o = bmm_enc(s, a_dev, b_dev,
                         batch=batch, M=M, N=N, K=K, b_broadcast=False)
        gpu_out = o.download(np.float32, (batch, M, N))
        o.free()
        expected = np.einsum("bik,bkj->bij", A, B_mat)
        np.testing.assert_allclose(gpu_out, expected, rtol=1e-4, atol=1e-4)
    finally:
        a_dev.free(); b_dev.free()


def test_bmm_enc_f16_handles_batch_greater_than_one():
    if not session_available() or not _ML_DTYPES:
        pytest.skip("encode-session or ml_dtypes unavailable")
    batch, M, N, K = 3, 4, 8, 6
    rng = np.random.default_rng(0xBA7C16)
    A = rng.standard_normal((batch, M, K), dtype=np.float32) * 0.3
    B_mat = rng.standard_normal((batch, K, N), dtype=np.float32) * 0.3
    a_dev = device_tensor(_to_f16(A))
    b_dev = device_tensor(_to_f16(B_mat))
    try:
        with batched_session() as s:
            o = bmm_enc_f16(s, a_dev, b_dev,
                             batch=batch, M=M, N=N, K=K, b_broadcast=False)
        gpu_out = _from_f16(o.download(np.uint16, (batch, M, N)))
        o.free()
        expected = np.einsum("bik,bkj->bij", A, B_mat)
        np.testing.assert_allclose(gpu_out, expected, rtol=1e-2, atol=1e-2)
    finally:
        a_dev.free(); b_dev.free()


# ---- f16 Llama-style attention block on one cb -------------------------

def test_f16_llama_attention_block_on_one_command_buffer():
    if not session_available() or not _ML_DTYPES:
        pytest.skip("encode-session or ml_dtypes unavailable")
    B, S, D = 1, 8, 16
    scale = 1.0 / np.sqrt(D)
    eps = 1e-5
    rng = np.random.default_rng(0xF16A77)

    X = rng.standard_normal((B * S, D), dtype=np.float32) * 0.1
    gamma = rng.standard_normal((D,), dtype=np.float32)
    Wq = rng.standard_normal((D, D), dtype=np.float32) * 0.05
    Wk = rng.standard_normal((D, D), dtype=np.float32) * 0.05
    Wv = rng.standard_normal((D, D), dtype=np.float32) * 0.05
    Wo = rng.standard_normal((D, D), dtype=np.float32) * 0.05
    Theta = (np.arange(B * S * D, dtype=np.float32) * 0.001
             ).reshape(B * S, D)

    devs = [
        device_tensor(_to_f16(X)),
        device_tensor(_to_f16(gamma)),
        device_tensor(_to_f16(Wq.reshape(1, D, D))),
        device_tensor(_to_f16(Wk.reshape(1, D, D))),
        device_tensor(_to_f16(Wv.reshape(1, D, D))),
        device_tensor(_to_f16(Wo.reshape(1, D, D))),
        device_tensor(_to_f16(Theta)),
    ]
    x_dev, g_dev, wq_dev, wk_dev, wv_dev, wo_dev, theta_dev = devs

    try:
        before = session_commit_count()
        with batched_session() as s:
            x_n = rmsnorm_enc_f16(s, x_dev, g_dev,
                                   rows=B * S, cols=D, eps=eps)
            q = bmm_enc_f16(s, x_n, wq_dev,
                             batch=1, M=B * S, N=D, K=D)
            k = bmm_enc_f16(s, x_n, wk_dev,
                             batch=1, M=B * S, N=D, K=D)
            v = bmm_enc_f16(s, x_n, wv_dev,
                             batch=1, M=B * S, N=D, K=D)
            q_r = rope_enc_f16(s, q, theta_dev, M=B * S, K=D)
            k_r = rope_enc_f16(s, k, theta_dev, M=B * S, K=D)
            attn = flash_attn_enc_f16(s, q_r, k_r, v,
                                       B=B, Sq=S, Sk=S, D=D,
                                       scale=scale, causal=False)
            out = bmm_enc_f16(s, attn, wo_dev,
                               batch=1, M=B * S, N=D, K=D)
        after = session_commit_count()
        assert (after - before) == 1, (
            f"f16 Llama attention block expected 1 commit, got "
            f"delta={after - before}")
        gpu_out = _from_f16(
            out.download(np.uint16, (1, B * S, D))).reshape(B, S, D)
        for t in (x_n, q, k, v, q_r, k_r, attn, out):
            t.free()
        # Sanity: output is well-formed (no NaN/inf) and finite-magnitude.
        assert np.isfinite(gpu_out).all()
        assert gpu_out.shape == (B, S, D)
    finally:
        for d in devs:
            d.free()
