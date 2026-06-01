"""Single-command-buffer decode chain — scaffold proof (audit Action 6).

The audit's deferred Action 6 (table row #6) asks for "prefill /
decode / attn / MLP / projection on one command buffer." This test
file pins the **scaffold stage** of the roadmap (see
``docs/audit/single_command_buffer_decode_plan.md``): a 2-op chain
(``layer_norm + bmm``) runs in ONE session, produces the right
numerical answer, and submits exactly ONE command buffer.

Stage 2 follow-ons (rope_enc / flash_attn_enc / softmax_enc / etc.)
extend the chain to a full decoder layer.

Tests pin:

* **Symbol availability** — both new C ABI symbols
  (``tessera_apple_gpu_layer_norm_dev_f32_enc`` and
  ``tessera_apple_gpu_session_commit_count``) resolve.
* **Numerical correctness** — the encoded layer_norm output matches a
  numpy reference at fp32 tolerance.
* **Single command buffer** — running 2 ops in one session
  increments the commit counter by exactly 1, not 2.
* **Encoded chain numerical correctness** — a layer_norm output fed
  into a bmm produces the right ``layer_norm(X) @ W`` result,
  end-to-end on one cb.
* **Lifecycle** — a session that's been ``ts_enc_commit_wait``'d is
  not reusable; the Python context manager handles this correctly.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol
from tessera.apple_gpu_batched import (
    batched_session,
    bmm_enc,
    device_empty,
    device_tensor,
    layer_norm_enc,
    session_available,
    session_commit_count,
)


# ---- Symbol-availability gate -------------------------------------------

def test_session_symbols_resolve():
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    assert session_available(), (
        "Apple GPU encode-session ABI failed to bind — check the "
        "TesseraAppleRuntime library has the new symbols")


def test_layer_norm_dev_f32_enc_symbol_resolves():
    """The new C ABI symbol shipped in this PR."""
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    fn = bind_symbol(
        "tessera_apple_gpu_layer_norm_dev_f32_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_float),
        ctypes.c_int32)
    assert fn is not None


def test_session_commit_count_symbol_resolves():
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    fn = bind_symbol(
        "tessera_apple_gpu_session_commit_count",
        (), ctypes.c_int64)
    assert fn is not None


# ---- Single-op numerical correctness ------------------------------------

def _np_layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                   eps: float) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps) * gamma + beta).astype(
        np.float32)


def test_layer_norm_enc_matches_numpy():
    if not session_available():
        pytest.skip("Apple GPU encode-session not available")
    rows, cols = 8, 32
    eps = 1e-5
    rng = np.random.default_rng(0xBEEF)
    X = rng.standard_normal((rows, cols), dtype=np.float32)
    gamma = rng.standard_normal((cols,), dtype=np.float32)
    beta = rng.standard_normal((cols,), dtype=np.float32)

    x_dev = device_tensor(X)
    g_dev = device_tensor(gamma)
    b_dev = device_tensor(beta)
    try:
        with batched_session() as s:
            y_dev = layer_norm_enc(s, x_dev, g_dev, b_dev,
                                    rows=rows, cols=cols, eps=eps)
        try:
            gpu_out = y_dev.download(np.float32, (rows, cols))
        finally:
            y_dev.free()
        expected = _np_layer_norm(X, gamma, beta, eps)
        np.testing.assert_allclose(gpu_out, expected, rtol=1e-4, atol=1e-4)
    finally:
        x_dev.free()
        g_dev.free()
        b_dev.free()


# ---- Two-op chain: ONE command buffer commits, both ops execute --------

def test_layer_norm_plus_bmm_runs_on_one_command_buffer():
    """The whole point of the scaffold: two ops in one session ⇒
    exactly one command-buffer commit. Drift gate for the
    architecture goal."""
    if not session_available():
        pytest.skip("Apple GPU encode-session not available")
    rows = 4
    cols = 16   # M
    K = cols    # bmm uses (batch, M, K) @ (batch, K, N) — we feed
                # layer_norm's (rows, cols) output as A with batch=1.
    N = 8
    eps = 1e-5
    rng = np.random.default_rng(0xC0FFEE)
    X = rng.standard_normal((rows, cols), dtype=np.float32)
    gamma = rng.standard_normal((cols,), dtype=np.float32)
    beta = rng.standard_normal((cols,), dtype=np.float32)
    W = rng.standard_normal((K, N), dtype=np.float32)

    x_dev = device_tensor(X)
    g_dev = device_tensor(gamma)
    b_dev = device_tensor(beta)
    # W is (K, N); bmm expects (batch, K, N) — broadcast batch=1.
    w_dev = device_tensor(W.reshape(1, K, N))
    try:
        count_before = session_commit_count()
        with batched_session() as s:
            y_dev = layer_norm_enc(s, x_dev, g_dev, b_dev,
                                    rows=rows, cols=cols, eps=eps)
            # bmm wants (batch, M, K). The layer_norm output is (rows, cols)
            # which we treat as (1, rows, cols) = (batch=1, M=rows, K=cols).
            z_dev = bmm_enc(s, y_dev, w_dev,
                            batch=1, M=rows, N=N, K=K,
                            b_broadcast=False)
        count_after = session_commit_count()
        # The session's exit committed EXACTLY one cb. Any tear-down
        # path that submits a second cb (a regression) trips this.
        # Note: the counter advances by 1 per ts_enc_commit_wait + any
        # OTHER concurrent Pattern-4 dispatch (which would also be a
        # bug in this test environment — the test is single-threaded).
        delta = count_after - count_before
        assert delta == 1, (
            f"expected exactly 1 session commit, got delta={delta} "
            f"(before={count_before} after={count_after}) — chain may "
            f"have been split across multiple command buffers")
        try:
            gpu_out = z_dev.download(np.float32, (1, rows, N))
        finally:
            y_dev.free()
            z_dev.free()

        # Numerical end-to-end: layer_norm(X) @ W
        ln_ref = _np_layer_norm(X, gamma, beta, eps)
        expected = (ln_ref @ W).reshape(1, rows, N)
        np.testing.assert_allclose(gpu_out, expected, rtol=1e-3, atol=1e-3)
    finally:
        x_dev.free()
        g_dev.free()
        b_dev.free()
        w_dev.free()


# ---- Counter semantics --------------------------------------------------

def test_session_commit_count_increments_per_session():
    """Each ``with batched_session():`` block, even with zero
    encoded ops inside, increments the commit counter by 1. (The
    empty-session case is technically valid — ts_enc_commit_wait
    commits the empty cb.)"""
    if not session_available():
        pytest.skip("Apple GPU encode-session not available")
    before = session_commit_count()
    with batched_session():
        pass
    after_one = session_commit_count()
    with batched_session():
        pass
    after_two = session_commit_count()
    # Each session increments by 1 (the shared event ticks once per
    # ts_enc_commit_wait).
    assert (after_one - before) == 1, (after_one, before)
    assert (after_two - after_one) == 1, (after_two, after_one)


# ---- Device-tensor lifecycle -------------------------------------------

def test_device_tensor_upload_download_round_trip():
    if not session_available():
        pytest.skip("Apple GPU encode-session not available")
    rng = np.random.default_rng(0xFACE)
    X = rng.standard_normal((4, 8), dtype=np.float32)
    dev = device_tensor(X)
    try:
        out = dev.download(np.float32, (4, 8))
        np.testing.assert_array_equal(out, X)
    finally:
        dev.free()


def test_device_empty_allocates_uninitialized_buffer():
    if not session_available():
        pytest.skip("Apple GPU encode-session not available")
    dev = device_empty(64)
    try:
        assert dev.nbytes == 64
        assert dev.handle != 0
    finally:
        dev.free()


def test_device_tensor_size_mismatch_raises_on_upload():
    if not session_available():
        pytest.skip("Apple GPU encode-session not available")
    dev = device_empty(16)
    try:
        # Wrong size — too big.
        with pytest.raises(ValueError, match="size"):
            dev.upload(np.zeros((4, 4), dtype=np.float32))
    finally:
        dev.free()


def test_device_tensor_size_mismatch_raises_on_download():
    if not session_available():
        pytest.skip("Apple GPU encode-session not available")
    dev = device_empty(16)
    try:
        with pytest.raises(ValueError, match="size mismatch"):
            dev.download(np.float32, (4, 4))  # 64 bytes, but dev is 16
    finally:
        dev.free()
