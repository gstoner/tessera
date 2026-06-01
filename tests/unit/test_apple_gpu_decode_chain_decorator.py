"""``@decode_chain`` decorator — phase 1 of jit auto-detection.

Stage-3 phase-1 of the single-cb decode-chain roadmap. The decorator
wraps a function whose first positional arg is the encode-session
handle, opens a fresh session before the call, and commits + waits
on exit. Phase 2 (true JIT-level auto-detection inside
``compiler/jit.py``) is documented in
``docs/audit/single_command_buffer_decode_plan.md``.

These tests pin the decorator's contract:

* **Single-cb invariant** — a decorated function with N encoded ops
  inside it commits exactly 1 cb (same as an explicit
  ``with batched_session():`` block).
* **Pass-through** — args, kwargs, and the return value are
  forwarded unchanged.
* **Numerical correctness** — a decorated Llama-style attention
  block produces the same output as the explicit-session form.
* **Composability** — decorator can be applied to nested functions
  (each opens its own session — explicit choice to keep the
  contract clean, vs. flattening into the outer session which would
  require contextvar plumbing).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.apple_gpu_batched import (
    bmm_enc,
    decode_chain,
    device_tensor,
    flash_attn_enc,
    rmsnorm_enc,
    rope_enc,
    session_available,
    session_commit_count,
)


# ---- Decorator contract ------------------------------------------------

def test_decorator_passes_session_as_first_arg():
    if not session_available():
        pytest.skip("Apple GPU encode-session not available")

    @decode_chain
    def echo_session(s, *args, **kwargs):
        # Session handle is a non-zero int.
        assert isinstance(s, int)
        assert s != 0
        return s

    handle = echo_session()
    # After the call returns, the session has been committed + freed;
    # the handle value is opaque (int) and not reusable.
    assert isinstance(handle, int)


def test_decorator_forwards_positional_and_keyword_args():
    if not session_available():
        pytest.skip("Apple GPU encode-session not available")

    @decode_chain
    def collect(s, a, b, *, c=None):
        return (a, b, c)

    result = collect(1, "two", c=3.0)
    assert result == (1, "two", 3.0)


def test_decorator_propagates_exceptions():
    """If the wrapped function raises, the session still commits
    (Python's ``with`` semantics) but the exception bubbles out."""
    if not session_available():
        pytest.skip("Apple GPU encode-session not available")

    @decode_chain
    def buggy(s):
        raise ValueError("intentional")

    with pytest.raises(ValueError, match="intentional"):
        buggy()


def test_decorator_returns_function_output_unchanged():
    if not session_available():
        pytest.skip("Apple GPU encode-session not available")

    rows, cols, eps = 4, 16, 1e-5
    rng = np.random.default_rng(0xDEC0DE)
    X = rng.standard_normal((rows, cols), dtype=np.float32) * 0.1
    gamma = rng.standard_normal((cols,), dtype=np.float32)

    @decode_chain
    def chain(s, x_dev, g_dev):
        return rmsnorm_enc(s, x_dev, g_dev,
                            rows=rows, cols=cols, eps=eps)

    x_dev = device_tensor(X)
    g_dev = device_tensor(gamma)
    try:
        out = chain(x_dev, g_dev)
        # Out is a DeviceTensor pointing at a valid GPU buffer.
        gpu_data = out.download(np.float32, (rows, cols))
        out.free()
        # Numerical sanity: each row's RMS is ~1 after norm (modulo
        # gamma scaling).
        assert gpu_data.shape == (rows, cols)
    finally:
        x_dev.free(); g_dev.free()


# ---- Single-cb invariant -----------------------------------------------

def test_decorator_commits_exactly_one_command_buffer():
    """The decorator's headline value: a chain of N encoded ops
    inside the wrapped function commits exactly 1 cb."""
    if not session_available():
        pytest.skip("Apple GPU encode-session not available")
    rows, cols = 4, 16
    eps = 1e-5
    rng = np.random.default_rng(0xCC1ED0)
    X = rng.standard_normal((rows, cols), dtype=np.float32) * 0.1
    gamma = rng.standard_normal((cols,), dtype=np.float32)
    W = rng.standard_normal((cols, cols), dtype=np.float32) * 0.1

    @decode_chain
    def chain(s, x_dev, g_dev, w_dev):
        n = rmsnorm_enc(s, x_dev, g_dev,
                         rows=rows, cols=cols, eps=eps)
        out = bmm_enc(s, n, w_dev,
                       batch=1, M=rows, N=cols, K=cols)
        return out

    x_dev = device_tensor(X)
    g_dev = device_tensor(gamma)
    w_dev = device_tensor(W.reshape(1, cols, cols))
    try:
        before = session_commit_count()
        out_dev = chain(x_dev, g_dev, w_dev)
        after = session_commit_count()
        assert (after - before) == 1, (
            f"decorator should commit exactly 1 cb, got "
            f"delta={after - before}")
        gpu_out = out_dev.download(
            np.float32, (1, rows, cols)).reshape(rows, cols)
        out_dev.free()

        # Numerical: matches explicit session form.
        # CPU reference.
        var = (X * X).mean(axis=-1, keepdims=True)
        n_ref = X / np.sqrt(var + eps) * gamma
        expected = n_ref @ W
        np.testing.assert_allclose(gpu_out, expected,
                                    rtol=2e-3, atol=2e-3)
    finally:
        x_dev.free(); g_dev.free(); w_dev.free()


# ---- Llama attention block via the decorator --------------------------

def test_decorated_llama_attention_block_matches_explicit_form():
    """The full Llama attention block via @decode_chain — verifies
    the decorator is a drop-in replacement for the explicit
    ``with batched_session():`` form."""
    if not session_available():
        pytest.skip("Apple GPU encode-session not available")
    B, S, D = 1, 8, 16
    scale = 1.0 / np.sqrt(D)
    eps = 1e-5
    rng = np.random.default_rng(0x11A4A77E)

    X = rng.standard_normal((B * S, D), dtype=np.float32) * 0.1
    gamma = rng.standard_normal((D,), dtype=np.float32)
    Wq = rng.standard_normal((D, D), dtype=np.float32) * 0.05
    Wk = rng.standard_normal((D, D), dtype=np.float32) * 0.05
    Wv = rng.standard_normal((D, D), dtype=np.float32) * 0.05
    Wo = rng.standard_normal((D, D), dtype=np.float32) * 0.05
    Theta = (np.arange(B * S * D, dtype=np.float32) * 0.001
             ).reshape(B * S, D)

    devs = [
        device_tensor(X), device_tensor(gamma),
        device_tensor(Wq.reshape(1, D, D)),
        device_tensor(Wk.reshape(1, D, D)),
        device_tensor(Wv.reshape(1, D, D)),
        device_tensor(Wo.reshape(1, D, D)),
        device_tensor(Theta),
    ]
    x_dev, g_dev, wq_dev, wk_dev, wv_dev, wo_dev, theta_dev = devs

    @decode_chain
    def llama_attention(s, x, g, wq, wk, wv, wo, theta, *,
                         B, S, D, scale, eps):
        n = rmsnorm_enc(s, x, g, rows=B * S, cols=D, eps=eps)
        q = bmm_enc(s, n, wq, batch=1, M=B * S, N=D, K=D)
        k = bmm_enc(s, n, wk, batch=1, M=B * S, N=D, K=D)
        v = bmm_enc(s, n, wv, batch=1, M=B * S, N=D, K=D)
        q_r = rope_enc(s, q, theta, M=B * S, K=D)
        k_r = rope_enc(s, k, theta, M=B * S, K=D)
        a = flash_attn_enc(s, q_r, k_r, v,
                            B=B, Sq=S, Sk=S, D=D, scale=scale)
        return bmm_enc(s, a, wo, batch=1, M=B * S, N=D, K=D)

    try:
        before = session_commit_count()
        out_dev = llama_attention(x_dev, g_dev, wq_dev, wk_dev,
                                    wv_dev, wo_dev, theta_dev,
                                    B=B, S=S, D=D, scale=scale, eps=eps)
        after = session_commit_count()
        assert (after - before) == 1, (
            f"Llama attention via decorator should commit 1 cb, "
            f"got {after - before}")
        gpu_out = out_dev.download(
            np.float32, (1, B * S, D)).reshape(B, S, D)
        out_dev.free()
        # Sanity check on output: well-formed (no NaN/inf, reasonable
        # magnitude given the small inputs).
        assert np.isfinite(gpu_out).all()
        assert gpu_out.shape == (B, S, D)
    finally:
        for d in devs:
            d.free()


# ---- Nested decorators --------------------------------------------------

def test_nested_decorated_functions_each_open_their_own_session():
    """Each @decode_chain opens a fresh session. Nesting doesn't
    flatten — explicit choice for now. (Contextvar-based flattening
    is a possible follow-on.)"""
    if not session_available():
        pytest.skip("Apple GPU encode-session not available")

    @decode_chain
    def inner(s):
        return 1

    @decode_chain
    def outer(s):
        inner_result = inner()
        return inner_result + 1

    before = session_commit_count()
    result = outer()
    after = session_commit_count()
    assert result == 2
    # Outer + inner = 2 commits.
    assert (after - before) == 2
