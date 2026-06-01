"""tessera.ops.flash_attn — eager numpy reference + interception
forwarding contract.

The base ``tessera.ops.flash_attn`` ships a real numpy reference (a
naive O(N²) attention with optional causal mask, dropout, and
deterministic-seeding hooks). Earlier the auto_batch interception
wrapper raised ``NotImplementedError`` in eager mode — wrong; the
reference is genuinely usable.

These tests pin the eager path's behavior + the wrapper's compose:

* **Eager numpy reference** — Q @ K^T → softmax → @ V matches a
  hand-rolled reference at fp32 tolerance.
* **Causal mask** — when ``causal=True``, the q-row sees only k-rows
  up to its own index.
* **Scale override** — ``scale=`` overrides the default 1/sqrt(D).
* **Dropout=0 idempotence** — dropout_p=0 produces identical output
  across calls.
* **Trace-mode forwarding** — the wrapper passes scale + causal
  through to apple_gpu_ops correctly.
* **Trace-mode rejects dropout** — dropout_p>0 inside @auto_batch
  raises (the encode-session kernel is dropout-free).
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera
import tessera.apple_gpu_ops as agpu
from tessera.apple_gpu_batched import session_available


# ---- Eager numpy reference --------------------------------------------

def _np_flash_attn(Q, K, V, scale, causal=False):
    scores = np.matmul(Q, K.swapaxes(-1, -2)) * scale
    if causal:
        q_len, k_len = scores.shape[-2], scores.shape[-1]
        mask = np.triu(np.ones((q_len, k_len), dtype=bool),
                        k=1 + max(k_len - q_len, 0))
        scores = np.where(mask, -np.inf, scores)
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = e / e.sum(axis=-1, keepdims=True)
    return np.matmul(weights, V)


def test_eager_flash_attn_matches_handrolled_reference():
    B, Sq, Sk, D = 1, 8, 8, 16
    rng = np.random.default_rng(0x4A77)
    Q = rng.standard_normal((B, Sq, D), dtype=np.float32) * 0.1
    K = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1
    V = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1
    scale = 1.0 / np.sqrt(D)
    out = tessera.ops.flash_attn(Q, K, V, scale=scale, causal=False)
    expected = _np_flash_attn(Q, K, V, scale, causal=False)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_eager_flash_attn_default_scale_is_inv_sqrt_d():
    """Calling without ``scale=`` uses 1/sqrt(D) (matches the runtime
    apple_gpu_ops default)."""
    B, Sq, Sk, D = 1, 4, 4, 16
    rng = np.random.default_rng(0x5CA1E)
    Q = rng.standard_normal((B, Sq, D), dtype=np.float32) * 0.1
    K = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1
    V = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1
    out_default = tessera.ops.flash_attn(Q, K, V)
    out_explicit = tessera.ops.flash_attn(Q, K, V, scale=1.0 / np.sqrt(D))
    np.testing.assert_array_equal(out_default, out_explicit)


def test_eager_flash_attn_causal_mask_blocks_future_keys():
    """With causal=True, the first q-row should see ONLY k_row=0
    — so its output equals V[0]."""
    Q = np.zeros((1, 3, 4), dtype=np.float32)
    Q[0, 0] = 1.0  # query row 0 is non-zero
    K = np.zeros((1, 3, 4), dtype=np.float32)
    K[0, 0] = 1.0  # the only key that matters for q[0]
    V = np.array([[[1.0, 2.0, 3.0, 4.0],
                    [9.0, 9.0, 9.0, 9.0],
                    [9.0, 9.0, 9.0, 9.0]]], dtype=np.float32)
    out = tessera.ops.flash_attn(Q, K, V, scale=1.0, causal=True)
    # q[0] sees only k[0]/v[0]; softmax over one (-inf, -inf-masked)
    # entry gives weight=1 on V[0].
    np.testing.assert_allclose(out[0, 0], V[0, 0])


def test_eager_flash_attn_dropout_zero_is_deterministic():
    """With dropout_p=0 the output is deterministic across calls."""
    rng = np.random.default_rng(0xDEAD)
    Q = rng.standard_normal((1, 4, 8), dtype=np.float32)
    K = rng.standard_normal((1, 4, 8), dtype=np.float32)
    V = rng.standard_normal((1, 4, 8), dtype=np.float32)
    a = tessera.ops.flash_attn(Q, K, V, dropout_p=0.0)
    b = tessera.ops.flash_attn(Q, K, V, dropout_p=0.0)
    np.testing.assert_array_equal(a, b)


def test_eager_flash_attn_rejects_invalid_dropout():
    """dropout_p must be in [0, 1)."""
    Q = np.zeros((1, 2, 4), dtype=np.float32)
    K = np.zeros((1, 2, 4), dtype=np.float32)
    V = np.zeros((1, 2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="dropout_p"):
        tessera.ops.flash_attn(Q, K, V, dropout_p=1.0)


# ---- Wrapper composition ----------------------------------------------

def test_trace_mode_forwards_scale_and_causal_to_apple_gpu_ops():
    """When the wrapper routes to apple_gpu_ops under @auto_batch,
    it passes scale + causal through. Compare against the explicit
    apple_gpu_ops call with the same scale + causal — outputs must
    match bit-for-bit."""
    if not session_available():
        pytest.skip("encode-session unavailable")
    B, Sq, Sk, D = 1, 6, 6, 16
    rng = np.random.default_rng(0xF77FA)
    Q = rng.standard_normal((B, Sq, D), dtype=np.float32) * 0.1
    K = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1
    V = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1
    # Non-default scale to verify forwarding.
    custom_scale = 0.123

    @agpu.auto_batch
    def via_tessera_ops(q, k, v):
        return tessera.ops.flash_attn(q, k, v,
                                       B=B, Sq=Sq, Sk=Sk, D=D,
                                       scale=custom_scale, causal=True)

    @agpu.auto_batch
    def via_apple_gpu_ops(q, k, v):
        return agpu.flash_attn(q, k, v,
                                B=B, Sq=Sq, Sk=Sk, D=D,
                                scale=custom_scale, causal=True)

    a = via_tessera_ops(Q, K, V)
    b = via_apple_gpu_ops(Q, K, V)
    arr_a = a.download(np.float32, (B, Sq, D))
    arr_b = b.download(np.float32, (B, Sq, D))
    a.free(); b.free()
    np.testing.assert_array_equal(arr_a, arr_b)


def test_trace_mode_rejects_dropout():
    """The encode-session flash_attn kernel is dropout-free. The
    wrapper raises if dropout_p > 0 inside @auto_batch (no silent
    incorrect output)."""
    if not session_available():
        pytest.skip("encode-session unavailable")
    Q = np.zeros((1, 4, 8), dtype=np.float32)
    K = np.zeros((1, 4, 8), dtype=np.float32)
    V = np.zeros((1, 4, 8), dtype=np.float32)

    @agpu.auto_batch
    def fn(q, k, v):
        return tessera.ops.flash_attn(q, k, v,
                                       B=1, Sq=4, Sk=4, D=8,
                                       dropout_p=0.5)

    with pytest.raises(ValueError, match="dropout_p>0 is not supported"):
        fn(Q, K, V)


def test_trace_mode_with_dropout_zero_works():
    """dropout_p=0 (the default) is allowed inside @auto_batch — the
    wrapper just ignores it (the encode kernel doesn't apply dropout
    so the value is moot)."""
    if not session_available():
        pytest.skip("encode-session unavailable")
    B, Sq, Sk, D = 1, 4, 4, 8
    rng = np.random.default_rng(0xD400FF)
    Q = rng.standard_normal((B, Sq, D), dtype=np.float32) * 0.1
    K = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1
    V = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1

    @agpu.auto_batch
    def fn(q, k, v):
        return tessera.ops.flash_attn(q, k, v,
                                       B=B, Sq=Sq, Sk=Sk, D=D,
                                       dropout_p=0.0)

    out = fn(Q, K, V)
    arr = out.download(np.float32, (B, Sq, D))
    out.free()
    assert np.isfinite(arr).all()


def test_eager_and_trace_outputs_agree_at_fp32_tolerance():
    """The HEADLINE consistency check: tessera.ops.flash_attn eager
    output ≈ tessera.ops.flash_attn under @auto_batch (modulo the
    online-softmax fp32 precision difference)."""
    if not session_available():
        pytest.skip("encode-session unavailable")
    B, Sq, Sk, D = 1, 8, 8, 16
    rng = np.random.default_rng(0xEAA77)
    Q = rng.standard_normal((B, Sq, D), dtype=np.float32) * 0.1
    K = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1
    V = rng.standard_normal((B, Sk, D), dtype=np.float32) * 0.1

    eager = tessera.ops.flash_attn(Q, K, V, causal=False)

    @agpu.auto_batch
    def via_trace(q, k, v):
        return tessera.ops.flash_attn(q, k, v,
                                       B=B, Sq=Sq, Sk=Sk, D=D,
                                       causal=False)

    traced_dev = via_trace(Q, K, V)
    traced = traced_dev.download(np.float32, (B, Sq, D))
    traced_dev.free()
    np.testing.assert_allclose(traced, eager, rtol=2e-3, atol=2e-3)
