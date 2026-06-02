"""JIT Phase 2.1b — @jit(target='apple_gpu', auto_batch=True) integration.

Pins the user-facing `auto_batch` kwarg on `tessera.jit`:

* **End-to-end** — a function decorated with
  ``@tessera.jit(target='apple_gpu', auto_batch=True)`` using
  ``apple_gpu_ops.*`` inside runs as ONE command buffer with correct
  output.
* **Opt-in only** — when ``auto_batch=False`` (default), the decorator
  is a passthrough that doesn't activate trace capture.
* **Target check** — ``auto_batch=True`` with ``target != "apple_gpu"``
  raises ``TesseraJitError`` at decoration time. Honest scoping.

Phase 2.1c (open): full ``tessera.ops.*`` interception. The kwarg
landed here just composes the JitFn with the apple_gpu_ops trace
decorator; users still need to call ``tessera.apple_gpu_ops.*``
inside the function body. Documented in
``docs/audit/backend/apple/APPLE_AUDIT.md``.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

import tessera
from tessera.apple_gpu_batched import session_available, session_commit_count
from tessera.compiler.jit import TesseraJitError
import tessera.apple_gpu_ops as agpu


# ---- Target check at decoration time -----------------------------------

def test_auto_batch_requires_apple_gpu_target():
    """auto_batch=True with target != apple_gpu raises immediately."""
    src = textwrap.dedent("""\
        def fn(x):
            return x
        """)
    with pytest.raises(TesseraJitError, match="auto_batch=True"):
        @tessera.jit(target="apple_cpu", auto_batch=True,
                     source=src)
        def fn(x):
            return x


def test_auto_batch_requires_a_target():
    """auto_batch=True without target also raises (target_kind is
    None, which isn't 'apple_gpu')."""
    src = textwrap.dedent("""\
        def fn(x):
            return x
        """)
    with pytest.raises(TesseraJitError, match="auto_batch=True"):
        @tessera.jit(auto_batch=True, source=src)
        def fn(x):
            return x


# ---- Default behavior is unchanged -------------------------------------

def test_auto_batch_false_is_default_and_passthrough():
    """Without auto_batch=True, @jit(target='apple_gpu') behaves
    exactly as before — no trace activation."""
    if not session_available():
        pytest.skip("encode-session unavailable")
    src = textwrap.dedent("""\
        def add_one(x):
            return x + 1
        """)

    @tessera.jit(target="apple_gpu", source=src)
    def add_one(x):
        return x + 1

    out = add_one(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(out, np.array([2.0, 3.0, 4.0]))


# ---- End-to-end @jit + auto_batch --------------------------------------

def test_jit_auto_batch_single_op_chain_commits_one_cb():
    """A function calling apple_gpu_ops.rmsnorm under
    @jit(target='apple_gpu', auto_batch=True) commits exactly 1 cb
    and produces correct numerical output. End-to-end proof that the
    kwarg wires through to apple_gpu_ops.auto_batch."""
    if not session_available():
        pytest.skip("encode-session unavailable")

    rows, cols, eps = 4, 16, 1e-5
    src = textwrap.dedent("""\
        def norm(x, gamma):
            return agpu.rmsnorm(x, gamma, rows=4, cols=16, eps=1e-5)
        """)

    @tessera.jit(target="apple_gpu", auto_batch=True,
                 source=src)
    def norm(x, gamma):
        return agpu.rmsnorm(x, gamma, rows=rows, cols=cols, eps=eps)

    rng = np.random.default_rng(0x317A6)
    X = rng.standard_normal((rows, cols), dtype=np.float32) * 0.1
    gamma = rng.standard_normal((cols,), dtype=np.float32)
    before = session_commit_count()
    out_dev = norm(X, gamma)
    after = session_commit_count()
    assert (after - before) == 1, (
        f"@jit + auto_batch should commit 1 cb, got delta="
        f"{after - before}")
    gpu = out_dev.download(np.float32, (rows, cols))
    out_dev.free()
    var = (X * X).mean(axis=-1, keepdims=True)
    expected = X / np.sqrt(var + eps) * gamma
    np.testing.assert_allclose(gpu, expected, rtol=1e-4, atol=1e-4)


def test_jit_auto_batch_full_attention_block_one_cb():
    """Headline: a full Llama-style attention block decorated with
    @jit(target='apple_gpu', auto_batch=True), using apple_gpu_ops.*
    inside, commits exactly 1 cb for the whole 8-op chain."""
    if not session_available():
        pytest.skip("encode-session unavailable")

    B, S, D = 1, 8, 16
    scale = 1.0 / np.sqrt(D)
    eps = 1e-5

    src = textwrap.dedent("""\
        def attention(x, gamma, wq, wk, wv, wo, theta):
            n = agpu.rmsnorm(x, gamma, rows=8, cols=16, eps=1e-5)
            q = agpu.bmm(n, wq, batch=1, M=8, N=16, K=16)
            k = agpu.bmm(n, wk, batch=1, M=8, N=16, K=16)
            v = agpu.bmm(n, wv, batch=1, M=8, N=16, K=16)
            q_r = agpu.rope(q, theta, M=8, K=16)
            k_r = agpu.rope(k, theta, M=8, K=16)
            a = agpu.flash_attn(q_r, k_r, v,
                                 B=1, Sq=8, Sk=8, D=16,
                                 scale=0.25, causal=False)
            return agpu.bmm(a, wo, batch=1, M=8, N=16, K=16)
        """)

    @tessera.jit(target="apple_gpu", auto_batch=True,
                 source=src)
    def attention(x, gamma, wq, wk, wv, wo, theta):
        n = agpu.rmsnorm(x, gamma, rows=B * S, cols=D, eps=eps)
        q = agpu.bmm(n, wq, batch=1, M=B * S, N=D, K=D)
        k = agpu.bmm(n, wk, batch=1, M=B * S, N=D, K=D)
        v = agpu.bmm(n, wv, batch=1, M=B * S, N=D, K=D)
        q_r = agpu.rope(q, theta, M=B * S, K=D)
        k_r = agpu.rope(k, theta, M=B * S, K=D)
        a = agpu.flash_attn(q_r, k_r, v,
                             B=B, Sq=S, Sk=S, D=D,
                             scale=scale, causal=False)
        return agpu.bmm(a, wo, batch=1, M=B * S, N=D, K=D)

    rng = np.random.default_rng(0x317A77)
    X = rng.standard_normal((B * S, D), dtype=np.float32) * 0.1
    gamma = rng.standard_normal((D,), dtype=np.float32)
    Wq = rng.standard_normal((1, D, D), dtype=np.float32) * 0.05
    Wk = rng.standard_normal((1, D, D), dtype=np.float32) * 0.05
    Wv = rng.standard_normal((1, D, D), dtype=np.float32) * 0.05
    Wo = rng.standard_normal((1, D, D), dtype=np.float32) * 0.05
    Theta = (np.arange(B * S * D, dtype=np.float32) * 0.001
             ).reshape(B * S, D)

    before = session_commit_count()
    out_dev = attention(X, gamma, Wq, Wk, Wv, Wo, Theta)
    after = session_commit_count()
    assert (after - before) == 1, (
        f"@jit + auto_batch on Llama attention block expected 1 cb, "
        f"got delta={after - before}")
    gpu = out_dev.download(np.float32, (1, B * S, D))
    out_dev.free()
    assert np.isfinite(gpu).all()


def test_jit_auto_batch_preserves_jitfn_attributes():
    """The decorated callable is still a JitFn (with .target,
    .compile_bundle, etc.) — auto_batch wraps the inner fn, not the
    JitFn wrapper. Users can introspect via the standard JIT
    diagnostics."""
    if not session_available():
        pytest.skip("encode-session unavailable")
    src = textwrap.dedent("""\
        def fn(x):
            return agpu.silu(x, n=4)
        """)

    @tessera.jit(target="apple_gpu", auto_batch=True,
                 source=src)
    def fn(x):
        return agpu.silu(x, n=4)

    # Standard JitFn attributes still accessible.
    assert fn.target == "apple_gpu"
    assert hasattr(fn, "graph_ir")
    assert hasattr(fn, "compile_bundle")
