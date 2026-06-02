"""Glass-jaw #7 + #9 (2026-06-01) — ``max_ops_per_cb`` knob.

Threads the chunking budget through the whole warm-up / production
surface so callers can tune the ops-per-command-buffer cap:

* ``precompile_chain(trace, max_ops_per_cb=W)`` — substrate.
* ``@auto_batch(max_ops_per_cb=N)`` — production-run budget on the
  decorator (was hardwired to ``run_trace``'s ``DEFAULT_OPS_PER_CB``).
* ``wrapped.warmup(..., max_ops_per_cb=W)`` — per-call warm budget,
  falling back to the decorator's budget, then to ``1``.

These tests pin the WIRING (signatures + budget precedence + single-cb
commit counts), not GPU numerics — they run on the substrate and the
commit-count probe, so they exercise real chunking behavior.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from tessera.apple_gpu_batched import (
    session_available,
    session_commit_count,
)
from tessera.apple_gpu_chain import (
    DEFAULT_OPS_PER_CB,
    OpRecord,
    TraceRef,
    precompile_chain,
)
import tessera.apple_gpu_ops as agpu


# ---- Signature wiring (no GPU needed) ---------------------------------

def test_precompile_chain_accepts_max_ops_per_cb():
    sig = inspect.signature(precompile_chain)
    assert "max_ops_per_cb" in sig.parameters
    # Default is 1 (per-op cbs, cliff-safe).
    assert sig.parameters["max_ops_per_cb"].default == 1


def test_auto_batch_supports_bare_and_parametrized_forms():
    # Bare form returns a wrapped callable with .warmup.
    @agpu.auto_batch
    def f(x):
        return x
    assert callable(f) and hasattr(f, "warmup")

    # Parametrized form returns a decorator that yields the same shape.
    @agpu.auto_batch(max_ops_per_cb=8)
    def g(x):
        return x
    assert callable(g) and hasattr(g, "warmup")


# ---- Budget precedence (substrate, GPU-gated) -------------------------

def _decoder_trace(N, layers, D, eps=1e-5):
    """Build a trace of `layers` rmsnorm+silu pairs over rows=N, cols=D."""
    rng = np.random.default_rng(0xC0DEC)
    X = rng.standard_normal((N, D), dtype=np.float32) * 0.1
    g = rng.standard_normal((D,), dtype=np.float32)
    trace = []
    prev = X
    for i in range(layers):
        trace.append(OpRecord("rmsnorm", "f32", inputs=[prev, g],
                               shape_kwargs=dict(rows=N, cols=D, eps=eps)))
        trace.append(OpRecord("silu", "f32",
                              inputs=[TraceRef(len(trace) - 1)],
                              shape_kwargs=dict(n=N * D)))
        prev = TraceRef(len(trace) - 1)
    return trace


def test_precompile_chain_at_budget_1_commits_per_op():
    if not session_available():
        pytest.skip("encode-session unavailable")
    trace = _decoder_trace(N=2, layers=3, D=16)  # 6 ops
    before = session_commit_count()
    n = precompile_chain(trace, max_ops_per_cb=1)
    after = session_commit_count()
    assert n == 6
    # 6 ops at budget 1 → 6 separate cbs.
    assert (after - before) == 6, (after - before)


def test_precompile_chain_at_large_budget_commits_one_cb():
    if not session_available():
        pytest.skip("encode-session unavailable")
    trace = _decoder_trace(N=2, layers=3, D=16)  # 6 ops
    before = session_commit_count()
    n = precompile_chain(trace, max_ops_per_cb=64)
    after = session_commit_count()
    assert n == 6
    # 6 ops, budget 64 → single cb.
    assert (after - before) == 1, (after - before)


def test_warmup_explicit_budget_overrides_decorator():
    """``warmup(max_ops_per_cb=W)`` wins over the decorator budget."""
    if not session_available():
        pytest.skip("encode-session unavailable")

    N, D, eps = 2, 16, 1e-5
    rng = np.random.default_rng(0x5A1)
    X = rng.standard_normal((N, D), dtype=np.float32) * 0.1
    g = rng.standard_normal((D,), dtype=np.float32)

    @agpu.auto_batch(max_ops_per_cb=64)
    def step(x, gamma):
        a = agpu.rmsnorm(x, gamma, rows=N, cols=D, eps=eps)
        b = agpu.silu(a, n=N * D)
        c = agpu.rmsnorm(b, gamma, rows=N, cols=D, eps=eps)
        return c  # 3 ops

    # Explicit warm budget 1 → 3 cbs (overrides the decorator's 64).
    before = session_commit_count()
    n = step.warmup(X, g, max_ops_per_cb=1)
    after = session_commit_count()
    assert n == 3
    assert (after - before) == 3, (after - before)


def test_warmup_defaults_to_decorator_budget():
    """With no explicit warmup budget, warm at the decorator's
    ``max_ops_per_cb`` (so warm-up exercises prod chunking)."""
    if not session_available():
        pytest.skip("encode-session unavailable")

    N, D, eps = 2, 16, 1e-5
    rng = np.random.default_rng(0x5A2)
    X = rng.standard_normal((N, D), dtype=np.float32) * 0.1
    g = rng.standard_normal((D,), dtype=np.float32)

    @agpu.auto_batch(max_ops_per_cb=64)
    def step(x, gamma):
        a = agpu.rmsnorm(x, gamma, rows=N, cols=D, eps=eps)
        b = agpu.silu(a, n=N * D)
        c = agpu.rmsnorm(b, gamma, rows=N, cols=D, eps=eps)
        return c  # 3 ops

    before = session_commit_count()
    n = step.warmup(X, g)  # no explicit budget → decorator's 64
    after = session_commit_count()
    assert n == 3
    assert (after - before) == 1, (after - before)  # single cb at budget 64


def test_bare_auto_batch_warmup_defaults_to_per_op():
    """The bare ``@auto_batch`` form warms at budget 1 (per-op cbs) —
    the cliff-safe default."""
    if not session_available():
        pytest.skip("encode-session unavailable")

    N, D, eps = 2, 16, 1e-5
    rng = np.random.default_rng(0x5A3)
    X = rng.standard_normal((N, D), dtype=np.float32) * 0.1
    g = rng.standard_normal((D,), dtype=np.float32)

    @agpu.auto_batch
    def step(x, gamma):
        a = agpu.rmsnorm(x, gamma, rows=N, cols=D, eps=eps)
        b = agpu.silu(a, n=N * D)
        c = agpu.rmsnorm(b, gamma, rows=N, cols=D, eps=eps)
        return c  # 3 ops

    before = session_commit_count()
    n = step.warmup(X, g)  # bare form → budget 1
    after = session_commit_count()
    assert n == 3
    assert (after - before) == 3, (after - before)


def test_parametrized_production_call_respects_budget():
    """A parametrized ``@auto_batch(max_ops_per_cb=2)`` production
    call chunks at 2 ops/cb."""
    if not session_available():
        pytest.skip("encode-session unavailable")

    N, D, eps = 2, 16, 1e-5
    rng = np.random.default_rng(0x5A4)
    X = rng.standard_normal((N, D), dtype=np.float32) * 0.1
    g = rng.standard_normal((D,), dtype=np.float32)

    @agpu.auto_batch(max_ops_per_cb=2)
    def step(x, gamma):
        a = agpu.rmsnorm(x, gamma, rows=N, cols=D, eps=eps)
        b = agpu.silu(a, n=N * D)
        c = agpu.rmsnorm(b, gamma, rows=N, cols=D, eps=eps)
        d = agpu.silu(c, n=N * D)
        return d  # 4 ops

    before = session_commit_count()
    out = step(X, g)
    after = session_commit_count()
    out.free()
    # 4 ops at budget 2 → 2 cbs.
    assert (after - before) == 2, (after - before)


def test_default_budget_is_default_ops_per_cb():
    """Bare ``@auto_batch`` production runs at DEFAULT_OPS_PER_CB —
    a small chain commits one cb."""
    if not session_available():
        pytest.skip("encode-session unavailable")
    assert DEFAULT_OPS_PER_CB >= 2  # sanity

    N, D, eps = 2, 16, 1e-5
    rng = np.random.default_rng(0x5A5)
    X = rng.standard_normal((N, D), dtype=np.float32) * 0.1
    g = rng.standard_normal((D,), dtype=np.float32)

    @agpu.auto_batch
    def step(x, gamma):
        a = agpu.rmsnorm(x, gamma, rows=N, cols=D, eps=eps)
        return agpu.silu(a, n=N * D)  # 2 ops

    before = session_commit_count()
    out = step(X, g)
    after = session_commit_count()
    out.free()
    assert (after - before) == 1, (after - before)
