"""P1 (2026-06-02) — canonical Apple one-command-buffer route via @jit.

The audit's APPLE_AUDIT "Next Work" item 4 — *finish the canonical
``@jit(target="apple_gpu", auto_batch=True)`` one-command-buffer route* —
is exercised end-to-end here. The contract:

* A user writes a normal decode function with the **canonical
  ``tessera.ops.*`` surface** (not the lower-level ``apple_gpu_ops.*``).
* ``@jit(target="apple_gpu", auto_batch=True)`` trace-captures the whole
  body and runs it as **one command buffer per encode segment** — the
  ``tessera.ops.*`` interception shim (installed globally at import)
  forwards to the encode session while a trace is active.
* ``max_ops_per_cb`` threads the chunking budget (Glass-jaw #7) through
  the decorator so a deep chain splits into K cbs transparently.

These tests are module-level (so ``@jit`` can inspect their source) and
pin: one-cb commit count, the chunking knob, numerical correctness vs a
numpy reference, and the misuse guards.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.apple_gpu_batched import session_available, session_commit_count

_ROWS, _COLS, _EPS = 4, 16, 1e-5


# ── Canonical decode functions (tessera.ops.* surface, module-level) ──
#
# Shape kwargs are literals (not module globals): a real decode function
# naturally passes shapes as literals or args, so this is the canonical
# style; _ROWS/_COLS stay in the numpy reference below. (P3 2026-06-09 —
# under auto_batch the AST Graph IR is no longer emitted at all: the tracer
# runs the body directly, so there's no Graph IR operand resolver in play.
# Auto-detection + emission-skip are covered by
# test_apple_gpu_jit_auto_batch_autodetect.py.)

@ts.jit(target="apple_gpu", auto_batch=True)
def _decode_default(x, g):
    a = ts.ops.rmsnorm(x, g, rows=4, cols=16, eps=1e-5)
    b = ts.ops.silu(a, n=64)
    c = ts.ops.rmsnorm(b, g, rows=4, cols=16, eps=1e-5)
    return ts.ops.silu(c, n=64)  # 4 encode-eligible ops


@ts.jit(target="apple_gpu", auto_batch=True, max_ops_per_cb=1)
def _decode_chunked(x, g):
    a = ts.ops.rmsnorm(x, g, rows=4, cols=16, eps=1e-5)
    b = ts.ops.silu(a, n=64)
    c = ts.ops.rmsnorm(b, g, rows=4, cols=16, eps=1e-5)
    return ts.ops.silu(c, n=64)  # 4 ops, 1 per cb → 4 cbs


def _np_reference(x, g):
    def rmsnorm(v):
        return v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + _EPS) * g

    def silu(v):
        return v / (1.0 + np.exp(-v))
    return silu(rmsnorm(silu(rmsnorm(x))))


def _inputs():
    rng = np.random.default_rng(0xDEC0DE)
    x = rng.standard_normal((_ROWS, _COLS), dtype=np.float32) * 0.1
    g = rng.standard_normal((_COLS,), dtype=np.float32)
    return x, g


# ── One-command-buffer contract ───────────────────────────────────────

def test_canonical_decode_runs_on_one_cb():
    if not session_available():
        pytest.skip("encode-session unavailable")
    x, g = _inputs()
    before = session_commit_count()
    out = _decode_default(x, g)
    after = session_commit_count()
    # 4 encode ops under the default budget → a single command buffer.
    assert (after - before) == 1, f"expected 1 cb, got {after - before}"
    arr = out.download(np.float32, (_ROWS, _COLS))
    assert np.isfinite(arr).all()


def test_max_ops_per_cb_threads_through_jit():
    if not session_available():
        pytest.skip("encode-session unavailable")
    x, g = _inputs()
    before = session_commit_count()
    out = _decode_chunked(x, g)
    after = session_commit_count()
    # 4 ops at budget 1 → 4 separate command buffers.
    assert (after - before) == 4, f"expected 4 cbs, got {after - before}"
    out.download(np.float32, (_ROWS, _COLS))


# ── Numerical correctness ─────────────────────────────────────────────

def test_canonical_decode_matches_numpy():
    if not session_available():
        pytest.skip("encode-session unavailable")
    x, g = _inputs()
    got = _decode_default(x, g).download(np.float32, (_ROWS, _COLS))
    ref = _np_reference(x, g)
    # fp32 GPU rmsnorm/silu vs numpy reference.
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_chunked_and_default_agree_numerically():
    """Chunking is a cb-segmentation detail — the numerical result must
    be identical whether the chain runs in one cb or four."""
    if not session_available():
        pytest.skip("encode-session unavailable")
    x, g = _inputs()
    a = _decode_default(x, g).download(np.float32, (_ROWS, _COLS))
    b = _decode_chunked(x, g).download(np.float32, (_ROWS, _COLS))
    np.testing.assert_array_equal(a, b)


# ── Misuse guards (no GPU needed) ─────────────────────────────────────

def test_max_ops_per_cb_without_auto_batch_raises():
    from tessera.compiler.jit import TesseraJitError
    with pytest.raises(TesseraJitError, match="only meaningful with"):
        @ts.jit(target="apple_gpu", max_ops_per_cb=8)
        def _bad(x):
            return x


def test_auto_batch_on_non_apple_target_raises():
    from tessera.compiler.jit import TesseraJitError
    with pytest.raises(TesseraJitError, match="only supports"):
        @ts.jit(target="x86", auto_batch=True)
        def _bad(x):
            return x
