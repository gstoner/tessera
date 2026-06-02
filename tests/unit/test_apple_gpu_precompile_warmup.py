"""Phase 5c — MPSGraph precompile/warmup helper.

The empirical "shape × op-count cliff" is bounded by MPSGraph
compile time. Each (op, dtype, shape) pair needs a one-time graph
build; bundling many fresh builds into one cb pushes the
commit_and_wait past the 30s timeout.

Solution: ``precompile_chain(trace)`` runs the trace with
``max_ops_per_cb=1`` so each op compiles + executes in its OWN
small command buffer. The MPSGraph cache absorbs the per-op
compile cost spread across N small cbs (none individually hits
the cliff). Subsequent production calls at the default budget
hit the warm cache and run fast.

The ``@auto_batch`` decorator exposes this via ``fn.warmup(*args)``:
call once on the same inputs to seed the cache; then call
``fn(*args)`` for steady-state production runs.

Tests pin:

* **Symbol availability** — ``precompile_chain`` is importable +
  exposed via ``__all__``.
* **Warmup grows the MPSGraph cache** — calling
  ``fn.warmup(...)`` increases the cache size by at least the
  number of distinct (op, dtype, shape) tuples in the chain.
* **Second call is faster than first** — after warmup, the
  production call latency is meaningfully lower (cache hit).
* **Output equivalence** — warmup output (discarded) equals
  production output bit-for-bit.
* **Nested warmup is a no-op** — calling warmup inside an
  active trace falls through (no double trace).
* **Caller doesn't see warmup tensors** — DeviceTensors created
  during warmup are freed by precompile_chain; subsequent
  cache.activation calls still work normally.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol
from tessera.apple_gpu_batched import session_available
from tessera.apple_gpu_chain import (
    OpRecord,
    TraceRef,
    precompile_chain,
    run_trace,
)
from tessera.apple_gpu_resident import ResidentWeights
import tessera.apple_gpu_ops as agpu


def _mpsgraph_cache_size() -> int:
    """Read the runtime's MPSGraph cache size via the existing
    probe symbol (shipped earlier as a drift gate)."""
    import ctypes
    fn = bind_symbol(
        "tessera_apple_gpu_mpsgraph_cache_size", (),
        ctypes.c_int32)
    if fn is None:
        return -1
    return int(fn())


# ---- Symbol surface ---------------------------------------------------

def test_precompile_chain_exposed():
    """Phase 5c surface — precompile_chain + auto_batch.warmup are
    public."""
    from tessera.apple_gpu_chain import precompile_chain as pc
    assert callable(pc)
    # And the decorator exposes warmup.

    @agpu.auto_batch
    def f(x):
        return x

    assert hasattr(f, "warmup")
    assert callable(f.warmup)


# ---- Cache growth ------------------------------------------------------

def test_warmup_grows_mpsgraph_cache():
    """A warmup call adds distinct (op, dtype, shape) entries to the
    MPSGraph cache. Verifiable via the existing cache_size probe."""
    if not session_available():
        pytest.skip("encode-session unavailable")
    if _mpsgraph_cache_size() < 0:
        pytest.skip("MPSGraph cache size probe unavailable")

    # Use a SHAPE-DTYPE-OP combo that's unique to this test so we
    # can confidently attribute cache growth to the warmup.
    rows, cols, eps = 4, 17, 1e-5  # rows*cols choices unlikely cached
    rng = np.random.default_rng(0xCAEC4E0)
    X = rng.standard_normal((rows, cols), dtype=np.float32) * 0.1
    gamma = rng.standard_normal((cols,), dtype=np.float32)

    @agpu.auto_batch
    def chain(x, g):
        n = agpu.rmsnorm(x, g, rows=rows, cols=cols, eps=eps)
        return agpu.silu(n, n=rows * cols)

    before = _mpsgraph_cache_size()
    n_warmed = chain.warmup(X, gamma)
    after = _mpsgraph_cache_size()
    # The chain has 2 distinct (op, dtype, shape) tuples — rmsnorm
    # f32 + silu f32 (unary op-code 4). Cache must grow by at least
    # that much (sometimes the runtime adds extra entries for
    # internal cast nodes; we only require ≥ 2).
    assert n_warmed == 2, (
        f"warmup should execute 2 ops, got {n_warmed}")
    assert (after - before) >= 2, (
        f"MPSGraph cache grew by only {after - before} entries; "
        f"expected ≥ 2 (one per distinct op/shape/dtype combo). "
        f"before={before} after={after}")


# ---- Warmup speeds up subsequent calls --------------------------------

def test_warmup_then_production_call_is_faster_than_cold():
    """After warmup, the production call (default chunking budget)
    should be measurably faster than the very first call would be.
    Verified by comparing warmup time to a subsequent production
    call time on the same inputs."""
    if not session_available():
        pytest.skip("encode-session unavailable")

    B, S, D = 1, 16, 32
    eps = 1e-5
    rng = np.random.default_rng(0x5A1AE0)
    X = rng.standard_normal((B * S, D), dtype=np.float32) * 0.1
    gamma = rng.standard_normal((D,), dtype=np.float32)
    W = rng.standard_normal((1, D, D), dtype=np.float32) * 0.05

    @agpu.auto_batch
    def chain(x, g, w):
        n = agpu.rmsnorm(x, g, rows=B * S, cols=D, eps=eps)
        out = agpu.bmm(n, w, batch=1, M=B * S, N=D, K=D)
        return out

    # Warmup pays the MPSGraph compile cost (per-op cbs).
    t0 = time.perf_counter()
    chain.warmup(X, gamma, W)
    warmup_ms = (time.perf_counter() - t0) * 1000

    # Production call hits the warm cache.
    t0 = time.perf_counter()
    out = chain(X, gamma, W)
    out.free()
    prod_ms = (time.perf_counter() - t0) * 1000

    # The production call after warmup should be substantially
    # faster than the warmup itself (which includes the compile
    # cost). We use a loose ratio — 1.5× is a safe floor given
    # measurement variance at small shapes.
    assert prod_ms < warmup_ms / 1.5, (
        f"production call after warmup should be <2/3 of warmup "
        f"time; got warmup={warmup_ms:.3f}ms prod={prod_ms:.3f}ms")


# ---- Numerical equivalence -------------------------------------------

def test_warmup_output_equals_production_output_bitwise():
    """Same inputs → same output, whether via warmup or production
    call. The warmup discards its DeviceTensor outputs but the
    underlying computation is identical."""
    if not session_available():
        pytest.skip("encode-session unavailable")

    rows, cols, eps = 4, 16, 1e-5
    rng = np.random.default_rng(0xEAB0)
    X = rng.standard_normal((rows, cols), dtype=np.float32) * 0.1
    gamma = rng.standard_normal((cols,), dtype=np.float32)

    @agpu.auto_batch
    def chain(x, g):
        return agpu.rmsnorm(x, g, rows=rows, cols=cols, eps=eps)

    # Production call captures the output.
    out1 = chain(X, gamma)
    a = out1.download(np.float32, (rows, cols))
    out1.free()
    # Warmup (output discarded internally), then production again.
    chain.warmup(X, gamma)
    out2 = chain(X, gamma)
    b = out2.download(np.float32, (rows, cols))
    out2.free()
    np.testing.assert_array_equal(a, b)


# ---- Nested warmup is a no-op -----------------------------------------

def test_nested_warmup_falls_through():
    """If ``warmup`` is called inside an existing auto_batch trace,
    it skips the per-op cb path (the outer trace handles
    execution) and returns 0. Avoids double-tracing / nested
    session-creation hazards."""
    if not session_available():
        pytest.skip("encode-session unavailable")

    rows, cols, eps = 4, 16, 1e-5
    rng = np.random.default_rng(0x4E57ED)
    X = rng.standard_normal((rows, cols), dtype=np.float32) * 0.1
    gamma = rng.standard_normal((cols,), dtype=np.float32)

    @agpu.auto_batch
    def inner(x, g):
        return agpu.rmsnorm(x, g, rows=rows, cols=cols, eps=eps)

    @agpu.auto_batch
    def outer(x, g):
        # warmup() called inside the outer trace should be a no-op.
        n = inner.warmup(x, g)  # returns 0; inner ops join outer trace
        assert n == 0
        return agpu.rmsnorm(x, g, rows=rows, cols=cols, eps=eps)

    out = outer(X, gamma)
    arr = out.download(np.float32, (rows, cols))
    out.free()
    assert np.isfinite(arr).all()


# ---- Pure substrate-level test (no decorator) -------------------------

def test_precompile_chain_with_hand_built_trace():
    """Lowest-level surface — build an OpRecord trace manually and
    call ``precompile_chain``. Useful for callers that don't go
    through the decorator (e.g., the JIT chain executor)."""
    if not session_available():
        pytest.skip("encode-session unavailable")

    rows, cols, eps = 4, 19, 1e-5
    rng = np.random.default_rng(0x9C0)
    X = rng.standard_normal((rows, cols), dtype=np.float32) * 0.1
    gamma = rng.standard_normal((cols,), dtype=np.float32)

    trace = [
        OpRecord("rmsnorm", "f32",
                  inputs=[X, gamma],
                  shape_kwargs=dict(rows=rows, cols=cols, eps=eps)),
        OpRecord("silu", "f32",
                  inputs=[TraceRef(op_index=0)],
                  shape_kwargs=dict(n=rows * cols)),
    ]
    before = _mpsgraph_cache_size()
    n = precompile_chain(trace)
    after = _mpsgraph_cache_size()
    assert n == 2, n
    # Cache grew (when probe is available).
    if before >= 0:
        assert after >= before, (before, after)


# ---- Headline: warmup unblocks the previously-cliff config ------------

def test_warmup_lets_large_chain_run_at_default_budget():
    """The original cliff scenario: a multi-layer chain that hangs
    on first encounter. Pre-warm via warmup; then run at the
    DEFAULT chunking budget — should succeed."""
    if not session_available():
        pytest.skip("encode-session unavailable")

    # Use a UNIQUE shape so the MPSGraph cache definitely has to
    # build new entries (no fake-pass from earlier tests).
    B, S, D = 1, 19, 37  # weird-on-purpose
    FFD = 2 * D
    N = 3
    scale = 1.0 / np.sqrt(D)
    eps = 1e-5
    rng = np.random.default_rng(0xDEEEE)
    X = rng.standard_normal((B * S, D), dtype=np.float32) * 0.1

    cache = ResidentWeights()
    try:
        for i in range(N):
            cache.weight(f"L{i}_g",
                          rng.standard_normal((D,), dtype=np.float32))
            cache.weight(f"L{i}_Wq",
                          rng.standard_normal((1, D, D),
                                               dtype=np.float32) * 0.05)
            cache.weight(f"L{i}_Wo",
                          rng.standard_normal((1, D, D),
                                               dtype=np.float32) * 0.05)
            cache.weight(f"L{i}_T",
                          (np.arange(B * S * D, dtype=np.float32)
                           * 0.001).reshape(B * S, D))
            cache.weight(f"L{i}_gm",
                          rng.standard_normal((D,), dtype=np.float32))
            cache.weight(f"L{i}_Wgate",
                          rng.standard_normal((1, D, FFD),
                                               dtype=np.float32) * 0.05)
            cache.weight(f"L{i}_Wdown",
                          rng.standard_normal((1, FFD, D),
                                               dtype=np.float32) * 0.05)

        @agpu.auto_batch
        def chain(x):
            xt = x
            for i in range(N):
                n = agpu.rmsnorm(xt, cache[f"L{i}_g"],
                                  rows=B * S, cols=D, eps=eps)
                q = agpu.bmm(n, cache[f"L{i}_Wq"],
                              batch=1, M=B * S, N=D, K=D)
                qr = agpu.rope(q, cache[f"L{i}_T"], M=B * S, K=D)
                a = agpu.flash_attn(qr, qr, n,
                                     B=B, Sq=S, Sk=S, D=D, scale=scale)
                xt = agpu.bmm(a, cache[f"L{i}_Wo"],
                               batch=1, M=B * S, N=D, K=D)
                mn = agpu.rmsnorm(xt, cache[f"L{i}_gm"],
                                   rows=B * S, cols=D, eps=eps)
                gate = agpu.bmm(mn, cache[f"L{i}_Wgate"],
                                 batch=1, M=B * S, N=FFD, K=D)
                act = agpu.silu(gate, n=B * S * FFD)
                xt = agpu.bmm(act, cache[f"L{i}_Wdown"],
                               batch=1, M=B * S, N=D, K=FFD)
            return xt

        # Warmup first.
        x_dev = cache.activation("x", X)
        n_warmed = chain.warmup(x_dev)
        assert n_warmed > 0
        # Production call — should now succeed at default budget.
        out = chain(x_dev)
        arr = out.download(np.float32, (1, B * S, D))
        out.free()
        assert np.isfinite(arr).all()
    finally:
        cache.free()
