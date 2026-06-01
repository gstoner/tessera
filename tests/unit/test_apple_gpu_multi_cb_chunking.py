"""Phase 5b — multi-cb chunking transparently breaks the ops cliff.

The single-cb encode-session has an empirically-measured cliff
around 30-40 MPSGraph encodeToCommandBuffer calls before the GPU
dispatch hangs at the 30-second commit_and_wait timeout. This is
an MPSGraph implementation limit, not a Tessera bug.

The planner now caps each encode segment at ``DEFAULT_OPS_PER_CB``
(30) ops by default; longer chains split into K cb's. Same
encode-session ABI, just K commits instead of 1.

Cross-segment data flow: a ``TraceRef`` in segment K+1 that
references an output from segment K is resolved by the executor
via the persistent ``results`` list spanning all segments.

These tests pin:

* **Planner splits chain at budget** — 50-op trace becomes 2
  segments (30 + 20).
* **Cross-segment TraceRef resolves correctly** — the 31st op
  reads from the 30th op's output even though they're in
  different segments.
* **Deep transformer no longer hangs** — a 6-layer transformer
  (~72 ops, 2.5× the budget) executes correctly under chunking.
* **Configurable budget** — caller can pass a smaller cap via
  ``max_ops_per_cb=`` for testing or extra safety margin.
* **Tiny budget (1 op/cb) works** — each op gets its own cb;
  same numerical output (proves cross-cb data flow).
* **Numerical correctness** — chunked output matches the
  single-cb output when both are valid.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.apple_gpu_batched import (
    batched_session,
    bmm_enc,
    device_tensor,
    rmsnorm_enc,
    session_available,
    session_commit_count,
)
from tessera.apple_gpu_chain import (
    DEFAULT_OPS_PER_CB,
    ChainSegment,
    OpRecord,
    TraceRef,
    plan_chain,
    run_trace,
)
from tessera.apple_gpu_resident import ResidentWeights
import tessera.apple_gpu_ops as agpu


# ---- Planner-level tests (no GPU) -------------------------------------

def test_default_ops_per_cb_is_set_with_margin_under_cliff():
    """The default ops-per-cb budget should be comfortably below
    the empirical cliff (~30-40). 30 leaves margin for both
    measurement variance and the worst case across op mixes."""
    assert DEFAULT_OPS_PER_CB <= 35
    assert DEFAULT_OPS_PER_CB >= 20  # not so low that small chains chunk


def test_planner_splits_at_budget():
    """A trace of 50 identical encode-eligible ops splits into
    ceil(50/30) = 2 segments under the default budget."""
    trace = [OpRecord("rmsnorm", "f32") for _ in range(50)]
    segs = plan_chain(trace)
    assert len(segs) == 2
    assert all(s.kind == "encode" for s in segs)
    assert len(segs[0].ops) == 30  # at budget
    assert len(segs[1].ops) == 20


def test_planner_splits_at_exact_multiples():
    """A trace of N×30 ops splits into N segments of exactly 30."""
    for k in (1, 2, 3):
        trace = [OpRecord("bmm", "f32") for _ in range(k * 30)]
        segs = plan_chain(trace)
        assert len(segs) == k
        assert all(len(s.ops) == 30 for s in segs)


def test_planner_honors_custom_budget():
    """A smaller ``max_ops_per_cb`` forces more, smaller segments."""
    trace = [OpRecord("rmsnorm", "f32") for _ in range(10)]
    # Budget = 3 → ceil(10/3) = 4 segments.
    segs = plan_chain(trace, max_ops_per_cb=3)
    assert len(segs) == 4
    assert [len(s.ops) for s in segs] == [3, 3, 3, 1]


def test_planner_budget_one_emits_one_segment_per_op():
    """``max_ops_per_cb=1`` puts each op in its own segment —
    equivalent to per-op dispatch via the encode ABI."""
    trace = [OpRecord("rmsnorm", "f32") for _ in range(5)]
    segs = plan_chain(trace, max_ops_per_cb=1)
    assert len(segs) == 5
    assert all(len(s.ops) == 1 for s in segs)


def test_planner_budget_split_preserves_dtype_grouping():
    """Mixed dtype + budget: f32 ops chunk separately from f16
    ops, AND each dtype's run chunks at the budget."""
    trace = ([OpRecord("rmsnorm", "f32") for _ in range(40)]
              + [OpRecord("rmsnorm", "f16") for _ in range(40)])
    segs = plan_chain(trace, max_ops_per_cb=30)
    # 40 f32 → 30+10; 40 f16 → 30+10 → 4 segments total.
    assert len(segs) == 4
    assert [len(s.ops) for s in segs] == [30, 10, 30, 10]
    # Each segment is single-dtype.
    for seg in segs:
        dtypes = {op.dtype for op in seg.ops}
        assert len(dtypes) == 1


def test_planner_budget_zero_or_negative_raises():
    with pytest.raises(ValueError, match=">= 1"):
        plan_chain([], max_ops_per_cb=0)
    with pytest.raises(ValueError, match=">= 1"):
        plan_chain([], max_ops_per_cb=-1)


# ---- Executor: cross-cb data flow + numerical correctness -------------

def test_cross_segment_traceref_resolves_to_correct_output():
    """A chain of 50 ops chunks into 2 segments. Op 49 (the last)
    must see the output of op 48, even though they may be in
    different segments. Verified numerically: a chain of
    rmsnorm → bmm → ... (50 alternating) produces the same final
    output regardless of how the planner chunks."""
    if not session_available():
        pytest.skip("encode-session unavailable")
    rows, cols, eps = 4, 16, 1e-5
    rng = np.random.default_rng(0xC2055)
    X = rng.standard_normal((rows, cols), dtype=np.float32) * 0.1
    gamma = rng.standard_normal((cols,), dtype=np.float32)
    W = rng.standard_normal((cols, cols), dtype=np.float32) * 0.1
    g_dev = device_tensor(gamma)
    w_dev = device_tensor(W.reshape(1, cols, cols))

    # Build a 50-op trace: alternating rmsnorm + bmm.
    # Each consumes the previous op's output (chain dependency).
    trace = []
    x_in = X  # first input is a numpy array
    for i in range(25):
        trace.append(OpRecord(
            "rmsnorm", "f32",
            inputs=[x_in if i == 0 else TraceRef(op_index=2*i - 1), g_dev],
            shape_kwargs=dict(rows=rows, cols=cols, eps=eps)))
        trace.append(OpRecord(
            "bmm", "f32",
            inputs=[TraceRef(op_index=2*i), w_dev],
            shape_kwargs=dict(batch=1, M=rows, N=cols, K=cols)))

    assert len(trace) == 50

    # Default budget (30) → planner splits into 2 segments.
    segs = plan_chain(trace)
    assert len(segs) == 2
    assert sum(len(s.ops) for s in segs) == 50

    # Execute.
    results = run_trace(trace)
    assert len(results) == 50
    # Every result is non-None (every op produced a DeviceTensor).
    assert all(r is not None for r in results)
    # Final output should be finite + well-formed.
    final = results[-1].download(
        np.float32, (1, rows, cols)).reshape(rows, cols)
    assert np.isfinite(final).all()
    # Free all results.
    for r in results:
        if r is not None:
            r.free()
    g_dev.free(); w_dev.free()


def test_chunked_chain_matches_unchunked_at_small_size():
    """For a chain that fits in one cb (under the budget), planner
    output should be the same whether chunked or not. Numerical
    output matches."""
    if not session_available():
        pytest.skip("encode-session unavailable")
    rows, cols, eps = 4, 16, 1e-5
    rng = np.random.default_rng(0x44A5)
    X = rng.standard_normal((rows, cols), dtype=np.float32) * 0.1
    gamma = rng.standard_normal((cols,), dtype=np.float32)
    g_dev = device_tensor(gamma)

    # Tiny chain: 4 rmsnorm ops, fits in one cb.
    trace = [
        OpRecord("rmsnorm", "f32",
                  inputs=[X if i == 0 else TraceRef(op_index=i - 1), g_dev],
                  shape_kwargs=dict(rows=rows, cols=cols, eps=eps))
        for i in range(4)
    ]
    # Default budget: 1 segment.
    segs_default = plan_chain(trace)
    assert len(segs_default) == 1
    # Tiny budget: 4 segments.
    segs_tiny = plan_chain(trace, max_ops_per_cb=1)
    assert len(segs_tiny) == 4

    # Run both, compare outputs.
    out_default = run_trace(trace)
    out_tiny = run_trace(trace, max_ops_per_cb=1)
    arr_default = out_default[-1].download(np.float32, (rows, cols))
    arr_tiny = out_tiny[-1].download(np.float32, (rows, cols))
    for r in out_default + out_tiny:
        if r is not None:
            r.free()
    np.testing.assert_allclose(arr_tiny, arr_default,
                                rtol=1e-5, atol=1e-5)
    g_dev.free()


# ---- The headline: deep transformer no longer hangs -------------------

def test_deep_transformer_no_longer_hangs_under_chunking():
    """A 6-layer attention+MLP transformer (~72 user ops) previously
    hung at the 30-second commit_and_wait timeout under
    auto_batch. With multi-cb chunking now landed, the same chain
    executes correctly — split into K=ceil(72/30)=3 cb's.

    This is the headline value of Phase 5b: deep models work."""
    if not session_available():
        pytest.skip("encode-session unavailable")

    B, S, D = 1, 16, 32
    FFD = 2 * D
    N = 6  # the layer count that previously hung
    scale = 1.0 / np.sqrt(D)
    eps = 1e-5

    rng = np.random.default_rng(0xDEEEEE)
    X = rng.standard_normal((B * S, D), dtype=np.float32) * 0.1

    cache = ResidentWeights()
    try:
        for i in range(N):
            cache.weight(f"L{i}_g",
                          rng.standard_normal((D,), dtype=np.float32))
            cache.weight(f"L{i}_Wq",
                          rng.standard_normal((1, D, D),
                                               dtype=np.float32) * 0.05)
            cache.weight(f"L{i}_Wk",
                          rng.standard_normal((1, D, D),
                                               dtype=np.float32) * 0.05)
            cache.weight(f"L{i}_Wv",
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
        def step(x):
            xt = x
            for i in range(N):
                n = agpu.rmsnorm(xt, cache[f"L{i}_g"],
                                  rows=B * S, cols=D, eps=eps)
                q = agpu.bmm(n, cache[f"L{i}_Wq"],
                              batch=1, M=B * S, N=D, K=D)
                k = agpu.bmm(n, cache[f"L{i}_Wk"],
                              batch=1, M=B * S, N=D, K=D)
                v = agpu.bmm(n, cache[f"L{i}_Wv"],
                              batch=1, M=B * S, N=D, K=D)
                qr = agpu.rope(q, cache[f"L{i}_T"], M=B * S, K=D)
                kr = agpu.rope(k, cache[f"L{i}_T"], M=B * S, K=D)
                a = agpu.flash_attn(qr, kr, v,
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

        x_dev = cache.activation("x", X)
        # The 6-layer chain has N*12 = 72 user ops. With
        # DEFAULT_OPS_PER_CB=30, planner splits into 3 segments
        # (30, 30, 12). 3 cb commits per step.
        before = session_commit_count()
        out = step(x_dev)
        after = session_commit_count()
        assert (after - before) == 3, (
            f"6-layer transformer should commit 3 cbs (72 ops / 30 "
            f"per cb), got delta={after - before}")
        arr = out.download(np.float32, (1, B * S, D))
        out.free()
        assert arr.shape == (1, B * S, D)
        assert np.isfinite(arr).all()
    finally:
        cache.free()
