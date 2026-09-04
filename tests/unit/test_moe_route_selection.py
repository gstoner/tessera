"""The MoE SwiGLU composite selects its implementation in ONE place, explicitly.

It used to be three fall-through blocks each ending in `except Exception: pass`,
so which of three implementations ran was not predictable from the inputs and a
failure between them was invisible. The ordering also silently preferred the
slow ones: any uniform f16/bf16 input took `lowp`, and the route ledger could
select `single_fused`.

Measured on this M1 Max (best of 5 after warm-up, milliseconds):

    (T,K,H,N,E)              dtype   single_fused     lowp   composed
    (64,128,256,128,4)       f32           13.23        --       1.27
    (64,128,256,128,4)       f16              --     15.04       1.26
    (256,256,256,256,8)      f32           32.00        --       1.89
    (256,256,256,256,8)      f16              --     35.66       1.84
    (1024,512,512,512,8)     f16              --   1571.06      26.31

`composed` wins every case and the gap widens with size. It is also the more
accurate: 6.3e-8 relative error against an fp32 reference where `lowp` is
2.6e-4, because `composed` accumulates in fp32. So the low-precision default
was both ~12-60x slower and ~4000x less accurate than the path it displaced.

These tests pin the selection, not the speed -- a timing assertion here would
be a flaky perf gate on shared CI. The measurement lives in the comment and in
the plan entry, where it can be re-run deliberately.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as R

G = np.array([2, 2])


def _ops(dtype):
    rng = np.random.default_rng(3)
    return (rng.standard_normal((4, 8)).astype(dtype),
            rng.standard_normal((2, 8, 10)).astype(dtype),
            rng.standard_normal((2, 10, 6)).astype(dtype))


@pytest.fixture(autouse=True)
def _no_opt_ins(monkeypatch):
    monkeypatch.delenv("TESSERA_APPLE_MOE_FUSED", raising=False)
    monkeypatch.delenv("TESSERA_APPLE_MOE_LOWP", raising=False)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_composed_is_the_default_for_every_dtype(dtype):
    """The regression that matters: f16/bf16 must no longer opt itself into the
    slowest, least accurate implementation just by being low precision.

    Note what the defect actually was. `lowp` sat in a block AHEAD of the
    arbiter and preempted it unconditionally, so the route ledger -- which
    answers `composed` on the committed ledger, i.e. correctly -- never got to
    decide for the common low-precision shape. The fix restores the arbiter
    rather than removing it.
    """
    x, wg, wd = _ops(dtype)
    route, reason = R._apple_moe_select_route(x, wg, wd, G, {}, np)
    assert route == "composed", reason


def test_quantized_blocks_force_composed_even_under_an_opt_in(monkeypatch):
    """Per-GEMM quant semantics are the one thing the single-kernel paths
    cannot express, so `quant` outranks the escape hatches rather than racing
    them."""
    monkeypatch.setenv("TESSERA_APPLE_MOE_FUSED", "1")
    x, wg, wd = _ops(np.float32)
    route, reason = R._apple_moe_select_route(x, wg, wd, G, {"quant": "int8"}, np)
    assert route == "composed" and "quant" in reason


def test_the_alternatives_are_reachable_but_only_by_name(monkeypatch):
    """Both stay available for the rewrites they are waiting on; neither is
    implicit any more."""
    monkeypatch.setenv("TESSERA_APPLE_MOE_FUSED", "1")
    x32, wg32, wd32 = _ops(np.float32)
    assert R._apple_moe_select_route(x32, wg32, wd32, G, {}, np)[0] == "single_fused"

    monkeypatch.delenv("TESSERA_APPLE_MOE_FUSED")
    monkeypatch.setenv("TESSERA_APPLE_MOE_LOWP", "1")
    x16, wg16, wd16 = _ops(np.float16)
    assert R._apple_moe_select_route(x16, wg16, wd16, G, {}, np)[0] == "lowp"


def test_an_opt_in_the_shape_cannot_honour_says_so(monkeypatch):
    """Asking for `lowp` with f32 operands returns `composed` WITH a reason
    naming the mismatch, rather than silently ignoring the request."""
    monkeypatch.setenv("TESSERA_APPLE_MOE_LOWP", "1")
    x, wg, wd = _ops(np.float32)
    route, reason = R._apple_moe_select_route(x, wg, wd, G, {}, np)
    assert route == "composed"
    assert "f16" in reason or "bf16" in reason, reason


def test_every_reason_is_non_empty():
    """A selection without a reason is the state this replaced."""
    for dtype in (np.float32, np.float16):
        x, wg, wd = _ops(dtype)
        for kw in ({}, {"quant": "int8"}):
            _, reason = R._apple_moe_select_route(x, wg, wd, G, kw, np)
            assert reason and reason.strip()


def test_choosing_a_slower_route_is_recorded_not_silent(monkeypatch):
    """Decision #21: taking a route measured slower than the default must land
    in the dispatch fallback log under the op's name, so a machine running the
    slow lane can be found rather than guessed at."""
    monkeypatch.setenv("TESSERA_APPLE_MOE_LOWP", "1")
    rng = np.random.default_rng(11)
    x = rng.standard_normal((4, 8)).astype(np.float16)
    wg = rng.standard_normal((2, 8, 10)).astype(np.float16)
    wu = rng.standard_normal((2, 8, 10)).astype(np.float16)
    wd = rng.standard_normal((2, 10, 6)).astype(np.float16)
    before = len(R._DISPATCH_FALLBACK_LOG)
    R._apple_gpu_dispatch_moe_swiglu_block([x, wg, wu, wd, [2, 2]], {}, np)
    new = list(R._DISPATCH_FALLBACK_LOG)[before:]
    assert any(op == "apple_gpu.moe_swiglu_block" and "lowp" in reason
               for op, reason in new), new


def test_the_arbiter_is_still_consulted(monkeypatch):
    """Decision #28/#29: the strict route ledger row must keep its consumer.

    Deleting the `production_route_for` call would have made the MoE ledger row
    a declaration nothing reads, and would have "fixed" a slow route by
    removing the mechanism that is supposed to choose between routes. The
    arbiter decides; `lowp` is merely no longer allowed to jump the queue.
    """
    from tessera.compiler import apple_route_selector

    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return "composed"

    monkeypatch.setattr(apple_route_selector, "production_route_for", fake)
    x, wg, wd = _ops(np.float16)
    route, reason = R._apple_moe_select_route(x, wg, wd, G, {}, np)
    assert route == "composed" and "ledger" in reason
    assert seen["op"] == "retune_moe_swiglu"
    assert seen["shape"] == "4x8x10x6_e2", seen


def test_an_unreadable_ledger_is_not_fatal(monkeypatch):
    """A ledger that cannot be read falls back to the measured-best default
    WITH a reason, rather than raising or silently guessing."""
    from tessera.compiler import apple_route_selector

    def boom(**kwargs):
        raise RuntimeError("ledger corrupt")

    monkeypatch.setattr(apple_route_selector, "production_route_for", boom)
    x, wg, wd = _ops(np.float32)
    route, reason = R._apple_moe_select_route(x, wg, wd, G, {}, np)
    assert route == "composed" and "unavailable" in reason


def test_a_ledger_choice_the_shape_cannot_honour_is_declined(monkeypatch):
    """The fused kernel has a shape range. A ledger row that names it for an
    out-of-range shape must decline with a reason, not dispatch into it."""
    from tessera.compiler import apple_route_selector

    monkeypatch.setattr(apple_route_selector, "production_route_for",
                        lambda **k: "single_fused")
    rng = np.random.default_rng(4)
    big = R._apple_fused_score_cap() + 8
    x = rng.standard_normal((4, 8)).astype(np.float32)
    wg = rng.standard_normal((2, 8, big)).astype(np.float32)
    wd = rng.standard_normal((2, big, 6)).astype(np.float32)
    route, reason = R._apple_moe_select_route(x, wg, wd, G, {}, np)
    assert route == "composed" and "out of its range" in reason
