"""Phase E3 / Pillar 2 — flywheel ↔ autotune_v2 bridge (EVALUATOR_PLAN.md §6).

Portable: the autotuner is analytical, so the corpus→search connection is fully
checkable without Metal — FLOP-accounting agreement, device-grounded legal
candidate generation, and best-by-measurement selection.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.flywheel import AutotuneRecord, LatencyStats
from tessera.compiler.flywheel_autotune import (
    autotune_matmul,
    autotuner_for,
    best_record,
    gemm_workload_for,
    measured_tflops,
    pick_best,
)


def _mm(a, b):
    return ts.ops.matmul(a, b)


_MM = ts.jit(target="apple_gpu")(_mm)


def _rec(M, N, K, dtype, median_ms):
    flops = 2 * M * N * K
    achieved = flops / (median_ms * 1e-3) / 1e12 if median_ms else None
    return AutotuneRecord(
        1, "matmul", {"M": M, "N": N, "K": K}, dtype, "apple_gpu",
        "apple_gpu:apple-m1-max", {"dtype": dtype}, True, "",
        LatencyStats(median_ms, median_ms, median_ms, 5) if median_ms else None,
        achieved, 0.1, None, "sweep",
    )


def test_workload_maps_shape_and_dtype():
    w = gemm_workload_for(_rec(64, 128, 256, "f32", 1.0))
    assert (w.M, w.N, w.K) == (64, 128, 256)
    assert w.dtype == "fp32"                 # flywheel f32 → autotune_v2 fp32
    assert w.flops() == 2 * 64 * 128 * 256
    assert gemm_workload_for(_rec(8, 8, 8, "f16", 1.0)).dtype == "fp16"


def test_measured_tflops_agrees_with_record_accounting():
    """The flywheel and autotune_v2 must agree on FLOP accounting — a measured
    record validates the analytical model (cross-check)."""
    r = _rec(512, 512, 512, "f32", 2.0)
    assert measured_tflops(r) == pytest.approx(r.achieved_tflops)


def test_measured_tflops_none_for_non_native():
    r = _rec(8, 8, 8, "f32", 0.0)            # median_ms 0 → no latency
    assert measured_tflops(r) is None


def test_autotuner_exposes_legal_constrained_candidates():
    """The corpus record drives a device-grounded BaCO-style search; its legal
    candidate set (the constrained space) is non-empty."""
    tuner = autotuner_for(_rec(1024, 1024, 1024, "f32", 1.0))
    cands = tuner.legal_candidates()
    assert cands, "expected a non-empty legal candidate set"


def test_best_record_picks_fastest_native():
    corpus = [
        _rec(512, 512, 512, "f32", 2.0),
        _rec(512, 512, 512, "f32", 1.0),     # fastest
        _rec(512, 512, 512, "f32", 3.0),
    ]
    best = best_record(corpus)
    assert best is not None and best.latency.median_ms == 1.0


def test_best_record_none_when_no_native():
    assert best_record([_rec(8, 8, 8, "f32", 0.0)]) is None


def test_pick_best_by_latency_and_tflops():
    slow = _rec(512, 512, 512, "f32", 4.0)    # higher latency, lower tflops
    fast = _rec(512, 512, 512, "f16", 1.0)    # lower latency, higher tflops
    assert pick_best([slow, fast], by="latency") is fast
    assert pick_best([slow, fast], by="tflops") is fast
    with pytest.raises(ValueError, match="unknown selection objective"):
        pick_best([fast], by="bogus")


# ── Darwin: autotune over the measurable Apple knob space ────────────────────

@pytest.mark.skipif(sys.platform != "darwin", reason="Metal measurement is Darwin-only.")
def test_autotune_matmul_over_dtype_returns_best_and_corpus():
    rng = np.random.default_rng(20260612)
    best, corpus = autotune_matmul("apple_gpu", _MM, 512, 512, 512, rng,
                                   dtypes=("f32", "f16"), reps=6)
    assert {r.dtype for r in corpus} == {"f32", "f16"}     # both candidates measured
    assert all(r.latency is not None for r in corpus)      # both ran natively
    assert all(r.search_method == "autotune" for r in corpus)
    assert best is not None and best.dtype in ("f32", "f16")
    # the winner is the fastest measured candidate
    assert best.latency.median_ms == min(r.latency.median_ms for r in corpus)
