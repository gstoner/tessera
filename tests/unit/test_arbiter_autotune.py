"""Workstream D2 + D3 — measured autotune + arbiter dispatch log (backend-agnostic).

Host-free: fake candidates under a private target exercise the measured-selection
cache (D2) and the won/degraded/no_candidate dispatch log (D3) without a GPU. The
NVIDIA live gates for these live in test_nvidia_plugin.py.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from tessera.compiler.emit import autotune as AT
from tessera.compiler.emit import candidate as C
from tessera.compiler.emit.candidate import (
    OP_MATMUL,
    Candidate,
    Tier,
    arbiter_dispatch_histogram,
    register_candidate,
    reset_arbiter_dispatch_log,
    run_arbitrated,
)

_TGT = "d2d3_faketarget"


class _FakeRegion:
    dtype = "bfloat16"

    def reference(self, A, B):
        return np.asarray(A, np.float32) @ np.asarray(B, np.float32)


class _FakeCand(Candidate):
    op = OP_MATMUL
    target = _TGT

    def __init__(self, name, tag, tier=Tier.EMITTED, delay=0.0):
        self.name = name
        self.tier = tier
        self._tag = tag
        self._delay = delay
        self.runs = 0

    def run(self, region, A, B, *a, **k):
        self.runs += 1
        if self._delay:
            time.sleep(self._delay)
        return region.reference(A, B), self._tag


def _mm(m=4, k=4, n=4):
    rng = np.random.default_rng(0)
    return (rng.standard_normal((m, k)).astype(np.float32),
            rng.standard_normal((k, n)).astype(np.float32))


# ── D2: measured selection + measure-at-first-miss cache ─────────────────────

def test_measure_latency_is_monotonic_positive():
    slow = AT.measure_latency(lambda: time.sleep(0.002), reps=3, warmup=1)
    fast = AT.measure_latency(lambda: None, reps=3, warmup=1)
    assert slow > fast >= 0.0


def test_measured_arbitrate_picks_fastest_and_caches():
    fast = _FakeCand("fake_fast", "fake_real", delay=0.0)
    slow = _FakeCand("fake_slow", "fake_real", delay=0.005)
    register_candidate(slow)      # register slow first: tier-priority would tie →
    register_candidate(fast)      # measurement must be what picks the fast one
    region, cache = _FakeRegion(), AT.MeasureCache()
    A, B = _mm()
    win = AT.measured_arbitrate(region, OP_MATMUL, _TGT, A, B,
                                dims=(4, 4, 4), dtype="bfloat16",
                                cache=cache, reps=5, warmup=1, device="fakedev")
    assert win is not None and win.name == "fake_fast"       # measured, not tiered
    assert cache.misses == 1 and cache.size == 1
    rec = cache.to_dict()["records"][0]
    assert set(rec["candidates"]) == {"fake_fast", "fake_slow"}  # both were timed
    runs_after_first = fast.runs + slow.runs
    # Second call, same bucket → cache hit, no re-timing.
    win2 = AT.measured_arbitrate(region, OP_MATMUL, _TGT, A, B,
                                 dims=(4, 4, 4), dtype="bfloat16",
                                 cache=cache, reps=5, warmup=1, device="fakedev")
    assert win2.name == "fake_fast" and cache.hits == 1
    assert fast.runs + slow.runs == runs_after_first       # nothing re-measured


def test_measured_arbitrate_buckets_distinct_shapes_separately():
    register_candidate(_FakeCand("fake_only", "fake_real"))
    region, cache = _FakeRegion(), AT.MeasureCache()
    A, B = _mm()
    AT.measured_arbitrate(region, OP_MATMUL, _TGT, A, B, dims=(16, 16, 16),
                          cache=cache, reps=2, warmup=1, device="fakedev")
    AT.measured_arbitrate(region, OP_MATMUL, _TGT, A, B, dims=(64, 64, 64),
                          cache=cache, reps=2, warmup=1, device="fakedev")
    assert cache.size == 2       # different power-of-two buckets → distinct keys


# ── D3: arbiter dispatch log (won / degraded / no_candidate) ─────────────────

def test_dispatch_log_records_won_degraded_no_candidate():
    register_candidate(_FakeCand("fake_winner", "fake_real"))
    register_candidate(_FakeCand("fake_decliner", "reference"))   # declines at run
    region = _FakeRegion()
    A, B = _mm()
    reset_arbiter_dispatch_log()
    # won: a real-tag candidate runs
    _, t1 = run_arbitrated(region, OP_MATMUL, _TGT, A, B,
                           verify=False, force="fake_winner")
    # degraded: selected a candidate, but it declined to the reference at run time
    _, t2 = run_arbitrated(region, OP_MATMUL, _TGT, A, B,
                           verify=False, force="fake_decliner")
    # no_candidate: an (target, op) with nothing registered
    _, t3 = run_arbitrated(region, OP_MATMUL, "unregistered_target", A, B)
    assert (t1, t2, t3) == ("fake_real", "reference", "reference")
    h = arbiter_dispatch_histogram()
    assert h[(_TGT, OP_MATMUL)] == {"won": 1, "degraded": 1, "no_candidate": 0}
    assert h[("unregistered_target", OP_MATMUL)]["no_candidate"] == 1


def test_dispatch_histogram_filters_by_target():
    register_candidate(_FakeCand("fake_w2", "fake_real"))
    region = _FakeRegion()
    A, B = _mm()
    reset_arbiter_dispatch_log()
    run_arbitrated(region, OP_MATMUL, _TGT, A, B, verify=False, force="fake_w2")
    run_arbitrated(region, OP_MATMUL, "other_target", A, B)
    only = arbiter_dispatch_histogram(target=_TGT)
    assert set(only) == {(_TGT, OP_MATMUL)}
