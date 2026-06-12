"""Phase E3 / Pillar 2 — autotuning flywheel records (EVALUATOR_PLAN.md §6–§7).

Portable: roofline math, FLOP/byte accounting, record schema + the device_id
discipline. Darwin: record a real matmul on Metal — asserting structure (native
measured + roofline + residual on one row), never a fixed latency.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.flywheel import (
    NOMINAL_APPLE_GPU_PEAK,
    AutotuneRecord,
    DevicePeak,
    LatencyStats,
    default_device_id,
    detect_device_id,
    distill_dispatch,
    efficiency_trend,
    load_corpus,
    lookup_dispatch,
    matmul_flops_bytes,
    peak_for_device,
    record_matmul,
    roofline_ms,
    save_corpus,
    size_bucket,
    sweep_matmul,
)


def _rec(size, dtype, median_ms, *, sched=None, op="matmul", native=True):
    """Synthetic record helper for portable tests."""
    return AutotuneRecord(
        1, op, {"M": size, "N": size, "K": size}, dtype, "apple_gpu",
        "apple_gpu:apple-m1-max", sched or {"dtype": dtype, "size": size}, True, "",
        LatencyStats(median_ms, median_ms, median_ms, 5) if native else None,
        1.0 if native else None, 0.1, None, "sweep",
    )


def _mm(a, b):
    return ts.ops.matmul(a, b)


_MM = ts.jit(target="apple_gpu")(_mm)


# ── portable ─────────────────────────────────────────────────────────────────

def test_matmul_flops_and_bytes():
    flops, byts = matmul_flops_bytes(64, 64, 64, dtype_bytes=4)
    assert flops == 2.0 * 64 ** 3
    assert byts == (3 * 64 * 64) * 4


def test_roofline_picks_max_of_compute_and_memory():
    peak = DevicePeak("t", peak_tflops=8.0, peak_bw_gb_s=200.0)
    compute_bound = roofline_ms(2e12, 8.0, peak)        # huge flops, tiny bytes
    mem_bound = roofline_ms(8.0, 2e11, peak)            # tiny flops, huge bytes
    assert compute_bound > 0 and mem_bound > 0
    # compute-bound case: 2e12 / 8e12 s = 0.25 s = 250 ms
    assert abs(compute_bound - 250.0) < 1.0
    # memory-bound case: 2e11 / 2e11 s = 1 s = 1000 ms
    assert abs(mem_bound - 1000.0) < 1.0


def test_record_carries_device_id_and_residual():
    rec = AutotuneRecord(
        schema_version=1, op_chain="matmul", problem_shape={"M": 8, "N": 8, "K": 8},
        dtype="f32", target="apple_gpu", device_id="apple_gpu:x",
        schedule={"dtype": "f32"}, legal=True, violation_reason="",
        latency=LatencyStats(2.0, 1.5, 2.5, 10), achieved_tflops=1.0,
        roofline_predicted_ms=0.5, model_predicted_ms=None, search_method="manual",
    )
    assert rec.device_id  # required partition key — never aggregate across it
    assert rec.roofline_residual_ms == pytest.approx(1.5)   # 2.0 measured − 0.5 floor
    d = rec.to_dict()
    assert d["latency"]["reps"] == 10 and d["device_id"] == "apple_gpu:x"


def test_record_without_latency_has_no_residual():
    rec = AutotuneRecord(
        schema_version=1, op_chain="matmul", problem_shape={"M": 8, "N": 8, "K": 8},
        dtype="f32", target="nvidia_sm90", device_id="nvidia:x",
        schedule={}, legal=True, violation_reason="", latency=None,
        achieved_tflops=None, roofline_predicted_ms=0.5, model_predicted_ms=None,
        search_method="manual",
    )
    assert rec.roofline_residual_ms is None      # no perf signal for a non-native run
    assert rec.to_dict()["latency"] is None


def test_default_device_id_is_nonempty_and_targeted():
    did = default_device_id("apple_gpu")
    assert did and "apple_gpu" in did


# ── (1) per-chip calibration ─────────────────────────────────────────────────

def test_peak_for_device_matches_family_most_specific_first():
    assert peak_for_device("apple_gpu:apple-m1-max").name == "apple-m1-max"
    assert peak_for_device("apple_gpu:apple-m1-ultra").name == "apple-m1-ultra"
    assert peak_for_device("apple_gpu:apple-m1").name == "apple-m1"   # bare M1
    # unknown chip → nominal fallback
    assert peak_for_device("apple_gpu:apple-m9-quantum") is NOMINAL_APPLE_GPU_PEAK


def test_m1_max_peak_is_calibrated_not_nominal():
    p = peak_for_device("apple_gpu:apple-m1-max")
    assert p is not NOMINAL_APPLE_GPU_PEAK
    assert p.peak_tflops > NOMINAL_APPLE_GPU_PEAK.peak_tflops  # 10.4 > 8 nominal


@pytest.mark.skipif(sys.platform != "darwin", reason="chip detection is Darwin-only.")
def test_detect_device_id_resolves_the_real_chip_on_darwin():
    did = detect_device_id("apple_gpu")
    assert did.startswith("apple_gpu:apple-m")   # real chip, not the coarse fallback
    assert did != default_device_id("apple_gpu")


# ── (3) persist + distill ────────────────────────────────────────────────────

def test_size_bucket_boundaries():
    assert size_bucket(128) == "xs"
    assert size_bucket(512) == "s"
    assert size_bucket(2048) == "m"
    assert size_bucket(8192) == "l"


def test_corpus_json_round_trip(tmp_path):
    corpus = [_rec(512, "f32", 1.2), _rec(8, "f32", 0.0, native=False)]
    path = str(tmp_path / "corpus.json")
    save_corpus(corpus, path)
    back = load_corpus(path)
    assert len(back) == 2
    assert back[0].problem_shape == {"M": 512, "N": 512, "K": 512}
    assert back[0].latency is not None and back[0].latency.median_ms == 1.2
    assert back[1].latency is None      # non-native record survives the round-trip


def test_distill_picks_lowest_latency_per_class():
    # two f32 512-cube candidates (same bucket) — distill must pick the faster.
    corpus = [
        _rec(512, "f32", 2.0, sched={"variant": "slow"}),
        _rec(512, "f32", 1.0, sched={"variant": "fast"}),
        _rec(512, "f16", 0.8, sched={"variant": "f16"}),
    ]
    table = distill_dispatch(corpus)
    win = lookup_dispatch(table, "matmul", "f32", 512)
    assert win is not None and win["schedule"]["variant"] == "fast"
    assert win["median_ms"] == 1.0
    # f16 is a distinct class
    assert lookup_dispatch(table, "matmul", "f16", 512)["schedule"]["variant"] == "f16"
    # uncovered class → None (caller falls back)
    assert lookup_dispatch(table, "matmul", "f32", 99999) is None


def test_distill_skips_non_native_records():
    table = distill_dispatch([_rec(512, "f32", 0.0, native=False)])
    assert table == {}


def test_efficiency_trend_sorts_by_size():
    recs = [
        AutotuneRecord(1, "matmul", {"M": s, "N": s, "K": s}, "f32", "apple_gpu",
                       "d", {}, True, "", LatencyStats(1, 1, 1, 1), tflops,
                       0.1, None, "sweep")
        for s, tflops in [(1024, 0.9), (256, 0.03), (512, 0.23)]
    ]
    trend = efficiency_trend(recs)
    assert [s for s, _ in trend] == [256, 512, 1024]      # sorted by size
    assert [t for _, t in trend] == [0.03, 0.23, 0.9]


def test_efficiency_trend_skips_non_native_records():
    recs = [
        AutotuneRecord(1, "matmul", {"M": 8, "N": 8, "K": 8}, "f32", "nvidia_sm90",
                       "d", {}, True, "", None, None, 0.1, None, "sweep"),
    ]
    assert efficiency_trend(recs) == []


# ── Darwin: record a real matmul on Metal ────────────────────────────────────

@pytest.mark.skipif(sys.platform != "darwin", reason="Metal measurement is Darwin-only.")
def test_flywheel_records_a_real_matmul():
    rng = np.random.default_rng(20260612)
    a = rng.standard_normal((256, 256)).astype(np.float32)
    b = rng.standard_normal((256, 256)).astype(np.float32)

    rec = record_matmul("apple_gpu", _MM, (a, b), m=256, n=256, k=256, reps=10)

    assert rec.latency is not None and rec.latency.median_ms > 0.0
    assert rec.latency.p10_ms <= rec.latency.median_ms <= rec.latency.p90_ms
    assert rec.roofline_predicted_ms > 0.0
    assert rec.achieved_tflops is not None and rec.achieved_tflops > 0.0
    assert rec.roofline_residual_ms is not None
    assert "apple_gpu" in rec.device_id
    assert rec.problem_shape == {"M": 256, "N": 256, "K": 256}
    assert rec.to_dict()["latency"]["reps"] == rec.latency.reps
    # Honest regime check: a small matmul is launch-overhead-dominated, so measured
    # latency sits well ABOVE the compute/BW roofline floor (residual > 0).
    assert rec.latency.median_ms > rec.roofline_predicted_ms
    assert rec.roofline_residual_ms > 0.0


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal measurement is Darwin-only.")
def test_sweep_builds_a_corpus_with_rising_efficiency():
    """The candidate sweep is a corpus, and efficiency climbs with size as launch
    overhead amortizes — a host-independent trend (assert the trend, not numbers)."""
    rng = np.random.default_rng(20260612)
    records = sweep_matmul("apple_gpu", _MM, (256, 512, 1024), rng, dtype="f32", reps=6)

    assert len(records) == 3
    assert all(r.latency is not None for r in records)        # all ran natively
    assert all(r.search_method == "sweep" for r in records)
    assert all(r.schedule["size"] == r.problem_shape["M"] for r in records)

    trend = efficiency_trend(records)
    sizes = [s for s, _ in trend]
    tflops = [t for _, t in trend]
    assert sizes == [256, 512, 1024]
    # Efficiency rises with size (overhead fraction shrinks). Robust margin: the
    # largest is comfortably more efficient than the smallest.
    assert tflops[-1] > tflops[0] * 3.0, dict(trend)


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal measurement is Darwin-only.")
def test_sweep_records_f16_natively():
    rng = np.random.default_rng(1)
    records = sweep_matmul("apple_gpu", _MM, (512,), rng, dtype="f16", reps=6)
    assert len(records) == 1
    r = records[0]
    assert r.latency is not None and r.dtype == "f16"
    assert r.achieved_tflops is not None and r.achieved_tflops > 0.0
