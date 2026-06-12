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
    matmul_flops_bytes,
    record_matmul,
    roofline_ms,
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
