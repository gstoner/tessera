"""Workstream J / W7 — roofline attainment for the E2 hot-path ratchets.

The latency ratchet is a *relative* bar (did it get slower?). J adds the
*absolute* bar: % of peak. Each hot-path row's wall-clock median → achieved
TFLOP/s ÷ the device's grounded peak = ``pct_peak``, and a regression below an
``attainment_floor`` fails the gate (the absolute analog of the latency cap).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import roofline as R  # noqa: E402

_DEV = "rocm:gfx1151"


# ── FLOP / byte / attainment model ────────────────────────────────────────────

def test_matmul_flops_and_bytes():
    assert R.op_flops("matmul", "512x512x512") == 2 * 512**3
    assert R.op_bytes("matmul", "8x16x32", "f16") == (8*32 + 32*16 + 8*16) * 2


def test_flash_attn_flops():
    s = 512
    # Default causal → triangular score pairs S·(S+1)/2 (the recorded hot path).
    assert R.op_flops("flash_attn", "1x8x512x64") == \
        4 * 1 * 8 * (s * (s + 1) // 2) * 64
    # Dense S² only when causal=False is requested explicitly.
    assert R.op_flops("flash_attn", "1x8x512x64", causal=False) == \
        4 * 1 * 8 * s * s * 64
    # Causal is ~half the dense work (never charges the masked upper tiles).
    assert R.op_flops("flash_attn", "1x8x512x64") < \
        R.op_flops("flash_attn", "1x8x512x64", causal=False)


def test_gemm_f32_flops_and_bytes_match_matmul():
    # The register-blocked f32 kernel is a plain GEMM — same 2·M·N·K work as WMMA
    # matmul, gated against the f32 peak (not f16). Bytes are f32-wide.
    assert R.op_flops("gemm_f32", "512x512x512") == 2 * 512**3
    assert R.op_bytes("gemm_f32", "8x16x32", "f32") == (8*32 + 32*16 + 8*16) * 4
    # pct_peak divides by the f32 peak (29.7), not the f16 peak (59.4).
    pk = R.pct_peak("gemm_f32", "1024x1024x1024", "f32", 5.0, _DEV)
    ach = R.achieved_tflops("gemm_f32", "1024x1024x1024", 5.0)
    assert pk == pytest.approx(ach / 29.7, rel=1e-6)


def test_flash_attn_bwd_flops_is_2p5x_forward():
    fwd = R.op_flops("flash_attn", "1x16x1024x128")
    bwd = R.op_flops("flash_attn_bwd", "1x16x1024x128")
    assert bwd == int(2.5 * fwd)
    # 7 tensors of DRAM traffic (Q,K,V,dO in; dQ,dK,dV out), f16-wide.
    assert R.op_bytes("flash_attn_bwd", "1x8x512x64", "f16") == \
        7 * 1 * 8 * 512 * 64 * 2


def test_unmodeled_op_is_none():
    assert R.op_flops("softmax", "1024") is None
    assert R.op_bytes("layer_norm", "4x8", "f16") is None


def test_achieved_and_pct_peak():
    # 2048³ f16 at 10ms: 2·2048³ / 10ms = 1.717e10/0.01/1e12 ≈ 1.717 TF.
    ach = R.achieved_tflops("matmul", "2048x2048x2048", 10.0)
    assert ach == pytest.approx(2 * 2048**3 / 0.01 / 1e12, rel=1e-6)
    pk = R.pct_peak("matmul", "2048x2048x2048", "f16", 10.0, _DEV)
    assert pk == pytest.approx(ach / 59.4, rel=1e-6)


def test_pct_peak_none_for_unknown_device_or_dtype():
    assert R.pct_peak("matmul", "64x64x64", "f16", 1.0, "rocm:gfxZZZ") is None
    assert R.pct_peak("matmul", "64x64x64", "int4", 1.0, _DEV) is None


def test_peak_is_grounded_with_a_source():
    dev = R.DEVICE_PEAK[_DEV]
    assert "rocminfo" in dev["source"] and "2.9 GHz" in dev["source"]
    assert dev["peak_tflops"]["f16"] == 59.4 and dev["peak_bw_gb_s"] == 256.0


def test_annotate_rows_adds_fields_from_median():
    rows = [{"op": "matmul", "shape": "1024x1024x1024", "dtype": "f16",
             "mode": "wmma", "median_ms": 5.0},
            {"op": "softmax", "shape": "1024", "dtype": "f16", "mode": "x",
             "median_ms": 0.1}]
    R.annotate_rows(rows, _DEV)
    assert "pct_peak" in rows[0] and "achieved_tflops" in rows[0]
    assert "pct_peak" not in rows[1]           # unmodeled op untouched


# ── the attainment gate ────────────────────────────────────────────────────────

def _row(median_ms, floor=None):
    r = {"op": "matmul", "shape": "2048x2048x2048", "dtype": "f16",
         "mode": "wmma", "median_ms": median_ms}
    if floor is not None:
        r["attainment_floor"] = floor
    return r


def test_attainment_gate_passes_above_floor():
    base = {"rows": [_row(10.0, floor=0.02)]}       # ~0.0287 pct_peak
    assert R.evaluate_attainment([_row(10.0)], base, _DEV) == []


def test_attainment_gate_fails_below_floor():
    base = {"rows": [_row(10.0, floor=0.05)]}        # floor above achievable
    fails = R.evaluate_attainment([_row(10.0)], base, _DEV)
    assert len(fails) == 1 and "below floor" in fails[0]


def test_attainment_gate_flags_missing_measurement():
    base = {"rows": [_row(10.0, floor=0.02)]}
    fails = R.evaluate_attainment([], base, _DEV)
    assert len(fails) == 1 and "coverage" in fails[0]


def test_attainment_gate_ignores_rows_without_floor():
    base = {"rows": [_row(10.0)]}                    # no floor → not gated
    assert R.evaluate_attainment([_row(999.0)], base, _DEV) == []


# ── the committed gfx1151 baseline is self-consistent ─────────────────────────

def test_attainment_gate_reads_latency_ms_from_measured_rows():
    # Measured ratchet-report rows carry `latency_ms` (not `median_ms`) — the gate
    # must read it, else pct_peak is None and the row false-fails on coverage.
    base = {"rows": [_row(10.0, floor=0.02)]}
    measured = [{"op": "matmul", "shape": "2048x2048x2048", "dtype": "f16",
                 "mode": "wmma", "latency_ms": 10.0}]     # no median_ms
    assert R.evaluate_attainment(measured, base, _DEV) == []
    slow = [{"op": "matmul", "shape": "2048x2048x2048", "dtype": "f16",
             "mode": "wmma", "latency_ms": 100.0}]         # 10x slower → below floor
    fails = R.evaluate_attainment(slow, base, _DEV)
    assert len(fails) == 1 and "below floor" in fails[0]


def test_perf_gate_attainment_via_package_import(tmp_path):
    # P2: perf_gate must import roofline under package usage (from benchmarks import
    # perf_gate) — not only as a script. P1: the report rows carry latency_ms.
    from benchmarks import perf_gate
    base_path = _BENCH / "baselines" / "rocm_gfx1151_hot_paths.json"
    base = json.loads(base_path.read_text())
    rows = [{"op": r["op"], "shape": r["shape"], "dtype": r["dtype"],
             "mode": r["mode"], "latency_ms": r["median_ms"]}
            for r in base["rows"]]
    report = tmp_path / "report.json"
    report.write_text(json.dumps(rows))
    rc = perf_gate.main([str(report), "--baseline", str(base_path),
                         "--attainment", "--device", "rocm:gfx1151"])
    assert rc == 0


def test_committed_baseline_has_attainment_and_self_passes():
    p = _BENCH / "baselines" / "rocm_gfx1151_hot_paths.json"
    base = json.loads(p.read_text())
    modeled = [r for r in base["rows"] if R.op_flops(r["op"], r["shape"])]
    assert modeled, "expected FLOP-modeled rows (matmul/flash_attn)"
    for r in modeled:
        assert "pct_peak" in r and "attainment_floor" in r
        assert r["attainment_floor"] <= r["pct_peak"]     # floor is below current
    # Re-evaluating the baseline's own medians against itself must pass.
    assert R.evaluate_attainment(base["rows"], base, _DEV) == []
