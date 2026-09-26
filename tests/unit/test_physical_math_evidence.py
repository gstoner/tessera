from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.math import benchmark_physical_math as benchmark


_ROOT = Path(__file__).resolve().parents[2]
_BASELINES = _ROOT / "benchmarks" / "baselines"


def _packet(name: str) -> dict:
    return json.loads((_BASELINES / name).read_text())


def test_zen5_math_packet_records_retained_scan_selector() -> None:
    packet = _packet("math_physical_zen5_2026_08_06.json")
    assert packet["schema"] == "tessera.physical_math_evidence.v1"
    assert packet["selector_eligible"] is True
    assert packet["storage_dtypes"] == ["f32"]
    assert len(packet["rows"]) == 7

    policies = {
        row["op_name"]: row for row in packet["scan_selector_evidence"]
    }
    assert set(policies) == {"cumsum", "cumprod", "cummax", "cummin"}
    for name in ("cumsum", "cumprod"):
        assert policies[name]["selected_policy"] == "avx512_hillis_steele_16"
        assert policies[name]["speedup"] > 1.05
    for name in ("cummax", "cummin"):
        assert policies[name]["selected_policy"] == "scalar_recurrence_retained"


def test_gfx1151_math_packet_covers_dtypes_and_cache_gain() -> None:
    packet = _packet("math_physical_gfx1151_2026_08_06.json")
    assert packet["schema"] == "tessera.physical_math_evidence.v1"
    assert packet["selector_eligible"] is False
    # Committed 2026-08-06 evidence records the rule in force then; fresh
    # packets name the kernel-clock witness instead (checked below).
    assert packet["device_event_follow_up"] == "bare_metal_required"
    assert packet["storage_dtypes"] == ["f32", "f16", "bf16"]
    assert len(packet["dtype_rows"]) == 21
    assert {row["dtype"] for row in packet["dtype_rows"]} == {
        "f32", "f16", "bf16"
    }
    assert all(
        row["max_abs_error"] <= row["error_limit"]
        for row in packet["dtype_rows"]
    )
    assert len(packet["f32_cache_comparison"]) == 7
    assert all(row["speedup"] > 1.4 for row in packet["f32_cache_comparison"])


def _generated_rows(_rt, target, dtype, _iterations):
    families = (
        ("unary", "sqrt"),
        ("transcendental" if target == "x86" else "unary", "exp"),
        ("binary", "add"), ("binary", "div"), ("reduce", "sum"),
        ("scan", "cumsum"), ("scan", "cummax"),
    )
    return [
        {"dtype": dtype, "family": family, "op_name": op_name,
         "warm_median_ms": 1.0, "max_abs_error": 0.0, "error_limit": 0.01}
        for family, op_name in families
    ]


def test_generator_emits_complete_x86_packet_schema(monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "_measure_dtype", _generated_rows)
    monkeypatch.setattr(
        benchmark, "_x86_scan_selector_evidence",
        lambda _rt, _iterations: [{"op_name": "cumsum"}],
    )
    packet = benchmark._run("x86", "all", 4)
    assert packet["selector_eligible"] is False
    assert packet["promotion_eligible"] is False
    assert packet["storage_dtypes"] == ["f32"]
    assert len(packet["rows"]) == 7
    assert packet["scan_selector_evidence"] == [{"op_name": "cumsum"}]


def test_generator_emits_complete_rocm_packet_schema(monkeypatch) -> None:
    from tessera import runtime as rt

    monkeypatch.setattr(benchmark, "_measure_dtype", _generated_rows)
    monkeypatch.setattr(rt, "_rocm_device_name", lambda: "gfx1151")
    monkeypatch.setattr(
        benchmark, "_rocm_cache_comparison",
        lambda _rt, rows, _iterations: [
            {"family": row["family"], "op_name": row["op_name"], "speedup": 2.0}
            for row in rows
        ],
    )
    packet = benchmark._run("rocm", "all", 4)
    assert packet["selector_eligible"] is False
    assert packet["device_event_follow_up"] == "kernel_clock_witness_required"
    assert packet["storage_dtypes"] == ["f32", "f16", "bf16"]
    assert len(packet["dtype_rows"]) == 21
    assert {row["dtype"] for row in packet["dtype_rows"]} == {
        "f32", "f16", "bf16"
    }
    assert len(packet["f32_cache_comparison"]) == 7


def test_generator_rejects_partial_rocm_packet(monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "_measure_dtype", _generated_rows)
    with pytest.raises(ValueError, match="must aggregate"):
        benchmark._run("rocm", "f32", 4)


@pytest.mark.parametrize("kind", ["reference_cpu", None])
def test_math_probe_rejects_unproven_native_execution(kind):
    from types import SimpleNamespace
    rt = SimpleNamespace(launch=lambda *args: {"ok": True, "execution_kind": kind})
    with pytest.raises(RuntimeError, match="observed native_gpu"):
        benchmark._checked_launch(rt, "rocm", object(), ())


def test_math_probe_rejects_nan_and_wrong_shape(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    monkeypatch.setattr(benchmark, "_cases", lambda *args: [("unary", "sqrt", (np.ones(2),), {}, lambda: np.ones(2))])
    rt = SimpleNamespace(RuntimeArtifact=lambda **kw: kw)
    for output in (np.array([np.nan, 1]), np.ones((1,2))):
        rt.launch = lambda *args: {"ok": True, "execution_kind": "native_cpu", "output": output}
        with pytest.raises(RuntimeError, match="nonfinite"):
            benchmark._measure_dtype(rt, "x86", "f32", 1)
