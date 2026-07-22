from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta

import numpy as np
import pytest


def _module():
    path = (Path(__file__).resolve().parents[2] / "benchmarks" / "apple_gpu" /
            "benchmark_legacy_retune.py")
    spec = importlib.util.spec_from_file_location("benchmark_legacy_retune", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_retune_report_keeps_reference_and_missing_device_intervals_explicit(
    monkeypatch,
):
    module = _module()
    oracle = np.arange(4, dtype=np.float32)
    case = module.Case(
        "reduction", "retune_test", "4", "f32",
        module.Route("native", lambda: oracle.copy(), True, True, "fake.native"),
        module.Route("reference", lambda: oracle.copy(), False, False,
                     "reference_cpu"),
        oracle, 16, 16, "test_logical_io",
    )
    monkeypatch.setattr(module, "_cases", lambda seed: [case])
    monkeypatch.setattr(module.rt.DeviceTensor, "is_metal", staticmethod(lambda: False))
    monkeypatch.setattr(module, "set_dispatch_telemetry_enabled", lambda enabled: True)
    monkeypatch.setattr(module, "live_apple_route_context", lambda: type(
        "Context", (), {"device": "apple7", "as_mapping": lambda self: {}})())
    monkeypatch.setattr(module, "clear_dispatch_telemetry", lambda: None)
    monkeypatch.setattr(module, "read_dispatch_telemetry", lambda: {
        "device_time_ns": None, "resources": None,
    })
    report = module.run_report(reps=2, trials=3)
    assert report["schema_version"] == 1
    native, reference = report["runs"]
    assert native["native_dispatched"] is False
    assert reference["execution_kind"] == "reference_cpu"
    assert native["telemetry"]["device_time_median_ns"] is None
    assert reference["telemetry"]["device_time_scope"] == \
        "unavailable_multi_dispatch_or_mapped_memory"
    assert native["telemetry"]["transport"]["scope"] == \
        "logical_host_visible_io_not_device_bandwidth"
    assert native["telemetry"]["transport"]["logical_total_bytes"] > 0


def test_retune_distinguishes_grouped_gap_from_owned_low_precision_moe_abi():
    module = _module()
    candidates = module.low_precision_candidate_status()
    assert {(row["family"], row["dtype"]) for row in candidates} == {
        ("grouped_gemm", "f16"), ("grouped_gemm", "bf16"),
        ("moe_swiglu", "f16"), ("moe_swiglu", "bf16"),
    }
    grouped = [row for row in candidates if row["family"] == "grouped_gemm"]
    moe = [row for row in candidates if row["family"] == "moe_swiglu"]
    assert all(row["status"] == "unsupported_no_owned_same_abi" for row in grouped)
    assert all(row["status"] == "owned_same_abi_ready_for_measurement" for row in moe)


def test_low_precision_profile_uses_native_moe_incumbent_and_reference_oracle():
    module = _module()
    cases = module._low_precision_moe_cases(seed=7)
    assert {(case.op, case.dtype) for case in cases} == {
        ("retune_moe_swiglu_lowp", "f16"),
        ("retune_moe_swiglu_lowp", "bf16"),
    }
    assert all(case.incumbent.name == "single_fused_lowp" for case in cases)
    assert all(case.incumbent.native and case.incumbent.complete_device_scope
               for case in cases)
    assert all(not case.candidate.native for case in cases)
    assert all(case.rtol == pytest.approx(6e-2) and case.atol == pytest.approx(6e-2)
               for case in cases)


def test_retune_ledger_does_not_promote_reference_evidence(monkeypatch):
    module = _module()
    context = {
        "device": "apple7", "physical_device": "fake", "os_version": "1",
        "sdk_version": "1", "compiler_fingerprint": "sha256:c",
        "runtime_fingerprint": "sha256:r",
    }
    row = {
        "family": "reduction", "op": "retune_reduce_sum", "shape": "4",
        "dtype": "f32", "device": "apple7", "reps": 2,
        "numerically_validated": True,
        "telemetry": {
            "end_to_end_median_ns": 100,
            "device_time_median_ns": None,
            "paired_trial_end_to_end_medians_ns": [100, 100, 100],
            "paired_trial_device_medians_ns": None,
            "device_time_coverage": 0.0,
            "resources": {"api": "test"},
        },
    }
    incumbent = {**row, "route": "mpsgraph", "native_dispatched": True,
                 "execution_kind": "native_gpu"}
    reference = {**row, "route": "numpy_reference", "native_dispatched": False,
                 "execution_kind": "reference_cpu",
                 "telemetry": {**row["telemetry"], "end_to_end_median_ns": 1,
                               "paired_trial_end_to_end_medians_ns": [1, 1, 1]}}
    report = {"schema_version": 1, "context": context,
              "runs": [incumbent, reference]}
    ledger = module.build_strict_ledger([report, report])
    decision = next(row for row in ledger["decisions"]
                    if row["timing_domain"] == "end_to_end")
    assert decision["selected_route"] == "mpsgraph"
    assert decision["selected_evidence"]["provenance"] == "native_gpu"
    assert len(ledger["source_report_digests"]) == 2


def test_moe_dispatch_consumes_the_strict_exact_row(monkeypatch):
    from tessera import _apple_gpu_backend as backend
    from tessera import runtime
    from tessera.compiler import apple_route_selector

    called: list[tuple[int, ...]] = []

    def select(**kwargs):
        assert kwargs["op"] == "retune_moe_swiglu"
        assert kwargs["shape"] == "16x32x64x32_e4"
        return "single_fused"

    def fused(x, wg, wu, wd, expert_ids):
        called.append(tuple(x.shape))
        return np.zeros((x.shape[0], wd.shape[2]), np.float32)

    monkeypatch.setattr(apple_route_selector, "production_route_for", select)
    monkeypatch.setattr(backend, "gpu_moe_swiglu_block", fused)
    x = np.zeros((16, 32), np.float32)
    wg = np.zeros((4, 32, 64), np.float32)
    wu = np.zeros_like(wg)
    wd = np.zeros((4, 64, 32), np.float32)
    gs = np.full(4, 4, np.int64)
    out = runtime._apple_gpu_dispatch_moe_swiglu_block(
        (x, wg, wu, wd, gs), {"grouped_kind": "contiguous"}, np)
    assert called == [(16, 32)]
    assert out.shape == (16, 32)


def test_committed_retune_corpus_and_strict_ledger_are_consistent():
    from tessera.compiler.apple_route_selector import (
        AppleRouteContext, load_strict_route_ledger)

    root = Path(__file__).resolve().parents[2]
    corpus = json.loads((root / "benchmarks/baselines/apple7_legacy_retune_two_run.json")
                        .read_text(encoding="utf-8"))
    assert corpus["schema"] == "tessera.apple.legacy-retune.v1"
    assert len(corpus["reports"]) == 2
    assert {row["family"] for report in corpus["reports"]
            for row in report["runs"]} == {
        "grouped_gemm", "moe", "reduction", "kv_movement", "mla_decode",
        "decode",
    }
    ledger_path = root / "benchmarks/baselines/apple_strict_route_ledger.json"
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    context = AppleRouteContext(**payload["context"])
    measured = datetime.fromisoformat(payload["measured_at"].replace("Z", "+00:00"))
    admitted = load_strict_route_ledger(
        ledger_path, context=context, now=measured + timedelta(days=1))
    assert admitted.rejected == ()
    assert len(admitted.routes) == 16
    assert admitted.routes[(
        "apple7", "retune_moe_swiglu", "16x32x64x32_e4", "f32",
        "end_to_end",
    )] == "composed"
    assert admitted.routes[(
        "apple7", "retune_mla_decode", "1x4x1x64x16x8x16x32", "f32",
        "end_to_end",
    )] == "explicit"


def test_committed_low_precision_moe_corpus_has_complete_native_intervals():
    root = Path(__file__).resolve().parents[2]
    corpus = json.loads((root / "benchmarks/baselines/apple7_lowp_moe_retune_two_run.json")
                        .read_text(encoding="utf-8"))
    assert len(corpus["reports"]) == 2
    native_rows = [
        row for report in corpus["reports"] for row in report["runs"]
        if row["route"] == "single_fused_lowp"
    ]
    assert {(row["dtype"], row["shape"]) for row in native_rows} == {
        ("f16", "16x32x64x32_e4"), ("bf16", "16x32x64x32_e4"),
        ("f16", "32x64x128x64_e4"), ("bf16", "32x64x128x64_e4"),
    }
    assert all(row["native_dispatched"] and row["numerically_validated"]
               for row in native_rows)
    assert all(row["telemetry"]["device_time_scope"] ==
               "complete_route_command_buffer" and
               row["telemetry"]["device_time_coverage"] == 1.0
               for row in native_rows)
    ledger = json.loads((root / "benchmarks/baselines/apple7_lowp_moe_strict_v2_route_ledger.json")
                        .read_text(encoding="utf-8"))
    assert len(ledger["decisions"]) == 8
    assert not ledger["ineligible_decisions"]
    assert {row["selected_route"] for row in ledger["decisions"]} == {
        "single_fused_lowp"}


@pytest.mark.hardware_apple_gpu
def test_strict_retune_ledger_admits_on_its_exact_live_apple_host():
    from tests._support.apple import require_apple_metal
    from tessera.compiler.apple_route_selector import (
        live_apple_route_context, load_strict_route_ledger,
        production_route_decision)

    require_apple_metal()
    root = Path(__file__).resolve().parents[2]
    path = root / "benchmarks/baselines/apple_strict_route_ledger.json"
    context = live_apple_route_context()
    admitted = load_strict_route_ledger(path, context=context)
    assert admitted.rejected == ()
    assert len(admitted.routes) == 16
    decision = production_route_decision(
        op="retune_moe_swiglu", shape="16x32x64x32_e4", dtype="f32",
        incumbent_route="composed", context=context, ledger_path=path)
    assert decision.route == "composed"
    assert decision.selected_from_ledger is True
    assert decision.citation is not None and "#decision[" in decision.citation
