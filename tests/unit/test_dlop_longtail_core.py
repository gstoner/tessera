"""DLOP-Bench-style long-tail operator fusion benchmark — dispatch-count +
metamorphic equivalence (fused ≡ eager-decomposed)."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from benchmarks.common import ExecutionKind, RuntimeStatus  # noqa: E402
from benchmarks.dlop_longtail_core import (  # noqa: E402
    LONGTAIL_OPS,
    DlopLongtailConfig,
    build_report,
    run_core,
    telemetry,
)


def test_every_composite_is_metamorphically_equivalent():
    # the fused reference must equal the eager decomposition for every op
    rows = run_core(DlopLongtailConfig(seed=3))
    assert rows and all(r.correctness.passed for r in rows)
    assert all(r.execution_kind is ExecutionKind.REFERENCE for r in rows)
    assert all(r.runtime_status is RuntimeStatus.EXECUTABLE for r in rows)


def test_dispatch_reduction_reflects_decomposition():
    rows = {r.operator.name: r for r in run_core()}
    # attention_block: 4 primitives → 1 fused (flash_attn) → 4× reduction
    att = rows["attention_block"]
    assert att.metrics["eager_dispatches"] == 4
    assert att.metrics["fused_dispatches"] == 1
    assert att.metrics["dispatch_reduction_x"] == 4.0
    # swiglu_ffn: 5 primitives → 1 fused (moe_swiglu_block)
    assert rows["swiglu_ffn"].metrics["dispatch_reduction_x"] == 5.0


def test_host_composed_op_has_no_fused_lane():
    # bbox2delta has no dedicated fused kernel — honest 1× (no reduction) and a
    # named gap, not a fake fusion win.
    bbox = {r.operator.name: r for r in run_core()}["bbox2delta"]
    assert bbox.metrics["fused_apple_gpu_lane"] == "none"
    assert bbox.metrics["dispatch_reduction_x"] == 1.0
    assert "host-composed" in bbox.reason


def test_fusible_ops_point_at_real_apple_gpu_lanes():
    lanes = {r.operator.name: r.metrics["fused_apple_gpu_lane"] for r in run_core()}
    assert lanes["attention_block"] == "flash_attn"
    assert lanes["swiglu_ffn"] == "moe_swiglu_block"


def test_build_report_summary():
    report = build_report()
    assert report["ops"] == len(LONGTAIL_OPS)
    assert report["all_metamorphic_equivalent"] is True
    assert report["fusible_ops"] == 3                 # attention, swiglu, rmsnorm
    assert report["max_dispatch_reduction_x"] >= 5.0
    assert "bbox2delta" in report["host_composed_gaps"]


def test_telemetry_one_event_per_row():
    rows = run_core()
    events = telemetry(rows)
    assert len(events) == len(rows)
    assert all(isinstance(e, dict) for e in events)


def test_runs_are_deterministic():
    a = build_report(run_core(DlopLongtailConfig(seed=7)))
    b = build_report(run_core(DlopLongtailConfig(seed=7)))
    assert a == b


def test_synthesized_fusion_rows_are_metamorphic_and_collapse_dispatch():
    from benchmarks.dlop_longtail_core import synthesized_fusion_rows

    rows = synthesized_fusion_rows(seed=3)
    assert rows and all(r.correctness.passed for r in rows)         # synth == unfused
    for r in rows:
        assert r.metrics["fused_dispatches"] == 1
        assert r.metrics["dispatch_reduction_x"] == 1 + len(r.operator.name.split("_")) - 2 \
            or r.metrics["dispatch_reduction_x"] >= 2.0              # >=2x collapse
        assert r.metrics["metamorphic_equivalent"] is True
    # at least one chain is NOT in the hand-written catalog — the generalization
    assert any(not r.metrics["in_handwritten_catalog"] for r in rows)
