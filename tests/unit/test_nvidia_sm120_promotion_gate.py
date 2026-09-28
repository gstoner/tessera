"""Mechanical honesty gate for sm_120 differentiation promotions."""
from __future__ import annotations

import json
from pathlib import Path

from tessera.compiler.execution_matrix import lookup


ROOT = Path(__file__).resolve().parents[2]
TYPED = (ROOT / "src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia"
         / "sm120_differentiation_target_ir.mlir")
BASELINE = ROOT / "benchmarks/baselines/nvidia_sm120_hot_paths.json"
DASHBOARD = ROOT / "docs/audit/backend/nvidia/SM120_DIFFERENTIATION_DASHBOARD.md"


def test_promoted_lanes_have_typed_artifact_benchmark_and_dashboard_evidence():
    typed = TYPED.read_text()
    for op in ("mma_fused", "mma_attention", "fpquant",
               "nvfp4_block_scale_mma"):
        assert f"tessera_nvidia.{op}" in typed

    rows = json.loads(BASELINE.read_text())["rows"]
    modes = {row["mode"] for row in rows}
    assert {"mma_sync_fused", "mma_sync_attention", "cuda_fpquant"} <= modes
    assert all(row["median_ms"] > 0 and row["max_latency_ms"] > row["median_ms"]
               for row in rows)

    dashboard = DASHBOARD.read_text()
    assert "**runtime-promoted; Target-IR column open**" in dashboard
    assert "**runtime-promoted (storage); Target-IR column open**" in dashboard
    assert "**blocked at runtime-dispatch gate**" in dashboard
    assert "passes** fixed-tile unit and non-uniform scale oracle" in dashboard


def test_fixture_only_target_ops_are_not_cited_as_promotion_evidence():
    """TILE-LATENT-DEFECTS-2026-09-27: a lit fixture that merely parses an op
    is not "Typed IR + verifier" evidence for a lane whose kernels run
    through Python candidates. While no compiler path produces
    ``mma_fused`` / ``mma_attention`` / ``fpquant``, the dashboard must mark
    that column open and must not call those lanes fully **promoted**. When a
    producer lands (ODS triage WIRE slice 7), update the dashboard and this
    test together.
    """
    producers = [
        ROOT / "src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
        ROOT / "python/tessera/compiler/target_ir.py",
    ]
    produced = "\n".join(path.read_text() for path in producers)
    dashboard = DASHBOARD.read_text()
    for op in ("mma_fused", "mma_attention", "fpquant"):
        if f"tessera_nvidia.{op}" in produced:
            continue  # a producer exists; this gate no longer applies to it
        assert f"`tessera_nvidia.{op}` is fixture-only" in dashboard, op
    assert "| **promoted** |" not in dashboard
    assert "| **promoted (storage)** |" not in dashboard


def test_fpquant_provenance_is_native_and_nvfp4_is_not_runtime_promoted():
    row = lookup("nvidia_sm120", "nvidia_fpquant_compiled")
    assert row is not None
    assert row.executable and row.execution_kind == "native_gpu"
    assert row.device_proof == "device_verified_jit"
    assert row.numerical_fixture == "tests/device/nvidia/test_fpquant.py"

    # An emitted PTX kernel is not a RuntimeArtifact compiler path.  This remains
    # None until a launch ABI and passing direct comparison are both present.
    assert lookup("nvidia_sm120", "nvidia_nvfp4_block_scale_mma") is None
