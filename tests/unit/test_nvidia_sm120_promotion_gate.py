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
    assert "**promoted**" in dashboard
    assert "**promoted (storage)**" in dashboard
    assert "**blocked at runtime-dispatch gate**" in dashboard
    assert "passes** fixed-tile unit and non-uniform scale oracle" in dashboard


def test_fpquant_provenance_is_native_and_nvfp4_is_not_runtime_promoted():
    row = lookup("nvidia_sm120", "nvidia_fpquant_compiled")
    assert row is not None
    assert row.executable and row.execution_kind == "native_gpu"
    assert row.device_proof == "device_verified_jit"
    assert row.numerical_fixture == "tests/unit/test_nvidia_fpquant_compiled.py"

    # An emitted PTX kernel is not a RuntimeArtifact compiler path.  This remains
    # None until a launch ABI and passing direct comparison are both present.
    assert lookup("nvidia_sm120", "nvidia_nvfp4_block_scale_mma") is None
