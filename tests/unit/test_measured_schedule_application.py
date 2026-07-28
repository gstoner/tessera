from __future__ import annotations

import pytest

from tessera.compiler.frontend import lower_text_to_graph_ir
from tessera.compiler.matmul_pipeline import build_cpu_plan
from tessera.compiler.autotune_v2 import TuningConfig, TuningResult


SOURCE = """
module measured {
  func gemm(A: tensor<64x32xfp32>, B: tensor<32x96xfp32>)
      -> tensor<64x96xfp32> {
    C = op.matmul(A, B);
    return C;
  }
}
"""


def test_measured_schedule_changes_physical_schedule_and_tile_ir() -> None:
    module = lower_text_to_graph_ir(SOURCE)
    result = TuningResult(
        config=TuningConfig(64, 32, 16, 2, 3),
        latency_ms=0.031,
        tflops=7.0,
        method="measured",
    )
    measured = result.to_measured_schedule(target="rocm")
    plan = build_cpu_plan(
        module, target_kind="rocm", measured_schedule=measured
    )
    assert plan is not None
    assert plan.tile == (64, 32, 16)
    assert plan.selected_schedule["method"] == "measured"
    assert "tile_m = 64" in plan.schedule_ir
    assert "num_warps = 2" in plan.schedule_ir
    assert "num_stages = 3" in plan.schedule_ir
    assert 'cost_model = "measured"' in plan.schedule_ir
    assert "tile_m = 64" in plan.tile_ir


def test_modeled_tuning_result_cannot_drive_lowering() -> None:
    result = TuningResult(
        config=TuningConfig(64, 32, 16),
        latency_ms=0.031,
        tflops=7.0,
        method="roofline",
    )
    with pytest.raises(ValueError, match="device-measured"):
        result.to_measured_schedule(target="rocm")


def test_measured_schedule_is_target_and_evidence_gated() -> None:
    module = lower_text_to_graph_ir(SOURCE)
    with pytest.raises(ValueError, match="method='measured'"):
        build_cpu_plan(
            module,
            target_kind="rocm",
            measured_schedule={
                "method": "roofline",
                "target": "rocm",
                "latency_ms": 1.0,
                "config": {},
            },
        )
    with pytest.raises(ValueError, match="does not match"):
        build_cpu_plan(
            module,
            target_kind="rocm",
            measured_schedule={
                "method": "measured",
                "target": "nvidia_sm120",
                "latency_ms": 1.0,
                "config": {},
            },
        )
