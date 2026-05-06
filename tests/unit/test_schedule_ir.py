from __future__ import annotations

import pytest

from tessera.compiler.frontend import lower_text_to_graph_ir
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRMesh, GraphIRModule, IRArg, IROp, TENSOR_OPAQUE
from tessera.compiler.schedule_ir import (
    ScheduleFunction,
    ScheduleIRModule,
    ScheduleIRVerificationError,
    ScheduleOp,
    lower_graph_to_schedule_ir,
)


def test_schedule_ir_renders_core_mesh_pipeline_stage_yield_ops():
    module = ScheduleIRModule(functions=[
        ScheduleFunction(
            "main",
            body=[
                ScheduleOp("schedule.mesh.define", {"sym_name": "M", "dims": [2, 4], "axis_names": ["dp", "tp"]}),
                ScheduleOp(
                    "schedule.mesh.region",
                    {"mesh": "M", "axis": "dp"},
                    body=[
                        ScheduleOp(
                            "schedule.pipeline.region",
                            {"schedule": "1f1b", "micro_batches": 4},
                            body=[
                                ScheduleOp("schedule.stage", {"devices": ["gpu:0"]}, body=[ScheduleOp("schedule.yield")]),
                                ScheduleOp("schedule.yield"),
                            ],
                        ),
                        ScheduleOp("schedule.yield"),
                    ],
                ),
            ],
        )
    ])
    assert module.verify().ok
    text = module.to_mlir()
    assert "schedule.mesh.define" in text
    assert "schedule.mesh.region" in text
    assert "schedule.pipeline.region" in text
    assert "schedule.stage" in text
    assert "schedule.yield" in text


def test_schedule_ir_verifier_rejects_bad_regions_and_pipeline_attrs():
    no_yield = ScheduleIRModule(functions=[
        ScheduleFunction(
            "bad",
            body=[
                ScheduleOp("schedule.mesh.define", {"sym_name": "M", "dims": [2], "axis_names": ["dp"]}),
                ScheduleOp("schedule.mesh.region", {"mesh": "M", "axis": "dp"}, body=[]),
            ],
        )
    ])
    result = no_yield.verify()
    assert not result.ok
    assert "SCHEDULE_IR_REGION_TERMINATOR" in result.format()

    bad_pipeline = ScheduleIRModule(functions=[
        ScheduleFunction(
            "bad_pipe",
            body=[
                ScheduleOp(
                    "schedule.pipeline.region",
                    {"schedule": "", "micro_batches": 0},
                    body=[ScheduleOp("schedule.yield")],
                )
            ],
        )
    ])
    result = bad_pipeline.verify()
    assert not result.ok
    assert "SCHEDULE_IR_PIPELINE_SCHEDULE" in result.format()
    assert "SCHEDULE_IR_PIPELINE_MICRO_BATCHES" in result.format()


def test_lower_graph_to_schedule_ir_preserves_meshes_and_tiles_matmul():
    graph = GraphIRModule(
        meshes=[GraphIRMesh("data", axes=("dp",), shape=(2,))],
        functions=[
            GraphIRFunction(
                "main",
                args=[IRArg("A", TENSOR_OPAQUE), IRArg("B", TENSOR_OPAQUE)],
                body=[
                    IROp("C", "tessera.matmul", ["%A", "%B"], ["tensor<*x?>", "tensor<*x?>"], "tensor<*x?>")
                ],
            )
        ],
    )
    schedule = lower_graph_to_schedule_ir(graph, tile=(64, 32, 16), target_kind="cpu")
    assert schedule.verify().ok
    text = schedule.to_mlir()
    assert "schedule.mesh.define" in text
    assert "schedule.mesh.region" in text
    assert "schedule.tile" in text
    assert "schedule.knob" in text
    assert "tile_m = 64" in text
    assert "tile_n = 32" in text
    assert "tile_k = 16" in text
    assert 'cost_model = "roofline"' in text
    assert "movement" in text


def test_textual_frontend_schedule_pipeline_lowers_to_region_stage_yield():
    graph = lower_text_to_graph_ir("""
    module demo {
      mesh m = mesh<axes=[dp], shape=[2]>;
      func main(A: tensor<4x8xfp32>, B: tensor<8x2xfp32>) -> tensor<4x2xfp32> {
        C = op.matmul(A, B);
        schedule.pipeline(C) @{schedule = "gpipe", micro_batches = 2, depth = 2};
        return C;
      }
    }
    """)
    schedule = lower_graph_to_schedule_ir(graph, tile=(4, 2, 8))
    assert schedule.verify().ok
    text = schedule.to_mlir()
    assert "schedule.pipeline.region" in text
    assert "micro_batches = 2" in text
    assert text.count("schedule.stage") == 2
    assert "schedule.mesh.region" in text


def test_schedule_ir_to_mlir_raises_on_invalid_module():
    module = ScheduleIRModule(functions=[
        ScheduleFunction("bad", body=[ScheduleOp("schedule.yield")])
    ])
    with pytest.raises(ScheduleIRVerificationError, match="SCHEDULE_IR_YIELD_OUTSIDE_REGION"):
        module.to_mlir()
