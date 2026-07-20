from __future__ import annotations

import pytest

from tessera.compiler.frontend import lower_text_to_graph_ir
from tessera.compiler.schedule_ir import ScheduleFunction, ScheduleIRModule, ScheduleOp, lower_graph_to_schedule_ir
from tessera.compiler.tile_ir import (
    TileFunction,
    TileIRModule,
    TileIRVerificationError,
    TileOp,
    lower_schedule_to_tile_ir,
)


def test_tile_ir_renders_core_tile_attention_and_queue_ops():
    module = TileIRModule(functions=[
        TileFunction(
            "fa4",
            body=[
                TileOp("tessera.queue.create", {"queue_id": 0, "depth": 2}),
                TileOp("tile.async_copy", {"source": "tessera.flash_attn", "result": "O", "ordinal": 0, "stage": 0, "vector": 16}),
                TileOp("tessera.queue.push", {"queue_id": 0, "stage": 0}),
                TileOp("tessera.queue.pop", {"queue_id": 0, "stage": 0}),
                TileOp("tessera_attn.online_softmax", {"source": "tessera.flash_attn", "result": "O", "ordinal": 0, "policy": "safe"}),
                TileOp("tile.wait_async", {"stage": 0}),
                TileOp("tessera.queue.barrier", {"queue_id": 0, "scope": "warpgroup"}),
            ],
        )
    ])
    assert module.verify().ok
    text = module.to_mlir()
    assert "tessera.queue.create" in text
    assert "tile.async_copy" in text
    assert "tessera.queue.push" in text
    assert "tessera.queue.pop" in text
    assert "tessera_attn.online_softmax" in text
    assert "tile.wait_async" in text
    assert "tessera.queue.barrier" in text


def test_tile_ir_verifier_rejects_bad_async_queue_and_attention_contracts():
    module = TileIRModule(functions=[
        TileFunction(
            "bad",
            body=[
                TileOp("tile.async_copy", {"stage": -1, "vector": 0}),
                TileOp("tessera.queue.pop", {"queue_id": 7}),
                TileOp("tessera_attn.online_softmax", {"source": "tessera.flash_attn"}),
            ],
        )
    ])
    result = module.verify()
    assert not result.ok
    text = result.format()
    assert "TILE_IR_ASYNC_STAGE" in text
    assert "TILE_IR_ASYNC_VECTOR" in text
    assert "TILE_IR_UNDEFINED_QUEUE" in text
    assert "TILE_IR_ATTN_POLICY" in text


def test_lower_schedule_to_tile_ir_materializes_matmul_and_prefetch():
    schedule = ScheduleIRModule(functions=[
        ScheduleFunction(
            "main",
            body=[
                ScheduleOp("schedule.tile", {"source": "tessera.matmul", "result": "C", "ordinal": 0, "tile_m": 64, "tile_n": 32, "tile_k": 16}),
                ScheduleOp("schedule.prefetch", {"source": "tessera.kv_cache.append", "result": "Cache", "ordinal": 1, "into": "shared", "overlap": "compute"}),
            ],
        )
    ])
    tile = lower_schedule_to_tile_ir(schedule)
    assert tile.verify().ok
    text = tile.to_mlir()
    assert "tile.mma" in text
    assert "tile.async_copy" in text
    assert "tensor_core_mma" in text
    assert "shared_memory_bytes" in text
    assert "register_estimate" in text


def test_schedule_debug_artifact_and_barrier_marker_reach_tile_ir():
    schedule = ScheduleIRModule(functions=[
        ScheduleFunction(
            "main",
            body=[
                ScheduleOp("schedule.debug_artifact", {"name": "%x", "capture": "value_summary", "ordinal": 0}),
                ScheduleOp("schedule.marker", {"marker": "tessera.barrier", "ordinal": 1}),
            ],
        )
    ])

    tile = lower_schedule_to_tile_ir(schedule)

    assert tile.verify().ok
    text = tile.to_mlir()
    assert "tile.debug_artifact" in text
    assert "tile.debug_barrier" in text
    assert "tessera.queue.barrier" in text


def test_frontend_graph_schedule_tile_pipeline_for_flash_attention_has_fa4_and_queues():
    graph = lower_text_to_graph_ir("""
    module demo {
      func main(Q: tensor<?xfp32>, K: tensor<?xfp32>, V: tensor<?xfp32>) -> tensor<?xfp32> {
        O = op.flash_attn(Q, K, V) @{causal = true};
        return O;
      }
    }
    """)
    schedule = lower_graph_to_schedule_ir(graph, target_kind="nvidia_sm90")
    tile = lower_schedule_to_tile_ir(schedule, target_kind="nvidia_sm90")
    assert tile.verify().ok
    text = tile.to_mlir()
    assert "tile.async_copy" in text
    assert "tessera.queue.create" in text
    assert "tessera.queue.push" in text
    assert "tessera.queue.pop" in text
    assert "tessera_attn.scaled_dot_product" in text
    assert "tessera_attn.online_softmax" in text
    assert "tessera_attn.lse_save" in text
    assert "tessera_attn.attend_v" in text
    assert "tile.wait_async" in text
    assert "tile.debug_barrier" in text


def test_tile_ir_to_mlir_blocks_invalid_module():
    module = TileIRModule(functions=[
        TileFunction("bad", body=[TileOp("tile.wait_async", {"stage": -1})])
    ])
    with pytest.raises(TileIRVerificationError, match="TILE_IR_WAIT_STAGE"):
        module.to_mlir()


def test_schedule_to_tile_preserves_placement_metadata():
    """G1: Schedule→Tile must PRESERVE mesh / layout / schedule-plan metadata
    (previously dropped) so Tile IR explains real placement + movement, while the
    compute inside a mesh region still flows through. The metadata markers must
    NOT leak into Target lowering as spurious compute ops."""
    from tessera.compiler.target_ir import lower_tile_to_target_ir

    module = ScheduleIRModule(functions=[
        ScheduleFunction("main", body=[
            ScheduleOp("schedule.mesh.define",
                       {"sym_name": "M", "dims": [2], "axis_names": ["dp"]}),
            ScheduleOp("schedule.layout",
                       {"operands": ["x"], "layout": "row_major", "ordinal": 0}),
            ScheduleOp("schedule.mesh.region", {"mesh": "M", "axis": "dp"}, body=[
                ScheduleOp("schedule.pipeline.region",
                           {"schedule": "1f1b", "micro_batches": 2}, body=[
                               ScheduleOp("schedule.stage", {"devices": ["gpu:0"]},
                                          body=[ScheduleOp("schedule.yield")]),
                               ScheduleOp("schedule.yield"),
                           ]),
                ScheduleOp("schedule.yield"),
            ]),
            ScheduleOp("schedule.artifact", {"hash": "h", "arch": "cpu"}),
        ])
    ])
    assert module.verify().ok
    tile = lower_schedule_to_tile_ir(module, target_kind="cpu")
    text = tile.to_mlir()
    # placement / layout / plan metadata preserved into Tile IR
    assert "tile.mesh.define" in text
    assert "tile.layout" in text
    assert "tile.artifact" in text
    assert "tile.mesh.region" in text
    assert "dp" in text                       # mesh-axis association carried through

    # Target lowering still succeeds and does NOT emit the metadata markers as
    # compute (they're filtered like tile.debug_artifact).
    target_text = lower_tile_to_target_ir(tile, target_kind="cpu").to_mlir()
    assert "tile.mesh" not in target_text
    assert "tile.artifact" not in target_text
    assert "tile.layout" not in target_text
