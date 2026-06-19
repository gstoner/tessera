from __future__ import annotations

from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, TENSOR_OPAQUE
from tessera.compiler.schedule_ir import lower_graph_to_schedule_ir
from tessera.compiler.target_ir import lower_tile_to_target_ir
from tessera.compiler.tile_ir import lower_schedule_to_tile_ir


def _msa_graph(**attrs):
    return GraphIRModule(functions=[
        GraphIRFunction(
            "decode",
            args=[
                IRArg("Q", TENSOR_OPAQUE),
                IRArg("K", TENSOR_OPAQUE),
                IRArg("V", TENSOR_OPAQUE),
            ],
            body=[
                IROp(
                    "O",
                    "tessera.msa_sparse_attention",
                    ["%Q", "%K", "%V"],
                    ["tensor<*xbf16>", "tensor<*xbf16>", "tensor<*xbf16>"],
                    "tensor<*xbf16>",
                    kwargs=attrs,
                )
            ],
            return_values=["%O"],
        )
    ])


def test_msa_graph_lowers_to_kv_outer_sparse_schedule_contract():
    schedule = lower_graph_to_schedule_ir(
        _msa_graph(
            block_size=64,
            top_k_blocks=8,
            num_attention_heads=8,
            num_kv_heads=2,
            head_dim=128,
            mode="decode",
            tile_q=1,
            tile_kv=128,
        ),
        target_kind="nvidia_sm90",
    )

    assert schedule.verify().ok
    text = schedule.to_mlir()
    assert "schedule.attn.kv_outer_sparse" in text
    assert 'target_op = "tessera.attn.msa_kv_outer_sparse"' in text
    assert 'block_ids_layout = "B,Hkv,Sq,top_k"' in text
    assert 'kv_traversal = "kv_outer"' in text
    assert 'mode = "decode"' in text
    assert "top_k = 8" in text
    assert "block_size = 64" in text
    assert "gqa_group_size = 4" in text


def test_msa_kv_outer_sparse_reaches_tile_and_nvidia_target_ir():
    schedule = lower_graph_to_schedule_ir(
        _msa_graph(
            block_size=128,
            top_k_blocks=16,
            num_attention_heads=64,
            num_kv_heads=4,
            head_dim=128,
            mode="decode",
            tile_q=1,
            tile_kv=128,
        ),
        target_kind="nvidia_sm90",
    )
    tile = lower_schedule_to_tile_ir(schedule, target_kind="nvidia_sm90")

    assert tile.verify().ok
    tile_text = tile.to_mlir()
    assert "tessera.attn.msa_kv_outer_sparse" in tile_text
    assert 'selected_block_layout = "B,Hkv,Sq,top_k"' in tile_text

    target = lower_tile_to_target_ir(tile, target_kind="nvidia_sm90")
    assert target.verify().ok
    target_text = target.to_mlir()
    assert "tessera_nvidia.cuda_kernel" in target_text
    assert 'kernel = "msa_kv_outer_sparse"' in target_text
    assert 'status = "artifact_only"' in target_text
    assert 'mode = "decode"' in target_text
    assert 'block_ids_layout = "B,Hkv,Sq,top_k"' in target_text
    assert 'kv_traversal = "kv_outer"' in target_text
    assert "gqa_group_size = 16" in target_text
