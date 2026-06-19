from __future__ import annotations

from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, TENSOR_OPAQUE
from tessera.compiler.schedule_ir import lower_graph_to_schedule_ir
from tessera.compiler.target_ir import lower_tile_to_target_ir
from tessera.compiler.tile_ir import lower_schedule_to_tile_ir


def _multimodal_jepa_graph():
    return GraphIRModule(functions=[
        GraphIRFunction(
            "multimodal_jepa",
            args=[
                IRArg("text", TENSOR_OPAQUE),
                IRArg("media", TENSOR_OPAQUE),
                IRArg("target", TENSOR_OPAQUE),
            ],
            body=[
                IROp(
                    "spliced",
                    "tessera.splice_embeddings",
                    ["%text", "%media"],
                    ["tensor<*xbf16>", "tensor<*xbf16>"],
                    "tensor<*xbf16>",
                    kwargs={"execution": "projected_embeddings"},
                ),
                IROp(
                    "mask",
                    "tessera.jepa_mask_blocks_2d",
                    ["%spliced"],
                    ["tensor<*xbf16>"],
                    "tensor<*xi1>",
                    kwargs={"block_size": 16, "mask_ratio": 0.4},
                ),
                IROp(
                    "ctx",
                    "tessera.jepa.gather_context",
                    ["%spliced", "%mask"],
                    ["tensor<*xbf16>", "tensor<*xi1>"],
                    "tensor<*xbf16>",
                ),
                IROp(
                    "pred",
                    "tessera.jepa.latent_predict",
                    ["%ctx"],
                    ["tensor<*xbf16>"],
                    "tensor<*xbf16>",
                    kwargs={"latent_space": "continuous"},
                ),
                IROp(
                    "loss",
                    "tessera.jepa.l2_loss",
                    ["%pred", "%target"],
                    ["tensor<*xbf16>", "tensor<*xbf16>"],
                    "tensor<xf32>",
                ),
            ],
            return_values=["%loss"],
        )
    ])


def test_multimodal_and_jepa_graph_contracts_reach_schedule_and_tile_ir():
    schedule = lower_graph_to_schedule_ir(_multimodal_jepa_graph(), target_kind="nvidia_sm90")

    assert schedule.verify().ok
    schedule_text = schedule.to_mlir()
    assert "schedule.media.splice_embeddings" in schedule_text
    assert 'execution = "projected_embeddings"' in schedule_text
    assert "schedule.jepa.mask_blocks_2d" in schedule_text
    assert "schedule.jepa.gather_context" in schedule_text
    assert "schedule.jepa.latent_predict" in schedule_text
    assert "schedule.jepa.l2_loss" in schedule_text
    assert 'latent_space = "continuous"' in schedule_text

    tile = lower_schedule_to_tile_ir(schedule, target_kind="nvidia_sm90")

    assert tile.verify().ok
    tile_text = tile.to_mlir()
    assert "tile.media.splice_embeddings" in tile_text
    assert 'lowering = "multimodal_contract"' in tile_text
    assert "tile.jepa.mask_blocks_2d" in tile_text
    assert 'lowering = "latent_prediction_contract"' in tile_text


def test_multimodal_and_jepa_contracts_reach_nvidia_target_ir():
    schedule = lower_graph_to_schedule_ir(_multimodal_jepa_graph(), target_kind="nvidia_sm90")
    tile = lower_schedule_to_tile_ir(schedule, target_kind="nvidia_sm90")
    target = lower_tile_to_target_ir(tile, target_kind="nvidia_sm90")

    assert target.verify().ok
    text = target.to_mlir()
    assert 'kernel = "splice_embeddings_contract"' in text
    assert 'contract = "splice_embeddings"' in text
    assert 'execution = "projected_embeddings"' in text
    assert 'kernel = "jepa_mask_blocks_2d_contract"' in text
    assert 'kernel = "jepa_latent_predict_contract"' in text
    assert 'kernel = "jepa_l2_loss_contract"' in text
    assert 'latent_space = "continuous"' in text
