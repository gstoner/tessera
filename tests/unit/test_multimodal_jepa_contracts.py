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
                IRArg("image", TENSOR_OPAQUE),
                IRArg("video", TENSOR_OPAQUE),
                IRArg("online_state", TENSOR_OPAQUE),
                IRArg("target_state", TENSOR_OPAQUE),
                IRArg("target", TENSOR_OPAQUE),
            ],
            body=[
                IROp(
                    "image_pixels",
                    "tessera.image_preprocess",
                    ["%image"],
                    ["tensor<*xf32>"],
                    "tensor<*xf32>",
                    kwargs={"image_size": 224, "patch_size": 14, "status": "composed"},
                ),
                IROp(
                    "video_pixels",
                    "tessera.video_frame_sample",
                    ["%video"],
                    ["tensor<*xf32>"],
                    "tensor<*xf32>",
                    kwargs={"frames": 4, "temporal_patch_size": 2, "status": "composed"},
                ),
                IROp(
                    "image_patches",
                    "tessera.patch_embed",
                    ["%image_pixels"],
                    ["tensor<*xf32>"],
                    "tensor<*xbf16>",
                    kwargs={"patch_size": 14, "status": "composed"},
                ),
                IROp(
                    "video_patches",
                    "tessera.patch_embed",
                    ["%video_pixels"],
                    ["tensor<*xf32>"],
                    "tensor<*xbf16>",
                    kwargs={"patch_size": 14, "status": "composed"},
                ),
                IROp(
                    "image_tokens",
                    "tessera.patch_merge",
                    ["%image_patches"],
                    ["tensor<*xbf16>"],
                    "tensor<*xbf16>",
                    kwargs={"spatial_merge_size": 2, "status": "composed"},
                ),
                IROp(
                    "video_tokens",
                    "tessera.patch_merge",
                    ["%video_patches"],
                    ["tensor<*xbf16>"],
                    "tensor<*xbf16>",
                    kwargs={"spatial_merge_size": 2, "temporal_patch_size": 2, "status": "composed"},
                ),
                IROp(
                    "image_projected",
                    "tessera.media_project",
                    ["%image_tokens"],
                    ["tensor<*xbf16>"],
                    "tensor<*xbf16>",
                    kwargs={"projector": "minimax_m3", "status": "composed"},
                ),
                IROp(
                    "video_projected",
                    "tessera.media_project",
                    ["%video_tokens"],
                    ["tensor<*xbf16>"],
                    "tensor<*xbf16>",
                    kwargs={"projector": "minimax_m3", "status": "composed"},
                ),
                IROp(
                    "spliced",
                    "tessera.splice_embeddings",
                    ["%text", "%image_projected", "%video_projected"],
                    ["tensor<*xbf16>", "tensor<*xbf16>", "tensor<*xbf16>"],
                    "tensor<*xbf16>",
                    kwargs={"execution": "projected_embeddings", "status": "composed"},
                ),
                IROp(
                    "mask",
                    "tessera.jepa.mask_blocks_2d",
                    ["%spliced"],
                    ["tensor<*xbf16>"],
                    "tensor<*xi1>",
                    kwargs={"block_size": 16, "mask_ratio": 0.4, "seed": 123, "rng": "philox"},
                ),
                IROp(
                    "ctx",
                    "tessera.jepa.gather_context",
                    ["%spliced", "%mask"],
                    ["tensor<*xbf16>", "tensor<*xi1>"],
                    "tensor<*xbf16>",
                ),
                IROp(
                    "tgt",
                    "tessera.jepa.gather_targets",
                    ["%spliced", "%mask"],
                    ["tensor<*xbf16>", "tensor<*xi1>"],
                    "tensor<*xbf16>",
                ),
                IROp(
                    "stop_tgt",
                    "tessera.jepa.stop_gradient",
                    ["%tgt"],
                    ["tensor<*xbf16>"],
                    "tensor<*xbf16>",
                    kwargs={"gradient": "blocked"},
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
                    ["%pred", "%stop_tgt"],
                    ["tensor<*xbf16>", "tensor<*xbf16>"],
                    "tensor<xf32>",
                ),
                IROp(
                    "ema",
                    "tessera.jepa.ema_update",
                    ["%target_state", "%online_state"],
                    ["tensor<*xbf16>", "tensor<*xbf16>"],
                    "tensor<*xbf16>",
                    kwargs={"decay": 0.996, "stateful": True},
                ),
                IROp(
                    "train_state",
                    "tessera.jepa.train_step",
                    ["%loss", "%ema"],
                    ["tensor<xf32>", "tensor<*xbf16>"],
                    "tensor<*xbf16>",
                    kwargs={"mask_rng_deterministic": True, "ema_update": True, "latent_loss": "l2"},
                ),
                IROp(
                    "decoded",
                    "tessera.jepa.selective_decode",
                    ["%pred", "%target"],
                    ["tensor<*xbf16>", "tensor<*xbf16>"],
                    "tensor<*xbf16>",
                    kwargs={"gating": "latent_score", "optional": True, "branches": ("retrieval", "classification", "decode")},
                ),
            ],
            return_values=["%loss", "%train_state", "%decoded"],
        )
    ])


def test_multimodal_and_jepa_graph_contracts_reach_schedule_and_tile_ir():
    schedule = lower_graph_to_schedule_ir(_multimodal_jepa_graph(), target_kind="nvidia_sm90")

    assert schedule.verify().ok
    schedule_text = schedule.to_mlir()
    assert "schedule.media.splice_embeddings" in schedule_text
    assert "schedule.media.image_preprocess" in schedule_text
    assert "schedule.media.video_frame_sample" in schedule_text
    assert "schedule.media.patch_embed" in schedule_text
    assert "schedule.media.patch_merge" in schedule_text
    assert "schedule.media.media_project" in schedule_text
    assert 'execution = "projected_embeddings"' in schedule_text
    assert "schedule.jepa.mask_blocks_2d" in schedule_text
    assert "schedule.jepa.gather_context" in schedule_text
    assert "schedule.jepa.gather_targets" in schedule_text
    assert "schedule.jepa.stop_gradient" in schedule_text
    assert "schedule.jepa.ema_update" in schedule_text
    assert "schedule.jepa.latent_predict" in schedule_text
    assert "schedule.jepa.l2_loss" in schedule_text
    assert "schedule.jepa.train_step" in schedule_text
    assert "schedule.jepa.selective_decode" in schedule_text
    assert 'latent_space = "continuous"' in schedule_text
    assert "mask_rng_deterministic = true" in schedule_text

    tile = lower_schedule_to_tile_ir(schedule, target_kind="nvidia_sm90")

    assert tile.verify().ok
    tile_text = tile.to_mlir()
    assert "tile.media.patch_embed" in tile_text
    assert "tile.media.patch_merge" in tile_text
    assert "tile.media.media_project" in tile_text
    assert "tile.media.splice_embeddings" in tile_text
    assert 'lowering = "multimodal_contract"' in tile_text
    assert "tile.jepa.mask_blocks_2d" in tile_text
    assert "tile.jepa.train_step" in tile_text
    assert "tile.jepa.selective_decode" in tile_text
    assert 'lowering = "latent_prediction_contract"' in tile_text


def test_multimodal_and_jepa_contracts_reach_nvidia_target_ir():
    schedule = lower_graph_to_schedule_ir(_multimodal_jepa_graph(), target_kind="nvidia_sm90")
    tile = lower_schedule_to_tile_ir(schedule, target_kind="nvidia_sm90")
    target = lower_tile_to_target_ir(tile, target_kind="nvidia_sm90")

    assert target.verify().ok
    text = target.to_mlir()
    assert 'kernel = "splice_embeddings_contract"' in text
    assert 'kernel = "patch_embed_contract"' in text
    assert 'kernel = "patch_merge_contract"' in text
    assert 'kernel = "media_project_contract"' in text
    assert 'contract = "splice_embeddings"' in text
    assert 'execution = "projected_embeddings"' in text
    assert 'kernel = "jepa_mask_blocks_2d_contract"' in text
    assert 'kernel = "jepa_ema_update_contract"' in text
    assert 'kernel = "jepa_latent_predict_contract"' in text
    assert 'kernel = "jepa_l2_loss_contract"' in text
    assert 'kernel = "jepa_train_step_contract"' in text
    assert 'kernel = "jepa_selective_decode_contract"' in text
    assert 'latent_space = "continuous"' in text
    assert "decay = 0.996" in text
    assert "stateful = true" in text
    assert 'gating = "latent_score"' in text
    assert "optional = true" in text
    assert 'branches = ["retrieval", "classification", "decode"]' in text
    assert "mask_rng_deterministic = true" in text
    assert "ema_update = true" in text
    assert 'latent_loss = "l2"' in text
