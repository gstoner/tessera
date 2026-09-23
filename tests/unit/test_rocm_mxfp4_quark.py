"""Quark checkpoint metadata is not an executable gfx1201 W4A8 proof."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from tessera.compiler.rocm_mxfp4 import (
    MXFP4_CHECKPOINT_LAYOUT_V1,
    MXFP4_QUARK_REORDER_LAYOUT_V1,
    convert_weight_layout,
    mxfp4_weight_layout,
)
from tessera.compiler.rocm_mxfp4_native import select_mxfp4_route
from tessera.compiler.rocm_mxfp4_quark import assess_quark_mxfp4_projection


# Header/config excerpt from amd/Qwen3.8-27B-Quark-AWQ-MXFP4 at
# 5233554c5fa56afda40150556b95573c2d7d29c0.  The pinned full config
# SHA-256 is 4a139d3e01df039e17b4c8b8362f7944674d76051a8cd96f213c6bc332d69f55;
# the 219640-byte safetensors header SHA-256 is
# 01453e3d08a6275905b19dd15ac582fba065a551bc392ca22af0588d476f2083.
GATE = "model.language_model.layers.0.mlp.gate_proj"
DOWN = "model.language_model.layers.0.mlp.down_proj"
QUARK_CONFIG = {
    "quantization_config": {
        "quant_method": "quark",
        "exclude": ["model.visual.pos_embed", "lm_head"],
        "export": {"weight_format": "real_quantized", "pack_method": "reorder"},
        "global_quant_config": {
            "weight": {
                "dtype": "fp4", "group_size": 32, "scale_format": "e8m0",
                "is_dynamic": False,
            },
            "input_tensors": {
                "dtype": "fp4", "group_size": 32, "scale_format": "e8m0",
                "is_dynamic": True,
            },
        },
        "layer_quant_config": {},
        "layer_type_quant_config": {},
    }
}


@pytest.mark.parametrize(
    ("module", "weight_shape", "scale_shape", "expected_n", "expected_k"),
    [
        (GATE, [17408, 2560], [17408, 160], 17408, 5120),
        (DOWN, [5120, 8704], [5120, 544], 5120, 17408),
    ],
)
def test_pinned_qwen_projections_are_metadata_only(
    module: str,
    weight_shape: list[int],
    scale_shape: list[int],
    expected_n: int,
    expected_k: int,
) -> None:
    assessment = assess_quark_mxfp4_projection(
        QUARK_CONFIG,
        module=module,
        m=128,
        weight={"dtype": "U8", "shape": weight_shape},
        scale={"dtype": "U8", "shape": scale_shape},
    )
    assert (assessment.n, assessment.k) == (expected_n, expected_k)
    assert assessment.weight_bytes == expected_n * expected_k // 2
    assert assessment.scale_bytes == expected_n * expected_k // 32
    assert assessment.source_layout == MXFP4_QUARK_REORDER_LAYOUT_V1
    assert assessment.activation_storage == "mxfp4_e2m1"
    assert not assessment.route.accepted
    assert assessment.route.abi_id is None
    assert assessment.route.as_dict()["activation_storage"] == "mxfp4_e2m1"
    assert len(assessment.unresolved_contracts) == 2


def test_quark_layout_and_w4a4_are_independent_refusals() -> None:
    assert not mxfp4_weight_layout(MXFP4_QUARK_REORDER_LAYOUT_V1).conversion_supported
    with pytest.raises(ValueError, match="no proved Tessera conversion"):
        convert_weight_layout(
            np.zeros((16, 16), dtype=np.uint8),
            source=MXFP4_QUARK_REORDER_LAYOUT_V1,
            destination=MXFP4_CHECKPOINT_LAYOUT_V1,
        )
    layout = select_mxfp4_route(
        128, 17408, 5120, requested_layout=MXFP4_QUARK_REORDER_LAYOUT_V1,
    )
    assert not layout.accepted
    assert "byte-level converter" in layout.reason
    activation = select_mxfp4_route(
        128, 17408, 5120, activation_storage="mxfp4_e2m1",
    )
    assert not activation.accepted
    assert "not the proved W4A8 FP8 ABI" in activation.reason
    assert select_mxfp4_route(128, 17408, 5120).accepted


@pytest.mark.parametrize(
    "mutation", ["scale", "packing", "activation", "exclude", "dtype", "override"]
)
def test_quark_metadata_drift_fails_closed(mutation: str) -> None:
    config = deepcopy(QUARK_CONFIG)
    weight = {"dtype": "U8", "shape": [17408, 2560]}
    scale = {"dtype": "U8", "shape": [17408, 160]}
    module = GATE
    if mutation == "scale":
        scale["shape"] = [17408, 159]
    elif mutation == "packing":
        config["quantization_config"]["export"]["pack_method"] = "order"
    elif mutation == "activation":
        config["quantization_config"]["global_quant_config"]["input_tensors"][
            "dtype"
        ] = "fp8"
    elif mutation == "exclude":
        config["quantization_config"]["exclude"].append(GATE)
    elif mutation == "dtype":
        weight["dtype"] = "BF16"
    else:
        config["quantization_config"]["layer_quant_config"][GATE] = {}
    with pytest.raises(ValueError):
        assess_quark_mxfp4_projection(
            config, module=module, m=128, weight=weight, scale=scale,
        )


def test_glm_fp8_block_scales_are_not_a_quark_mxfp4_checkpoint() -> None:
    # GLM-5.3-Flash uses F8_E4M3 weights and F32 128x128 inverse scales.
    with pytest.raises(ValueError, match="Quark real-quantized reorder"):
        assess_quark_mxfp4_projection(
            {"quantization_config": {
                "quant_method": "fp8",
                "export": {},
                "global_quant_config": {"weight": {}, "input_tensors": {}},
            }},
            module="model.language_model.layers.10.mlp.experts.0.gate_proj",
            m=128,
            weight={"dtype": "F8_E4M3", "shape": [2048, 4096]},
            scale={"dtype": "F32", "shape": [16, 32]},
        )
