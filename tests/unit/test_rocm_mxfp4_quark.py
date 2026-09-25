"""Quark checkpoint metadata is not an executable gfx1201 W4A8 proof."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest

from tessera.compiler.rocm_mxfp4 import (
    MXFP4_CHECKPOINT_LAYOUT_V1,
    MXFP4_QUARK_REORDER_LAYOUT_V1,
    convert_weight_layout,
    mxfp4_weight_layout,
)
from tessera.compiler.rocm_mxfp4_native import select_mxfp4_route
from tessera.compiler.rocm_mxfp4_quark import (
    assess_quark_mxfp4_projection,
    decode_quark_low_even_hypothesis,
)


# Header/config excerpt from amd/Qwen3.8-27B-Quark-AWQ-MXFP4 at
# 5233554c5fa56afda40150556b95573c2d7d29c0.  The pinned full config
# SHA-256 is 4a139d3e01df039e17b4c8b8362f7944674d76051a8cd96f213c6bc332d69f55;
# the 219640-byte safetensors header SHA-256 is
# 01453e3d08a6275905b19dd15ac582fba065a551bc392ca22af0588d476f2083.
GATE = "model.language_model.layers.0.mlp.gate_proj"
DOWN = "model.language_model.layers.0.mlp.down_proj"
QUARK_CONFIG = {
    "quantization_config": {
        "version": "0.13+unknown",
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


def test_quark_candidate_oracle_all_e2m1_codes_and_low_even_nibble() -> None:
    packed = np.array([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2], dtype=np.uint8)
    scales = np.array([[127]], dtype=np.uint8)
    actual = decode_quark_low_even_hypothesis(packed, scales)
    values = np.array(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(actual[0], np.tile(values, 2))


@pytest.mark.parametrize(
    ("weight_hex", "scale_hex", "weight_sha", "scale_sha", "first_eight"),
    [
        (
            "ca2c11c63a2569da03400522a305571192d9eafd0d6fee42b9c12db5cbee3b"
            "6c41441e2c94a73c305d5ba52a1cba4e5e9ce94dc2265b5154db02d49bddba4273",
            "77777777",
            "3d2eb71ad973ad544f41d22be1d9ab9c82e0d67fa6862c415d84fff70f591a55",
            "6b4a1673b225e8bf5f093b91be8c864427df32ca41b17cc0b82112b8f0185e41",
            [-1, -2, -2, 1, 0.5, 0.5, 4, -2],
        ),
        (
            "0abf6eefaf61a5773fa7cbfdb0e3cd9dc12640410d09ba99991d3ca3105e90a3"
            "390255c30a5120a4c10103a92e35c9a14a152ae0ea21d0554f20d24440d4aa59",
            "76777777",
            "a461a6ce74bdcb5190cfac9ab90b018405cc91ebffcca9dfa4889ccd8a7ccd9b",
            "9b8b7e4c918c83e315361b93cdac8821c918c77d3b5eab00e76546d121cffd57",
            [-1, 0, -6, -1.5, -4, 4, -6, -4],
        ),
    ],
)
def test_pinned_checkpoint_byte_ranges_support_only_candidate_decode(
    weight_hex: str, scale_hex: str, weight_sha: str, scale_sha: str,
    first_eight: list[float],
) -> None:
    # HF revision 5233554c5fa56afda40150556b95573c2d7d29c0, first row,
    # K=0..127. Gate/down absolute ranges and source URI are recorded in the
    # baseline README; no checkpoint download is needed in CI.
    weight_bytes = bytes.fromhex(weight_hex)
    scale_bytes = bytes.fromhex(scale_hex)
    assert sha256(weight_bytes).hexdigest() == weight_sha
    assert sha256(scale_bytes).hexdigest() == scale_sha
    weight = np.frombuffer(weight_bytes, dtype=np.uint8).reshape(1, 64)
    scales = np.frombuffer(scale_bytes, dtype=np.uint8).reshape(1, 4)
    candidate = decode_quark_low_even_hypothesis(weight, scales)
    np.testing.assert_array_equal(
        candidate[0, :8], np.asarray(first_eight) * 2.0 ** (int(scales[0, 0]) - 127),
    )


@pytest.mark.parametrize("code", [0, 255])
def test_quark_candidate_oracle_refuses_unproved_scale_edges(code: int) -> None:
    with pytest.raises(ValueError, match="0/255 semantics"):
        decode_quark_low_even_hypothesis(
            np.zeros((1, 16), dtype=np.uint8), np.array([[code]], dtype=np.uint8)
        )


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
    assert assessment.as_dict()["activation_storage"] == "mxfp4_e2m1"
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
    assert "not an executable gfx1201 MXFP4 ABI" in layout.reason
    activation = assess_quark_mxfp4_projection(
        QUARK_CONFIG,
        module=GATE,
        m=128,
        weight={"dtype": "U8", "shape": [17408, 2560]},
        scale={"dtype": "U8", "shape": [17408, 160]},
    )
    assert not activation.route.accepted
    assert "dynamic MXFP4 activations" in activation.route.reason
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


def test_quark_glob_exclusion_is_honored() -> None:
    config = deepcopy(QUARK_CONFIG)
    config["quantization_config"]["exclude"] = ["*mlp.gate_proj"]
    with pytest.raises(ValueError, match="excluded"):
        assess_quark_mxfp4_projection(
            config, module=GATE, m=128,
            weight={"dtype": "U8", "shape": [17408, 2560]},
            scale={"dtype": "U8", "shape": [17408, 160]},
        )
    # A glob that does not match leaves the projection assessable.
    config["quantization_config"]["exclude"] = ["*lm_head"]
    assert assess_quark_mxfp4_projection(
        config, module=GATE, m=128,
        weight={"dtype": "U8", "shape": [17408, 2560]},
        scale={"dtype": "U8", "shape": [17408, 160]},
    ).route.accepted is False


def test_quark_probe_is_not_a_proved_scheduled_abi() -> None:
    from tessera import runtime as rt
    from tessera.compiler.rocm_mxfp4_quark_native import GFX1201_QUARK_W4A4_PROBE_ABI

    assert GFX1201_QUARK_W4A4_PROBE_ABI not in rt._gfx1201_proved_scheduled_abis()
    assert GFX1201_QUARK_W4A4_PROBE_ABI in rt._gfx1201_manual_probe_abis()


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1200", "gfx90a"])
def test_quark_probe_refuses_unowned_architectures(arch: str) -> None:
    from tessera.compiler.rocm_mxfp4_quark_native import package_quark_w4a4_probe

    with pytest.raises(ValueError, match="gfx1151 or gfx1201"):
        package_quark_w4a4_probe(1, 2, 32, arch=arch)


def test_quark_probe_launch_refuses_a_foreign_live_device(monkeypatch) -> None:
    import ml_dtypes
    from tessera import runtime as rt
    from tessera.compiler.rocm_mxfp4_quark_native import launch_quark_w4a4_probe

    monkeypatch.setattr(rt, "_rocm_live_arch", lambda: "gfx1100")
    buffers = {
        "a_packed": np.zeros((1, 16), np.uint8),
        "b_packed": np.zeros((2, 16), np.uint8),
        "a_scale": np.full((1, 1), 127, np.uint8),
        "b_scale": np.full((2, 1), 127, np.uint8),
        "output": np.zeros((1, 2), ml_dtypes.bfloat16),
    }
    with pytest.raises(RuntimeError, match="gfx1151 or gfx1201"):
        launch_quark_w4a4_probe(buffers)
