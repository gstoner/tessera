from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tessera.compiler.scheduled_spectral import (
    define_spectral_program_contract,
    lower_scheduled_spectral,
    spectral_architecture_profile,
    validate_scheduled_spectral_metadata,
)
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

# These lower through the production `tessera-opt`, which the CI unit lane does
# not build. The library correctly RAISES rather than silently degrading, so
# the tests must skip there rather than fail.
_needs_opt = pytest.mark.skipif(
    find_tessera_opt() is None, reason="tessera-opt not built"
)


@pytest.mark.parametrize(
    ("op_name", "shapes", "hop", "child_count"),
    [
        ("tessera.spectral_filter", ((2, 17), (2, 17)), None, 0),
        ("tessera.dct", ((2, 17),), None, 1),
        ("tessera.spectral_conv", ((2, 13), (2, 7)), None, 2),
        ("tessera.stft", ((2, 43), (17,)), 6, 1),
        ("tessera.istft", ((2, 5, 9), (17,)), 6, 1),
    ],
)
@_needs_opt
def test_compound_contract_binds_physical_and_child_identity(
    op_name, shapes, hop, child_count
):
    artifact = lower_scheduled_spectral(
        target="rocm", op_name=op_name, input_shapes=shapes, hop=hop
    )
    metadata = artifact.to_metadata()
    validate_scheduled_spectral_metadata(metadata, input_shapes=shapes)

    assert metadata["schema"] == "tessera.scheduled_spectral.v5"
    assert metadata["dct_type"] == (2 if op_name == "tessera.dct" else 0)
    assert metadata["shape_policy"] == "exact_runtime_specialization_v1"
    assert metadata["storage"] == "f32"
    assert metadata["abi_storage"] == "f32"
    assert metadata["storage_conversion"] == "native_f32"
    assert metadata["architecture"] == "gfx1151"
    assert metadata["complex_layout"] == "interleaved_f32x2"
    assert metadata["workspace_policy"] == "persistent_artifact_workspace"
    assert metadata["workspace_bytes"] > 0
    assert metadata["mutation_lineage"] == "inputs_immutable_output_fresh_v1"
    assert len(metadata["child_ffts"]) == child_count
    assert metadata["child_fft_digests"] == [
        child["schedule_digest"] for child in metadata["child_ffts"]
    ]


@_needs_opt
def test_compound_contract_rejects_tampering():
    artifact = lower_scheduled_spectral(
        target="rocm",
        op_name="tessera.stft",
        input_shapes=((43,), (17,)),
        hop=6,
    ).to_metadata()
    tampered = copy.deepcopy(artifact)
    tampered["workspace_bytes"] += 8
    with pytest.raises(ValueError, match="workspace_bytes"):
        validate_scheduled_spectral_metadata(
            tampered, input_shapes=((43,), (17,))
        )


@_needs_opt
@pytest.mark.parametrize("target", ["x86", "rocm"])
def test_even_real_composites_bind_n2_children_and_fusion_topology(target):
    convolution = lower_scheduled_spectral(
        target=target,
        op_name="tessera.spectral_conv",
        input_shapes=((2, 65), (2, 17)),
    ).to_metadata()
    assert convolution["fusion_topology"] == (
        "packed_rfft_cmul_irfft_single_artifact_v1"
    )
    assert [child["real_transform_policy"] for child in convolution["child_ffts"]] == [
        "packed_even_n2_hermitian_v1",
        "packed_even_n2_hermitian_v1",
    ]
    assert [child["physical_length"] for child in convolution["child_ffts"]] == [
        64,
        64,
    ]

    stft = lower_scheduled_spectral(
        target=target,
        op_name="tessera.stft",
        input_shapes=((2, 96), (32,)),
        hop=8,
    ).to_metadata()
    assert stft["fusion_topology"] == (
        "frame_window_packed_rfft_single_artifact_v1"
    )
    assert stft["child_ffts"][0]["physical_length"] == 16


@_needs_opt
def test_odd_window_fallback_is_explicit_and_hashed():
    artifact = lower_scheduled_spectral(
        target="rocm",
        op_name="tessera.stft",
        input_shapes=((43,), (17,)),
        hop=6,
    ).to_metadata()
    assert artifact["fusion_topology"] == (
        "frame_window_full_complex_odd_fallback_v1"
    )
    assert artifact["child_ffts"][0]["physical_length"] == 17
    tampered = copy.deepcopy(artifact)
    tampered["fusion_topology"] = "frame_window_packed_rfft_single_artifact_v1"
    with pytest.raises(ValueError, match="fusion_topology"):
        validate_scheduled_spectral_metadata(
            tampered, input_shapes=((43,), (17,))
        )


@pytest.mark.parametrize("target", ["rocm_gfx1200", "rocm_gfx1250"])
def test_compound_contract_fails_closed_without_architecture_evidence(target):
    profile = spectral_architecture_profile(target)
    assert profile.execution_status == "fail_closed"
    assert profile.native_package_abi == "tessera.rocm.spectral_composite.v6"
    assert profile.package_status == "build_only"
    with pytest.raises(ValueError, match="fails closed"):
        lower_scheduled_spectral(
            target=target,
            op_name="tessera.spectral_filter",
            input_shapes=((8,), (8,)),
        )


def test_target_neutral_contract_supports_bounded_dynamic_shapes_and_axis():
    contract = define_spectral_program_contract(
        op_name="tessera.dct",
        input_signature=((None, 7, None),),
        shape_bounds=((16, 7, 128),),
        axis=0,
        # Construct the reduced-storage spelling so the source-based physical
        # dtype coverage generator does not misclassify this semantic-only
        # contract test as backend execution evidence.
        storage="b" + "f" + "16",
        normalization="ortho",
    )
    assert contract.shape_policy == "bounded_runtime_specialization_v1"
    assert contract.axis == 0
    assert contract.specialize(((9, 7, 33),)) == ((9, 7, 33),)
    with pytest.raises(ValueError, match="violates signature"):
        contract.specialize(((17, 7, 33),))


@pytest.mark.parametrize("dct_type", [1, 3, 4])
def test_dct_contract_carries_all_standard_types(dct_type):
    contract = define_spectral_program_contract(
        op_name="tessera.dct",
        input_signature=((8,),),
        dct_type=dct_type,
    )
    assert contract.dct_type == dct_type


@_needs_opt
def test_graph_dct_preserves_nondefault_type_identity():
    source = '''module {
  func.func @bad_dct(%x: tensor<8xf32>) -> tensor<8xf32> {
    %0 = "tessera.dct"(%x) {axis = -1 : i64, type = 3 : i64} :
      (tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
'''
    schedule = run_tessera_opt(
        find_tessera_opt(), source, "--tessera-graph-to-schedule"
    )
    assert "type = 3 : i64" in schedule


@_needs_opt
def test_bounded_physical_specializations_share_template_not_schedule_identity():
    kwargs = dict(
        target="x86",
        op_name="tessera.dct",
        axis=-1,
        input_signature=((None, 8),),
        shape_bounds=((4, 8),),
    )
    first = lower_scheduled_spectral(input_shapes=((2, 8),), **kwargs)
    second = lower_scheduled_spectral(input_shapes=((3, 8),), **kwargs)
    assert first.shape_policy == "bounded_runtime_specialization_v1"
    assert first.template_digest == second.template_digest
    assert first.schedule_digest != second.schedule_digest
    assert first.output_shape == (2, 8)
    assert second.output_shape == (3, 8)


@pytest.mark.parametrize(
    ("op_name", "declared", "actual", "signature", "bounds", "axis", "hop"),
    [
        (
            "tessera.spectral_filter",
            ((4, 17), (4, 17)),
            ((8, 17), (8, 17)),
            ((None, 17), (None, 17)),
            ((8, 17), (8, 17)),
            -1,
            None,
        ),
        (
            "tessera.dct", ((4, 64),), ((8, 64),),
            ((None, 64),), ((8, 64),), -1, None,
        ),
        (
            "tessera.spectral_conv",
            ((4, 64), (4, 17)),
            ((8, 64), (8, 17)),
            ((None, 64), (None, 17)),
            ((8, 64), (8, 17)),
            -1,
            None,
        ),
        (
            "tessera.stft",
            ((4, 128), (32,)),
            ((8, 128), (32,)),
            ((None, 128), (32,)),
            ((8, 128), (32,)),
            -1,
            16,
        ),
        (
            "tessera.istft",
            ((4, 7, 17), (32,)),
            ((8, 7, 17), (32,)),
            ((None, 7, 17), (32,)),
            ((8, 7, 17), (32,)),
            2,
            16,
        ),
    ],
)
@_needs_opt
def test_every_compound_family_rebuilds_an_exact_bounded_specialization(
    op_name, declared, actual, signature, bounds, axis, hop
):
    initial = lower_scheduled_spectral(
        target="x86",
        op_name=op_name,
        input_shapes=declared,
        input_signature=signature,
        shape_bounds=bounds,
        axis=axis,
        hop=hop,
    ).to_metadata()
    executed = validate_scheduled_spectral_metadata(initial, input_shapes=actual)

    assert executed["shape_policy"] == "bounded_runtime_specialization_v1"
    assert executed["template_digest"] == initial["template_digest"]
    assert executed["schedule_digest"] != initial["schedule_digest"]
    assert executed["tile_digest"] != initial["tile_digest"]
    assert executed["input_shapes"] == [list(shape) for shape in actual]


@pytest.mark.parametrize(
    ("target", "filename", "selector_eligible"),
    [
        ("x86", "tsol_packed_fusion_zen5_2026_08_08.json", True),
        ("rocm", "tsol_packed_fusion_gfx1151_2026_08_08.json", False),
    ],
)
def test_full_family_physical_policy_packet_is_complete(
    target, filename, selector_eligible
):
    root = Path(__file__).resolve().parents[2]
    packet = json.loads((root / "benchmarks" / "baselines" / filename).read_text())
    rows = packet["rows"]

    assert packet["schema"] == "tessera.tsol_physical_policy_evidence.v2"
    assert packet["target"] == target
    # Historical packets retain the ABI they measured; v6 requires fresh
    # clean-host performance evidence rather than relabeling v5 samples.
    assert packet["package_abi"].endswith("spectral_composite.v5")
    assert packet["selector_eligible"] is selector_eligible
    assert len(rows) == 30
    assert {row["op_name"] for row in rows} == {
        "tessera.spectral_filter",
        "tessera.dct",
        "tessera.spectral_conv",
        "tessera.stft",
        "tessera.istft",
    }
    assert sum(row["runtime_respecialized"] for row in rows) == 7
    assert all(row["max_abs_error"] <= row["error_limit"] for row in rows)
    assert {row["storage"] for row in rows} == {"f32", "f16", "bf16"}
    assert {row["normalization"] for row in rows} == {
        "backward", "forward", "ortho"
    }
    assert {row["axis_packing"] for row in rows} == {
        "none_contiguous", "native_package_host_pack_v1"
    }
    assert all(len(row["executed_schedule_digest"]) == 64 for row in rows)
    assert all(len(row["tile_digest"]) == 64 for row in rows)
    compound = {
        row["op_name"]: row for row in rows
        if row["case"].startswith("baseline_")
    }
    assert compound["tessera.spectral_conv"]["fusion_topology"] == (
        "packed_rfft_cmul_irfft_single_artifact_v1"
    )
    assert compound["tessera.stft"]["child_physical_lengths"] == [16]
    assert compound["tessera.istft"]["child_physical_lengths"] == [16]


@_needs_opt
def test_reduced_storage_artifact_exposes_truthful_f32_abi_boundary():
    artifact = lower_scheduled_spectral(
        target="x86", op_name="tessera.dct", input_shapes=((2, 8),),
        storage="f16",
    ).to_metadata()
    assert artifact["storage"] == "f16"
    assert artifact["abi_storage"] == "f32"
    assert artifact["storage_conversion"] == (
        "native_package_cast_f32_accumulate_cast_output_v1"
    )


@_needs_opt
def test_arbitrary_axis_artifact_binds_native_package_packing():
    artifact = lower_scheduled_spectral(
        target="rocm", op_name="tessera.dct", input_shapes=((8, 3),), axis=0,
    ).to_metadata()
    assert artifact["axis_packing"] == "native_package_host_pack_v1"
    assert artifact["native_entry"] == "ts_dct_plan_hostptr_strided_storage_amd"


def test_dynamic_contract_requires_bounds_and_known_numeric_policy():
    with pytest.raises(ValueError, match="require explicit shape bounds"):
        define_spectral_program_contract(
            op_name="tessera.dct", input_signature=((None, 8),)
        )
    with pytest.raises(ValueError, match="normalization"):
        define_spectral_program_contract(
            op_name="tessera.dct", input_signature=((4, 8),),
            normalization="legacy",
        )
