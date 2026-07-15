"""ROCM-5 architecture-owned physical fragment descriptor tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler.rocm_fragment import (
    FragmentFamily,
    RegisterFormat,
    select_fragment_layout,
)
from tessera.compiler.rocm_mma import select_mma
from tessera.compiler.rocm_target import AMDArch, TesseraROCmTargetError


def test_gfx1151_owns_duplicated_gfx11_fragment_map():
    d = select_fragment_layout(AMDArch.GFX_1151, "fp16", (16, 16, 16))
    assert d.family is FragmentFamily.RDNA3_WMMA
    assert d.matrix_op == "wmma"
    assert d.wave_size == 32
    assert d.input_elements_per_lane == 16
    assert d.input_registers_per_lane == 8
    assert d.accumulator_elements_per_lane == 8
    assert d.input_lane_replication == 2
    assert d.input_format is RegisterFormat.WMMA_INPUT_GFX11
    assert d.accumulator_mapping == "gfx11_col_per_lane"


@pytest.mark.parametrize("arch", [AMDArch.GFX_1200, AMDArch.GFX_1201])
def test_rdna4_owns_dense_nonduplicated_fragment_map(arch):
    d = select_fragment_layout(arch, "bf16", (16, 16, 16))
    assert d.family is FragmentFamily.RDNA4_WMMA
    assert d.input_elements_per_lane == 8
    assert d.input_registers_per_lane == 4
    assert d.input_lane_replication == 1
    assert d.input_format is RegisterFormat.SOA
    assert d.accumulator_mapping == "soa_row_per_lane"
    assert d.intrinsic_abi == "abc_3arg_gfx12"


def test_rdna4_fp8_is_packed_and_int4_uses_k32():
    fp8 = select_fragment_layout(
        AMDArch.GFX_1201, "fp8_e4m3", (16, 16, 16))
    assert fp8.input_elements_per_lane == 8
    assert fp8.input_registers_per_lane == 2
    assert fp8.input_format is RegisterFormat.SOA_INT

    int4 = select_fragment_layout(AMDArch.GFX_1201, "int4", (16, 16, 32))
    assert int4.input_elements_per_lane == 16
    assert int4.input_registers_per_lane == 2
    assert int4.acc_dtype == "int32"


def test_gfx125x_v2_does_not_alias_rdna4():
    d = select_fragment_layout(AMDArch.GFX_1250, "bf16", (16, 16, 32))
    assert d.family is FragmentFamily.GFX125X_WMMA_V2
    assert d.input_elements_per_lane == 16
    assert d.intrinsic_abi == "mods_reuse_8arg_gfx125x"
    assert d.materialization_ready


@pytest.mark.parametrize(
    ("arch", "family"),
    [
        (AMDArch.GFX_90A, FragmentFamily.CDNA2_MFMA),
        (AMDArch.GFX_942, FragmentFamily.CDNA3_MFMA),
        (AMDArch.GFX_950, FragmentFamily.CDNA4_MFMA),
    ],
)
def test_cdna_owns_wave64_mfma_fragment_map(arch, family):
    d = select_fragment_layout(arch, "bf16", (16, 16, 16))
    assert d.family is family
    assert d.matrix_op == "mfma"
    assert d.wave_size == 64
    assert d.input_elements_per_lane == 4
    assert d.accumulator_elements_per_lane == 4
    assert d.input_lane_replication == 1
    assert d.accumulator_mapping == "soa_row_per_lane"


def test_family_dtype_guards_are_named_and_do_not_fallback():
    with pytest.raises(TesseraROCmTargetError, match="ILLEGAL_RDNA3"):
        select_fragment_layout(AMDArch.GFX_1151, "fp8_e4m3", (16, 16, 16))
    with pytest.raises(TesseraROCmTargetError, match="UNSUPPORTED_CDNA2"):
        select_fragment_layout(AMDArch.GFX_90A, "fp8_e4m3", (16, 16, 32))
    with pytest.raises(TesseraROCmTargetError, match="UNSUPPORTED_CDNA3"):
        select_fragment_layout(AMDArch.GFX_942, "fp4_e2m1", (16, 16, 64))


def test_mma_metadata_carries_the_physical_fragment_contract():
    md = select_mma(AMDArch.GFX_1201, "fp16").as_metadata_dict()
    physical = md["physical_fragment"]
    assert physical["family"] == "rdna4_wmma"
    assert physical["input_elements_per_lane"] == 8
    assert physical["input_lane_replication"] == 1


def test_int4_selection_uses_family_specific_k():
    assert select_mma(AMDArch.GFX_1151, "int4").shape == (16, 16, 16)
    assert select_mma(AMDArch.GFX_1201, "int4").shape == (16, 16, 32)


def test_python_and_cpp_fragment_contract_names_do_not_drift():
    root = Path(__file__).resolve().parents[2]
    header = (root / "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion"
              / "ROCMFragmentLayout.h").read_text()
    descriptors = [
        select_fragment_layout(AMDArch.GFX_1151, "fp16", (16, 16, 16)),
        select_fragment_layout(AMDArch.GFX_1201, "fp16", (16, 16, 16)),
        select_fragment_layout(AMDArch.GFX_1250, "fp16", (16, 16, 32)),
        select_fragment_layout(AMDArch.GFX_90A, "fp16", (16, 16, 16)),
        select_fragment_layout(AMDArch.GFX_942, "fp16", (16, 16, 16)),
        select_fragment_layout(AMDArch.GFX_950, "fp16", (16, 16, 16)),
    ]
    for descriptor in descriptors:
        assert f'"{descriptor.family.value}"' in header
        assert f'"{descriptor.intrinsic_abi}"' in header
        assert f'"{descriptor.input_format.value}"' in header
