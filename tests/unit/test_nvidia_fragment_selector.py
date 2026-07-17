from __future__ import annotations

import pytest

from tessera.compiler.nvidia_fragment import (
    FragmentFamily,
    NvidiaFragmentError,
    RegisterPacking,
    select_sm120_fragment_layout,
)


@pytest.mark.parametrize(
    "dtype,shape,packing,a_regs,b_regs,mnemonic",
    [
        ("f16", (16, 8, 16), RegisterPacking.PAIR_F16, 4, 2, ".f16.f16.f32"),
        ("bf16", (16, 8, 16), RegisterPacking.PAIR_F16, 4, 2, ".bf16.bf16.f32"),
        ("tf32", (16, 8, 8), RegisterPacking.SCALAR_F32, 4, 2, ".tf32.tf32.f32"),
        ("fp8_e4m3", (16, 8, 32), RegisterPacking.PACKED_X4_I8, 4, 2, ".e4m3.e4m3.f32"),
        ("fp8_e5m2", (16, 8, 32), RegisterPacking.PACKED_X4_I8, 4, 2, ".e5m2.e5m2.f32"),
        ("int8", (16, 8, 32), RegisterPacking.PACKED_X4_I8, 4, 2, ".s8.s8.s32"),
    ],
)
def test_sm120_fragment_selector_owns_cuda_physical_abi(
    dtype: str, shape: tuple[int, int, int], packing: RegisterPacking,
    a_regs: int, b_regs: int, mnemonic: str,
) -> None:
    desc = select_sm120_fragment_layout(dtype, shape)
    assert desc.arch == "sm_120a"
    assert desc.family is FragmentFamily.MMA_SYNC
    assert desc.warp_size == 32
    assert desc.input_packing is packing
    assert (desc.a_registers_per_lane, desc.b_registers_per_lane) == (a_regs, b_regs)
    assert desc.accumulator_registers_per_lane == 4
    assert mnemonic in desc.instruction_family
    assert desc.typed_tile_materialization_ready


def test_sm120_nvfp4_is_block_scaled_and_typed_materialization_ready() -> None:
    desc = select_sm120_fragment_layout("nvfp4", (16, 8, 64))
    assert desc.family is FragmentFamily.MMA_SYNC_BLOCK_SCALE
    assert desc.block_scaled
    assert desc.scale_dtype == "ue4m3"
    assert desc.scale_vector == "4X"
    assert desc.input_packing is RegisterPacking.PACKED_X8_E2M1
    assert "mxf4nvf4.block_scale" in desc.instruction_family
    assert desc.typed_tile_materialization_ready


def test_sm120_f16_accumulation_has_packed_accumulator_registers() -> None:
    desc = select_sm120_fragment_layout("f16", (16, 8, 16), "f16")
    assert desc.acc_dtype == "f16"
    assert desc.accumulator_elements_per_lane == 4
    assert desc.accumulator_registers_per_lane == 2
    assert ".f16.f16.f16.f16" in desc.instruction_family


@pytest.mark.parametrize(
    "dtype,shape",
    [("f16", (16, 16, 16)), ("tf32", (16, 8, 16)),
     ("nvfp4", (16, 8, 32)), ("fp6_e3m2", (16, 8, 32))],
)
def test_sm120_fragment_selector_rejects_unproven_contracts(
    dtype: str, shape: tuple[int, int, int]
) -> None:
    with pytest.raises(NvidiaFragmentError, match="sm_120a|fragments"):
        select_sm120_fragment_layout(dtype, shape)


def test_sm120_fragment_selector_rejects_unsupported_accumulation() -> None:
    with pytest.raises(NvidiaFragmentError, match="do not support"):
        select_sm120_fragment_layout("bf16", (16, 8, 16), "f16")


def test_sm120_fragment_selector_metadata_is_resource_ready() -> None:
    row = select_sm120_fragment_layout("e4m3", (16, 8, 32)).as_metadata_dict()
    assert row["arch"] == "sm_120a"
    assert row["shape"] == [16, 8, 32]
    assert row["instruction_family"].startswith("mma.sync.aligned")
    assert row["block_scaled"] is False
