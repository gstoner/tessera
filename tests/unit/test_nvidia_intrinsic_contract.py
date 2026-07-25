"""Truth-table guards for SM120 integer/cast/SIMD API coverage."""
from __future__ import annotations

import pytest

from tessera.compiler.nvidia_intrinsic_contract import (
    SM120_INTRINSIC_CONTRACTS,
    sm120_intrinsic_contract,
)
from tessera.compiler.nvidia_native import (
    PROMOTED_SM120_CUDA_INTRINSICS,
    package_cuda_intrinsic,
)


def test_required_cuda_intrinsic_categories_are_total() -> None:
    assert {row.category for row in SM120_INTRINSIC_CONTRACTS} == {
        "integer_math", "integer_bits", "integer_dot", "cast", "packed_simd",
    }
    assert len({row.key for row in SM120_INTRINSIC_CONTRACTS}) == len(
        SM120_INTRINSIC_CONTRACTS
    )


def test_cuda_compile_proof_does_not_imply_tessera_selection() -> None:
    assert all(row.cuda_compile_state == "ready" for row in SM120_INTRINSIC_CONTRACTS)
    assert {
        row.key for row in SM120_INTRINSIC_CONTRACTS if row.selectable
    } == {
        "integer_min_max_abs",
        "funnel_shift_l_wrap",
        "integer_packed_dot",
        "numeric_cast_rn_rd_ru_rz",
        "packed_simd_add_u16x2",
    }
    assert {
        row.key
        for row in SM120_INTRINSIC_CONTRACTS
        if row.target_ir_state == row.runtime_state == "planned"
    } == {
        "integer_bit_ops",
        "bit_reinterpret_cast",
        "packed_simd_2x16_4x8",
    }
    assert all(row.ptx_operand_storage for row in SM120_INTRINSIC_CONTRACTS)


def test_semantic_hazards_are_retained() -> None:
    assert "undefined" in sm120_intrinsic_contract(
        "numeric_cast_rn_rd_ru_rz"
    ).invalid_contract
    assert "signedness" in sm120_intrinsic_contract(
        "integer_packed_dot"
    ).invalid_contract
    assert "saturation" in sm120_intrinsic_contract(
        "packed_simd_2x16_4x8"
    ).invalid_contract


def test_unknown_intrinsic_family_fails_closed() -> None:
    with pytest.raises(ValueError, match="no SM120 intrinsic contract"):
        sm120_intrinsic_contract("generic_simd")


def test_runtime_packager_fails_closed_outside_promoted_subset() -> None:
    assert PROMOTED_SM120_CUDA_INTRINSICS == {
        "abs_i32",
        "min_i32",
        "max_i32",
        "funnelshift_l_wrap_u32",
        "dp2a_lo_s32",
        "dp4a_s32",
        "cvt_f32_i32_rn",
        "cvt_f32_i32_rd",
        "cvt_f32_i32_ru",
        "cvt_f32_i32_rz",
        "vadd2_u16x2",
    }
    with pytest.raises(ValueError, match="not in the exact-device promoted subset"):
        package_cuda_intrinsic(kind="vaddss4_s8x4", count=16)
