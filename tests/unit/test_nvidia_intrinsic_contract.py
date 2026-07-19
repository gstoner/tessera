"""Truth-table guards for SM120 integer/cast/SIMD API coverage."""
from __future__ import annotations

import pytest

from tessera.compiler.nvidia_intrinsic_contract import (
    SM120_INTRINSIC_CONTRACTS,
    sm120_intrinsic_contract,
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
    assert all(row.target_ir_state == "planned" for row in SM120_INTRINSIC_CONTRACTS)
    assert not any(row.selectable for row in SM120_INTRINSIC_CONTRACTS)
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
