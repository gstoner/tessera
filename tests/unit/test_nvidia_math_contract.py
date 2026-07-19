"""Host-free checks for explicit CUDA floating-point semantic routes."""
from __future__ import annotations

import pytest

from tessera.compiler.nvidia_math_contract import (
    CUDA_MATH_CONTRACT_VERSION,
    cuda_math_contract,
)


def test_ieee_operator_contract_is_not_fast_math() -> None:
    contract = cuda_math_contract("ieee_operator")
    assert contract.rounding == "round_to_nearest_ties_to_even"
    assert contract.ftz is False
    assert contract.approximate is False
    assert contract.max_ulp == 0.0
    assert contract.requires_nonzero_atol is False


def test_libdevice_accuracy_remains_function_specific() -> None:
    contract = cuda_math_contract("cuda_libdevice")
    assert contract.ftz is False
    assert contract.approximate is True
    assert contract.max_ulp is None
    assert "function-specific" in contract.accuracy_scope


def test_softmax_ex2_route_records_ptx_bound_and_subnormal_policy() -> None:
    contract = cuda_math_contract("ptx_ex2_approx_f32")
    assert contract.ftz is False
    assert contract.approximate is True
    assert contract.max_ulp == 2.0
    assert contract.requires_nonzero_atol is True


def test_unknown_math_route_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown CUDA math route"):
        cuda_math_contract("fastish")  # type: ignore[arg-type]


def test_math_contract_has_versioned_cache_identity() -> None:
    assert CUDA_MATH_CONTRACT_VERSION == "tessera.nvidia.cuda_math.v1"
