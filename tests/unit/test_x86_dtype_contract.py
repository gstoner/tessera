from tessera.compiler.capabilities import get_target_capability
from tessera.compiler.x86_dtype_contract import (
    X86_DTYPE_CONTRACTS,
    x86_dtype_contract,
    x86_ready_storage_dtypes,
)
from tessera.dtype import canonical_dtypes, planned_gated_dtypes


def test_x86_contract_is_total_over_dtype_vocabulary() -> None:
    expected = canonical_dtypes() | planned_gated_dtypes()
    actual = [row.storage for row in X86_DTYPE_CONTRACTS]
    assert len(actual) == len(set(actual))
    assert set(actual) == expected


def test_x86_ready_rows_match_capability_registry() -> None:
    registered = set(get_target_capability("x86").supported_dtypes) - {"f32"}
    assert registered == x86_ready_storage_dtypes()


def test_low_precision_roles_are_explicit() -> None:
    assert x86_dtype_contract("bf16").accumulator == "fp32"
    assert x86_dtype_contract("int8").accumulator == "int32"
    assert x86_dtype_contract("uint8").tessera_state == "planned_gated"
    assert x86_dtype_contract("fp8_e4m3").vector_state == "emulated"
    assert x86_dtype_contract("fp8_e4m3").tessera_state == "planned_gated"


def test_current_host_profile_does_not_conflate_amx_or_fp8() -> None:
    features = set(get_target_capability("x86").features)
    assert "amx" not in features and "amx_tile" not in features
    assert "avx512_vnni" in features and "avx512_bf16" in features
    assert x86_dtype_contract("bf16").available_on(features)
    assert x86_dtype_contract("int8").available_on(features)
    assert not x86_dtype_contract("fp8_e4m3").required_features

