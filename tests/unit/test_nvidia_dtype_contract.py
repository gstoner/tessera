"""Totality and separation guards for the SM120 dtype contract."""

from __future__ import annotations

import pytest

from tessera.dtype import canonical_dtypes
from tessera.compiler.capabilities import get_target_capability
from tessera.compiler.gpu_target import GPUTargetProfile, ISA
from tessera.compiler.nvidia_dtype_contract import (
    SM120_DTYPE_CONTRACTS,
    sm120_dtype_contract,
    sm120_supported_storage_dtypes,
)
from tessera.compiler.nvidia_fragment import (
    NvidiaFragmentError,
    select_sm120_fragment_layout,
)
from tessera.compiler.mma_selector import (
    MmaSelectorError,
    get_isa,
    select_mma,
)


_FLOAT_STORAGE = {
    "fp64",
    "fp32",
    "fp16",
    "bf16",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp6_e2m3",
    "fp6_e3m2",
    "fp4_e2m1",
    "nvfp4",
}


def test_sm120_contract_covers_every_canonical_float_storage_once() -> None:
    rows = [row.storage for row in SM120_DTYPE_CONTRACTS if row.storage != "fp32"]
    assert set(rows) & _FLOAT_STORAGE == _FLOAT_STORAGE - {"fp32"}
    assert _FLOAT_STORAGE <= canonical_dtypes()
    assert {row.key for row in SM120_DTYPE_CONTRACTS if row.storage == "fp32"} == {
        "fp32_ieee",
        "fp32_tf32",
    }


def test_tf32_is_only_an_fp32_math_mode() -> None:
    ieee = sm120_dtype_contract("fp32")
    tf32 = sm120_dtype_contract("fp32", math_mode="tf32")
    assert ieee.tensor_core == "unsupported"
    assert ieee.runtime_state == "ready"
    assert tf32.tensor_core == "math_mode"
    assert tf32.tensor_format == "tf32"
    assert tf32.fragment_shape == (16, 8, 8)
    assert all(row.storage != "tf32" for row in SM120_DTYPE_CONTRACTS)


def test_ptx_storage_and_tensor_operand_types_are_explicit() -> None:
    fundamental_storage = {row.storage for row in SM120_DTYPE_CONTRACTS if row.storage_format_kind == "fundamental"}
    assert fundamental_storage == {"fp64", "fp32", "fp16", "int8"}
    assert sm120_dtype_contract("bf16").ptx_storage_register == ".b16"
    assert sm120_dtype_contract("bf16").scalar_vector == "conversion_only"
    assert "restricted" in sm120_dtype_contract("int8").note
    for storage in (
        "bf16",
        "fp8_e4m3",
        "fp8_e5m2",
        "fp6_e2m3",
        "fp6_e3m2",
        "fp4_e2m1",
        "nvfp4",
    ):
        row = sm120_dtype_contract(storage)
        assert row.storage_format_kind == "alternate"
        assert row.ptx_storage_register.startswith(".b")
    assert sm120_dtype_contract("fp32", math_mode="tf32").tensor_operand_register == ".b32"


def test_every_tensor_format_names_its_physical_fragment_register() -> None:
    for row in SM120_DTYPE_CONTRACTS:
        if row.tensor_core == "unsupported":
            assert row.tensor_operand_register is None
            continue
        assert row.tensor_operand_register in {".f64", ".b32"}
        assert row.tensor_format_kind in {"fundamental", "alternate", "subbyte"}


def test_fp64_target_ir_tile_and_runtime_are_ready() -> None:
    fp64 = sm120_dtype_contract("fp64")
    assert fp64.tensor_format == "f64"
    assert fp64.fragment_shape == (8, 8, 4)
    assert fp64.target_ir_state == "ready"
    assert fp64.compiler_state == "ready"
    assert fp64.runtime_state == "ready"
    assert fp64.selectable


@pytest.mark.parametrize(
    ("storage", "math_mode", "selector_dtype", "shape"),
    [
        ("fp64", None, "fp64", (8, 8, 4)),
        ("fp32", "tf32", "tf32", (16, 8, 8)),
        ("fp16", None, "fp16", (16, 8, 16)),
        ("bf16", None, "bf16", (16, 8, 16)),
        ("fp8_e4m3", None, "fp8_e4m3", (16, 8, 32)),
        ("fp8_e5m2", None, "fp8_e5m2", (16, 8, 32)),
        ("fp6_e2m3", None, "fp6_e2m3", (16, 8, 32)),
        ("fp6_e3m2", None, "fp6_e3m2", (16, 8, 32)),
        ("fp4_e2m1", None, "fp4_e2m1", (16, 8, 64)),
        ("nvfp4", None, "nvfp4", (16, 8, 64)),
        ("int8", None, "int8", (16, 8, 32)),
    ],
)
def test_every_ready_tensor_contract_has_a_physical_fragment_selector(
    storage: str,
    math_mode: str | None,
    selector_dtype: str,
    shape: tuple[int, int, int],
) -> None:
    contract = sm120_dtype_contract(storage, math_mode=math_mode)
    assert contract.selectable
    fragment = select_sm120_fragment_layout(selector_dtype, shape)
    assert fragment.shape == contract.fragment_shape


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        ("fp6_e2m3", (16, 8, 32)),
        ("fp6_e3m2", (16, 8, 32)),
        ("fp4_e2m1", (16, 8, 64)),
    ],
)
def test_block_scaled_formats_have_proven_fragment_abis(dtype, shape) -> None:
    assert select_sm120_fragment_layout(dtype, shape).block_scaled


def test_ieee_fp32_never_silently_selects_tf32() -> None:
    with pytest.raises(NvidiaFragmentError, match="explicit TF32 math-mode"):
        select_sm120_fragment_layout("fp32", (16, 8, 8))
    with pytest.raises(MmaSelectorError, match="math_mode='tf32'"):
        select_mma(get_isa("nvidia", "sm_120"), "fp32")
    assert select_mma(
        get_isa("nvidia", "sm_120"),
        "fp32",
        math_mode="tf32",
    ).shape == (16, 8, 8)


def test_nvfp4_and_ocp_fp4_have_distinct_scale_contracts() -> None:
    nvfp4 = sm120_dtype_contract("nvfp4")
    mxfp4 = sm120_dtype_contract("fp4_e2m1")
    assert nvfp4.scale_format == "ue4m3"
    assert mxfp4.scale_format == "ue8m0"
    assert nvfp4.scale_vector == "4X"
    assert mxfp4.scale_vector == "2X"
    assert nvfp4.selectable
    assert mxfp4.selectable
    isa = get_isa("nvidia", "sm_120")
    assert select_mma(isa, "nvfp4").shape == (16, 8, 64)
    assert select_mma(isa, "fp4_e2m1").shape == (16, 8, 64)


def test_required_sm120_tensor_core_families_have_target_ir_contracts() -> None:
    families = {
        "tf32": [sm120_dtype_contract("fp32", math_mode="tf32")],
        "bf16": [sm120_dtype_contract("bf16")],
        "fp16": [sm120_dtype_contract("fp16")],
        "fp8": [sm120_dtype_contract("fp8_e4m3"), sm120_dtype_contract("fp8_e5m2")],
        "fp6": [sm120_dtype_contract("fp6_e2m3"), sm120_dtype_contract("fp6_e3m2")],
        "fp4": [sm120_dtype_contract("fp4_e2m1"), sm120_dtype_contract("nvfp4")],
        "int8": [sm120_dtype_contract("int8")],
    }
    assert set(families) == {"tf32", "bf16", "fp16", "fp8", "fp6", "fp4", "int8"}
    assert all(
        row.tensor_core != "unsupported" and row.target_ir_state == "ready"
        for rows in families.values()
        for row in rows
    )


def test_target_capability_and_hardware_report_derive_from_contract() -> None:
    target = get_target_capability("nvidia_sm120")
    assert set(target.supported_dtypes) == set(sm120_supported_storage_dtypes())
    report = GPUTargetProfile(isa=ISA.SM_120).tensor_core_dtypes
    assert "tf32" in report
    assert "fp64" in report and "fp6_e2m3" in report and "fp4_e2m1" in report
    assert "tf32" not in target.supported_dtypes
