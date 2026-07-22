"""Totality and ISA-evidence guards for gfx1151 datatype support."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.dtype import canonical_dtypes, planned_gated_dtypes
from tessera.compiler.capabilities import get_target_capability
from tessera.compiler.rocm_dtype_contract import (
    GFX1151_DTYPE_CONTRACTS,
    GFX1151_INT4_STORAGE,
    gfx1151_dtype_contract,
    gfx1151_operation_dtype_readiness,
    gfx1151_ready_storage_dtypes,
)
from tessera.compiler.rocm_target import AMDArch, ROCmTargetProfile


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RDNA35_INSTRUCTIONS = (
    _REPO_ROOT / "docs/reference/isa/rdna/rdna35/instructions.json"
)


def test_gfx1151_contract_covers_all_canonical_and_planned_dtypes_once() -> None:
    expected = canonical_dtypes() | planned_gated_dtypes()
    actual = [row.storage for row in GFX1151_DTYPE_CONTRACTS]
    assert len(actual) == len(set(actual))
    assert set(actual) == expected


def test_gfx1151_ready_rows_match_both_target_registries() -> None:
    ready = gfx1151_ready_storage_dtypes()
    assert ready == ROCmTargetProfile(AMDArch.GFX_1151).dtype_set
    assert ready == set(get_target_capability("rocm_gfx1151").supported_dtypes)


def test_every_positive_isa_claim_names_an_archived_rdna35_opcode() -> None:
    instructions = json.loads(_RDNA35_INSTRUCTIONS.read_text())
    names = {item["name"] for item in instructions}
    for row in GFX1151_DTYPE_CONTRACTS:
        assert set(row.isa_evidence) <= names, row.storage


def test_gfx1151_matrix_roles_are_not_conflated_with_storage_support() -> None:
    assert gfx1151_dtype_contract("fp32").matrix == "accumulator_only"
    assert gfx1151_dtype_contract("int32").matrix == "accumulator_only"
    assert gfx1151_dtype_contract("fp64").scalar_vector == "native"
    assert gfx1151_dtype_contract("fp64").tessera_target_state == "unregistered"
    assert gfx1151_dtype_contract("int4").matrix == "native_input"
    assert gfx1151_dtype_contract("int4").tessera_target_state == "ready"
    assert gfx1151_dtype_contract("int4").rocm_toolchain_state == "validated"


def test_gpu_rocm_and_tessera_support_layers_remain_independent() -> None:
    fp64 = gfx1151_dtype_contract("fp64")
    assert fp64.scalar_vector == "native"
    assert fp64.rocm_toolchain_state == "validated"
    assert fp64.tessera_target_state == "unregistered"

    fp8 = gfx1151_dtype_contract("fp8_e4m3")
    assert fp8.scalar_vector == "unsupported"
    assert fp8.rocm_toolchain_state == "unsupported"
    assert fp8.tessera_target_state == "not_applicable"

    fp16 = gfx1151_dtype_contract("fp16")
    assert fp16.rocm_toolchain_state == "validated"
    assert fp16.tessera_target_state == "ready"


def test_rdna35_rejects_all_low_precision_float_matrix_formats() -> None:
    for storage in (
        "fp8_e4m3",
        "fp8_e5m2",
        "fp6_e2m3",
        "fp6_e3m2",
        "fp4_e2m1",
        "nvfp4",
        "mxfp8",
        "mxfp6",
        "mxfp4",
    ):
        row = gfx1151_dtype_contract(storage)
        assert row.scalar_vector == "unsupported"
        assert row.matrix == "unsupported"
        assert not row.matrix_selectable


def test_only_registered_native_matrix_inputs_are_selectable() -> None:
    selectable = {
        row.storage for row in GFX1151_DTYPE_CONTRACTS if row.matrix_selectable
    }
    assert selectable == {"fp16", "bf16", "int8", "int4"}


def test_fp64_and_wide_integer_per_operation_assessment_is_explicit() -> None:
    assert gfx1151_operation_dtype_readiness("fp64").state == "assessed_unavailable"
    assert gfx1151_operation_dtype_readiness("int16").state == "assessed_unavailable"
    assert gfx1151_operation_dtype_readiness("int32").state == "abi_only"
    assert "wmma_accumulator" in gfx1151_operation_dtype_readiness("int32").runtime_operations
    assert gfx1151_operation_dtype_readiness("int64").runtime_operations == ("shape_scalar",)


def test_unsigned_widths_are_validated_but_remain_unregistered() -> None:
    for storage in ("uint8", "uint16", "uint32", "uint64"):
        contract = gfx1151_dtype_contract(storage)
        readiness = gfx1151_operation_dtype_readiness(storage)
        assert contract.tessera_target_state == "planned_gated"
        assert readiness.state == "planned_gated"
        assert not readiness.runtime_operations


def test_first_class_int4_storage_is_signed_and_nibble_packed() -> None:
    assert GFX1151_INT4_STORAGE.logical == "int4"
    assert GFX1151_INT4_STORAGE.container == "int8"
    assert GFX1151_INT4_STORAGE.factor == 2
    assert GFX1151_INT4_STORAGE.signedness == "signed_twos_complement"
    assert gfx1151_operation_dtype_readiness("int4").state == "ready"


def test_fp8_and_bf8_have_explicit_target_and_runtime_rejection() -> None:
    for storage in ("fp8_e4m3", "fp8_e5m2"):
        readiness = gfx1151_operation_dtype_readiness(storage)
        assert readiness.state == "rejected"
        assert not readiness.target_ir_operations
        assert not readiness.runtime_operations


@pytest.mark.parametrize("storage", ("fp8_e4m3", "fp8_e5m2"))
def test_gfx1151_runtime_builder_rejects_fp8_and_bf8_by_name(storage: str) -> None:
    with pytest.raises(ValueError, match="ROCM_TILE_UNSUPPORTED_DTYPE"):
        rt._build_compiled_gemm_hsaco(1, 1, storage)


def test_signed_int4_host_packing_order_and_range() -> None:
    packed = rt._pack_signed_int4(np.array([-8, -1, 0, 7, 3], dtype=np.int8))
    assert packed.view(np.uint8).tolist() == [0xF8, 0x70, 0x03]
    with pytest.raises(ValueError, match=r"\[-8, 7\]"):
        rt._pack_signed_int4(np.array([8], dtype=np.int8))


@pytest.mark.compiler_tool
@pytest.mark.compiler_rocm
@pytest.mark.parametrize(
    "storage,llvm_type,instruction",
    (
        ("fp64", "double", "fadd double %x, 1.000000e+00"),
        ("int16", "i16", "sdiv i16 %x, 3"),
        ("int32", "i32", "sdiv i32 %x, 3"),
        ("int64", "i64", "sdiv i64 %x, 3"),
        ("uint8", "i8", "udiv i8 %x, 3"),
        ("uint16", "i16", "udiv i16 %x, 3"),
        ("uint32", "i32", "udiv i32 %x, 3"),
        ("uint64", "i64", "udiv i64 %x, 3"),
    ),
)
def test_llvm23_gfx1151_scalar_width_and_signedness_probe(
    storage: str, llvm_type: str, instruction: str,
) -> None:
    llc = shutil.which("llc-23") or "/usr/lib/llvm-23/bin/llc"
    if not Path(llc).is_file():
        pytest.skip("LLVM 23 llc is unavailable")
    source = f"""
        define amdgpu_kernel void @dtype_probe(ptr addrspace(1) %src,
                                                ptr addrspace(1) %dst) {{
          %x = load {llvm_type}, ptr addrspace(1) %src
          %y = {instruction}
          store {llvm_type} %y, ptr addrspace(1) %dst
          ret void
        }}
    """
    result = subprocess.run(
        [llc, "-mtriple=amdgcn-amd-amdhsa", "-mcpu=gfx1151", "-filetype=null", "-"],
        input=source, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, f"{storage}: {result.stderr}"
