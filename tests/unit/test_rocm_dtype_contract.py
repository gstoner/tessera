"""Totality and ISA-evidence guards for gfx1151 datatype support."""

from __future__ import annotations

import json
from pathlib import Path

from tessera.dtype import canonical_dtypes, planned_gated_dtypes
from tessera.compiler.capabilities import get_target_capability
from tessera.compiler.rocm_dtype_contract import (
    GFX1151_DTYPE_CONTRACTS,
    gfx1151_dtype_contract,
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
    assert gfx1151_dtype_contract("int4").tessera_target_state == "planned_gated"
    assert gfx1151_dtype_contract("int4").rocm_toolchain_state == "validated"


def test_gpu_rocm_and_tessera_support_layers_remain_independent() -> None:
    fp64 = gfx1151_dtype_contract("fp64")
    assert fp64.scalar_vector == "native"
    assert fp64.rocm_toolchain_state == "available_unvalidated"
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
    assert selectable == {"fp16", "bf16", "int8"}
