"""Cross-generation AMD ISA/dtype totality and instruction-selection tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tessera.dtype import canonical_dtypes, planned_gated_dtypes
from tessera.compiler.dtype_flow_audit import _amd_matmul_isa_state
from tessera.compiler.rocm_isa_contract import (
    AMD_ARCHITECTURE_CONTRACTS,
    AMD_DTYPE_CONTRACTS,
    amd_architecture_contract,
    amd_dtype_contract,
    select_amd_matrix_instruction,
)
from tessera.compiler.rocm_exact_device_proofs import GFX1201_PUBLIC_PROOFS
from tessera.compiler.rocm_target import AMDArch, TesseraROCmTargetError


ROOT = Path(__file__).resolve().parents[2]


def test_registered_architecture_contracts_are_dtype_total() -> None:
    expected = canonical_dtypes() | planned_gated_dtypes()
    registered = {
        AMDArch.GFX_1151,
        AMDArch.GFX_1200,
        AMDArch.GFX_1201,
        AMDArch.GFX_1250,
        AMDArch.GFX_1251,
    }
    assert set(AMD_ARCHITECTURE_CONTRACTS) == registered
    assert set(AMD_DTYPE_CONTRACTS) == registered
    for arch, rows in AMD_DTYPE_CONTRACTS.items():
        assert set(rows) == expected, arch.name


def test_gfx1201_matmul_dtype_flow_uses_its_exact_isa_contract() -> None:
    state = _amd_matmul_isa_state("matmul", "rocm_gfx1201", "fp16")
    assert state is not None
    assert state.status == "ready"
    assert state.source == "rocm_isa_contract[GFX_1201]"


def test_rdna4_dense_dtype_state_is_exact_target_scoped() -> None:
    proved = next(
        set(proof.dtypes)
        for proof in GFX1201_PUBLIC_PROOFS
        if proof.op_name == "tessera.matmul"
    )
    assert proved == {"fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "int8", "int4"}
    for storage in proved:
        assert amd_dtype_contract(
            AMDArch.GFX_1201, storage
        ).dense_matrix == "ready"
        assert select_amd_matrix_instruction(
            AMDArch.GFX_1201, storage
        ).compiler_state == "ready"
        assert amd_dtype_contract(
            AMDArch.GFX_1200, storage
        ).dense_matrix == "artifact_only"
        assert select_amd_matrix_instruction(
            AMDArch.GFX_1200, storage
        ).compiler_state == "artifact_only"

    # Accumulator/result formats do not become input formats merely because
    # every proved matrix route produces one of them.
    for storage in ("fp32", "int32"):
        with pytest.raises(
            TesseraROCmTargetError, match="ROCM_TILE_UNSUPPORTED_DTYPE"
        ):
            select_amd_matrix_instruction(AMDArch.GFX_1201, storage)


def test_architecture_identity_does_not_infer_wave_or_matrix_path() -> None:
    rdna35 = amd_architecture_contract(AMDArch.GFX_1151)
    gfx1200 = amd_architecture_contract(AMDArch.GFX_1200)
    gfx1201 = amd_architecture_contract(AMDArch.GFX_1201)
    mi455x = amd_architecture_contract(AMDArch.GFX_1250)
    mi430x = amd_architecture_contract(AMDArch.GFX_1251)
    assert (rdna35.family, rdna35.wave_size, rdna35.matrix_pipeline) == (
        "rdna35", 32, "valu_wmma")
    assert (gfx1200.family, gfx1200.wave_size, gfx1200.sparse_matrix) == (
        "rdna4", 32, True)
    assert gfx1200.product == "Radeon RX 9050 / RX 9060 series"
    assert gfx1201.product == (
        "Radeon RX 9070 series / Radeon AI PRO R9000 series"
    )
    assert gfx1200.cost_model_identity != gfx1201.cost_model_identity
    assert (mi455x.family, mi455x.product, mi455x.matrix_pipeline) == (
        "cdna5", "AMD Instinct MI455X", "xdl_wmma")
    assert (mi430x.family, mi430x.product) == ("cdna5", "AMD Instinct MI430X")
    assert mi455x.cost_model_identity != mi430x.cost_model_identity
    assert amd_dtype_contract(AMDArch.GFX_1250, "fp64").dense_matrix == "unsupported"
    assert amd_dtype_contract(AMDArch.GFX_1251, "fp64").dense_matrix == "planned_gated"
    assert select_amd_matrix_instruction(
        AMDArch.GFX_1251, "fp64"
    ).mnemonic == "V_WMMA_F64_16X16X4_F64"


def test_architecture_constants_match_cdna5_contract() -> None:
    for arch in (AMDArch.GFX_1250, AMDArch.GFX_1251):
        row = amd_architecture_contract(arch)
        assert row.wave_size == 32
        assert row.lds_bytes == 320 * 1024
        assert row.vgpr_budget == 1024
        assert row.dense_matrix and row.sparse_matrix and row.microscaling


def test_gfx1151_rejects_fp8_and_sparse_without_fallback() -> None:
    with pytest.raises(TesseraROCmTargetError, match="ROCM_TILE_UNSUPPORTED_DTYPE"):
        select_amd_matrix_instruction(AMDArch.GFX_1151, "fp8_e4m3")
    with pytest.raises(TesseraROCmTargetError, match="ROCM_TILE_UNSUPPORTED_DTYPE"):
        select_amd_matrix_instruction(AMDArch.GFX_1151, "fp16", sparse=True)


def test_gfx1151_and_rdna4_targets_select_exact_architecture_ops() -> None:
    assert select_amd_matrix_instruction(
        AMDArch.GFX_1151, "fp16").mnemonic == "V_WMMA_F32_16X16X16_F16"
    assert select_amd_matrix_instruction(
        AMDArch.GFX_1151, "int4").mnemonic == "V_WMMA_I32_16X16X16_IU4"
    assert select_amd_matrix_instruction(
        AMDArch.GFX_1200, "fp8_e4m3", "fp8_e5m2"
    ).mnemonic == "V_WMMA_F32_16X16X16_FP8_BF8"
    assert select_amd_matrix_instruction(
        AMDArch.GFX_1200, "fp16", sparse=True
    ).mnemonic == "V_SWMMAC_F32_16X16X32_F16"
    assert select_amd_matrix_instruction(
        AMDArch.GFX_1201, "fp8_e4m3", "fp8_e5m2"
    ).mnemonic == "V_WMMA_F32_16X16X16_FP8_BF8"
    assert select_amd_matrix_instruction(
        AMDArch.GFX_1201, "fp16", sparse=True
    ).mnemonic == "V_SWMMAC_F32_16X16X32_F16"


def test_cdna5_selects_dense_sparse_and_scaled_xdl_ops() -> None:
    assert select_amd_matrix_instruction(
        AMDArch.GFX_1250, "bf16").mnemonic == "V_WMMA_F32_16X16X32_BF16"
    assert select_amd_matrix_instruction(
        AMDArch.GFX_1250, "fp8_e4m3", "fp8_e5m2"
    ).mnemonic == "V_WMMA_F32_16X16X64_FP8_BF8"
    assert select_amd_matrix_instruction(
        AMDArch.GFX_1250, "int8", sparse=True
    ).mnemonic == "V_SWMMAC_I32_16X16X128_IU8"
    scaled = select_amd_matrix_instruction(
        AMDArch.GFX_1250, "mxfp4", scaled=True)
    assert scaled.mnemonic == "V_WMMA_SCALE_F32_16X16X128_F8F6F4"
    assert scaled.requires_scale_operand
    assert scaled.compiler_state == "planned_gated"


def test_cdna5_int4_dot_does_not_imply_int4_wmma() -> None:
    row = amd_dtype_contract(AMDArch.GFX_1250, "int4")
    assert row.scalar_vector == "artifact_only"
    assert row.dense_matrix == "unsupported"
    with pytest.raises(TesseraROCmTargetError, match="no matrix path for int4"):
        select_amd_matrix_instruction(AMDArch.GFX_1250, "int4")


def test_selected_rdna_ops_exist_in_machine_readable_archives() -> None:
    for arch, version, cases in (
        (AMDArch.GFX_1151, "rdna35", (("fp16", None, False), ("int4", None, False))),
        (AMDArch.GFX_1200, "rdna4", (("fp16", None, True), ("fp8_e4m3", "fp8_e5m2", False))),
        (AMDArch.GFX_1201, "rdna4", (("fp16", None, True), ("fp8_e4m3", "fp8_e5m2", False))),
    ):
        rows = json.loads((ROOT / f"docs/reference/isa/rdna/{version}/instructions.json").read_text())
        names = {row["name"] for row in rows}
        for dtype, other, sparse in cases:
            selected = select_amd_matrix_instruction(
                arch, dtype, other, sparse=sparse)
            assert selected.mnemonic in names


def test_selected_cdna5_ops_exist_in_reviewed_primary_source_index() -> None:
    index = (ROOT / "docs/reference/isa/PRIMARY_SOURCES_INDEX.md").read_text()
    cases = (
        select_amd_matrix_instruction(AMDArch.GFX_1250, "fp16"),
        select_amd_matrix_instruction(AMDArch.GFX_1250, "int8", sparse=True),
        select_amd_matrix_instruction(AMDArch.GFX_1250, "mxfp4", scaled=True),
    )
    for selected in cases:
        assert selected.mnemonic.removeprefix("V_") in index
